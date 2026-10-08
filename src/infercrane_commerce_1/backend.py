"""Content-bound native AutoJev backend for Commerce-1.

The backend never resolves a Hub model identifier. It accepts only a complete
local checkpoint, a self-hashed per-primitive calibration receipt, the exact
bundled AutoJev source closure, and a local evidence file. ``runtime.json`` in
``model_dir`` binds those bytes before the model implementation is imported.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Protocol, cast

from .runtime import (
    MAX_INPUT_TOKENS,
    MODEL_ID,
    QUESTION_MICROBATCH_SIZE,
    ContextLimitExceededError,
    DecisionRuntime,
    ModelIdentity,
    RuntimePrediction,
    RuntimeUnavailableError,
)

RUNTIME_SCHEMA = "infercrane-commerce-1-native-autojev-runtime/v1"
CALIBRATION_SCHEMA = "infercrane-commerce-1-v3-soft-calibration/v1"
BACKEND_ID = "autojev-native-v1"
BASE_MODEL = "Qwen/Qwen3.8-27B"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
RUNTIME_MANIFEST = "runtime.json"
CALIBRATION_METHOD = (
    "bounded per-primitive temperature scaling on calibration-only soft targets"
)
PRIMITIVES = ("choice", "noul", "score")
SOURCE_CLOSURE = (
    "autojev/__init__.py",
    "autojev/model.py",
    "autojev/types.py",
)
REQUIRED_CHECKPOINT_FILES = {
    "chat_template.jinja",
    "config.json",
    "decision_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "readout.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
}
ARTIFACT_TRANSFORM_SCHEMA = "infercrane-commerce-1-v3-checkpoint-transform/v1"
ARTIFACT_TRANSFORM_STATUS = "PASS_HERMETIC_CHECKPOINT_TRANSFORM"
DECISION_CONFIG_FIELDS = {
    "format_version",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
}


class _PreparedBatch(Protocol):
    counts: Sequence[int]
    input_tokens: int


class _Tensor(Protocol):
    def __truediv__(self, value: float) -> _Tensor: ...

    def softmax(self, dimension: int) -> _Tensor: ...

    def detach(self) -> _Tensor: ...

    def cpu(self) -> _Tensor: ...

    def tolist(self) -> list[list[float]]: ...


class _AutoJevModel(Protocol):
    def prepare(
        self, rows: Sequence[Mapping[str, Any]], *, max_length: int
    ) -> _PreparedBatch: ...

    def __call__(self, batch: _PreparedBatch) -> _Tensor: ...


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise RuntimeUnavailableError(
            "runtime identity is not canonical JSON"
        ) from error
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RuntimeUnavailableError("runtime file cannot be read") from error
    return digest.hexdigest()


def _digest(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeUnavailableError(f"{description} must be a lowercase SHA-256")
    return value


def _exact(value: Mapping[str, Any], fields: set[str], description: str) -> None:
    if set(value) != fields:
        raise RuntimeUnavailableError(f"{description} fields changed")


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeUnavailableError(f"{description} must be an object")
    return cast(Mapping[str, Any], value)


def _json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeUnavailableError(f"{description} is unavailable") from error
    if not isinstance(value, dict):
        raise RuntimeUnavailableError(f"{description} must be an object")
    return value


def _safe_root(path: Path, description: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeUnavailableError(f"{description} must be a local directory")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise RuntimeUnavailableError(f"{description} cannot be resolved") from error


def _safe_relative_path(root: Path, value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeUnavailableError(f"{description} must be a relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise RuntimeUnavailableError(f"{description} must be a safe relative path")
    candidate = root.joinpath(*pure.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise RuntimeUnavailableError(
            f"{description} escapes its package root"
        ) from error
    current = candidate
    while current != root:
        if current.is_symlink():
            raise RuntimeUnavailableError(f"{description} cannot contain symlinks")
        current = current.parent
    return resolved


def _regular_file(root: Path, value: Any, description: str) -> Path:
    path = _safe_relative_path(root, value, description)
    if path.is_symlink() or not path.is_file():
        raise RuntimeUnavailableError(f"{description} must be a regular file")
    return path


def _file_manifest(root: Path, description: str) -> dict[str, str]:
    files: dict[str, str] = {}
    try:
        entries = sorted(root.rglob("*"))
    except OSError as error:
        raise RuntimeUnavailableError(f"{description} cannot be inventoried") from error
    for path in entries:
        if path.is_symlink():
            raise RuntimeUnavailableError(f"{description} cannot contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _file_sha256(path)
        elif not path.is_dir():
            raise RuntimeUnavailableError(f"{description} contains a special file")
    if not files:
        raise RuntimeUnavailableError(f"{description} is empty")
    return files


def _manifest_files(value: Any, description: str) -> dict[str, str]:
    raw = _mapping(value, description)
    result: dict[str, str] = {}
    for name, digest in raw.items():
        if not isinstance(name, str):
            raise RuntimeUnavailableError(f"{description} has a non-string path")
        pure = PurePosixPath(name)
        if not name or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise RuntimeUnavailableError(f"{description} has an unsafe path")
        result[name] = _digest(digest, f"{description}[{name!r}]")
    if not result:
        raise RuntimeUnavailableError(f"{description} is empty")
    return result


def _verify_checkpoint(root: Path, receipt: Mapping[str, Any]) -> tuple[Path, str]:
    _exact(receipt, {"path", "artifact_sha256", "files"}, "checkpoint binding")
    checkpoint = _safe_relative_path(root, receipt.get("path"), "checkpoint path")
    if not checkpoint.is_dir():
        raise RuntimeUnavailableError("checkpoint path must be a directory")
    expected = _manifest_files(receipt.get("files"), "checkpoint file manifest")
    observed = _file_manifest(checkpoint, "checkpoint")
    if observed != expected:
        raise RuntimeUnavailableError("checkpoint file manifest changed")
    if not set(observed) >= REQUIRED_CHECKPOINT_FILES or not any(
        name.startswith("model-") and name.endswith(".safetensors") for name in observed
    ):
        raise RuntimeUnavailableError(
            "checkpoint is not a complete native AutoJev layout"
        )
    artifact_sha256 = _digest(
        receipt.get("artifact_sha256"), "checkpoint artifact identity"
    )
    if _canonical_sha256(observed) != artifact_sha256:
        raise RuntimeUnavailableError("checkpoint artifact identity changed")
    decision_config = _json_object(
        checkpoint / "decision_config.json", "decision checkpoint config"
    )
    if (
        set(decision_config) != DECISION_CONFIG_FIELDS
        or decision_config.get("format_version") != 1
        or decision_config.get("base_model") != BASE_MODEL
        or decision_config.get("revision") != BASE_REVISION
        or decision_config.get("attention_mode", "causal")
        not in {"causal", "noncausal_full_attention"}
        or decision_config.get("pooling", "last") not in {"last", "mean"}
    ):
        raise RuntimeUnavailableError("decision checkpoint identity changed")
    codes = decision_config.get("codes")
    token_ids = decision_config.get("token_ids")
    temperature = decision_config.get("temperature")
    if (
        not isinstance(codes, list)
        or not codes
        or any(not isinstance(code, str) or not code for code in codes)
        or len(set(codes)) != len(codes)
        or not isinstance(token_ids, list)
        or len(token_ids) != len(codes)
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        )
        or len(set(token_ids)) != len(token_ids)
        or isinstance(temperature, bool)
        or not isinstance(temperature, int | float)
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise RuntimeUnavailableError("decision checkpoint runtime fields changed")
    return checkpoint, artifact_sha256


def _transform_inventory(value: Any, description: str) -> dict[str, dict[str, Any]]:
    raw = _mapping(value, description)
    result: dict[str, dict[str, Any]] = {}
    for name, record_value in raw.items():
        if not isinstance(name, str):
            raise RuntimeUnavailableError(f"{description} has a non-string path")
        pure = PurePosixPath(name)
        if not name or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise RuntimeUnavailableError(f"{description} has an unsafe path")
        record = _mapping(record_value, f"{description}[{name!r}]")
        _exact(record, {"sha256", "size_bytes"}, f"{description}[{name!r}]")
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RuntimeUnavailableError(f"{description}[{name!r}] size changed")
        result[name] = {
            "sha256": _digest(
                record.get("sha256"), f"{description}[{name!r}] identity"
            ),
            "size_bytes": size,
        }
    if not result:
        raise RuntimeUnavailableError(f"{description} is empty")
    return result


def _artifact_from_transform_inventory(
    inventory: Mapping[str, Mapping[str, Any]],
) -> str:
    return _canonical_sha256(
        {name: record["sha256"] for name, record in sorted(inventory.items())}
    )


def _verify_artifact_transform(
    root: Path,
    receipt: Mapping[str, Any],
    *,
    checkpoint: Path,
    release_artifact_sha256: str,
) -> tuple[str, str]:
    _exact(
        receipt,
        {
            "path",
            "sha256",
            "transform_sha256",
            "evaluated_artifact_sha256",
            "release_artifact_sha256",
        },
        "artifact transform binding",
    )
    path = _regular_file(root, receipt.get("path"), "artifact transform path")
    if _file_sha256(path) != _digest(
        receipt.get("sha256"), "artifact transform file identity"
    ):
        raise RuntimeUnavailableError("artifact transform file identity changed")
    transform = _json_object(path, "artifact transform attestation")
    _exact(
        transform,
        {
            "schema_version",
            "status",
            "decision_config_file",
            "retained_decision_config_fields",
            "removed_decision_config_keys",
            "original",
            "release",
            "unchanged_files",
            "attestation_sha256",
        },
        "artifact transform attestation",
    )
    transform_sha256 = _digest(
        transform.get("attestation_sha256"), "artifact transform identity"
    )
    transform_body = {
        key: value for key, value in transform.items() if key != "attestation_sha256"
    }
    removed = transform.get("removed_decision_config_keys")
    if (
        transform.get("schema_version") != ARTIFACT_TRANSFORM_SCHEMA
        or transform.get("status") != ARTIFACT_TRANSFORM_STATUS
        or transform.get("decision_config_file") != "decision_config.json"
        or transform.get("retained_decision_config_fields")
        != [
            "format_version",
            "base_model",
            "revision",
            "codes",
            "token_ids",
            "temperature",
            "attention_mode",
            "pooling",
        ]
        or not isinstance(removed, list)
        or removed != sorted(removed)
        or any(not isinstance(name, str) or not name for name in removed)
        or _canonical_sha256(transform_body) != transform_sha256
        or receipt.get("transform_sha256") != transform_sha256
    ):
        raise RuntimeUnavailableError("artifact transform attestation changed")

    sides: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
    for side_name in ("original", "release"):
        side = _mapping(transform.get(side_name), f"artifact transform {side_name}")
        _exact(
            side,
            {"artifact_sha256", "decision_config_sha256", "files"},
            f"artifact transform {side_name}",
        )
        inventory = _transform_inventory(
            side.get("files"), f"artifact transform {side_name} files"
        )
        artifact = _digest(
            side.get("artifact_sha256"),
            f"artifact transform {side_name} artifact",
        )
        if _artifact_from_transform_inventory(inventory) != artifact or side.get(
            "decision_config_sha256"
        ) != inventory.get("decision_config.json", {}).get("sha256"):
            raise RuntimeUnavailableError(
                f"artifact transform {side_name} identity changed"
            )
        sides[side_name] = (artifact, inventory)

    evaluated_artifact_sha256, original_files = sides["original"]
    observed_release_artifact, release_files = sides["release"]
    observed_checkpoint = _file_manifest(checkpoint, "release checkpoint")
    observed_records = {
        name: {
            "sha256": digest,
            "size_bytes": (checkpoint / name).stat().st_size,
        }
        for name, digest in observed_checkpoint.items()
    }
    unchanged = _transform_inventory(
        transform.get("unchanged_files"), "artifact transform unchanged files"
    )
    expected_unchanged = {
        name: record
        for name, record in release_files.items()
        if name != "decision_config.json"
    }
    if (
        release_files != observed_records
        or unchanged != expected_unchanged
        or set(original_files) != set(release_files)
        or any(original_files[name] != record for name, record in unchanged.items())
        or observed_release_artifact != release_artifact_sha256
        or receipt.get("release_artifact_sha256") != release_artifact_sha256
        or receipt.get("evaluated_artifact_sha256") != evaluated_artifact_sha256
    ):
        raise RuntimeUnavailableError("artifact transform bytes changed")
    return evaluated_artifact_sha256, transform_sha256


def _finite_number(value: Any, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeUnavailableError(f"{description} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise RuntimeUnavailableError(f"{description} is outside its finite bound")
    return result


def _validate_metrics(value: Any, description: str) -> int:
    metrics = _mapping(value, description)
    _exact(
        metrics,
        {"count", "soft_nll", "soft_brier", "hard_accuracy", "hard_ece"},
        description,
    )
    count = metrics.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise RuntimeUnavailableError(f"{description}.count must be positive")
    for field in ("soft_nll", "soft_brier", "hard_accuracy", "hard_ece"):
        number = _finite_number(metrics.get(field), f"{description}.{field}")
        if number < 0 or (field in {"hard_accuracy", "hard_ece"} and number > 1):
            raise RuntimeUnavailableError(f"{description}.{field} is outside its bound")
    return count


def _verify_calibration(
    root: Path, receipt: Mapping[str, Any], *, artifact_sha256: str
) -> tuple[dict[str, float], str]:
    _exact(
        receipt,
        {"path", "file_sha256", "calibration_sha256"},
        "calibration binding",
    )
    path = _regular_file(root, receipt.get("path"), "calibration path")
    if _file_sha256(path) != _digest(
        receipt.get("file_sha256"), "calibration file identity"
    ):
        raise RuntimeUnavailableError("calibration file identity changed")
    calibration = _json_object(path, "calibration receipt")
    _exact(
        calibration,
        {
            "schema_version",
            "method",
            "rows",
            "primitives",
            "artifact_sha256",
            "input_sha256",
            "raw_logits_sha256",
            "calibration_sha256",
        },
        "calibration receipt",
    )
    calibration_sha256 = _digest(
        calibration.get("calibration_sha256"), "calibration identity"
    )
    body = {
        key: value for key, value in calibration.items() if key != "calibration_sha256"
    }
    if (
        calibration.get("schema_version") != CALIBRATION_SCHEMA
        or calibration.get("method") != CALIBRATION_METHOD
        or calibration.get("artifact_sha256") != artifact_sha256
        or _canonical_sha256(body) != calibration_sha256
        or receipt.get("calibration_sha256") != calibration_sha256
    ):
        raise RuntimeUnavailableError("calibration identity or lineage changed")
    _digest(calibration.get("input_sha256"), "calibration input identity")
    _digest(calibration.get("raw_logits_sha256"), "calibration logits identity")
    primitives = _mapping(calibration.get("primitives"), "calibration primitives")
    if set(primitives) != set(PRIMITIVES):
        raise RuntimeUnavailableError("calibration primitive coverage changed")
    temperatures: dict[str, float] = {}
    total = 0
    for primitive in PRIMITIVES:
        fitted = _mapping(primitives[primitive], f"calibration {primitive}")
        _exact(
            fitted,
            {"temperature", "untempered", "calibrated"},
            f"calibration {primitive}",
        )
        temperatures[primitive] = _finite_number(
            fitted.get("temperature"),
            f"calibration {primitive} temperature",
            positive=True,
        )
        if not 0.05 <= temperatures[primitive] <= 20.0:
            raise RuntimeUnavailableError(
                f"calibration {primitive} temperature is outside its fitted bound"
            )
        before = _validate_metrics(
            fitted.get("untempered"), f"calibration {primitive}.untempered"
        )
        after = _validate_metrics(
            fitted.get("calibrated"), f"calibration {primitive}.calibrated"
        )
        if before != after:
            raise RuntimeUnavailableError("calibration row counts changed")
        if (
            float(fitted["calibrated"]["soft_nll"])
            > float(fitted["untempered"]["soft_nll"]) + 1e-12
        ):
            raise RuntimeUnavailableError("calibration soft NLL regressed")
        total += after
    rows = calibration.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows != total:
        raise RuntimeUnavailableError("calibration total row count changed")
    return temperatures, calibration_sha256


def _verify_source(root: Path, receipt: Mapping[str, Any]) -> Path:
    _exact(receipt, {"path", "files", "files_sha256"}, "source binding")
    source = _safe_relative_path(root, receipt.get("path"), "source path")
    if not source.is_dir():
        raise RuntimeUnavailableError("source path must be a directory")
    expected = _manifest_files(receipt.get("files"), "source file manifest")
    if set(expected) != set(SOURCE_CLOSURE):
        raise RuntimeUnavailableError("AutoJev source closure changed")
    # Inventory the entire import root, not only the three expected closure
    # files.  The directory is inserted into ``sys.path`` below, so an
    # unbound sibling such as ``torch.py`` or ``transformers.py`` would
    # otherwise execute before the content-bound AutoJev module is imported.
    # ``_file_manifest`` also rejects every symlink and special file.
    observed = _file_manifest(source, "AutoJev source")
    if observed != expected or _canonical_sha256(observed) != _digest(
        receipt.get("files_sha256"), "source closure identity"
    ):
        raise RuntimeUnavailableError("AutoJev source closure identity changed")
    return source


def _verify_evidence(root: Path, receipt: Mapping[str, Any]) -> str:
    fields = set(receipt)
    if fields not in (
        {"path", "sha256"},
        {"path", "sha256", "attestation_sha256"},
    ):
        raise RuntimeUnavailableError("evidence binding fields changed")
    path = _regular_file(root, receipt.get("path"), "evidence path")
    expected = _digest(receipt.get("sha256"), "evidence identity")
    if _file_sha256(path) != expected:
        raise RuntimeUnavailableError("evidence identity changed")
    attestation_sha256 = receipt.get("attestation_sha256")
    if attestation_sha256 is not None:
        expected_attestation = _digest(
            attestation_sha256, "evidence attestation identity"
        )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeUnavailableError(
                "evidence attestation is not valid JSON"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeUnavailableError("evidence attestation must be an object")
        supplied = value.get("attestation_sha256")
        body = {key: item for key, item in value.items() if key != "attestation_sha256"}
        if supplied != expected_attestation or _canonical_sha256(body) != supplied:
            raise RuntimeUnavailableError("evidence attestation changed")
    return expected


def _module_path(module: ModuleType, description: str) -> Path:
    raw = getattr(module, "__file__", None)
    if not isinstance(raw, str):
        raise RuntimeUnavailableError(f"{description} has no local source file")
    try:
        return Path(raw).resolve(strict=True)
    except OSError as error:
        raise RuntimeUnavailableError(
            f"{description} source cannot be resolved"
        ) from error


def _import_decision_model(source_root: Path) -> type[Any]:
    expected_package = source_root / "autojev" / "__init__.py"
    expected_model = source_root / "autojev" / "model.py"
    expected_types = source_root / "autojev" / "types.py"
    loaded_package = sys.modules.get("autojev")
    if (
        loaded_package is not None
        and _module_path(loaded_package, "loaded AutoJev package") != expected_package
    ):
        raise RuntimeUnavailableError("another AutoJev source is already loaded")
    loaded_model = sys.modules.get("autojev.model")
    loaded_types = sys.modules.get("autojev.types")
    if (
        loaded_types is not None
        and _module_path(loaded_types, "loaded AutoJev types") != expected_types
    ):
        raise RuntimeUnavailableError("another AutoJev types source is already loaded")
    if loaded_model is not None:
        if _module_path(loaded_model, "loaded AutoJev model") != expected_model:
            raise RuntimeUnavailableError(
                "another AutoJev model source is already loaded"
            )
        decision_model = getattr(loaded_model, "DecisionModel", None)
        if not isinstance(decision_model, type):
            raise RuntimeUnavailableError("AutoJev DecisionModel is unavailable")
        imported_types = sys.modules.get("autojev.types")
        if (
            imported_types is None
            or _module_path(imported_types, "AutoJev types") != expected_types
        ):
            raise RuntimeUnavailableError(
                "AutoJev types import escaped its bound source root"
            )
        return decision_model

    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module("autojev.model")
    except Exception as error:
        raise RuntimeUnavailableError(
            "AutoJev runtime could not be imported"
        ) from error
    finally:
        with suppress(ValueError):
            sys.path.remove(str(source_root))
    if _module_path(module, "AutoJev model") != expected_model:
        raise RuntimeUnavailableError("AutoJev import escaped its bound source root")
    imported_package = sys.modules.get("autojev")
    imported_types = sys.modules.get("autojev.types")
    if (
        imported_package is None
        or _module_path(imported_package, "AutoJev package") != expected_package
        or imported_types is None
        or _module_path(imported_types, "AutoJev types") != expected_types
    ):
        raise RuntimeUnavailableError("AutoJev import escaped its bound source root")
    decision_model = getattr(module, "DecisionModel", None)
    if not isinstance(decision_model, type):
        raise RuntimeUnavailableError("AutoJev DecisionModel is unavailable")
    return decision_model


def _question_options(question: Mapping[str, Any]) -> int:
    primitive = question.get("type")
    criteria = question.get("criteria")
    if (primitive == "choice" and isinstance(criteria, Mapping)) or (
        primitive == "score"
        and isinstance(criteria, Sequence)
        and not isinstance(criteria, str | bytes)
    ):
        count = len(criteria)
    elif primitive == "noul":
        count = 2
    else:
        raise RuntimeUnavailableError("runtime received an unsupported question")
    if not 2 <= count <= 255:
        raise RuntimeUnavailableError("runtime question option count is invalid")
    return count


class NativeAutoJevRuntime:
    """Exact native checkpoint runtime with per-primitive calibration."""

    def __init__(
        self,
        *,
        model: _AutoJevModel,
        identity: ModelIdentity,
        temperatures: Mapping[str, float],
        batch_size: int = QUESTION_MICROBATCH_SIZE,
    ) -> None:
        if set(temperatures) != set(PRIMITIVES):
            raise RuntimeUnavailableError("runtime calibration coverage changed")
        if batch_size != QUESTION_MICROBATCH_SIZE:
            raise RuntimeUnavailableError(
                "runtime question microbatch differs from the qualified value"
            )
        self._model: _AutoJevModel | None = model
        self.identity = identity
        self._temperatures = dict(temperatures)
        self._batch_size = batch_size

    def predict(
        self,
        *,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> RuntimePrediction:
        model = self._model
        if model is None:
            raise RuntimeUnavailableError("runtime is closed")
        if not questions:
            raise RuntimeUnavailableError("runtime received no questions")
        grouped: dict[str, list[tuple[str, dict[str, Any], int]]] = {
            primitive: [] for primitive in PRIMITIVES
        }
        for question_id, supplied in questions.items():
            if not isinstance(question_id, str) or not isinstance(supplied, Mapping):
                raise RuntimeUnavailableError("runtime question shape is invalid")
            question = dict(supplied)
            primitive = question.get("type")
            if primitive not in PRIMITIVES:
                raise RuntimeUnavailableError(
                    "runtime received an unsupported question"
                )
            grouped[str(primitive)].append(
                (question_id, question, _question_options(question))
            )

        predicted: dict[str, list[float]] = {}
        input_tokens = 0
        try:
            for primitive in PRIMITIVES:
                entries = grouped[primitive]
                for start in range(0, len(entries), self._batch_size):
                    chunk = entries[start : start + self._batch_size]
                    rows = [
                        {"state": state, "question": question}
                        for _, question, _ in chunk
                    ]
                    prepared = model.prepare(rows, max_length=MAX_INPUT_TOKENS)
                    counts = list(prepared.counts)
                    expected = [count for _, _, count in chunk]
                    if counts != expected:
                        raise RuntimeUnavailableError(
                            "AutoJev changed question option coverage"
                        )
                    tokens = prepared.input_tokens
                    if (
                        isinstance(tokens, bool)
                        or not isinstance(tokens, int)
                        or tokens < 0
                    ):
                        raise RuntimeUnavailableError(
                            "AutoJev returned an invalid input token count"
                        )
                    input_tokens += tokens
                    tensor = model(prepared)
                    values = (
                        (tensor / self._temperatures[primitive])
                        .softmax(-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    if len(values) != len(chunk):
                        raise RuntimeUnavailableError(
                            "AutoJev changed question batch coverage"
                        )
                    for (question_id, _, count), distribution in zip(
                        chunk, values, strict=True
                    ):
                        if len(distribution) < count:
                            raise RuntimeUnavailableError(
                                "AutoJev returned too few option probabilities"
                            )
                        selected = [float(value) for value in distribution[:count]]
                        if any(
                            not math.isfinite(value) or value < 0 for value in selected
                        ) or not math.isclose(
                            math.fsum(selected), 1.0, rel_tol=0.0, abs_tol=1e-5
                        ):
                            raise RuntimeUnavailableError(
                                "AutoJev returned an invalid probability distribution"
                            )
                        predicted[question_id] = selected
        except RuntimeUnavailableError:
            raise
        except ValueError as error:
            expected = (
                f"Question branch exceeds the {MAX_INPUT_TOKENS}-token limit; "
                "no input was truncated."
            )
            if str(error) == expected:
                raise ContextLimitExceededError(
                    f"request exceeds the {MAX_INPUT_TOKENS}-token context limit"
                ) from error
            raise RuntimeUnavailableError("AutoJev inference failed") from error
        except Exception as error:
            raise RuntimeUnavailableError("AutoJev inference failed") from error
        if set(predicted) != set(questions):
            raise RuntimeUnavailableError("AutoJev changed question coverage")
        return RuntimePrediction(
            distributions={
                question_id: predicted[question_id] for question_id in questions
            },
            input_tokens=input_tokens,
        )

    def close(self) -> None:
        model = self._model
        self._model = None
        close = getattr(model, "close", None)
        if callable(close):
            close()


def create_runtime(
    *, model_dir: Path, base_dir: Path, offline: bool
) -> DecisionRuntime:
    """Validate and load one exact local native AutoJev release package."""

    if not offline:
        raise RuntimeUnavailableError("native Commerce-1 serving requires offline mode")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeUnavailableError("native AutoJev serving requires Python 3.12.x")
    model_root = _safe_root(model_dir, "model directory")
    source_package_root = _safe_root(base_dir, "base directory")
    manifest_path = model_root / RUNTIME_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RuntimeUnavailableError(
            "no qualified native runtime manifest is available"
        )
    manifest = _json_object(manifest_path, "native runtime manifest")
    _exact(
        manifest,
        {
            "schema_version",
            "model",
            "backend",
            "checkpoint",
            "artifact_transform",
            "calibration",
            "source",
            "evidence",
            "runtime_identity_sha256",
        },
        "native runtime manifest",
    )
    runtime_identity_sha256 = _digest(
        manifest.get("runtime_identity_sha256"), "runtime identity"
    )
    manifest_body = {
        key: value
        for key, value in manifest.items()
        if key != "runtime_identity_sha256"
    }
    if (
        manifest.get("schema_version") != RUNTIME_SCHEMA
        or manifest.get("model") != MODEL_ID
        or manifest.get("backend") != BACKEND_ID
        or _canonical_sha256(manifest_body) != runtime_identity_sha256
    ):
        raise RuntimeUnavailableError("native runtime identity changed")

    checkpoint, artifact_sha256 = _verify_checkpoint(
        model_root, _mapping(manifest.get("checkpoint"), "checkpoint binding")
    )
    evaluated_artifact_sha256, _transform_sha256 = _verify_artifact_transform(
        model_root,
        _mapping(manifest.get("artifact_transform"), "artifact transform binding"),
        checkpoint=checkpoint,
        release_artifact_sha256=artifact_sha256,
    )
    temperatures, _calibration_sha256 = _verify_calibration(
        model_root,
        _mapping(manifest.get("calibration"), "calibration binding"),
        artifact_sha256=evaluated_artifact_sha256,
    )
    source_root = _verify_source(
        source_package_root, _mapping(manifest.get("source"), "source binding")
    )
    evidence_sha256 = _verify_evidence(
        model_root, _mapping(manifest.get("evidence"), "evidence binding")
    )

    # These flags are set before importing Transformers. The model constructor
    # receives a verified local directory, never a repository identifier.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    decision_model = _import_decision_model(source_root)
    try:
        model = cast(
            _AutoJevModel,
            decision_model(checkpoint=checkpoint, train=False, device=None),
        )
    except Exception as error:
        raise RuntimeUnavailableError(
            "native AutoJev checkpoint could not be loaded"
        ) from error
    identity = ModelIdentity(
        model=MODEL_ID,
        revision=artifact_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        evidence_sha256=evidence_sha256,
        backend=BACKEND_ID,
    )
    return NativeAutoJevRuntime(
        model=model,
        identity=identity,
        temperatures=temperatures,
    )
