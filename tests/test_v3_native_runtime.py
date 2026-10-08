from __future__ import annotations

import json
import math
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from infercrane_commerce_1 import backend, server
from infercrane_commerce_1.runtime import (
    API_SCHEMA_VERSION,
    MAX_INPUT_TOKENS,
    MODEL_ID,
    QUESTION_MICROBATCH_SIZE,
    RELEASE_STATUS,
    ContextLimitExceededError,
    ModelIdentity,
    RuntimePrediction,
    RuntimeUnavailableError,
)


def _write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _json(path: Path, value: dict[str, Any]) -> None:
    _write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def _metrics() -> dict[str, int | float]:
    return {
        "count": 1,
        "soft_nll": 0.5,
        "soft_brier": 0.2,
        "hard_accuracy": 1.0,
        "hard_ece": 0.1,
    }


def _package(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "immutable-v3-package"
    checkpoint = root / "model"
    source = root / "source"
    for name, value in {
        "chat_template.jinja": b"template",
        "config.json": b"{}\n",
        "decision_config.json": (
            json.dumps(
                {
                    "format_version": 1,
                    "base_model": backend.BASE_MODEL,
                    "revision": backend.BASE_REVISION,
                    "codes": ["A", "B", "C", "D"],
                    "token_ids": [1, 2, 3, 4],
                    "temperature": 1.0,
                    "attention_mode": "noncausal_full_attention",
                    "pooling": "last",
                }
            )
            + "\n"
        ).encode(),
        "model-00001-of-00001.safetensors": b"weights",
        "model.safetensors.index.json": b"{}\n",
        "processor_config.json": b"{}\n",
        "readout.safetensors": b"readout",
        "tokenizer.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
    }.items():
        _write(checkpoint / name, value)
    for name, value in {
        "autojev/__init__.py": b"",
        "autojev/model.py": b"class DecisionModel: ...\n",
        "autojev/types.py": b"DecisionInput = dict\n",
    }.items():
        _write(source / name, value)

    checkpoint_files = backend._file_manifest(checkpoint, "test checkpoint")
    artifact_sha256 = backend._canonical_sha256(checkpoint_files)
    checkpoint_records = {
        name: {
            "sha256": digest,
            "size_bytes": (checkpoint / name).stat().st_size,
        }
        for name, digest in checkpoint_files.items()
    }
    unchanged_files = {
        name: record
        for name, record in checkpoint_records.items()
        if name != "decision_config.json"
    }
    transform_body = {
        "schema_version": backend.ARTIFACT_TRANSFORM_SCHEMA,
        "status": backend.ARTIFACT_TRANSFORM_STATUS,
        "decision_config_file": "decision_config.json",
        "retained_decision_config_fields": [
            "format_version",
            "base_model",
            "revision",
            "codes",
            "token_ids",
            "temperature",
            "attention_mode",
            "pooling",
        ],
        "removed_decision_config_keys": [],
        "original": {
            "artifact_sha256": artifact_sha256,
            "decision_config_sha256": checkpoint_files["decision_config.json"],
            "files": checkpoint_records,
        },
        "release": {
            "artifact_sha256": artifact_sha256,
            "decision_config_sha256": checkpoint_files["decision_config.json"],
            "files": checkpoint_records,
        },
        "unchanged_files": unchanged_files,
    }
    transform = {
        **transform_body,
        "attestation_sha256": backend._canonical_sha256(transform_body),
    }
    transform_path = root / "evidence/checkpoint-transform-attestation.json"
    _json(transform_path, transform)
    calibration_body = {
        "schema_version": backend.CALIBRATION_SCHEMA,
        "method": backend.CALIBRATION_METHOD,
        "rows": 3,
        "primitives": {
            primitive: {
                "temperature": temperature,
                "untempered": _metrics(),
                "calibrated": _metrics(),
            }
            for primitive, temperature in {
                "choice": 2.0,
                "noul": 1.0,
                "score": 0.5,
            }.items()
        },
        "artifact_sha256": artifact_sha256,
        "input_sha256": "a" * 64,
        "raw_logits_sha256": "b" * 64,
    }
    calibration = {
        **calibration_body,
        "calibration_sha256": backend._canonical_sha256(calibration_body),
    }
    calibration_path = root / "calibration/calibration.json"
    evidence_path = root / "evidence/sharded-FINAL.json"
    _json(calibration_path, calibration)
    _json(evidence_path, {"status": "TEST_ONLY_QUALIFIED_FIXTURE"})
    source_files = {
        name: backend._file_sha256(source / name) for name in backend.SOURCE_CLOSURE
    }
    manifest_body = {
        "schema_version": backend.RUNTIME_SCHEMA,
        "model": MODEL_ID,
        "backend": backend.BACKEND_ID,
        "checkpoint": {
            "path": "model",
            "artifact_sha256": artifact_sha256,
            "files": checkpoint_files,
        },
        "artifact_transform": {
            "path": "evidence/checkpoint-transform-attestation.json",
            "sha256": backend._file_sha256(transform_path),
            "transform_sha256": transform["attestation_sha256"],
            "evaluated_artifact_sha256": artifact_sha256,
            "release_artifact_sha256": artifact_sha256,
        },
        "calibration": {
            "path": "calibration/calibration.json",
            "file_sha256": backend._file_sha256(calibration_path),
            "calibration_sha256": calibration["calibration_sha256"],
        },
        "source": {
            "path": "source",
            "files": source_files,
            "files_sha256": backend._canonical_sha256(source_files),
        },
        "evidence": {
            "path": "evidence/sharded-FINAL.json",
            "sha256": backend._file_sha256(evidence_path),
        },
    }
    manifest = {
        **manifest_body,
        "runtime_identity_sha256": backend._canonical_sha256(manifest_body),
    }
    manifest_path = root / backend.RUNTIME_MANIFEST
    _json(manifest_path, manifest)
    return {
        "root": root,
        "checkpoint": checkpoint,
        "source": source,
        "calibration": calibration_path,
        "evidence": evidence_path,
        "manifest": manifest_path,
        "artifact_sha256": artifact_sha256,
        "runtime_identity_sha256": manifest["runtime_identity_sha256"],
        "evidence_sha256": manifest["evidence"]["sha256"],
    }


class FakeTensor:
    def __init__(self, rows: list[list[float]], divisions: list[float]) -> None:
        self.rows = rows
        self.divisions = divisions

    def __truediv__(self, value: float) -> FakeTensor:
        self.divisions.append(value)
        return FakeTensor(
            [[item / value for item in row] for row in self.rows], self.divisions
        )

    def softmax(self, dimension: int) -> FakeTensor:
        assert dimension == -1
        rows = []
        for row in self.rows:
            maximum = max(row)
            values = [math.exp(value - maximum) for value in row]
            total = sum(values)
            rows.append([value / total for value in values])
        return FakeTensor(rows, self.divisions)

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeDecisionModel:
    instances: ClassVar[list[FakeDecisionModel]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.current: list[dict[str, Any]] = []
        self.divisions: list[float] = []
        self.batches: list[list[dict[str, Any]]] = []
        self.max_lengths: list[int] = []
        self.closed = False
        self.__class__.instances.append(self)

    def prepare(
        self, rows: list[dict[str, Any]], *, max_length: int
    ) -> SimpleNamespace:
        self.current = rows
        self.batches.append(rows)
        self.max_lengths.append(max_length)
        return SimpleNamespace(
            counts=[backend._question_options(row["question"]) for row in rows],
            input_tokens=11 * len(rows),
        )

    def __call__(self, _batch: SimpleNamespace) -> FakeTensor:
        logits = {
            "choice": [0.0, 2.0, 0.0],
            "noul": [0.0, 2.0],
            "score": [0.0, 1.0, 2.0],
        }
        return FakeTensor(
            [logits[row["question"]["type"]] for row in self.current],
            self.divisions,
        )

    def close(self) -> None:
        self.closed = True


class WideDecisionModel(FakeDecisionModel):
    def __call__(self, batch: SimpleNamespace) -> FakeTensor:
        return FakeTensor(
            [[float(index) for index in range(count)] for count in batch.counts],
            self.divisions,
        )


def test_native_v3_loads_self_contained_bundle_and_calibrates_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    FakeDecisionModel.instances.clear()
    monkeypatch.setattr(
        backend, "_import_decision_model", lambda _source: FakeDecisionModel
    )
    runtime = backend.create_runtime(
        model_dir=package["root"], base_dir=package["root"], offline=True
    )
    prediction = runtime.predict(
        state={"cart": ["sku-1"]},
        questions={
            "risk": {"type": "score", "criteria": ["low", "mid", "high"]},
            "route": {
                "type": "choice",
                "criteria": {"ship": "Ship", "hold": "Hold", "stop": "Stop"},
            },
            "review": {"type": "noul"},
        },
    )

    model = FakeDecisionModel.instances[0]
    assert model.kwargs == {
        "checkpoint": package["checkpoint"],
        "train": False,
        "device": None,
    }
    assert model.divisions == [2.0, 1.0, 0.5]
    assert model.max_lengths == [MAX_INPUT_TOKENS] * 3
    assert [len(rows) for rows in model.batches] == [QUESTION_MICROBATCH_SIZE] * 3
    assert prediction.input_tokens == 33
    assert runtime.identity.revision == package["artifact_sha256"]
    assert (
        runtime.identity.runtime_identity_sha256 == package["runtime_identity_sha256"]
    )
    assert runtime.identity.evidence_sha256 == package["evidence_sha256"]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    runtime.close()
    assert model.closed is True


def test_native_v3_rejects_unqualified_microbatch_and_maps_exact_context_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    FakeDecisionModel.instances.clear()
    monkeypatch.setattr(
        backend, "_import_decision_model", lambda _source: FakeDecisionModel
    )
    runtime = backend.create_runtime(
        model_dir=package["root"], base_dir=package["root"], offline=True
    )
    with pytest.raises(RuntimeUnavailableError, match="microbatch"):
        backend.NativeAutoJevRuntime(
            model=FakeDecisionModel(),
            identity=runtime.identity,
            temperatures={primitive: 1.0 for primitive in backend.PRIMITIVES},
            batch_size=2,
        )

    def over_context(
        _rows: list[dict[str, Any]], *, max_length: int
    ) -> SimpleNamespace:
        raise ValueError(
            f"Question branch exceeds the {max_length}-token limit; "
            "no input was truncated."
        )

    model = FakeDecisionModel.instances[0]
    monkeypatch.setattr(model, "prepare", over_context)
    with pytest.raises(
        ContextLimitExceededError,
        match=f"{MAX_INPUT_TOKENS}-token context limit",
    ):
        runtime.predict(
            state={"cart": ["sku-1"]},
            questions={"review": {"type": "noul"}},
        )


def test_native_v3_microbatches_multiple_questions_and_preserves_255_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    WideDecisionModel.instances.clear()
    monkeypatch.setattr(
        backend, "_import_decision_model", lambda _source: WideDecisionModel
    )
    runtime = backend.create_runtime(
        model_dir=package["root"], base_dir=package["root"], offline=True
    )
    prediction = runtime.predict(
        state={"cart": ["sku-1"]},
        questions={
            "wide": {
                "type": "choice",
                "criteria": {str(index): f"Option {index}" for index in range(255)},
            },
            "binary": {
                "type": "choice",
                "criteria": {"yes": "Yes", "no": "No"},
            },
        },
    )

    model = WideDecisionModel.instances[0]
    assert [len(rows) for rows in model.batches] == [1, 1]
    assert model.max_lengths == [MAX_INPUT_TOKENS, MAX_INPUT_TOKENS]
    assert list(prediction.distributions) == ["wide", "binary"]
    assert len(prediction.distributions["wide"]) == 255
    assert len(prediction.distributions["binary"]) == 2
    assert math.fsum(prediction.distributions["wide"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("checkpoint", "checkpoint file manifest changed"),
        ("calibration", "calibration file identity changed"),
        ("source", "source closure identity changed"),
        ("evidence", "evidence identity changed"),
    ],
)
def test_native_v3_rejects_changed_bytes_before_model_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    package = _package(tmp_path)
    paths = {
        "checkpoint": package["checkpoint"] / "readout.safetensors",
        "calibration": package["calibration"],
        "source": package["source"] / "autojev/model.py",
        "evidence": package["evidence"],
    }
    paths[target].write_bytes(paths[target].read_bytes() + b"changed")
    imported = False

    def loader(_source: Path) -> type[FakeDecisionModel]:
        nonlocal imported
        imported = True
        return FakeDecisionModel

    monkeypatch.setattr(backend, "_import_decision_model", loader)
    with pytest.raises(RuntimeUnavailableError, match=message):
        backend.create_runtime(
            model_dir=package["root"], base_dir=package["root"], offline=True
        )
    assert imported is False


def test_native_v3_rejects_unbound_source_file_before_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    _write(package["source"] / "torch.py", b"raise RuntimeError('executed')\n")
    imported = False

    def loader(_source: Path) -> type[FakeDecisionModel]:
        nonlocal imported
        imported = True
        return FakeDecisionModel

    monkeypatch.setattr(backend, "_import_decision_model", loader)
    with pytest.raises(
        RuntimeUnavailableError, match="source closure identity changed"
    ):
        backend.create_runtime(
            model_dir=package["root"], base_dir=package["root"], offline=True
        )
    assert imported is False


def test_native_v3_rejects_source_symlink_before_model_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    target = tmp_path / "unbound.py"
    target.write_text("raise RuntimeError('executed')\n")
    (package["source"] / "transformers.py").symlink_to(target)
    imported = False

    def loader(_source: Path) -> type[FakeDecisionModel]:
        nonlocal imported
        imported = True
        return FakeDecisionModel

    monkeypatch.setattr(backend, "_import_decision_model", loader)
    with pytest.raises(RuntimeUnavailableError, match="cannot contain symlinks"):
        backend.create_runtime(
            model_dir=package["root"], base_dir=package["root"], offline=True
        )
    assert imported is False


def test_native_v3_rejects_online_or_resigned_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _package(tmp_path)
    with pytest.raises(RuntimeUnavailableError, match="offline"):
        backend.create_runtime(
            model_dir=package["root"], base_dir=package["root"], offline=False
        )

    manifest = json.loads(package["manifest"].read_text())
    manifest["backend"] = "changed"
    _json(package["manifest"], manifest)
    monkeypatch.setattr(
        backend, "_import_decision_model", lambda _source: FakeDecisionModel
    )
    with pytest.raises(RuntimeUnavailableError, match="runtime identity changed"):
        backend.create_runtime(
            model_dir=package["root"], base_dir=package["root"], offline=True
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.identity = ModelIdentity(
            model=MODEL_ID,
            revision="a" * 64,
            runtime_identity_sha256="b" * 64,
            evidence_sha256="c" * 64,
            backend="fake-runtime-v1",
        )
        self.closed = False

    def predict(self, *, state: Any, questions: dict[str, dict[str, Any]]):
        del state
        values = {
            "choice": [0.1, 0.8, 0.1],
            "noul": [0.2, 0.8],
            "score": [0.02, 0.08, 0.2, 0.7],
        }
        return RuntimePrediction(
            distributions={
                name: values[question["type"]] for name, question in questions.items()
            },
            input_tokens=137,
        )

    def close(self) -> None:
        self.closed = True


def _payload() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "state": {"cart_total": 117, "approved_total": 94},
        "questions": {
            "route": {
                "type": "choice",
                "criteria": {
                    "continue": "Continue.",
                    "review": "Review.",
                    "block": "Stop.",
                },
            },
            "review": {"type": "noul"},
            "risk": {
                "type": "score",
                "criteria": ["low", "guarded", "high", "block"],
            },
        },
    }


def test_v3_systemone_health_and_model_contract() -> None:
    runtime = FakeRuntime()
    app = server.create_app(runtime)
    with TestClient(app) as client:
        health = client.get("/healthz")
        model = client.get("/v1/model")
        response = client.post("/v1/systemone", json=_payload())

    assert health.json() == {"status": "ok"}
    assert model.json()["revision"] == "a" * 64
    assert model.json()["runtime_identity_sha256"] == "b" * 64
    assert model.json()["evidence_sha256"] == "c" * 64
    assert "qualified" not in model.json()
    assert model.json()["qualification"]["status"] == ("external_attestation_required")
    assert model.json()["limits"]["maximum_input_tokens"] == MAX_INPUT_TOKENS
    assert (
        model.json()["limits"]["question_microbatch_size"] == QUESTION_MICROBATCH_SIZE
    )
    assert response.status_code == 200
    result = response.json()
    assert result["schema_version"] == API_SCHEMA_VERSION
    assert result["provider"] == "InferCrane"
    assert result["release_status"] == RELEASE_STATUS
    assert result["answers"]["route"]["choice"] == "review"
    assert result["answers"]["review"] == {"type": "noul", "noul": 0.8}
    assert result["answers"]["risk"]["score"] == pytest.approx(2.58)
    assert result["usage"] == {
        "input_tokens": 137,
        "output_tokens": 0,
        "cost": None,
    }
    assert runtime.closed is True


def test_v3_metadata_is_installable_and_exactly_pinned() -> None:
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    pins = project["project"]["optional-dependencies"]["v3-runtime"]
    requirements = [
        line
        for line in (root / "src/infercrane_commerce_1/requirements.txt")
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert pins == [line.replace('"', "'") for line in requirements]
    assert project["project"]["scripts"] == {
        "commerce-1-v3-serve": "infercrane_commerce_1.server:main"
    }
    assert project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == [
        "src/infercrane_commerce_1",
    ]
