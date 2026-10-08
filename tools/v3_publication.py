#!/usr/bin/env python3
"""Fail-closed Commerce-1 publication planning.

This module performs no network mutation.  It validates one hermetic V3 model
package and its release evidence, creates an InferCrane-only private rehearsal
plan, verifies immutable private GitHub and Hugging Face receipts, and only then
creates a public-transition plan.  Plans are evidence, not publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

PACKAGE_SCHEMA = "infercrane-commerce-1-v3-full-checkpoint-package/v3"
CURRENT_PACKAGE_SCHEMA = "infercrane-commerce-1-full-checkpoint-package/v5"
PACKAGE_STATUS = "STAGED_NOT_UPLOADED"
MODEL_ID = "infercrane/Commerce-1"
QUALIFICATION_SCHEMA = "infercrane-commerce-1-v3-native-runtime-qualification/v2"
CURRENT_QUALIFICATION_SCHEMA = (
    "infercrane-commerce-1-v3-native-runtime-qualification/v4"
)
PUBLIC_QUALIFICATION_SCHEMA = "infercrane-commerce-1-public-runtime-qualification/v2"
PUBLIC_V03_SCHEMA = "infercrane-commerce-1-v3-public-decision-index-0-3/v1"
OFFICIAL_V03_SCHEMA = (
    "infercrane-commerce-1-v3-official-decision-index-0.3-attestation/v1"
)
CURRENT_OFFICIAL_V03_SCHEMA = (
    "infercrane-commerce-1-v3-official-decision-index-0.3-board-evidence/v2"
)
GATE_SCHEMA = "infercrane-commerce-1-v3-publication-gate/v1"
PRIVATE_PLAN_SCHEMA = "infercrane-commerce-1-v3-private-rehearsal/v1"
REMOTE_RECEIPT_SCHEMA = "infercrane-commerce-1-v3-private-remote-verification/v1"
PRIVATE_VERIFICATION_SCHEMA = "infercrane-commerce-1-v3-private-verification/v1"
PUBLIC_PLAN_SCHEMA = "infercrane-commerce-1-v3-public-transition-plan/v1"
OWNER_AUTHORIZATION_SCHEMA = "infercrane-commerce-1-owner-publication-authorization/v2"
TRAINING_EVALUATION_BRIDGE_SCHEMA = (
    "infercrane-commerce-1-v3-training-evaluation-bridge/v1"
)
PROMOTED_FINAL_SCHEMA = "infercrane-commerce-1-v3-sharded-training-final/v1"
PROMOTED_FINAL_STATUS = "SUCCEEDED_SHARDED_EVIDENCE_BOUND"
PROMOTED_AUDIT_SCHEMA = "infercrane-commerce-1-v3-sharded-final-audit/v1"
PROMOTED_AUDIT_STATUS = "PASS_SHARDED_EVIDENCE_LINEAGE"
IDENTITY_SCHEMA_V2 = "infercrane-commerce-1-v3-source-runtime-identity/v2"
TRAINING_DATA_LINEAGE_SCHEMA = "infercrane-commerce-1-training-data-lineage/v2"
PUBLIC_EVALUATION_SCHEMA = "infercrane-commerce-1-public-evaluation/v1"
PUBLIC_EVALUATION_PATH = "evidence/public-evaluation.json"
ARTIFACT_TRANSFORM_PATH = "evidence/checkpoint-transform-attestation.json"
ARTIFACT_TRANSFORM_SCHEMA = "infercrane-commerce-1-v3-checkpoint-transform/v1"
ARTIFACT_TRANSFORM_STATUS = "PASS_HERMETIC_CHECKPOINT_TRANSFORM"
TRAINING_DATA_LINEAGE_PATH = "training-data/lineage.json"
DECISION_CONFIG_FIELDS = (
    "format_version",
    "base_model",
    "revision",
    "codes",
    "token_ids",
    "temperature",
    "attention_mode",
    "pooling",
)
RUNTIME_DEPENDENCY_SCHEMA = "infercrane-commerce-1-public-runtime-dependencies/v1"
PUBLIC_PYPROJECT_SHA256 = (
    "51bfd0132b441fe96b4f4ef174ca5bc830eebef4af35781e0abf176311d49fa9"
)
PUBLIC_UV_LOCK_SHA256 = (
    "94a077404d91de6a00981ca2f9206e44ad4c3da1a0010cb8ad7bd4e5681ae4a2"
)
PUBLIC_REQUIREMENTS_SHA256 = (
    "e7a578acd6f953959c999e97600187d8bfb0f40fb2eeb95a60a82f0ccfb3a829"
)
PUBLIC_RUNTIME_SOURCE_FILES = {
    "__init__.py": "0bc1f77c0220fb620cad20563e4b8095b435e31e5de11f1a477bb142b501d99b",
    "__main__.py": "cdacda4fbd364afed5cf7a1d7c8c745fec4fbeaf636842bbe4f64cfb6e65424b",
    "api.py": "f0a12ca30d4bcf57bcf8728deb5a4b50e733fdf5048296517ffdf39924b3568c",
    "backend.py": "f4e383a731215554be8da6262501fc5550ff92b8dbeca88dbe914d4085bdd968",
    "openai_compat.py": (
        "975089a89e3344504f09cedaa4f405b7071d21f5b68f0360bef847f2425751a0"
    ),
    "py.typed": "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
    "requirements.txt": (
        "e7a578acd6f953959c999e97600187d8bfb0f40fb2eeb95a60a82f0ccfb3a829"
    ),
    "runtime.py": "c8a8e2eec3209cf35f260d32474b33cc8b1b3b96dd64f3d4e79b79740cd109b5",
    "server.py": "6b01946574d4145595c7e8a91041f8e5862e16fe46912d7916574149314376c8",
}
PUBLIC_RUNTIME_SOURCE_SHA256 = (
    "8c1a997842be34a60e6d96b1ba9a786c0387cfcb8d3b0e951cb27cf90b2982b2"
)
RUNTIME_PACKAGE_VERSIONS = {
    "accelerate": "1.15.0",
    "fastapi": "0.141.1",
    "fla-core": "0.5.2",
    "flash-linear-attention": "0.5.2",
    "huggingface-hub": "1.31.0",
    "kernels": "0.16.1",
    "numpy": "2.5.3",
    "pillow": "12.3.0",
    "pydantic": "2.12.4",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.2",
    "torch": "2.14.0",
    "torchvision": "0.29.0",
    "transformers": "5.17.0",
    "triton": "3.8.0",
    "uvicorn": "0.52.4",
}

GITHUB_REPOSITORY = "infercrane/commerce-1"
HF_REPOSITORY = MODEL_ID
OFFICIAL_REPOSITORY = "apolinario/decision-index"
OFFICIAL_BOARD = "multimodalart/jev-decision-index"
KIT_REVISION = "9eb2dbe2a358004c8782c66e40a83ac07b953fec"
# Legacy-only synthetic fixture values. Protected evaluator, authorization,
# provider, and spend identities are deliberately absent from this public repo.
EVALUATOR_ATTEMPT = "0" * 32
EVALUATION_AUTHORIZATION = "0" * 64
EVALUATION_RESERVATION = "0.01"
EVALUATION_CAP = "0.01"
RELEASE_BASELINE_DECISION_INDEX = 52.55
CANONICAL_CONTRACT_SHA256 = (
    "5928baf94f9ee870a259fe56e0827285d529593d680e8255cd6b0731957c6d33"
)
CANONICAL_EVALUATED_ARTIFACT_SHA256 = (
    "b7fba0cceb1b57834fecacfbcb5e361d94fa185e877e891f4e43fc25c6458a1d"
)
# Compatibility name for legacy callers.  This is the evaluated artifact,
# never the metadata-projected release artifact.
CANONICAL_ARTIFACT_SHA256 = CANONICAL_EVALUATED_ARTIFACT_SHA256
CANONICAL_CALIBRATION_FILE_SHA256 = (
    "4164c7a2dee99f02920bde48b3ea989e45ef8abe8de4399c1c67e700dc7edf93"
)
CANONICAL_RELEASE_MANIFEST_SHA256 = "0" * 64
CANONICAL_NATIVE_QUALIFICATION_ATTEMPT = "0" * 32
CANONICAL_NATIVE_QUALIFICATION_FILE_SHA256 = "0" * 64
CANONICAL_NATIVE_QUALIFICATION_RECEIPT_SHA256 = "0" * 64
CANONICAL_NATIVE_RUNTIME_IDENTITY_SHA256 = "0" * 64
CANONICAL_V021_DECISION_INDEX = 60.99
CANONICAL_V021_FINAL_RECEIPT_SHA256 = "0" * 64
CANONICAL_V021_RESULTS_SHA256 = (
    "a8e6241e86614c95d224090bb361c8ec92153256fe2e114ed6e9f6041bc69ad4"
)
CANONICAL_V021_SCORES_SHA256 = (
    "3b7873ef330b1b6d9dfa8a246b028d3dd392adf6eb4d706370aa8e371a5c3765"
)
PUBLIC_RUNTIME_ROOT = "public-runtime"
CURRENT_NOTICE_ROOT = "licenses"
CURRENT_NOTICE_FILES = {
    "training/nvidia-helpsteer2-README.md": (
        "835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b"
    ),
    "training/anthropic-hh-rlhf-README.md": (
        "f75f40db0268656ba07736ec8e59a9720c1910ce554c85354cec74b1c8bda175"
    ),
    "training/allenai-qasc-README.md": (
        "c90ae4776b98cd6fc0a22558784c43b07e1f0b00a090dda99797d6b6efab4a3e"
    ),
    "training/kubernetes-LICENSE": (
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
    ),
    "models/qwen-LICENSE": (
        "689f0c220c9b4e857f35ca7d0dca51397e5a545dc4ad3987438c023e24ff0ab6"
    ),
    "models/pplx-decider-LICENSE": (
        "689f0c220c9b4e857f35ca7d0dca51397e5a545dc4ad3987438c023e24ff0ab6"
    ),
    "models/pplx-decider-NOTICE": (
        "3792cbe9e964a7ed65fff010d3f94bb27c6e726e2940ab6b0e1db5d30df92510"
    ),
    "terms/CC-BY-4.0.txt": (
        "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411"
    ),
    "terms/anthropic-hh-rlhf-MIT-LICENSE": (
        "d4fb18db48757e273261a5597b8a09381de8073da060474ce33d0643e06c875e"
    ),
}
OWNER_RELEASE_CONFIRMATION = "AUTHORIZE_INFERCRANE_COMMERCE_1_V3_RELEASE"
INITIAL_RELEASE_STATE = "INITIAL_PUBLIC_RELEASE_PENDING_OFFICIAL_V0_3"
OFFICIAL_RELEASE_STATE = "OFFICIAL_V0_3_VERIFIED"
# The exact authority statements remain in protected release-control storage.
# Public tooling binds their canonical digest without redistributing operator
# utterances or identities.
OWNER_AUTHORITY_STATEMENTS_SHA256 = (
    "4968d695a7860896816b4c7f39ee2a5c1fdb32102896d41466dc67f5d9b53e2a"
)
BLOCKER_DISPOSITIONS = {
    "six-source-commercial-training-and-redistribution": (
        "RESOLVED_BY_BOUND_OWNER_AUTHORITY"
    ),
    "training-provenance-has-no-legal-conclusion": "ACCEPTED_KNOWN_DISCLOSURE",
    "training-provenance-does-not-authorize-publication": (
        "SUPERSEDED_FOR_THIS_ARTIFACT_BY_OWNER_AUTHORIZATION"
    ),
    "quality-runtime-and-evaluation-evidence": "RESOLVED_BY_BOUND_EVIDENCE",
    "mandatory-license-and-attribution-notices": (
        "RESOLVED_BY_BOUND_PROVENANCE_AND_LICENSES"
    ),
}

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_ATTEMPT = re.compile(r"[0-9a-f]{32}\Z")
_PR_URL = re.compile(
    r"https://github\.com/apolinario/decision-index/pull/[1-9][0-9]*\Z"
)
CURRENT_V03_WEIGHTS = {
    "public": "0.20",
    "equated_private_same_skills": "0.50",
    "equated_private_new_domains": "0.30",
}
AUDITED_V03_EQUATING = {
    "formula_id": "v0.3-final",
    "decided_utc": "2026-10-06",
    "bases": 21,
    "public": {"mean": 23.5, "sd": 16.54},
    "same_skills": {"mean": 24.27, "sd": 16.35},
    "new_domains": {"mean": 23.68, "sd": 12.29},
    "tie_band": 0.9,
    "gap_band": {"bases": 21, "sd": 0.91},
}


class V3PublicationError(ValueError):
    """Publication evidence is absent, mutable, detached, or unsafe."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise V3PublicationError(f"{label} must be a lowercase SHA-256")
    return value


def _revision(value: Any, label: str) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise V3PublicationError(f"{label} must be an immutable lowercase commit")
    return value


def _positive_money(value: Any, label: str) -> Decimal:
    if not isinstance(value, str):
        raise V3PublicationError(f"{label} must be a positive decimal string")
    try:
        amount = Decimal(value)
    except InvalidOperation as error:
        raise V3PublicationError(
            f"{label} must be a positive decimal string"
        ) from error
    if not amount.is_finite() or amount <= 0 or format(amount, "f") != value:
        raise V3PublicationError(f"{label} must be a positive decimal string")
    return amount


def _object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V3PublicationError(f"{label} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V3PublicationError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise V3PublicationError(f"{label} must be a JSON object")
    return value


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    supplied = _sha(value.get(field), f"{label}.{field}")
    body = {key: item for key, item in value.items() if key != field}
    if supplied != _json_sha256(body):
        raise V3PublicationError(f"{label} self-hash changed")
    return supplied


def _safe_relative(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise V3PublicationError(f"{label} must be a relative path")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise V3PublicationError(f"{label} must be a safe relative path")
    return value


def _utc_timestamp(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise V3PublicationError(f"{label} must be an ISO-8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V3PublicationError(
            f"{label} must be an ISO-8601 UTC timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise V3PublicationError(f"{label} must be an ISO-8601 UTC timestamp")


def _regular_inventory(
    root: Path, *, exclude: frozenset[str] = frozenset()
) -> dict[str, dict[str, Any]]:
    if root.is_symlink() or not root.is_dir():
        raise V3PublicationError("package must be a non-symlink directory")
    resolved = root.resolve(strict=True)
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved.rglob("*")):
        name = path.relative_to(resolved).as_posix()
        if path.is_symlink():
            raise V3PublicationError(f"package contains a symlink: {name}")
        if path.is_file():
            if name not in exclude:
                result[name] = {
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
        elif not path.is_dir():
            raise V3PublicationError(f"package contains a special file: {name}")
    return result


def _packaged_file(root: Path, relative: Any, label: str) -> Path:
    name = _safe_relative(relative, label)
    if root.is_symlink() or not root.is_dir():
        raise V3PublicationError(f"{label} package root is not a regular directory")
    resolved_root = root.resolve(strict=True)
    path = root / name
    cursor = root
    for part in PurePosixPath(name).parts:
        cursor /= part
        if cursor.is_symlink():
            raise V3PublicationError(f"{label} crosses a symlink")
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as error:
        raise V3PublicationError(f"{label} is not packaged") from error
    if (
        not resolved_path.is_relative_to(resolved_root)
        or not resolved_path.is_file()
        or not path.is_file()
    ):
        raise V3PublicationError(f"{label} is not packaged")
    return path


def _require_fields(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if set(value) != fields:
        raise V3PublicationError(f"{label} fields changed")


def _transform_inventory(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise V3PublicationError(f"{label} is empty or invalid")
    result: dict[str, dict[str, Any]] = {}
    for name, supplied in value.items():
        _safe_relative(name, f"{label} path")
        if not isinstance(supplied, Mapping) or set(supplied) != {
            "sha256",
            "size_bytes",
        }:
            raise V3PublicationError(f"{label} record changed")
        size = supplied.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise V3PublicationError(f"{label} size changed")
        result[name] = {
            "sha256": _sha(supplied.get("sha256"), f"{label} {name}"),
            "size_bytes": size,
        }
    return result


def _artifact_from_transform_inventory(
    inventory: Mapping[str, Mapping[str, Any]],
) -> str:
    return _json_sha256(
        {name: item["sha256"] for name, item in sorted(inventory.items())}
    )


def _validate_artifact_transform(
    root: Path,
    binding: Any,
    *,
    release_checkpoint: Mapping[str, Mapping[str, Any]],
    release_artifact_sha256: str,
) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise V3PublicationError("V5 artifact transform binding is absent")
    _require_fields(
        binding,
        {
            "path",
            "sha256",
            "transform_sha256",
            "evaluated_artifact_sha256",
            "release_artifact_sha256",
        },
        "V5 artifact transform binding",
    )
    if binding.get("path") != ARTIFACT_TRANSFORM_PATH:
        raise V3PublicationError("V5 artifact transform path changed")
    path = _packaged_file(root, binding.get("path"), "V5 artifact transform")
    if binding.get("sha256") != _file_sha256(path):
        raise V3PublicationError("V5 artifact transform bytes changed")
    attestation = _object(path, "V5 artifact transform")
    _require_fields(
        attestation,
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
        "V5 artifact transform",
    )
    transform_sha256 = _self_hash(
        attestation, "attestation_sha256", "V5 artifact transform"
    )
    removed = attestation.get("removed_decision_config_keys")
    if (
        attestation.get("schema_version") != ARTIFACT_TRANSFORM_SCHEMA
        or attestation.get("status") != ARTIFACT_TRANSFORM_STATUS
        or attestation.get("decision_config_file") != "decision_config.json"
        or attestation.get("retained_decision_config_fields")
        != list(DECISION_CONFIG_FIELDS)
        or not isinstance(removed, list)
        or removed != sorted(removed)
        or any(not isinstance(name, str) or not name for name in removed)
        or set(removed) & set(DECISION_CONFIG_FIELDS)
        or binding.get("transform_sha256") != transform_sha256
    ):
        raise V3PublicationError("V5 artifact transform contract changed")

    sides: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
    for side_name in ("original", "release"):
        side = attestation.get(side_name)
        if not isinstance(side, Mapping):
            raise V3PublicationError(f"V5 artifact transform {side_name} changed")
        _require_fields(
            side,
            {"artifact_sha256", "decision_config_sha256", "files"},
            f"V5 artifact transform {side_name}",
        )
        files = _transform_inventory(
            side.get("files"), f"V5 artifact transform {side_name} files"
        )
        artifact = _sha(
            side.get("artifact_sha256"),
            f"V5 artifact transform {side_name} artifact",
        )
        if _artifact_from_transform_inventory(files) != artifact or side.get(
            "decision_config_sha256"
        ) != files.get("decision_config.json", {}).get("sha256"):
            raise V3PublicationError(
                f"V5 artifact transform {side_name} identity changed"
            )
        sides[side_name] = (artifact, files)

    evaluated_artifact_sha256, original_files = sides["original"]
    observed_release_artifact, release_files = sides["release"]
    unchanged = _transform_inventory(
        attestation.get("unchanged_files"), "V5 artifact transform unchanged files"
    )
    expected_unchanged = {
        name: item
        for name, item in release_files.items()
        if name != "decision_config.json"
    }
    if (
        release_files != dict(release_checkpoint)
        or unchanged != expected_unchanged
        or set(original_files) != set(release_files)
        or any(original_files[name] != item for name, item in unchanged.items())
        or observed_release_artifact != release_artifact_sha256
        or binding.get("release_artifact_sha256") != release_artifact_sha256
        or binding.get("evaluated_artifact_sha256") != evaluated_artifact_sha256
    ):
        raise V3PublicationError("V5 artifact transform identity changed")
    return {
        "file_sha256": _file_sha256(path),
        "transform_sha256": transform_sha256,
        "evaluated_artifact_sha256": evaluated_artifact_sha256,
        "release_artifact_sha256": observed_release_artifact,
    }


def _validate_runtime_dependencies(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V3PublicationError("runtime dependency identity is absent")
    expected = {
        "schema_version",
        "pyproject_sha256",
        "uv_lock_sha256",
        "requirements_sha256",
        "runtime_source_files",
        "runtime_source_sha256",
        "python_version",
        "python_implementation",
        "libc",
        "packages",
        "identity_sha256",
    }
    _require_fields(value, expected, "runtime dependency identity")
    identity_sha256 = _self_hash(
        value, "identity_sha256", "runtime dependency identity"
    )
    source_files = value.get("runtime_source_files")
    if (
        value.get("schema_version") != RUNTIME_DEPENDENCY_SCHEMA
        or value.get("pyproject_sha256") != PUBLIC_PYPROJECT_SHA256
        or value.get("uv_lock_sha256") != PUBLIC_UV_LOCK_SHA256
        or value.get("requirements_sha256") != PUBLIC_REQUIREMENTS_SHA256
        or value.get("runtime_source_sha256") != PUBLIC_RUNTIME_SOURCE_SHA256
        or source_files != PUBLIC_RUNTIME_SOURCE_FILES
        or _json_sha256(source_files) != PUBLIC_RUNTIME_SOURCE_SHA256
        or value.get("packages") != RUNTIME_PACKAGE_VERSIONS
        or value.get("python_implementation") != "CPython"
        or not isinstance(value.get("python_version"), str)
        or not value["python_version"].startswith("3.12.")
        or not isinstance(value.get("libc"), str)
        or not value["libc"].startswith("glibc-")
    ):
        raise V3PublicationError("runtime dependency identity changed")
    return {**dict(value), "identity_sha256": identity_sha256}


def _validate_legacy_package(package: Path, public_repo: Path) -> dict[str, Any]:
    if package.is_symlink():
        raise V3PublicationError("V3 package cannot be a symlink")
    root = package.resolve(strict=True)
    manifest = _object(root / "release-manifest.json", "V3 release manifest")
    manifest_sha = _self_hash(manifest, "manifest_sha256", "V3 release manifest")
    if (
        manifest.get("schema_version") != PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("upload_performed") is not False
        or manifest.get("publication_performed") is not False
    ):
        raise V3PublicationError(
            "V3 release manifest is not an unpublished staging package"
        )

    inventory = _regular_inventory(root, exclude=frozenset({"release-manifest.json"}))
    if manifest.get("files") != inventory:
        raise V3PublicationError("V3 release package inventory changed")
    checksum = _packaged_file(root, "SHA256SUMS", "checksum inventory")
    expected_lines = "".join(
        f"{item['sha256']}  {name}\n"
        for name, item in inventory.items()
        if name != "SHA256SUMS"
    ).encode()
    if checksum.read_bytes() != expected_lines:
        raise V3PublicationError("V3 checksum inventory changed")

    evidence = manifest.get("evidence_receipts")
    required_evidence = {
        "final",
        "final_audit",
        "sharded_final",
        "spend_upper_bound",
        "training_evaluation_bridge",
    }
    if not isinstance(evidence, Mapping) or not required_evidence <= set(evidence):
        raise V3PublicationError("V3 package lost required evidence")
    packaged_evidence: dict[str, dict[str, Any]] = {}
    for role in sorted(required_evidence):
        binding = evidence.get(role)
        if not isinstance(binding, Mapping) or "packaged_path" not in binding:
            raise V3PublicationError(f"V3 evidence is not hermetic: {role}")
        path = _packaged_file(root, binding["packaged_path"], f"V3 evidence {role}")
        if binding.get("sha256") != _file_sha256(path):
            raise V3PublicationError(f"V3 evidence bytes changed: {role}")
        packaged_evidence[role] = _object(path, f"V3 evidence {role}")
        hash_field = "audit_sha256" if role == "final_audit" else "receipt_sha256"
        _self_hash(packaged_evidence[role], hash_field, f"V3 evidence {role}")

    final = packaged_evidence["final"]
    audit = packaged_evidence["final_audit"]
    sharded = packaged_evidence["sharded_final"]
    spend = packaged_evidence["spend_upper_bound"]
    bridge = packaged_evidence["training_evaluation_bridge"]
    manifest_calibration = manifest.get("calibration")
    calibration_file_sha = (
        manifest_calibration.get("sha256")
        if isinstance(manifest_calibration, Mapping)
        else None
    )
    if (
        final.get("schema_version") != "infercrane-commerce-1-v3-training-evaluation/v3"
        or final.get("status") != "SUCCEEDED"
        or final.get("artifact_sha256") != manifest.get("artifact_sha256")
        or final.get("contract_sha256") != manifest.get("contract_sha256")
        or audit.get("schema_version") != "infercrane-commerce-1-v3-final-audit/v1"
        or audit.get("status") != "PASS_LOCAL_EVIDENCE_PENDING_OFFICIAL_V0_3"
        or audit.get("artifact_sha256") != manifest.get("artifact_sha256")
        or audit.get("contract_sha256") != manifest.get("contract_sha256")
        or audit.get("final_receipt_sha256") != final.get("receipt_sha256")
        or sharded.get("schema_version")
        != "infercrane-commerce-1-v3-sharded-evidence/v1"
        or sharded.get("status") != "PASS_COMPLETE_OFFICIAL_0_2_1"
        or sharded.get("artifact_sha256") != manifest.get("artifact_sha256")
        or sharded.get("contract_sha256") != manifest.get("contract_sha256")
        or sharded.get("attempt_id") != EVALUATOR_ATTEMPT
        or sharded.get("calibration_file_sha256") != calibration_file_sha
        or sharded.get("official_complete") is not True
        or sharded.get("selected_in_edition") != 120340
        or sharded.get("excluded_selected") != 442
        or sharded.get("selected_scoreable") != 119898
        or sharded.get("added") != 30419
        or sharded.get("merged") != 150317
        or spend.get("schema_version")
        != "infercrane-commerce-1-v3-spend-upper-bound/v1"
        or spend.get("within_campaign_hard_cap") is not True
        or sharded.get("spend_upper_bound_receipt_sha256")
        != spend.get("receipt_sha256")
        or bridge.get("schema_version") != TRAINING_EVALUATION_BRIDGE_SCHEMA
        or bridge.get("status") != "PASS_TRAINING_TO_EXACT_EVALUATION_LINEAGE"
        or bridge.get("artifact_sha256") != manifest.get("artifact_sha256")
        or bridge.get("contract_sha256") != manifest.get("contract_sha256")
        or bridge.get("calibration_file_sha256") != calibration_file_sha
        or bridge.get("training_final_receipt_sha256") != final.get("receipt_sha256")
        or bridge.get("final_audit_sha256") != audit.get("audit_sha256")
        or bridge.get("sharded_final_receipt_sha256") != sharded.get("receipt_sha256")
    ):
        raise V3PublicationError(
            "training FINAL, audit, and exact evaluation are not bridged"
        )

    for kind in ("source", "runtime"):
        binding = manifest.get(f"{kind}_identity")
        if not isinstance(binding, Mapping):
            raise V3PublicationError(f"V3 package has no {kind} identity")
        identity_path = _packaged_file(root, binding.get("path"), f"{kind} identity")
        identity = _object(identity_path, f"{kind} identity")
        files = identity.get("files")
        packaged_path = binding.get("packaged_path")
        if not isinstance(files, Mapping) or not files or packaged_path is None:
            raise V3PublicationError(f"V3 {kind} identity is not hermetic")
        source_root = root / _safe_relative(packaged_path, f"{kind} packaged_path")
        source_inventory = _regular_inventory(source_root)
        if set(source_inventory) != set(files):
            raise V3PublicationError(f"V3 {kind} source inventory changed")
        for name, digest in files.items():
            source = _packaged_file(source_root, name, f"{kind} source {name}")
            if _sha(digest, f"{kind} identity {name}") != _file_sha256(source):
                raise V3PublicationError(f"V3 {kind} source bytes changed: {name}")

    public_runtime = manifest.get("public_runtime")
    if (
        not isinstance(public_runtime, Mapping)
        or public_runtime.get("path") != PUBLIC_RUNTIME_ROOT
    ):
        raise V3PublicationError("V3 package has no bound public runtime")
    public_files = public_runtime.get("files")
    if not isinstance(public_files, Mapping) or not public_files:
        raise V3PublicationError("V3 public runtime inventory is absent")
    expected_public: dict[str, str] = {}
    local_root = public_repo.resolve(strict=True)
    for name, digest in public_files.items():
        relative = _safe_relative(name, f"public runtime {name}")
        packaged = _packaged_file(
            root / PUBLIC_RUNTIME_ROOT, relative, f"public runtime {name}"
        )
        local = _packaged_file(local_root, relative, f"local public runtime {name}")
        observed = _file_sha256(packaged)
        if (
            _sha(digest, f"public runtime {name}") != observed
            or _file_sha256(local) != observed
        ):
            raise V3PublicationError(
                f"public runtime differs from the packaged runtime: {name}"
            )
        expected_public[relative] = observed
    packaged_public_inventory = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / PUBLIC_RUNTIME_ROOT).items()
    }
    if packaged_public_inventory != expected_public:
        raise V3PublicationError("V3 public runtime has undeclared or missing files")
    if public_runtime.get("inventory_sha256") != _json_sha256(expected_public):
        raise V3PublicationError("public runtime inventory binding changed")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise V3PublicationError("V3 provenance binding is absent")
    provenance_path = _packaged_file(root, provenance.get("path"), "V3 provenance")
    if provenance.get("sha256") != _file_sha256(provenance_path):
        raise V3PublicationError("V3 provenance bytes changed")
    provenance_value = _object(provenance_path, "V3 provenance")
    provenance_sha = _self_hash(provenance_value, "provenance_sha256", "V3 provenance")
    training_sources = provenance_value.get("required_training_sources")
    model_lineage = provenance_value.get("model_lineage")
    if (
        provenance_value.get("model_id") != MODEL_ID
        or provenance_value.get("legal_conclusion") is not False
        or provenance_value.get("publication_approval") is not False
        or provenance_value.get("required_source_count") != 6
        or not isinstance(training_sources, list)
        or len(training_sources) != 6
        or not isinstance(model_lineage, Mapping)
        or set(model_lineage) != {"base", "initialization"}
    ):
        raise V3PublicationError("V3 training-time provenance disclosure changed")

    mandatory_notices: list[dict[str, Any]] = []
    for source in training_sources:
        if not isinstance(source, Mapping):
            raise V3PublicationError("V3 training source disclosure changed")
        mandatory_notices.append(
            {
                "subject_type": "training_source",
                "subject_id": source.get("source_id"),
                "license_declaration": source.get("license_declaration"),
                "license_evidence_sha256": _sha(
                    source.get("license_evidence_sha256"),
                    "training source license evidence",
                ),
                "notice_requirement": source.get("notice_requirement"),
            }
        )
    for role in ("base", "initialization"):
        item = model_lineage[role]
        if not isinstance(item, Mapping):
            raise V3PublicationError("V3 model lineage disclosure changed")
        mandatory_notices.append(
            {
                "subject_type": "model_lineage",
                "subject_id": role,
                "model_id": item.get("model_id"),
                "revision": _revision(item.get("revision"), f"{role} model revision"),
                "license_declaration": item.get("license_declaration"),
                "notice_requirement": item.get("notice_requirement"),
            }
        )
    if any(
        not isinstance(item.get("subject_id"), str)
        or not isinstance(item.get("license_declaration"), str)
        or not isinstance(item.get("notice_requirement"), str)
        or not item["notice_requirement"].strip()
        for item in mandatory_notices
    ):
        raise V3PublicationError("V3 mandatory notice disclosure is incomplete")
    mandatory_notices.sort(key=lambda item: (item["subject_type"], item["subject_id"]))

    notice_binding = manifest.get("mandatory_notices")
    if not isinstance(notice_binding, Mapping):
        raise V3PublicationError("V3 package has no mandatory notice inventory")
    notice_path = _packaged_file(
        root, notice_binding.get("path"), "mandatory notice inventory"
    )
    notice_value = _object(notice_path, "mandatory notice inventory")
    notice_receipt_sha = _self_hash(
        notice_value, "receipt_sha256", "mandatory notice inventory"
    )
    if (
        notice_binding.get("sha256") != _file_sha256(notice_path)
        or notice_binding.get("receipt_sha256") != notice_receipt_sha
        or notice_value.get("schema_version")
        != "infercrane-commerce-1-v3-mandatory-notices/v1"
        or notice_value.get("model_id") != MODEL_ID
        or notice_value.get("notices") != mandatory_notices
    ):
        raise V3PublicationError("V3 mandatory notice inventory changed")

    licenses = manifest.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {"project", "third_party"}:
        raise V3PublicationError("V3 package license bindings changed")
    license_shas: dict[str, str] = {}
    for role, expected_path in (
        ("project", "LICENSE"),
        ("third_party", "THIRD_PARTY.md"),
    ):
        binding = licenses.get(role)
        if not isinstance(binding, Mapping) or binding.get("path") != expected_path:
            raise V3PublicationError(f"V3 {role} license binding changed")
        path = _packaged_file(root, expected_path, f"V3 {role} license")
        observed = _file_sha256(path)
        if binding.get("sha256") != observed:
            raise V3PublicationError(f"V3 {role} license bytes changed")
        license_shas[role] = observed

    calibration = manifest.get("calibration")
    native_runtime = manifest.get("native_runtime")
    if not isinstance(calibration, Mapping) or not isinstance(native_runtime, Mapping):
        raise V3PublicationError(
            "V3 package lost calibration or native runtime binding"
        )
    calibration_path = _packaged_file(root, calibration.get("path"), "V3 calibration")
    runtime_path = _packaged_file(
        root, native_runtime.get("manifest_path"), "V3 native runtime"
    )
    if calibration.get("sha256") != _file_sha256(calibration_path):
        raise V3PublicationError("V3 calibration bytes changed")
    if native_runtime.get("manifest_sha256") != _file_sha256(runtime_path):
        raise V3PublicationError("V3 native runtime bytes changed")

    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "inventory_sha256": _json_sha256(inventory),
        "artifact_sha256": _sha(manifest.get("artifact_sha256"), "V3 artifact"),
        "contract_sha256": _sha(manifest.get("contract_sha256"), "V3 contract"),
        "calibration_file_sha256": _sha(calibration.get("sha256"), "V3 calibration"),
        "runtime_identity_sha256": _sha(
            native_runtime.get("runtime_identity_sha256"), "V3 native runtime identity"
        ),
        "provenance_sha256": provenance_sha,
        "provenance_file_sha256": _file_sha256(provenance_path),
        "mandatory_notices": mandatory_notices,
        "mandatory_notices_receipt_sha256": notice_receipt_sha,
        "license_shas": license_shas,
        "evidence": packaged_evidence,
    }


def _identity_files(root: Path, label: str) -> dict[str, str]:
    """Inventory one packaged/public source tree using the V4 identity rules."""

    if root.is_symlink() or not root.is_dir():
        raise V3PublicationError(f"{label} source root is unavailable")
    ignored_directories = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in ignored_directories for part in relative.parts):
            continue
        if path.name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise V3PublicationError(f"{label} source contains a symlink")
        if path.is_file():
            files[relative.as_posix()] = _file_sha256(path)
        elif not path.is_dir():
            raise V3PublicationError(f"{label} source contains a special file")
    if not files or not any(name.endswith(".py") for name in files):
        raise V3PublicationError(f"{label} source contains no Python files")
    return files


def _validate_current_package(package: Path, public_repo: Path) -> dict[str, Any]:
    """Validate the hermetic V4 package emitted by the current V3 pipeline."""

    if package.is_symlink():
        raise V3PublicationError("V3 package cannot be a symlink")
    root = package.resolve(strict=True)
    manifest = _object(root / "release-manifest.json", "V3 release manifest")
    manifest_sha = _self_hash(manifest, "manifest_sha256", "V3 release manifest")
    if (
        manifest.get("schema_version") != CURRENT_PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("evaluator_attempt_id") != EVALUATOR_ATTEMPT
        or manifest.get("upload_performed") is not False
        or manifest.get("publication_performed") is not False
    ):
        raise V3PublicationError("V4 package is not an unpublished staging package")

    inventory = _regular_inventory(root, exclude=frozenset({"release-manifest.json"}))
    if manifest.get("files") != inventory:
        raise V3PublicationError("V4 release package inventory changed")
    checksum_binding = manifest.get("checksums")
    if not isinstance(checksum_binding, Mapping):
        raise V3PublicationError("V4 checksum binding is absent")
    checksum = _packaged_file(root, checksum_binding.get("path"), "checksum inventory")
    expected_lines = "".join(
        f"{item['sha256']}  {name}\n"
        for name, item in inventory.items()
        if name != checksum_binding.get("path")
    ).encode()
    if (
        checksum_binding.get("path") != "SHA256SUMS"
        or checksum_binding.get("sha256") != _file_sha256(checksum)
        or checksum.read_bytes() != expected_lines
    ):
        raise V3PublicationError("V4 checksum inventory changed")

    evidence = manifest.get("evidence_receipts")
    required_evidence = {
        "evaluation_request",
        "final",
        "final_audit",
        "sharded_final",
        "spend_upper_bound",
    }
    if not isinstance(evidence, Mapping) or not required_evidence <= set(evidence):
        raise V3PublicationError("V4 package lost required evidence")
    packaged_evidence: dict[str, dict[str, Any]] = {}
    hash_fields = {
        "evaluation_request": "request_sha256",
        "final": "receipt_sha256",
        "final_audit": "audit_sha256",
        "sharded_final": "receipt_sha256",
        "spend_upper_bound": "receipt_sha256",
    }
    for role in sorted(evidence):
        binding = evidence.get(role)
        if not isinstance(binding, Mapping) or "packaged_path" not in binding:
            raise V3PublicationError(f"V4 evidence is not hermetic: {role}")
        path = _packaged_file(root, binding["packaged_path"], f"V4 evidence {role}")
        if binding.get("sha256") != _file_sha256(path):
            raise V3PublicationError(f"V4 evidence bytes changed: {role}")
        value = _object(path, f"V4 evidence {role}")
        hash_field = hash_fields.get(role)
        if hash_field is not None:
            claimed = _self_hash(value, hash_field, f"V4 evidence {role}")
            if binding.get(hash_field) != claimed:
                raise V3PublicationError(f"V4 evidence self-hash changed: {role}")
        packaged_evidence[role] = value

    request = packaged_evidence["evaluation_request"]
    final = packaged_evidence["final"]
    audit = packaged_evidence["final_audit"]
    sharded = packaged_evidence["sharded_final"]
    spend = packaged_evidence["spend_upper_bound"]
    calibration_binding = manifest.get("calibration")
    if not isinstance(calibration_binding, Mapping):
        raise V3PublicationError("V4 calibration binding is absent")
    calibration_path = _packaged_file(
        root, calibration_binding.get("path"), "V4 calibration"
    )
    calibration_value = _object(calibration_path, "V4 calibration")
    calibration_file_sha = _file_sha256(calibration_path)
    sharded_binding = final.get("sharded_evidence")
    publication = final.get("publication")
    training_lineage = final.get("training_lineage")
    if (
        request.get("schema_version")
        != "infercrane-commerce-1-v3-sharded-evidence-request/v1"
        or request.get("attempt_id") != EVALUATOR_ATTEMPT
        or request.get("artifact_sha256") != manifest.get("artifact_sha256")
        or request.get("calibration_file_sha256") != calibration_file_sha
        or sharded.get("schema_version")
        != "infercrane-commerce-1-v3-sharded-evidence/v1"
        or sharded.get("status") != "PASS_COMPLETE_OFFICIAL_0_2_1"
        or sharded.get("attempt_id") != EVALUATOR_ATTEMPT
        or sharded.get("request_sha256") != request.get("request_sha256")
        or sharded.get("contract_sha256") != manifest.get("contract_sha256")
        or sharded.get("artifact_sha256") != manifest.get("artifact_sha256")
        or sharded.get("calibration_file_sha256") != calibration_file_sha
        or sharded.get("official_complete") is not True
        or sharded.get("selected_in_edition") != 120340
        or sharded.get("excluded_selected") != 442
        or sharded.get("selected_scoreable") != 119898
        or sharded.get("added") != 30419
        or sharded.get("merged") != 150317
        or spend.get("schema_version")
        != "infercrane-commerce-1-v3-spend-upper-bound/v1"
        or spend.get("request_sha256") != request.get("request_sha256")
        or spend.get("within_campaign_hard_cap") is not True
        or sharded.get("spend_upper_bound_receipt_sha256")
        != spend.get("receipt_sha256")
        or final.get("schema_version") != PROMOTED_FINAL_SCHEMA
        or final.get("status") != PROMOTED_FINAL_STATUS
        or final.get("contract_sha256") != manifest.get("contract_sha256")
        or final.get("artifact_sha256") != manifest.get("artifact_sha256")
        or final.get("calibration") != calibration_value
        or final.get("legacy_serial_evaluation_used") is not False
        or final.get("publication_performed") is not False
        or not isinstance(sharded_binding, Mapping)
        or sharded_binding.get("attempt_id") != EVALUATOR_ATTEMPT
        or sharded_binding.get("request_sha256") != request.get("request_sha256")
        or sharded_binding.get("receipt_sha256") != sharded.get("receipt_sha256")
        or sharded_binding.get("spend_receipt_sha256") != spend.get("receipt_sha256")
        or sharded_binding.get("decision_index") != sharded.get("decision_index")
        or sharded_binding.get("rows") != 150317
        or sharded_binding.get("official_complete") is not True
        or not isinstance(training_lineage, Mapping)
        or not isinstance(publication, Mapping)
        or publication.get("artifact_sha256") != manifest.get("artifact_sha256")
        or publication.get("reproducible_score") != str(sharded.get("decision_index"))
        or publication.get("publication_performed") is not False
        or audit.get("schema_version") != PROMOTED_AUDIT_SCHEMA
        or audit.get("status") != PROMOTED_AUDIT_STATUS
        or audit.get("contract_sha256") != manifest.get("contract_sha256")
        or audit.get("artifact_sha256") != manifest.get("artifact_sha256")
        or audit.get("promoted_final_receipt_sha256") != final.get("receipt_sha256")
        or audit.get("sharded_final_receipt_sha256") != sharded.get("receipt_sha256")
        or audit.get("spend_receipt_sha256") != spend.get("receipt_sha256")
        or audit.get("request_sha256") != request.get("request_sha256")
        or audit.get("decision_index") != sharded.get("decision_index")
        or audit.get("publication") != publication
        or audit.get("publication_performed") is not False
    ):
        raise V3PublicationError("V4 training, evaluation, and audit lineage changed")

    # Commerce-1 is one immutable launch candidate.  Once its artifact is
    # present, accept only the exact complete 0.2.1 run that was scored by the
    # frozen official local scorer.  This prevents a later package assembly
    # from silently substituting another score/result set under the same model
    # name.  Synthetic test packages and future artifacts still follow the
    # structural validation above rather than inheriting these V3 byte pins.
    if manifest.get("artifact_sha256") == CANONICAL_ARTIFACT_SHA256:
        merge = sharded.get("merge")
        canonical_score = sharded.get("decision_index")
        if (
            manifest.get("contract_sha256") != CANONICAL_CONTRACT_SHA256
            or manifest.get("manifest_sha256") != CANONICAL_RELEASE_MANIFEST_SHA256
            or calibration_file_sha != CANONICAL_CALIBRATION_FILE_SHA256
            or sharded.get("receipt_sha256") != CANONICAL_V021_FINAL_RECEIPT_SHA256
            or isinstance(canonical_score, bool)
            or not isinstance(canonical_score, (int, float))
            or not math.isclose(
                float(canonical_score),
                CANONICAL_V021_DECISION_INDEX,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or sharded.get("scores_sha256") != CANONICAL_V021_SCORES_SHA256
            or not isinstance(merge, Mapping)
            or merge.get("results_sha256") != CANONICAL_V021_RESULTS_SHA256
        ):
            raise V3PublicationError(
                "canonical Commerce-1 public evaluation bytes changed"
            )

    identities: dict[str, dict[str, Any]] = {}
    for kind in ("source", "runtime"):
        binding = manifest.get(f"{kind}_identity")
        if not isinstance(binding, Mapping):
            raise V3PublicationError(f"V4 package has no {kind} identity")
        identity_path = _packaged_file(root, binding.get("path"), f"{kind} identity")
        identity = _object(identity_path, f"{kind} identity")
        tree_path = binding.get("tree_path")
        files = identity.get("files")
        if (
            identity.get("schema_version") != IDENTITY_SCHEMA_V2
            or identity.get("kind") != kind
            or identity.get("packaged_path") != tree_path
            or not isinstance(files, Mapping)
            or not files
            or binding.get("sha256") != _file_sha256(identity_path)
            or binding.get("identity_sha256")
            != _json_sha256({"kind": kind, "files": files})
            or identity.get("identity_sha256") != binding.get("identity_sha256")
        ):
            raise V3PublicationError(f"V4 {kind} identity changed")
        tree = root / _safe_relative(tree_path, f"V4 {kind} tree path")
        if _identity_files(tree, f"V4 packaged {kind}") != dict(files):
            raise V3PublicationError(f"V4 packaged {kind} source changed")
        identities[kind] = identity

    local_runtime = public_repo.resolve(strict=True) / "src/infercrane_commerce_1"
    if (
        _identity_files(local_runtime, "local public runtime")
        != identities["runtime"]["files"]
    ):
        raise V3PublicationError("public runtime differs from the qualified runtime")

    expected_public_runtime = {
        f"src/infercrane_commerce_1/{name}": digest
        for name, digest in identities["runtime"]["files"].items()
    }
    public_runtime = manifest.get("public_runtime")
    if (
        not isinstance(public_runtime, Mapping)
        or public_runtime.get("path") != PUBLIC_RUNTIME_ROOT
        or public_runtime.get("files") != expected_public_runtime
        or public_runtime.get("inventory_sha256")
        != _json_sha256(expected_public_runtime)
    ):
        raise V3PublicationError("V4 public runtime binding changed")
    packaged_public = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / PUBLIC_RUNTIME_ROOT).items()
    }
    if packaged_public != expected_public_runtime:
        raise V3PublicationError("V4 packaged public runtime changed")

    notice_binding = manifest.get("mandatory_notices")
    if (
        not isinstance(notice_binding, Mapping)
        or notice_binding.get("path") != CURRENT_NOTICE_ROOT
        or notice_binding.get("files") != CURRENT_NOTICE_FILES
        or notice_binding.get("inventory_sha256") != _json_sha256(CURRENT_NOTICE_FILES)
        or notice_binding.get("upstream_model_license_raw_sha256")
        != "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
        or notice_binding.get("upstream_model_license_normalization")
        != "CRLF_TO_LF_ONLY"
    ):
        raise V3PublicationError("V4 mandatory notice binding changed")
    packaged_notices = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / CURRENT_NOTICE_ROOT).items()
    }
    if packaged_notices != CURRENT_NOTICE_FILES:
        raise V3PublicationError("V4 mandatory notice bytes changed")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise V3PublicationError("V4 provenance binding is absent")
    provenance_path = _packaged_file(root, provenance.get("path"), "V4 provenance")
    if provenance.get("sha256") != _file_sha256(provenance_path):
        raise V3PublicationError("V4 provenance bytes changed")
    provenance_value = _object(provenance_path, "V4 provenance")
    provenance_sha = _self_hash(provenance_value, "provenance_sha256", "V4 provenance")
    training_sources = provenance_value.get("required_training_sources")
    model_lineage = provenance_value.get("model_lineage")
    if (
        provenance_value.get("model_id") != MODEL_ID
        or provenance_value.get("legal_conclusion") is not False
        or provenance_value.get("publication_approval") is not False
        or provenance_value.get("required_source_count") != 6
        or not isinstance(training_sources, list)
        or len(training_sources) != 6
        or not isinstance(model_lineage, Mapping)
        or set(model_lineage) != {"base", "initialization"}
    ):
        raise V3PublicationError("V4 training-time provenance disclosure changed")
    mandatory_notices: list[dict[str, Any]] = []
    for source in training_sources:
        if not isinstance(source, Mapping):
            raise V3PublicationError("V4 training source disclosure changed")
        mandatory_notices.append(
            {
                "subject_type": "training_source",
                "subject_id": source.get("source_id"),
                "license_declaration": source.get("license_declaration"),
                "license_evidence_sha256": _sha(
                    source.get("license_evidence_sha256"),
                    "training source license evidence",
                ),
                "notice_requirement": source.get("notice_requirement"),
            }
        )
    for role in ("base", "initialization"):
        item = model_lineage[role]
        if not isinstance(item, Mapping):
            raise V3PublicationError("V4 model lineage disclosure changed")
        mandatory_notices.append(
            {
                "subject_type": "model_lineage",
                "subject_id": role,
                "model_id": item.get("model_id"),
                "revision": _revision(item.get("revision"), f"{role} model revision"),
                "license_declaration": item.get("license_declaration"),
                "notice_requirement": item.get("notice_requirement"),
            }
        )
    if any(
        not isinstance(item.get("subject_id"), str)
        or not isinstance(item.get("license_declaration"), str)
        or not isinstance(item.get("notice_requirement"), str)
        or not item["notice_requirement"].strip()
        for item in mandatory_notices
    ):
        raise V3PublicationError("V4 mandatory notice disclosure is incomplete")
    mandatory_notices.sort(key=lambda item: (item["subject_type"], item["subject_id"]))
    notice_body = {
        "schema_version": "infercrane-commerce-1-v3-mandatory-notices/v1",
        "model_id": MODEL_ID,
        "notices": mandatory_notices,
    }

    licenses = manifest.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {"project", "third_party"}:
        raise V3PublicationError("V4 package license bindings changed")
    license_shas: dict[str, str] = {}
    for role, expected_path in (
        ("project", "LICENSE"),
        ("third_party", "THIRD_PARTY.md"),
    ):
        binding = licenses.get(role)
        path = _packaged_file(root, expected_path, f"V4 {role} license")
        observed = _file_sha256(path)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != expected_path
            or binding.get("sha256") != observed
        ):
            raise V3PublicationError(f"V4 {role} license bytes changed")
        license_shas[role] = observed

    native_runtime = manifest.get("native_runtime")
    if not isinstance(native_runtime, Mapping):
        raise V3PublicationError("V4 native runtime binding is absent")
    runtime_path = _packaged_file(
        root, native_runtime.get("manifest_path"), "V4 native runtime"
    )
    runtime_value = _object(runtime_path, "V4 native runtime")
    runtime_body = {
        key: item
        for key, item in runtime_value.items()
        if key != "runtime_identity_sha256"
    }
    if (
        calibration_binding.get("sha256") != calibration_file_sha
        or native_runtime.get("manifest_sha256") != _file_sha256(runtime_path)
        or native_runtime.get("runtime_identity_sha256")
        != runtime_value.get("runtime_identity_sha256")
        or runtime_value.get("runtime_identity_sha256") != _json_sha256(runtime_body)
        or runtime_value.get("model") != "infercrane/commerce-1"
        or runtime_value.get("checkpoint", {}).get("artifact_sha256")
        != manifest.get("artifact_sha256")
        or runtime_value.get("calibration", {}).get("file_sha256")
        != calibration_file_sha
        or runtime_value.get("evidence", {}).get("sha256")
        != evidence["sharded_final"].get("sha256")
    ):
        raise V3PublicationError("V4 native runtime identity changed")

    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "inventory_sha256": _json_sha256(inventory),
        "artifact_sha256": _sha(manifest.get("artifact_sha256"), "V4 artifact"),
        "contract_sha256": _sha(manifest.get("contract_sha256"), "V4 contract"),
        "calibration_file_sha256": _sha(calibration_file_sha, "V4 calibration"),
        "runtime_identity_sha256": _sha(
            native_runtime.get("runtime_identity_sha256"), "V4 native runtime identity"
        ),
        "provenance_sha256": provenance_sha,
        "provenance_file_sha256": _file_sha256(provenance_path),
        "mandatory_notices": mandatory_notices,
        "mandatory_notices_receipt_sha256": _json_sha256(notice_body),
        "license_shas": license_shas,
        "evidence": {
            role: {
                **dict(evidence[role]),
                **(
                    {
                        hash_fields[role]: packaged_evidence[role][hash_fields[role]],
                    }
                    if role in hash_fields
                    else {}
                ),
            }
            for role in evidence
        },
    }


def _validate_safe_current_package(package: Path, public_repo: Path) -> dict[str, Any]:
    """Validate the content-free V5 launch package.

    Raw requests, spend receipts, provider identifiers, orchestration source,
    and source-rights review records are protected build inputs.  V5 ships
    only self-hashed public evaluation and training-lineage attestations.
    """

    if package.is_symlink():
        raise V3PublicationError("V5 package cannot be a symlink")
    root = package.resolve(strict=True)
    manifest = _object(root / "release-manifest.json", "V5 release manifest")
    manifest_sha = _self_hash(manifest, "manifest_sha256", "V5 release manifest")
    if (
        manifest.get("schema_version") != CURRENT_PACKAGE_SCHEMA
        or manifest.get("status") != PACKAGE_STATUS
        or manifest.get("model_id") != MODEL_ID
        or manifest.get("contract_sha256") != CANONICAL_CONTRACT_SHA256
        or manifest.get("upload_performed") is not False
        or manifest.get("publication_performed") is not False
    ):
        raise V3PublicationError("V5 package is not an unpublished staging package")
    forbidden = {
        "evaluator_attempt_id",
        "evidence_receipts",
        "source_identity",
        "runtime_identity",
    }
    if forbidden & set(manifest):
        raise V3PublicationError("V5 package exposes protected release controls")

    inventory = _regular_inventory(root, exclude=frozenset({"release-manifest.json"}))
    if manifest.get("files") != inventory:
        raise V3PublicationError("V5 release package inventory changed")
    checksum_binding = manifest.get("checksums")
    if not isinstance(checksum_binding, Mapping):
        raise V3PublicationError("V5 checksum binding is absent")
    checksum = _packaged_file(root, checksum_binding.get("path"), "V5 checksums")
    expected_lines = "".join(
        f"{item['sha256']}  {name}\n"
        for name, item in inventory.items()
        if name != checksum_binding.get("path")
    ).encode()
    if (
        checksum_binding.get("path") != "SHA256SUMS"
        or checksum_binding.get("sha256") != _file_sha256(checksum)
        or checksum.read_bytes() != expected_lines
    ):
        raise V3PublicationError("V5 checksum inventory changed")

    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise V3PublicationError("V5 checkpoint binding is absent")
    checkpoint_inventory_sha256 = _self_hash(
        checkpoint, "inventory_sha256", "V5 checkpoint"
    )
    checkpoint_files = checkpoint.get("files")
    checkpoint_sizes = checkpoint.get("sizes_bytes")
    if not isinstance(checkpoint_files, Mapping) or not isinstance(
        checkpoint_sizes, Mapping
    ):
        raise V3PublicationError("V5 checkpoint inventory changed")
    observed_checkpoint = _regular_inventory(root / "model")
    expected_checkpoint = {
        name: {"sha256": digest, "size_bytes": checkpoint_sizes.get(name)}
        for name, digest in checkpoint_files.items()
    }
    release_artifact_sha256 = _sha(
        manifest.get("artifact_sha256"), "V5 release artifact"
    )
    if (
        expected_checkpoint != observed_checkpoint
        or set(checkpoint_files) != set(checkpoint_sizes)
        or _json_sha256(dict(sorted(checkpoint_files.items())))
        != release_artifact_sha256
        or checkpoint.get("artifact_sha256") != release_artifact_sha256
    ):
        raise V3PublicationError("V5 checkpoint identity changed")
    transform = _validate_artifact_transform(
        root,
        manifest.get("artifact_transform"),
        release_checkpoint=observed_checkpoint,
        release_artifact_sha256=release_artifact_sha256,
    )
    evaluated_artifact_sha256 = _sha(
        manifest.get("evaluated_artifact_sha256"), "V5 evaluated artifact"
    )
    if transform["evaluated_artifact_sha256"] != evaluated_artifact_sha256:
        raise V3PublicationError("V5 evaluated artifact binding changed")

    calibration_binding = manifest.get("calibration")
    if not isinstance(calibration_binding, Mapping):
        raise V3PublicationError("V5 calibration binding is absent")
    calibration_path = _packaged_file(
        root, calibration_binding.get("path"), "V5 calibration"
    )
    calibration_file_sha = _file_sha256(calibration_path)
    if calibration_binding.get("sha256") != calibration_file_sha:
        raise V3PublicationError("V5 calibration bytes changed")

    public_runtime = manifest.get("public_runtime")
    if (
        not isinstance(public_runtime, Mapping)
        or public_runtime.get("path") != PUBLIC_RUNTIME_ROOT
        or not isinstance(public_runtime.get("files"), Mapping)
    ):
        raise V3PublicationError("V5 public runtime binding is absent")
    expected_public_runtime = dict(public_runtime["files"])
    if public_runtime.get("inventory_sha256") != _json_sha256(expected_public_runtime):
        raise V3PublicationError("V5 public runtime inventory changed")
    packaged_public = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / PUBLIC_RUNTIME_ROOT).items()
    }
    if packaged_public != expected_public_runtime:
        raise V3PublicationError("V5 packaged public runtime changed")
    local_runtime = public_repo.resolve(strict=True) / "src/infercrane_commerce_1"
    local_public = {
        f"src/infercrane_commerce_1/{name}": digest
        for name, digest in _identity_files(
            local_runtime, "local public runtime"
        ).items()
    }
    if local_public != expected_public_runtime:
        raise V3PublicationError("public runtime differs from the qualified runtime")

    notice_binding = manifest.get("mandatory_notices")
    if (
        not isinstance(notice_binding, Mapping)
        or notice_binding.get("path") != CURRENT_NOTICE_ROOT
        or notice_binding.get("files") != CURRENT_NOTICE_FILES
        or notice_binding.get("inventory_sha256") != _json_sha256(CURRENT_NOTICE_FILES)
        or notice_binding.get("upstream_model_license_raw_sha256")
        != "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
        or notice_binding.get("upstream_model_license_normalization")
        != "CRLF_TO_LF_ONLY"
    ):
        raise V3PublicationError("V5 mandatory notice binding changed")
    packaged_notices = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / CURRENT_NOTICE_ROOT).items()
    }
    if packaged_notices != CURRENT_NOTICE_FILES:
        raise V3PublicationError("V5 mandatory notice bytes changed")

    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        raise V3PublicationError("V5 provenance binding is absent")
    provenance_path = _packaged_file(root, provenance.get("path"), "V5 provenance")
    if provenance.get("sha256") != _file_sha256(provenance_path):
        raise V3PublicationError("V5 provenance bytes changed")
    provenance_value = _object(provenance_path, "V5 provenance")
    provenance_sha = _self_hash(provenance_value, "provenance_sha256", "V5 provenance")
    if provenance.get("provenance_sha256") != provenance_sha:
        raise V3PublicationError("V5 provenance self-hash changed")
    training_sources = provenance_value.get("required_training_sources")
    model_lineage = provenance_value.get("model_lineage")
    if (
        provenance_value.get("model_id") != MODEL_ID
        or provenance_value.get("legal_conclusion") is not False
        or provenance_value.get("publication_approval") is not False
        or provenance_value.get("required_source_count") != 6
        or not isinstance(training_sources, list)
        or len(training_sources) != 6
        or not isinstance(model_lineage, Mapping)
        or set(model_lineage) != {"base", "initialization"}
    ):
        raise V3PublicationError("V5 training-time provenance disclosure changed")

    lineage_path = _packaged_file(
        root, TRAINING_DATA_LINEAGE_PATH, "V5 training-data lineage"
    )
    lineage = _object(lineage_path, "V5 training-data lineage")
    _self_hash(lineage, "lineage_sha256", "V5 training-data lineage")
    lineage_fields = {
        "schema_version",
        "status",
        "model_id",
        "contract_sha256",
        "protected_inputs",
        "checkpoint_data_provenance",
        "protected_owner_authority_sha256",
        "protected_authorization_records",
        "sources",
        "lineage_sha256",
    }
    if (
        set(lineage) != lineage_fields
        or manifest.get("training_data_lineage") != lineage
        or lineage.get("schema_version") != TRAINING_DATA_LINEAGE_SCHEMA
        or lineage.get("status") != "VERIFIED_PROTECTED_INPUTS_NOT_DISTRIBUTED"
        or lineage.get("model_id") != MODEL_ID
        or lineage.get("contract_sha256") != manifest.get("contract_sha256")
        or lineage.get("protected_owner_authority_sha256")
        != OWNER_AUTHORITY_STATEMENTS_SHA256
    ):
        raise V3PublicationError("V5 training-data lineage changed")
    authorization_records = lineage.get("protected_authorization_records")
    authorization_fields = {
        "authorization_record_sha256",
        "approval_record_sha256",
        "record_set_sha256",
    }
    if (
        not isinstance(authorization_records, Mapping)
        or set(authorization_records) != authorization_fields
    ):
        raise V3PublicationError("V5 protected authorization binding changed")
    for field in ("authorization_record_sha256", "approval_record_sha256"):
        _sha(
            authorization_records.get(field),
            f"V5 protected authorization {field}",
        )
    expected_record_set_sha256 = _json_sha256(
        {
            field: authorization_records[field]
            for field in (
                "authorization_record_sha256",
                "approval_record_sha256",
            )
        }
    )
    if authorization_records.get("record_set_sha256") != expected_record_set_sha256:
        raise V3PublicationError("V5 protected authorization record set changed")
    protected_inputs = lineage.get("protected_inputs")
    expected_protected = {
        "prepared_manifest": {"file_sha256", "manifest_sha256"},
        "merge_input_bundle": {"file_sha256", "bundle_sha256"},
    }
    if not isinstance(protected_inputs, Mapping) or set(protected_inputs) != set(
        expected_protected
    ):
        raise V3PublicationError("V5 protected training-input binding changed")
    for label, fields in expected_protected.items():
        value = protected_inputs.get(label)
        if not isinstance(value, Mapping) or set(value) != fields:
            raise V3PublicationError(f"V5 protected {label} binding changed")
        for field in fields:
            _sha(value.get(field), f"V5 protected {label} {field}")
    decision_config = _object(
        _packaged_file(root, "model/decision_config.json", "V5 decision config"),
        "V5 decision config",
    )
    checkpoint_data = lineage.get("checkpoint_data_provenance")
    if not isinstance(checkpoint_data, Mapping) or set(checkpoint_data) != {
        "train",
        "development",
        "temperature",
        "git_commit",
    }:
        raise V3PublicationError("V5 checkpoint training-data provenance changed")
    for field in ("train", "development", "temperature"):
        _sha(checkpoint_data.get(field), f"V5 checkpoint provenance {field}")
    if not _REVISION.fullmatch(str(checkpoint_data.get("git_commit"))):
        raise V3PublicationError("V5 checkpoint provenance commit changed")
    codes = decision_config.get("codes")
    token_ids = decision_config.get("token_ids")
    temperature = decision_config.get("temperature")
    if (
        set(decision_config) != set(DECISION_CONFIG_FIELDS)
        or isinstance(decision_config.get("format_version"), bool)
        or not isinstance(decision_config.get("format_version"), int)
        or any(
            not isinstance(decision_config.get(field), str)
            or not decision_config.get(field)
            for field in ("base_model", "revision", "attention_mode", "pooling")
        )
        or not isinstance(codes, list)
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
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0
    ):
        raise V3PublicationError(
            "V5 decision config is not the exact runtime projection"
        )

    lineage_sources = lineage.get("sources")
    if not isinstance(lineage_sources, list) or len(lineage_sources) != 6:
        raise V3PublicationError("V5 training source lineage changed")
    expected_source_fields = {
        "source_id",
        "repository",
        "revision",
        "data_sha256",
        "license",
        "license_evidence_sha256",
        "approved_roles",
        "commercial_use_permitted",
        "redistribution_permitted",
        "review_file_sha256",
        "review_receipt_sha256",
    }
    for source, disclosed in zip(lineage_sources, training_sources, strict=True):
        if not isinstance(source, Mapping) or not isinstance(disclosed, Mapping):
            raise V3PublicationError("V5 training source binding is invalid")
        if (
            set(source) != expected_source_fields
            or source.get("source_id") != disclosed.get("source_id")
            or source.get("repository") != disclosed.get("repository")
            or source.get("revision") != disclosed.get("revision")
            or source.get("license") != disclosed.get("license_declaration")
            or source.get("license_evidence_sha256")
            != disclosed.get("license_evidence_sha256")
            or source.get("approved_roles") != disclosed.get("requested_roles")
            or source.get("commercial_use_permitted") is not True
            or source.get("redistribution_permitted") is not True
        ):
            raise V3PublicationError("V5 training source binding changed")
        for field in ("data_sha256", "review_file_sha256", "review_receipt_sha256"):
            _sha(source.get(field), f"V5 training source {field}")

    evaluation_binding = manifest.get("public_evaluation")
    if not isinstance(evaluation_binding, Mapping):
        raise V3PublicationError("V5 public evaluation binding is absent")
    evaluation_path = _packaged_file(
        root, evaluation_binding.get("path"), "V5 public evaluation"
    )
    evaluation = _object(evaluation_path, "V5 public evaluation")
    evaluation_sha = _self_hash(
        evaluation, "attestation_sha256", "V5 public evaluation"
    )
    evaluation_fields = {
        "schema_version",
        "status",
        "model_id",
        "edition",
        "scope",
        "evaluation_authority",
        "contract_sha256",
        "evaluated_artifact_sha256",
        "release_artifact_sha256",
        "artifact_transform_sha256",
        "calibration_file_sha256",
        "suite_sha256",
        "selected_in_edition",
        "excluded_selected",
        "selected_scoreable",
        "added",
        "merged",
        "counts",
        "results_sha256",
        "scores_sha256",
        "decision_index",
        "reproduction_complete",
        "attestation_sha256",
    }
    score = evaluation.get("decision_index")
    if (
        set(evaluation) != evaluation_fields
        or evaluation_binding
        != {
            "path": PUBLIC_EVALUATION_PATH,
            "sha256": _file_sha256(evaluation_path),
            "attestation_sha256": evaluation_sha,
        }
        or evaluation.get("schema_version") != PUBLIC_EVALUATION_SCHEMA
        or evaluation.get("status") != "PASS_COMPLETE_LOCAL_REPRODUCTION"
        or evaluation.get("model_id") != MODEL_ID
        or evaluation.get("edition") != "0.2.1"
        or evaluation.get("scope") != "complete_public_suite"
        or evaluation.get("evaluation_authority") != "self_hosted"
        or evaluation.get("contract_sha256") != manifest.get("contract_sha256")
        or evaluation.get("evaluated_artifact_sha256") != evaluated_artifact_sha256
        or evaluation.get("release_artifact_sha256") != release_artifact_sha256
        or evaluation.get("artifact_transform_sha256") != transform["transform_sha256"]
        or evaluation.get("calibration_file_sha256") != calibration_file_sha
        or evaluation.get("selected_in_edition") != 120_340
        or evaluation.get("excluded_selected") != 442
        or evaluation.get("selected_scoreable") != 119_898
        or evaluation.get("added") != 30_419
        or evaluation.get("merged") != 150_317
        or evaluation.get("counts") != {"ok": 150_317}
        or evaluation.get("reproduction_complete") is not True
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise V3PublicationError("V5 public evaluation attestation changed")
    for field in ("suite_sha256", "results_sha256", "scores_sha256"):
        _sha(evaluation.get(field), f"V5 public evaluation {field}")
    if evaluated_artifact_sha256 == CANONICAL_EVALUATED_ARTIFACT_SHA256 and (
        calibration_file_sha != CANONICAL_CALIBRATION_FILE_SHA256
        or not math.isclose(
            float(score), CANONICAL_V021_DECISION_INDEX, rel_tol=0.0, abs_tol=1e-9
        )
        or evaluation.get("results_sha256") != CANONICAL_V021_RESULTS_SHA256
        or evaluation.get("scores_sha256") != CANONICAL_V021_SCORES_SHA256
    ):
        raise V3PublicationError("canonical Commerce-1 public evaluation changed")

    autojev = manifest.get("autojev_source")
    if not isinstance(autojev, Mapping) or autojev.get("path") != "source":
        raise V3PublicationError("V5 AutoJev source binding is absent")
    source_files = {
        name: item["sha256"]
        for name, item in _regular_inventory(root / "source").items()
    }
    if autojev.get("files") != source_files or autojev.get(
        "files_sha256"
    ) != _json_sha256(source_files):
        raise V3PublicationError("V5 AutoJev source closure changed")

    native_runtime = manifest.get("native_runtime")
    if not isinstance(native_runtime, Mapping):
        raise V3PublicationError("V5 native runtime binding is absent")
    runtime_path = _packaged_file(
        root, native_runtime.get("manifest_path"), "V5 native runtime"
    )
    runtime_value = _object(runtime_path, "V5 native runtime")
    runtime_body = {
        key: item
        for key, item in runtime_value.items()
        if key != "runtime_identity_sha256"
    }
    if (
        set(runtime_value)
        != {
            "schema_version",
            "model",
            "backend",
            "checkpoint",
            "artifact_transform",
            "calibration",
            "source",
            "evidence",
            "runtime_identity_sha256",
        }
        or native_runtime.get("manifest_sha256") != _file_sha256(runtime_path)
        or native_runtime.get("runtime_identity_sha256")
        != runtime_value.get("runtime_identity_sha256")
        or runtime_value.get("runtime_identity_sha256") != _json_sha256(runtime_body)
        or runtime_value.get("model") != "infercrane/commerce-1"
        or runtime_value.get("checkpoint", {}).get("artifact_sha256")
        != release_artifact_sha256
        or runtime_value.get("checkpoint", {}).get("files") != dict(checkpoint_files)
        or runtime_value.get("artifact_transform") != manifest.get("artifact_transform")
        or runtime_value.get("calibration", {}).get("file_sha256")
        != calibration_file_sha
        or runtime_value.get("source", {}).get("files") != source_files
        or runtime_value.get("evidence")
        != {
            "path": PUBLIC_EVALUATION_PATH,
            "sha256": _file_sha256(evaluation_path),
            "attestation_sha256": evaluation_sha,
        }
        or native_runtime.get("evidence_role") != "public_evaluation"
        or native_runtime.get("evidence_path") != PUBLIC_EVALUATION_PATH
    ):
        raise V3PublicationError("V5 native runtime identity changed")

    mandatory_notices: list[dict[str, Any]] = []
    for source in training_sources:
        mandatory_notices.append(
            {
                "subject_type": "training_source",
                "subject_id": source.get("source_id"),
                "license_declaration": source.get("license_declaration"),
                "license_evidence_sha256": _sha(
                    source.get("license_evidence_sha256"),
                    "training source license evidence",
                ),
                "notice_requirement": source.get("notice_requirement"),
            }
        )
    for role in ("base", "initialization"):
        item = model_lineage[role]
        if not isinstance(item, Mapping):
            raise V3PublicationError("V5 model lineage disclosure changed")
        mandatory_notices.append(
            {
                "subject_type": "model_lineage",
                "subject_id": role,
                "model_id": item.get("model_id"),
                "revision": _revision(item.get("revision"), f"{role} model revision"),
                "license_declaration": item.get("license_declaration"),
                "notice_requirement": item.get("notice_requirement"),
            }
        )
    if any(
        not isinstance(item.get("subject_id"), str)
        or not isinstance(item.get("license_declaration"), str)
        or not isinstance(item.get("notice_requirement"), str)
        or not item["notice_requirement"].strip()
        for item in mandatory_notices
    ):
        raise V3PublicationError("V5 mandatory notice disclosure is incomplete")
    mandatory_notices.sort(key=lambda item: (item["subject_type"], item["subject_id"]))
    notice_body = {
        "schema_version": "infercrane-commerce-1-v3-mandatory-notices/v1",
        "model_id": MODEL_ID,
        "notices": mandatory_notices,
    }

    licenses = manifest.get("licenses")
    if not isinstance(licenses, Mapping) or set(licenses) != {"project", "third_party"}:
        raise V3PublicationError("V5 package license bindings changed")
    license_shas: dict[str, str] = {}
    for role, expected_path in (
        ("project", "LICENSE"),
        ("third_party", "THIRD_PARTY.md"),
    ):
        binding = licenses.get(role)
        path = _packaged_file(root, expected_path, f"V5 {role} license")
        observed = _file_sha256(path)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != expected_path
            or binding.get("sha256") != observed
        ):
            raise V3PublicationError(f"V5 {role} license bytes changed")
        license_shas[role] = observed

    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "inventory_sha256": _json_sha256(inventory),
        "artifact_sha256": release_artifact_sha256,
        "release_artifact_sha256": release_artifact_sha256,
        "evaluated_artifact_sha256": evaluated_artifact_sha256,
        "artifact_transform_sha256": transform["transform_sha256"],
        "artifact_transform_file_sha256": transform["file_sha256"],
        "checkpoint_inventory_sha256": checkpoint_inventory_sha256,
        "contract_sha256": _sha(manifest.get("contract_sha256"), "V5 contract"),
        "calibration_file_sha256": _sha(calibration_file_sha, "V5 calibration"),
        "runtime_identity_sha256": _sha(
            native_runtime.get("runtime_identity_sha256"), "V5 native runtime identity"
        ),
        "runtime_manifest_sha256": _file_sha256(runtime_path),
        "provenance_sha256": provenance_sha,
        "provenance_file_sha256": _file_sha256(provenance_path),
        "mandatory_notices": mandatory_notices,
        "mandatory_notices_receipt_sha256": _json_sha256(notice_body),
        "license_shas": license_shas,
        "training_data_lineage": lineage,
        "training_data_lineage_file_sha256": _file_sha256(lineage_path),
        "public_evaluation": evaluation,
        "public_evaluation_file_sha256": _file_sha256(evaluation_path),
    }


def _validate_package(package: Path, public_repo: Path) -> dict[str, Any]:
    manifest = _object(package / "release-manifest.json", "V3 release manifest")
    if manifest.get("schema_version") == CURRENT_PACKAGE_SCHEMA:
        return _validate_safe_current_package(package, public_repo)
    return _validate_legacy_package(package, public_repo)


def _validate_public_qualification(
    value: Mapping[str, Any], package: Mapping[str, Any]
) -> str:
    expected = {
        "schema_version",
        "status",
        "model_id",
        "contract_sha256",
        "evaluated_artifact_sha256",
        "release_artifact_sha256",
        "artifact_transform_sha256",
        "calibration_file_sha256",
        "release_manifest_sha256",
        "package_files_sha256",
        "runtime_manifest_sha256",
        "runtime_identity_sha256",
        "public_evaluation_file_sha256",
        "public_evaluation_attestation_sha256",
        "maximum_input_tokens",
        "question_microbatch_size",
        "fixture_manifest_sha256",
        "fixture_count",
        "question_types",
        "parity",
        "artifact_transform_parity",
        "runtime_dependencies",
        "performance",
        "hardware",
        "offline",
        "network_required",
        "publication_performed",
        "benchmark_evidence_mutated",
        "qualified_at_utc",
        "protected_qualification_receipt_sha256",
        "attestation_sha256",
    }
    _require_fields(value, expected, "public runtime qualification")
    attestation_sha256 = _self_hash(
        value, "attestation_sha256", "public runtime qualification"
    )
    if (
        value.get("schema_version") != PUBLIC_QUALIFICATION_SCHEMA
        or value.get("status") != "PASS_NATIVE_RUNTIME_QUALIFICATION"
        or value.get("model_id") != MODEL_ID
        or value.get("contract_sha256") != package["contract_sha256"]
        or value.get("evaluated_artifact_sha256")
        != package["evaluated_artifact_sha256"]
        or value.get("release_artifact_sha256") != package["release_artifact_sha256"]
        or value.get("artifact_transform_sha256")
        != package["artifact_transform_sha256"]
        or value.get("calibration_file_sha256") != package["calibration_file_sha256"]
        or value.get("release_manifest_sha256") != package["manifest_sha256"]
        or value.get("package_files_sha256") != package["inventory_sha256"]
        or value.get("runtime_identity_sha256") != package["runtime_identity_sha256"]
        or value.get("runtime_manifest_sha256") != package["runtime_manifest_sha256"]
        or value.get("public_evaluation_file_sha256")
        != package["public_evaluation_file_sha256"]
        or value.get("public_evaluation_attestation_sha256")
        != package["public_evaluation"]["attestation_sha256"]
        or value.get("maximum_input_tokens") != 65_536
        or value.get("question_microbatch_size") != 1
        or value.get("fixture_count") != 3
        or value.get("question_types") != ["choice", "noul", "score"]
        or value.get("offline") is not True
        or value.get("network_required") is not False
        or value.get("publication_performed") is not False
        or value.get("benchmark_evidence_mutated") is not False
    ):
        raise V3PublicationError("public runtime qualification is detached")
    for field in (
        "runtime_manifest_sha256",
        "fixture_manifest_sha256",
        "protected_qualification_receipt_sha256",
    ):
        _sha(value.get(field), f"public runtime qualification {field}")
    _utc_timestamp(value.get("qualified_at_utc"), "public runtime qualification time")

    parity = value.get("parity")
    transform_parity = value.get("artifact_transform_parity")
    performance = value.get("performance")
    hardware = value.get("hardware")
    _validate_runtime_dependencies(value.get("runtime_dependencies"))
    if (
        not isinstance(parity, Mapping)
        or set(parity)
        != {
            "status",
            "absolute_tolerance",
            "maximum_absolute_difference",
            "comparisons",
            "native_outputs_sha256",
            "direct_outputs_sha256",
        }
        or parity.get("status") != "PASS_NATIVE_DIRECT_AUTOJEV_PARITY"
        or parity.get("absolute_tolerance") != 1e-7
        or not isinstance(transform_parity, Mapping)
        or set(transform_parity)
        != {
            "status",
            "absolute_tolerance",
            "maximum_absolute_difference",
            "comparisons",
            "evaluated_outputs_sha256",
            "release_outputs_sha256",
        }
        or transform_parity.get("status") != "PASS_EVALUATED_RELEASE_PARITY"
        or transform_parity.get("absolute_tolerance") != 1e-7
        or not isinstance(performance, Mapping)
        or performance.get("content_free") is not True
        or performance.get("fixture_manifest_sha256")
        != value.get("fixture_manifest_sha256")
        or performance.get("latency_unit") != "milliseconds_per_complete_fixture_set"
        or not isinstance(hardware, Mapping)
        or hardware.get("accelerator_count") != 1
        or "H200" not in str(hardware.get("accelerator"))
        or hardware.get("precision") != "bfloat16"
    ):
        raise V3PublicationError("public runtime qualification result changed")
    for item, fields, label in (
        (
            parity,
            ("native_outputs_sha256", "direct_outputs_sha256"),
            "native parity",
        ),
        (
            transform_parity,
            ("evaluated_outputs_sha256", "release_outputs_sha256"),
            "artifact transform parity",
        ),
    ):
        comparisons = item.get("comparisons")
        maximum = item.get("maximum_absolute_difference")
        if (
            isinstance(comparisons, bool)
            or not isinstance(comparisons, int)
            or comparisons <= 0
            or isinstance(maximum, bool)
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(maximum))
            or not 0 <= float(maximum) <= 1e-7
        ):
            raise V3PublicationError(f"public runtime qualification {label} changed")
        for field in fields:
            _sha(item.get(field), f"public runtime qualification {label} {field}")
    for field in (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "throughput_decisions_per_second",
    ):
        supplied = performance.get(field)
        if (
            isinstance(supplied, bool)
            or not isinstance(supplied, (int, float))
            or not math.isfinite(float(supplied))
            or float(supplied) <= 0
        ):
            raise V3PublicationError("public runtime qualification performance changed")
    return attestation_sha256


def _validate_qualification(
    value: Mapping[str, Any], package: Mapping[str, Any]
) -> str:
    schema = value.get("schema_version")
    if schema == PUBLIC_QUALIFICATION_SCHEMA:
        return _validate_public_qualification(value, package)

    if schema == QUALIFICATION_SCHEMA:
        expected = {
            "schema_version",
            "status",
            "attempt_id",
            "request_sha256",
            "contract_sha256",
            "artifact_sha256",
            "calibration_file_sha256",
            "evaluator_attempt_id",
            "evaluation_budget_authorization_sha256",
            "exact_evaluation_reservation_upper_bound_usd",
            "evaluation_campaign_hard_cap_usd",
            "release_manifest_sha256",
            "sharded_final_receipt_sha256",
            "spend_receipt_sha256",
            "runtime_identity",
            "fixture_manifest_sha256",
            "fixture_count",
            "question_types",
            "parity",
            "performance",
            "hardware",
            "offline",
            "network_required",
            "publication_performed",
            "benchmark_evidence_mutated",
            "qualified_at_utc",
            "receipt_sha256",
        }
        _require_fields(value, expected, "legacy runtime qualification")
        receipt_sha = _self_hash(
            value, "receipt_sha256", "legacy runtime qualification"
        )
        if (
            value.get("status") != "PASS_NATIVE_RUNTIME_QUALIFICATION"
            or value.get("artifact_sha256") != package["artifact_sha256"]
            or value.get("calibration_file_sha256")
            != package["calibration_file_sha256"]
            or value.get("contract_sha256") != package["contract_sha256"]
            or value.get("release_manifest_sha256") != package["manifest_sha256"]
        ):
            raise V3PublicationError("legacy runtime qualification is detached")
        return receipt_sha

    expected = {
        "schema_version",
        "status",
        "attempt_id",
        "request_sha256",
        "contract_sha256",
        "evaluated_artifact_sha256",
        "release_artifact_sha256",
        "artifact_transform_sha256",
        "calibration_file_sha256",
        "evaluator_attempt_id",
        "evaluation_budget_authorization_sha256",
        "exact_evaluation_reservation_upper_bound_usd",
        "evaluation_campaign_hard_cap_usd",
        "release_manifest_sha256",
        "sharded_final_receipt_sha256",
        "spend_receipt_sha256",
        "runtime_identity",
        "fixture_manifest_sha256",
        "fixture_count",
        "question_types",
        "parity",
        "artifact_transform_parity",
        "runtime_dependencies",
        "performance",
        "hardware",
        "offline",
        "network_required",
        "publication_performed",
        "benchmark_evidence_mutated",
        "qualified_at_utc",
        "receipt_sha256",
        "sharded_spend_receipt_sha256",
        "maximum_input_tokens",
        "question_microbatch_size",
    }
    _require_fields(value, expected, "runtime qualification")
    receipt_sha = _self_hash(value, "receipt_sha256", "runtime qualification")
    runtime_identity = value.get("runtime_identity")
    parity = value.get("parity")
    transform_parity = value.get("artifact_transform_parity")
    performance = value.get("performance")
    hardware = value.get("hardware")
    _validate_runtime_dependencies(value.get("runtime_dependencies"))
    evaluator_attempt = value.get("evaluator_attempt_id")
    if (
        not isinstance(evaluator_attempt, str)
        or _ATTEMPT.fullmatch(evaluator_attempt) is None
    ):
        raise V3PublicationError("runtime qualification evaluator identity is invalid")
    _sha(
        value.get("evaluation_budget_authorization_sha256"),
        "runtime qualification protected budget authorization",
    )
    reservation = _positive_money(
        value.get("exact_evaluation_reservation_upper_bound_usd"),
        "runtime qualification protected reservation",
    )
    cap = _positive_money(
        value.get("evaluation_campaign_hard_cap_usd"),
        "runtime qualification protected cap",
    )
    if reservation > cap:
        raise V3PublicationError("runtime qualification reservation exceeds its cap")
    if (
        schema != CURRENT_QUALIFICATION_SCHEMA
        or value.get("status") != "PASS_NATIVE_RUNTIME_QUALIFICATION"
        or value.get("evaluated_artifact_sha256")
        != package["evaluated_artifact_sha256"]
        or value.get("release_artifact_sha256") != package["release_artifact_sha256"]
        or value.get("artifact_transform_sha256")
        != package["artifact_transform_sha256"]
        or value.get("calibration_file_sha256") != package["calibration_file_sha256"]
        or value.get("contract_sha256") != package["contract_sha256"]
        or not isinstance(value.get("attempt_id"), str)
        or _ATTEMPT.fullmatch(value["attempt_id"]) is None
        or value.get("release_manifest_sha256") != package["manifest_sha256"]
        or value.get("offline") is not True
        or value.get("network_required") is not False
        or value.get("publication_performed") is not False
        or value.get("benchmark_evidence_mutated") is not False
        or value.get("maximum_input_tokens") != 65_536
        or value.get("question_microbatch_size") != 1
        or value.get("question_types") != ["choice", "noul", "score"]
        or not isinstance(runtime_identity, Mapping)
        or runtime_identity.get("model") != "infercrane/commerce-1"
        or runtime_identity.get("revision") != package["artifact_sha256"]
        or runtime_identity.get("runtime_identity_sha256")
        != package["runtime_identity_sha256"]
        or runtime_identity.get("backend") != "autojev-native-v1"
        or not isinstance(parity, Mapping)
        or parity.get("status") != "PASS_NATIVE_DIRECT_AUTOJEV_PARITY"
        or parity.get("absolute_tolerance") != 1e-7
        or isinstance(parity.get("maximum_absolute_difference"), bool)
        or not isinstance(parity.get("maximum_absolute_difference"), (int, float))
        or not 0 <= float(parity["maximum_absolute_difference"]) <= 1e-7
        or not isinstance(parity.get("comparisons"), int)
        or isinstance(parity.get("comparisons"), bool)
        or parity["comparisons"] <= 0
        or not isinstance(transform_parity, Mapping)
        or set(transform_parity)
        != {
            "status",
            "absolute_tolerance",
            "maximum_absolute_difference",
            "comparisons",
            "evaluated_outputs_sha256",
            "release_outputs_sha256",
        }
        or transform_parity.get("status") != "PASS_EVALUATED_RELEASE_PARITY"
        or transform_parity.get("absolute_tolerance") != 1e-7
        or isinstance(transform_parity.get("maximum_absolute_difference"), bool)
        or not isinstance(
            transform_parity.get("maximum_absolute_difference"), (int, float)
        )
        or not 0 <= float(transform_parity["maximum_absolute_difference"]) <= 1e-7
        or not isinstance(transform_parity.get("comparisons"), int)
        or isinstance(transform_parity.get("comparisons"), bool)
        or transform_parity["comparisons"] <= 0
        or not isinstance(performance, Mapping)
        or performance.get("content_free") is not True
        or performance.get("latency_unit") != "milliseconds_per_complete_fixture_set"
        or not isinstance(performance.get("sample_count"), int)
        or isinstance(performance.get("sample_count"), bool)
        or performance["sample_count"] <= 0
        or not isinstance(hardware, Mapping)
        or hardware.get("accelerator_count") != 1
        or "H200" not in str(hardware.get("accelerator"))
        or hardware.get("precision") != "bfloat16"
    ):
        raise V3PublicationError("runtime qualification is incomplete or detached")
    for field in ("native_outputs_sha256", "direct_outputs_sha256"):
        _sha(parity.get(field), f"runtime qualification parity {field}")
    for field in ("evaluated_outputs_sha256", "release_outputs_sha256"):
        _sha(
            transform_parity.get(field),
            f"runtime qualification artifact transform parity {field}",
        )
    for field in (
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_max_ms",
        "throughput_decisions_per_second",
    ):
        supplied = performance.get(field)
        if (
            isinstance(supplied, bool)
            or not isinstance(supplied, (int, float))
            or not math.isfinite(float(supplied))
            or float(supplied) <= 0
        ):
            raise V3PublicationError("runtime performance result is invalid")
    _sha(value.get("request_sha256"), "runtime qualification request")
    _sha(value.get("fixture_manifest_sha256"), "runtime qualification fixtures")
    _utc_timestamp(value.get("qualified_at_utc"), "runtime qualification time")
    _sha(
        value.get("sharded_final_receipt_sha256"),
        "runtime qualification protected sharded evidence",
    )
    _sha(
        value.get("spend_receipt_sha256"),
        "runtime qualification protected spend receipt",
    )
    _sha(
        value.get("sharded_spend_receipt_sha256"),
        "runtime qualification protected sharded spend",
    )
    if runtime_identity.get("evidence_sha256") != package.get(
        "public_evaluation_file_sha256"
    ):
        raise V3PublicationError("runtime qualification lost public evaluation lineage")
    return receipt_sha


def _validate_public_v03(value: Mapping[str, Any], package: Mapping[str, Any]) -> str:
    expected = {
        "schema_version",
        "status",
        "attempt_id",
        "request_sha256",
        "contract_sha256",
        "source_edition",
        "target_edition",
        "kit_revision",
        "artifact_sha256",
        "calibration_file_sha256",
        "runtime_sha256",
        "source_lineage_receipt_sha256",
        "spend_receipt_sha256",
        "inventory_sha256",
        "context_receipt_sha256",
        "dispatch_sha256",
        "worker_receipt_sha256",
        "source_was_relabelled",
        "reused_v021",
        "new_rebuilt_gsm8k",
        "merged",
        "merge",
        "official_public_complete",
        "public_decision_index",
        "official_full_score_claimed",
        "public_run_files",
        "receipt_sha256",
    }
    _require_fields(value, expected, "public Decision Index 0.3 FINAL")
    receipt_sha = _self_hash(value, "receipt_sha256", "public Decision Index 0.3 FINAL")
    score = value.get("public_decision_index")
    merge = value.get("merge")
    files = value.get("public_run_files")
    if (
        value.get("schema_version") != PUBLIC_V03_SCHEMA
        or value.get("status") != "PASS_COMPLETE_PUBLIC_DECISION_INDEX_0_3"
        or value.get("source_edition") != "0.2.1"
        or value.get("target_edition") != "0.3"
        or value.get("kit_revision") != KIT_REVISION
        or value.get("artifact_sha256")
        != package.get("evaluated_artifact_sha256", package["artifact_sha256"])
        or value.get("calibration_file_sha256") != package["calibration_file_sha256"]
        or value.get("contract_sha256") != package["contract_sha256"]
        or not isinstance(value.get("attempt_id"), str)
        or _ATTEMPT.fullmatch(value["attempt_id"]) is None
        or value.get("source_was_relabelled") is not False
        or value.get("official_public_complete") is not True
        or value.get("official_full_score_claimed") is not False
        or value.get("reused_v021") != 137540
        or value.get("new_rebuilt_gsm8k") != 2638
        or value.get("merged") != 140178
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0 <= float(score) <= 100
        or not isinstance(merge, Mapping)
        or merge.get("merged_rows") != 140178
        or merge.get("counts") != {"ok": 140178}
        or not isinstance(files, Mapping)
        or set(files)
        != {
            "results.jsonl",
            "environment.json",
            "status.json",
            "benchmark-summary.json",
            "index.json",
            "scores.json",
        }
        or files.get("results.jsonl") != merge.get("results_sha256")
    ):
        raise V3PublicationError(
            "public Decision Index 0.3 evidence is incomplete or detached"
        )
    for field in (
        "request_sha256",
        "runtime_sha256",
        "source_lineage_receipt_sha256",
        "spend_receipt_sha256",
        "inventory_sha256",
        "context_receipt_sha256",
        "dispatch_sha256",
        "worker_receipt_sha256",
    ):
        _sha(value.get(field), f"public Decision Index 0.3 {field}")
    _sha(merge.get("results_sha256"), "public Decision Index 0.3 results")
    for name, digest in files.items():
        _sha(digest, f"public Decision Index 0.3 file {name}")
    return receipt_sha


def _validate_legacy_official_v03(
    value: Mapping[str, Any], package: Mapping[str, Any]
) -> str:
    expected = {
        "schema_version",
        "status",
        "edition",
        "panel_id",
        "artifact_sha256",
        "official_repository_id",
        "official_submission_pr_url",
        "official_submission_pr_revision",
        "official_board_space_id",
        "official_board_space_revision",
        "maintainer_operator_identity_sha256",
        "self_attestation",
        "public_results_dataset_sha256",
        "maintainer_rescore_sha256",
        "public_complete",
        "private_same_skills_complete",
        "private_new_domains_complete",
        "decision_index",
        "public_score",
        "private_same_skills_score",
        "private_new_domains_score",
        "weights",
        "coverage",
        "errors",
        "unsupported",
        "latency_hardware",
        "latency_median_ms",
        "latency_mean_ms",
        "latency_p80_ms",
        "latency_sample_count",
        "held_out_answer_agreement_sha256",
        "evaluated_at_utc",
        "attestation_sha256",
    }
    _require_fields(value, expected, "official Decision Index 0.3 attestation")
    receipt_sha = _self_hash(
        value, "attestation_sha256", "official Decision Index 0.3 attestation"
    )
    scores = [
        value.get(name)
        for name in (
            "decision_index",
            "public_score",
            "private_same_skills_score",
            "private_new_domains_score",
        )
    ]
    latencies = [
        value.get(name)
        for name in ("latency_median_ms", "latency_mean_ms", "latency_p80_ms")
    ]
    if (
        value.get("schema_version") != OFFICIAL_V03_SCHEMA
        or value.get("status") != "OFFICIAL_MAINTAINER_V0_3_ATTESTED"
        or value.get("edition") != "0.3"
        or value.get("panel_id") != "decision-index-0.3"
        or value.get("artifact_sha256")
        != package.get("evaluated_artifact_sha256", package["artifact_sha256"])
        or value.get("official_repository_id") != OFFICIAL_REPOSITORY
        or value.get("official_board_space_id") != OFFICIAL_BOARD
        or not isinstance(value.get("official_submission_pr_url"), str)
        or _PR_URL.fullmatch(value["official_submission_pr_url"]) is None
        or value.get("self_attestation") is not False
        or value.get("public_complete") is not True
        or value.get("private_same_skills_complete") is not True
        or value.get("private_new_domains_complete") is not True
        or value.get("weights")
        != {
            "public": "0.20",
            "private_same_skills": "0.50",
            "private_new_domains": "0.30",
        }
        or value.get("coverage") != 1
        or value.get("errors") != 0
        or value.get("unsupported") != 0
        or value.get("latency_hardware") != "NVIDIA RTX PRO 6000"
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0 < float(item) <= 1000
            for item in latencies
        )
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or not 0 <= float(item) <= 100
            for item in scores
        )
        or not isinstance(value.get("latency_sample_count"), int)
        or isinstance(value.get("latency_sample_count"), bool)
        or value["latency_sample_count"] <= 0
    ):
        raise V3PublicationError(
            "official Decision Index 0.3 evidence is incomplete or detached"
        )
    _revision(value.get("official_submission_pr_revision"), "official PR revision")
    _revision(value.get("official_board_space_revision"), "official board revision")
    for field in (
        "artifact_sha256",
        "maintainer_operator_identity_sha256",
        "public_results_dataset_sha256",
        "maintainer_rescore_sha256",
        "held_out_answer_agreement_sha256",
    ):
        _sha(value.get(field), f"official Decision Index 0.3 {field}")
    _utc_timestamp(value.get("evaluated_at_utc"), "official evaluation time")
    expected_score = (
        float(value["public_score"]) * 0.2
        + float(value["private_same_skills_score"]) * 0.5
        + float(value["private_new_domains_score"]) * 0.3
    )
    if not math.isclose(float(value["decision_index"]), expected_score, abs_tol=0.051):
        raise V3PublicationError("official Decision Index 0.3 score does not reproduce")
    return receipt_sha


def _validate_current_official_v03(
    value: Mapping[str, Any], package: Mapping[str, Any]
) -> str:
    expected = {
        "schema_version",
        "status",
        "edition",
        "panel_id",
        "artifact_sha256",
        "candidate_repository_id",
        "candidate_revision",
        "official_repository_id",
        "official_submission_pr_url",
        "official_submission_pr_head_revision",
        "official_submission_pr_merged",
        "official_board_space_id",
        "official_board_space_revision",
        "official_board_index_url",
        "official_board_v03_url",
        "official_board_index_sha256",
        "official_board_v03_sha256",
        "official_board_entry_sha256",
        "results_dataset_receipt_sha256",
        "public_complete",
        "private_same_skills_complete",
        "private_new_domains_complete",
        "decision_index",
        "public_score_raw",
        "private_same_skills_score_raw",
        "private_new_domains_score_raw",
        "private_same_skills_score_equated",
        "private_new_domains_score_equated",
        "equating_provenance",
        "weights",
        "capacity",
        "latency_hardware",
        "latency_median_ms",
        "latency_mean_ms",
        "latency_p80_ms",
        "latency_sample_count",
        "observed_at_utc",
        "attestation_sha256",
    }
    _require_fields(value, expected, "official Decision Index 0.3 board evidence")
    receipt_sha = _self_hash(
        value,
        "attestation_sha256",
        "official Decision Index 0.3 board evidence",
    )
    if (
        value.get("schema_version") != CURRENT_OFFICIAL_V03_SCHEMA
        or value.get("status") != "OFFICIAL_BOARD_V0_3_EVIDENCE"
        or value.get("edition") != "0.3"
        or value.get("panel_id") != "decision-index-0.3"
        or value.get("artifact_sha256")
        != package.get("evaluated_artifact_sha256", package["artifact_sha256"])
        or value.get("official_repository_id") != OFFICIAL_REPOSITORY
        or value.get("official_board_space_id") != OFFICIAL_BOARD
        or not isinstance(value.get("official_submission_pr_url"), str)
        or _PR_URL.fullmatch(value["official_submission_pr_url"]) is None
        or not isinstance(value.get("official_submission_pr_merged"), bool)
        or value.get("public_complete") is not True
        or value.get("private_same_skills_complete") is not True
        or value.get("private_new_domains_complete") is not True
        or value.get("weights") != CURRENT_V03_WEIGHTS
        or value.get("equating_provenance") != AUDITED_V03_EQUATING
        or value.get("latency_hardware") != "NVIDIA RTX PRO 6000"
        or value.get("candidate_repository_id") != HF_REPOSITORY
    ):
        raise V3PublicationError(
            "official Decision Index 0.3 board evidence is incomplete or detached"
        )
    for field in (
        "artifact_sha256",
        "official_board_index_sha256",
        "official_board_v03_sha256",
        "official_board_entry_sha256",
        "results_dataset_receipt_sha256",
    ):
        _sha(value.get(field), f"official Decision Index 0.3 {field}")
    for field in (
        "candidate_revision",
        "official_submission_pr_head_revision",
        "official_board_space_revision",
    ):
        _revision(value.get(field), f"official Decision Index 0.3 {field}")
    board_revision = value["official_board_space_revision"]
    board_base = (
        f"https://huggingface.co/spaces/{OFFICIAL_BOARD}/resolve/{board_revision}/data"
    )
    if (
        value.get("official_board_index_url") != f"{board_base}/index.json"
        or value.get("official_board_v03_url") != f"{board_base}/v03.json"
    ):
        raise V3PublicationError("official board immutable URLs changed")

    scores = [
        value.get(name)
        for name in (
            "decision_index",
            "public_score_raw",
            "private_same_skills_score_raw",
            "private_new_domains_score_raw",
            "private_same_skills_score_equated",
            "private_new_domains_score_equated",
        )
    ]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or not 0 <= float(item) <= 100
        for item in scores
    ):
        raise V3PublicationError("official Decision Index 0.3 scores are invalid")

    capacity = value.get("capacity")
    capacity_fields = {
        "complete",
        "total",
        "answered",
        "errors",
        "unsupported",
        "coverage",
        "unsupported_counted_wrong",
        "declared_limits",
    }
    if not isinstance(capacity, Mapping) or set(capacity) != capacity_fields:
        raise V3PublicationError("official Decision Index 0.3 capacity changed")
    if (
        any(
            isinstance(capacity.get(field), bool)
            or not isinstance(capacity.get(field), int)
            or capacity[field] < 0
            for field in ("total", "answered", "errors", "unsupported")
        )
        or capacity["total"] <= 0
    ):
        raise V3PublicationError("official Decision Index 0.3 capacity is invalid")
    if (
        capacity.get("complete") is not True
        or capacity.get("unsupported_counted_wrong") is not True
        or capacity["answered"] + capacity["errors"] + capacity["unsupported"]
        != capacity["total"]
        or isinstance(capacity.get("coverage"), bool)
        or not isinstance(capacity.get("coverage"), (int, float))
        or not math.isclose(
            float(capacity["coverage"]),
            capacity["answered"] / capacity["total"],
            abs_tol=1e-9,
        )
        or not isinstance(capacity.get("declared_limits"), list)
        or any(
            not isinstance(item, str) or not item.strip()
            for item in capacity["declared_limits"]
        )
    ):
        raise V3PublicationError("official Decision Index 0.3 capacity is incomplete")

    latencies = [
        value.get(name)
        for name in ("latency_median_ms", "latency_mean_ms", "latency_p80_ms")
    ]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or not 0 < float(item) <= 1000
        for item in latencies
    ) or (
        isinstance(value.get("latency_sample_count"), bool)
        or not isinstance(value.get("latency_sample_count"), int)
        or value["latency_sample_count"] <= 0
    ):
        raise V3PublicationError("official Decision Index 0.3 latency is incomplete")
    _utc_timestamp(value.get("observed_at_utc"), "official board observation time")
    expected_score = (
        float(value["public_score_raw"]) * 0.20
        + float(value["private_same_skills_score_equated"]) * 0.50
        + float(value["private_new_domains_score_equated"]) * 0.30
    )
    if not math.isclose(float(value["decision_index"]), expected_score, abs_tol=0.051):
        raise V3PublicationError(
            "official Decision Index 0.3 equated score does not reproduce"
        )
    return receipt_sha


def _validate_official_v03(value: Mapping[str, Any], package: Mapping[str, Any]) -> str:
    if value.get("schema_version") == CURRENT_OFFICIAL_V03_SCHEMA:
        return _validate_current_official_v03(value, package)
    return _validate_legacy_official_v03(value, package)


def _public_evaluation_sha256(package: Mapping[str, Any]) -> str:
    """Return the public-suite lineage digest across safe and legacy packages."""

    public = package.get("public_evaluation")
    if isinstance(public, Mapping):
        return _sha(public.get("attestation_sha256"), "public evaluation attestation")
    evidence = package.get("evidence")
    if isinstance(evidence, Mapping) and isinstance(
        evidence.get("sharded_final"), Mapping
    ):
        return _sha(
            evidence["sharded_final"].get("receipt_sha256"),
            "legacy public evaluation receipt",
        )
    raise V3PublicationError("package has no public evaluation lineage")


def _validated_release_materials(
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    public_v03: Path | None,
    official_v03: Path | None,
) -> tuple[dict[str, Any], str, str | None, str | None, str, float | None]:
    validated = _validate_package(package, public_repo)
    qualification_value = _object(qualification, "runtime qualification")
    qualification_sha = _validate_qualification(qualification_value, validated)
    if (public_v03 is None) != (official_v03 is None):
        raise V3PublicationError(
            "public and official Decision Index 0.3 evidence must be supplied together"
        )
    if public_v03 is None or official_v03 is None:
        public_evaluation = validated.get("public_evaluation")
        if isinstance(public_evaluation, Mapping):
            public_score = public_evaluation.get("decision_index")
        else:
            public_score = validated["evidence"]["sharded_final"].get("decision_index")
        if (
            isinstance(public_score, bool)
            or not isinstance(public_score, (int, float))
            or not math.isfinite(float(public_score))
            or float(public_score) <= RELEASE_BASELINE_DECISION_INDEX
        ):
            raise V3PublicationError(
                "public Decision Index 0.2.1 does not beat the 52.55 release baseline"
            )
        return (
            validated,
            qualification_sha,
            None,
            None,
            INITIAL_RELEASE_STATE,
            None,
        )

    public_value = _object(public_v03, "public Decision Index 0.3 FINAL")
    official_value = _object(official_v03, "official Decision Index 0.3 attestation")
    public_sha = _validate_public_v03(public_value, validated)
    official_sha = _validate_official_v03(official_value, validated)
    official_public_score = (
        official_value["public_score_raw"]
        if official_value.get("schema_version") == CURRENT_OFFICIAL_V03_SCHEMA
        else official_value["public_score"]
    )
    if not math.isclose(
        float(public_value["public_decision_index"]),
        float(official_public_score),
        rel_tol=0.0,
        abs_tol=0.051,
    ):
        raise V3PublicationError(
            "official Decision Index 0.3 public score does not match the public run"
        )
    if float(official_value["decision_index"]) <= RELEASE_BASELINE_DECISION_INDEX:
        raise V3PublicationError(
            "official Decision Index 0.3 does not beat the 52.55 release baseline"
        )
    return (
        validated,
        qualification_sha,
        public_sha,
        official_sha,
        OFFICIAL_RELEASE_STATE,
        float(official_value["decision_index"]),
    )


def _blocker_dispositions(release_state: str) -> dict[str, str]:
    dispositions = dict(BLOCKER_DISPOSITIONS)
    dispositions["official-v0.3-private-evaluation"] = (
        "RESOLVED_BY_OFFICIAL_BOARD_EVIDENCE"
        if release_state == OFFICIAL_RELEASE_STATE
        else "DEFERRED_UNTIL_PUBLIC_IMMUTABLE_ENDPOINTS_EXIST"
    )
    return dispositions


def build_owner_authorization(
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    public_v03: Path | None,
    official_v03: Path | None,
    confirmation: str,
    authorized_at_utc: str,
) -> dict[str, Any]:
    """Create an exact owner authorization after every technical gate passes."""

    if confirmation != OWNER_RELEASE_CONFIRMATION:
        raise V3PublicationError("exact owner release confirmation is required")
    _utc_timestamp(authorized_at_utc, "owner authorization time")
    (
        validated,
        qualification_sha,
        public_sha,
        official_sha,
        release_state,
        official_full_score,
    ) = _validated_release_materials(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public_v03,
        official_v03=official_v03,
    )
    return _owner_authorization_from_validated_materials(
        validated=validated,
        qualification_sha=qualification_sha,
        public_sha=public_sha,
        official_sha=official_sha,
        release_state=release_state,
        official_full_score=official_full_score,
        authorized_at_utc=authorized_at_utc,
    )


def _owner_authorization_from_validated_materials(
    *,
    validated: Mapping[str, Any],
    qualification_sha: str,
    public_sha: str | None,
    official_sha: str | None,
    release_state: str,
    official_full_score: float | None,
    authorized_at_utc: str,
) -> dict[str, Any]:
    """Build the authorization without re-reading already validated package bytes."""

    disclosures = {
        "package_provenance_is_training_time_disclosure": True,
        "package_legal_conclusion": False,
        "package_publication_approval": False,
        "owner_accepts_no_legal_conclusion_claim": True,
    }
    body = {
        "schema_version": OWNER_AUTHORIZATION_SCHEMA,
        "status": "OWNER_AUTHORIZED_EXACT_V3_RELEASE",
        "release_state": release_state,
        "owner": "infercrane",
        "model_id": MODEL_ID,
        "owner_authorized_release": True,
        "legal_conclusion": False,
        "contract_sha256": validated["contract_sha256"],
        "artifact_sha256": validated["artifact_sha256"],
        "calibration_file_sha256": validated["calibration_file_sha256"],
        "release_manifest_sha256": validated["manifest_sha256"],
        "package_inventory_sha256": validated["inventory_sha256"],
        "runtime_identity_sha256": validated["runtime_identity_sha256"],
        "public_evaluation_attestation_sha256": _public_evaluation_sha256(validated),
        "runtime_qualification_sha256": qualification_sha,
        "public_v03_final_sha256": public_sha,
        "official_v03_attestation_sha256": official_sha,
        "official_v03_full_score": official_full_score,
        "official_v03_rank": None,
        "package_provenance_sha256": validated["provenance_sha256"],
        "package_provenance_file_sha256": validated["provenance_file_sha256"],
        "accepted_known_disclosures": disclosures,
        "authority_statements_sha256": OWNER_AUTHORITY_STATEMENTS_SHA256,
        "mandatory_notices_sha256": _json_sha256(validated["mandatory_notices"]),
        "mandatory_notices_receipt_sha256": validated[
            "mandatory_notices_receipt_sha256"
        ],
        "license_shas": validated["license_shas"],
        "blocker_dispositions": _blocker_dispositions(release_state),
        "authorized_at_utc": authorized_at_utc,
        "publication_performed": False,
    }
    return {**body, "receipt_sha256": _json_sha256(body)}


def _validate_owner_authorization(
    value: Mapping[str, Any],
    *,
    package: Mapping[str, Any],
    qualification_sha256: str,
    public_v03_sha256: str | None,
    official_v03_sha256: str | None,
    release_state: str,
    official_v03_full_score: float | None,
) -> str:
    expected = {
        "schema_version",
        "status",
        "release_state",
        "owner",
        "model_id",
        "owner_authorized_release",
        "legal_conclusion",
        "contract_sha256",
        "artifact_sha256",
        "calibration_file_sha256",
        "release_manifest_sha256",
        "package_inventory_sha256",
        "runtime_identity_sha256",
        "public_evaluation_attestation_sha256",
        "runtime_qualification_sha256",
        "public_v03_final_sha256",
        "official_v03_attestation_sha256",
        "official_v03_full_score",
        "official_v03_rank",
        "package_provenance_sha256",
        "package_provenance_file_sha256",
        "accepted_known_disclosures",
        "authority_statements_sha256",
        "mandatory_notices_sha256",
        "mandatory_notices_receipt_sha256",
        "license_shas",
        "blocker_dispositions",
        "authorized_at_utc",
        "publication_performed",
        "receipt_sha256",
    }
    _require_fields(value, expected, "owner publication authorization")
    receipt_sha = _self_hash(value, "receipt_sha256", "owner publication authorization")
    expected_disclosures = {
        "package_provenance_is_training_time_disclosure": True,
        "package_legal_conclusion": False,
        "package_publication_approval": False,
        "owner_accepts_no_legal_conclusion_claim": True,
    }
    if (
        value.get("schema_version") != OWNER_AUTHORIZATION_SCHEMA
        or value.get("status") != "OWNER_AUTHORIZED_EXACT_V3_RELEASE"
        or value.get("release_state") != release_state
        or value.get("owner") != "infercrane"
        or value.get("model_id") != MODEL_ID
        or value.get("owner_authorized_release") is not True
        or value.get("legal_conclusion") is not False
        or value.get("contract_sha256") != package["contract_sha256"]
        or value.get("artifact_sha256") != package["artifact_sha256"]
        or value.get("calibration_file_sha256") != package["calibration_file_sha256"]
        or value.get("release_manifest_sha256") != package["manifest_sha256"]
        or value.get("package_inventory_sha256") != package["inventory_sha256"]
        or value.get("runtime_identity_sha256") != package["runtime_identity_sha256"]
        or value.get("public_evaluation_attestation_sha256")
        != _public_evaluation_sha256(package)
        or value.get("runtime_qualification_sha256") != qualification_sha256
        or value.get("public_v03_final_sha256") != public_v03_sha256
        or value.get("official_v03_attestation_sha256") != official_v03_sha256
        or value.get("official_v03_full_score") != official_v03_full_score
        or value.get("official_v03_rank") is not None
        or value.get("package_provenance_sha256") != package["provenance_sha256"]
        or value.get("package_provenance_file_sha256")
        != package["provenance_file_sha256"]
        or value.get("accepted_known_disclosures") != expected_disclosures
        or value.get("authority_statements_sha256") != OWNER_AUTHORITY_STATEMENTS_SHA256
        or value.get("mandatory_notices_sha256")
        != _json_sha256(package["mandatory_notices"])
        or value.get("mandatory_notices_receipt_sha256")
        != package["mandatory_notices_receipt_sha256"]
        or value.get("license_shas") != package["license_shas"]
        or value.get("blocker_dispositions") != _blocker_dispositions(release_state)
        or value.get("publication_performed") is not False
    ):
        raise V3PublicationError("owner publication authorization is detached")
    _utc_timestamp(value.get("authorized_at_utc"), "owner authorization time")
    return receipt_sha


def build_evidence_gate(
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    public_v03: Path | None,
    official_v03: Path | None,
    authorization: Path,
) -> dict[str, Any]:
    (
        validated,
        qualification_sha,
        public_sha,
        official_sha,
        release_state,
        official_full_score,
    ) = _validated_release_materials(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public_v03,
        official_v03=official_v03,
    )
    return _evidence_gate_from_validated_materials(
        validated=validated,
        qualification_sha=qualification_sha,
        public_sha=public_sha,
        official_sha=official_sha,
        release_state=release_state,
        official_full_score=official_full_score,
        authorization=_object(authorization, "owner publication authorization"),
    )


def _evidence_gate_from_validated_materials(
    *,
    validated: Mapping[str, Any],
    qualification_sha: str,
    public_sha: str | None,
    official_sha: str | None,
    release_state: str,
    official_full_score: float | None,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the gate without re-reading already validated package bytes."""

    authorization_sha = _validate_owner_authorization(
        authorization,
        package=validated,
        qualification_sha256=qualification_sha,
        public_v03_sha256=public_sha,
        official_v03_sha256=official_sha,
        release_state=release_state,
        official_v03_full_score=official_full_score,
    )
    body = {
        "schema_version": GATE_SCHEMA,
        "status": (
            "PASS_OFFICIAL_V0_3_PUBLICATION_EVIDENCE"
            if release_state == OFFICIAL_RELEASE_STATE
            else "PASS_INITIAL_PUBLICATION_EVIDENCE"
        ),
        "release_state": release_state,
        "owner": "infercrane",
        "model_id": MODEL_ID,
        "artifact_sha256": validated["artifact_sha256"],
        "release_manifest_sha256": validated["manifest_sha256"],
        "package_inventory_sha256": validated["inventory_sha256"],
        "runtime_qualification_sha256": qualification_sha,
        "public_v03_final_sha256": public_sha,
        "official_v03_attestation_sha256": official_sha,
        "official_v03_full_score": official_full_score,
        "official_v03_rank": None,
        "owner_publication_authorization_sha256": authorization_sha,
        "network_used": False,
        "publication_performed": False,
    }
    return {**body, "receipt_sha256": _json_sha256(body)}


def _validate_receipt(
    value: Mapping[str, Any], schema: str, status: str, label: str
) -> str:
    if value.get("schema_version") != schema or value.get("status") != status:
        raise V3PublicationError(f"{label} status changed")
    return _self_hash(value, "receipt_sha256", label)


def _validate_gate(value: Mapping[str, Any]) -> str:
    expected = {
        "schema_version",
        "status",
        "release_state",
        "owner",
        "model_id",
        "artifact_sha256",
        "release_manifest_sha256",
        "package_inventory_sha256",
        "runtime_qualification_sha256",
        "public_v03_final_sha256",
        "official_v03_attestation_sha256",
        "official_v03_full_score",
        "official_v03_rank",
        "owner_publication_authorization_sha256",
        "network_used",
        "publication_performed",
        "receipt_sha256",
    }
    _require_fields(value, expected, "publication gate")
    release_state = value.get("release_state")
    expected_status = (
        "PASS_OFFICIAL_V0_3_PUBLICATION_EVIDENCE"
        if release_state == OFFICIAL_RELEASE_STATE
        else "PASS_INITIAL_PUBLICATION_EVIDENCE"
        if release_state == INITIAL_RELEASE_STATE
        else None
    )
    if expected_status is None:
        raise V3PublicationError("publication gate release state changed")
    receipt_sha = _validate_receipt(
        value, GATE_SCHEMA, expected_status, "publication gate"
    )
    if (
        value.get("owner") != "infercrane"
        or value.get("model_id") != MODEL_ID
        or value.get("network_used") is not False
        or value.get("publication_performed") is not False
    ):
        raise V3PublicationError("publication gate ownership or state changed")
    for field in (
        "artifact_sha256",
        "release_manifest_sha256",
        "package_inventory_sha256",
        "runtime_qualification_sha256",
        "owner_publication_authorization_sha256",
    ):
        _sha(value.get(field), f"publication gate {field}")
    public_v03 = value.get("public_v03_final_sha256")
    official_v03 = value.get("official_v03_attestation_sha256")
    full_score = value.get("official_v03_full_score")
    if release_state == INITIAL_RELEASE_STATE:
        if (
            public_v03 is not None
            or official_v03 is not None
            or full_score is not None
            or value.get("official_v03_rank") is not None
        ):
            raise V3PublicationError(
                "initial publication gate overstates v0.3 evidence"
            )
    else:
        _sha(public_v03, "publication gate public_v03_final_sha256")
        _sha(official_v03, "publication gate official_v03_attestation_sha256")
        if (
            isinstance(full_score, bool)
            or not isinstance(full_score, (int, float))
            or not math.isfinite(float(full_score))
            or value.get("official_v03_rank") is not None
        ):
            raise V3PublicationError("official publication gate score or rank changed")
    return receipt_sha


def _validate_private_plan(value: Mapping[str, Any]) -> str:
    expected = {
        "schema_version",
        "status",
        "publication_gate_sha256",
        "artifact_sha256",
        "package_inventory_sha256",
        "source_revision",
        "github",
        "huggingface",
        "network_used",
        "repositories_created",
        "publication_performed",
        "receipt_sha256",
    }
    _require_fields(value, expected, "private rehearsal")
    receipt_sha = _validate_receipt(
        value,
        PRIVATE_PLAN_SCHEMA,
        "PRIVATE_REHEARSAL_PLANNED",
        "private rehearsal",
    )
    if (
        value.get("github")
        != {"repository": GITHUB_REPOSITORY, "visibility": "private"}
        or value.get("huggingface")
        != {"repository": HF_REPOSITORY, "visibility": "private"}
        or value.get("network_used") is not False
        or value.get("repositories_created") is not False
        or value.get("publication_performed") is not False
    ):
        raise V3PublicationError(
            "private rehearsal owner, visibility, or state changed"
        )
    _sha(value.get("publication_gate_sha256"), "private rehearsal publication gate")
    _sha(value.get("artifact_sha256"), "private rehearsal artifact")
    _sha(value.get("package_inventory_sha256"), "private rehearsal inventory")
    _revision(value.get("source_revision"), "private rehearsal source revision")
    return receipt_sha


def _validate_private_verification(value: Mapping[str, Any]) -> str:
    expected = {
        "schema_version",
        "status",
        "private_rehearsal_sha256",
        "publication_gate_sha256",
        "artifact_sha256",
        "github_verification_sha256",
        "huggingface_verification_sha256",
        "publication_performed",
        "receipt_sha256",
    }
    _require_fields(value, expected, "private verification")
    receipt_sha = _validate_receipt(
        value,
        PRIVATE_VERIFICATION_SCHEMA,
        "PASS_PRIVATE_REHEARSAL",
        "private verification",
    )
    if value.get("publication_performed") is not False:
        raise V3PublicationError("private verification publication state changed")
    for field in (
        "private_rehearsal_sha256",
        "publication_gate_sha256",
        "artifact_sha256",
        "github_verification_sha256",
        "huggingface_verification_sha256",
    ):
        _sha(value.get(field), f"private verification {field}")
    return receipt_sha


def _validate_public_plan(
    value: Mapping[str, Any],
    *,
    gate: Mapping[str, Any] | None = None,
    private: Mapping[str, Any] | None = None,
) -> str:
    expected = {
        "schema_version",
        "status",
        "owner",
        "model_id",
        "artifact_sha256",
        "publication_gate_sha256",
        "private_verification_sha256",
        "github_repository",
        "huggingface_repository",
        "network_used",
        "repositories_created",
        "publication_performed",
        "receipt_sha256",
    }
    _require_fields(value, expected, "public transition plan")
    receipt_sha = _validate_receipt(
        value,
        PUBLIC_PLAN_SCHEMA,
        "PUBLIC_TRANSITION_AUTHORIZED_NOT_PERFORMED",
        "public transition plan",
    )
    if (
        value.get("owner") != "infercrane"
        or value.get("model_id") != MODEL_ID
        or value.get("github_repository") != GITHUB_REPOSITORY
        or value.get("huggingface_repository") != HF_REPOSITORY
        or value.get("network_used") is not False
        or value.get("repositories_created") is not False
        or value.get("publication_performed") is not False
    ):
        raise V3PublicationError("public transition ownership or state changed")
    for field in (
        "artifact_sha256",
        "publication_gate_sha256",
        "private_verification_sha256",
    ):
        _sha(value.get(field), f"public transition {field}")
    if gate is not None:
        gate_sha = _validate_gate(gate)
        if value.get("publication_gate_sha256") != gate_sha or value.get(
            "artifact_sha256"
        ) != gate.get("artifact_sha256"):
            raise V3PublicationError("public transition targets another gate")
    if private is not None:
        private_sha = _validate_private_verification(private)
        if value.get("private_verification_sha256") != private_sha or value.get(
            "artifact_sha256"
        ) != private.get("artifact_sha256"):
            raise V3PublicationError(
                "public transition targets another private verification"
            )
    return receipt_sha


def build_private_plan(
    gate: Mapping[str, Any], *, source_revision: str
) -> dict[str, Any]:
    gate_sha = _validate_gate(gate)
    body = {
        "schema_version": PRIVATE_PLAN_SCHEMA,
        "status": "PRIVATE_REHEARSAL_PLANNED",
        "publication_gate_sha256": gate_sha,
        "artifact_sha256": gate["artifact_sha256"],
        "package_inventory_sha256": gate["package_inventory_sha256"],
        "source_revision": _revision(source_revision, "source revision"),
        "github": {"repository": GITHUB_REPOSITORY, "visibility": "private"},
        "huggingface": {"repository": HF_REPOSITORY, "visibility": "private"},
        "network_used": False,
        "repositories_created": False,
        "publication_performed": False,
    }
    return {**body, "receipt_sha256": _json_sha256(body)}


def _validate_remote(
    value: Mapping[str, Any],
    *,
    service: str,
    repository: str,
    artifact_sha256: str,
    source_revision: str,
    package_inventory_sha256: str,
) -> str:
    expected = {
        "schema_version",
        "status",
        "service",
        "repository",
        "visibility",
        "revision",
        "source_revision",
        "artifact_sha256",
        "inventory_sha256",
        "verified",
        "verified_at_utc",
        "receipt_sha256",
    }
    if service == "huggingface":
        expected.add("hub_metadata")
    _require_fields(value, expected, f"{service} verification")
    receipt_sha = _self_hash(value, "receipt_sha256", f"{service} verification")
    if (
        value.get("schema_version") != REMOTE_RECEIPT_SCHEMA
        or value.get("status") != "IMMUTABLE_PRIVATE_REVISION_VERIFIED"
        or value.get("service") != service
        or value.get("repository") != repository
        or value.get("visibility") != "private"
        or value.get("verified") is not True
        or value.get("source_revision") != source_revision
        or value.get("artifact_sha256") != artifact_sha256
        or (service == "github" and value.get("revision") != source_revision)
        or (
            service == "huggingface"
            and value.get("inventory_sha256") != package_inventory_sha256
        )
    ):
        raise V3PublicationError(
            f"{service} private revision is unverified or detached"
        )
    _revision(value.get("revision"), f"{service} revision")
    _sha(value.get("inventory_sha256"), f"{service} inventory")
    if service == "huggingface":
        metadata = value.get("hub_metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != {".gitattributes"}:
            raise V3PublicationError("huggingface metadata binding changed")
        _sha(metadata.get(".gitattributes"), "huggingface .gitattributes")
    _utc_timestamp(value.get("verified_at_utc"), f"{service} verification time")
    return receipt_sha


def build_private_verification(
    plan: Mapping[str, Any], github: Mapping[str, Any], huggingface: Mapping[str, Any]
) -> dict[str, Any]:
    plan_sha = _validate_private_plan(plan)
    artifact = _sha(plan.get("artifact_sha256"), "private rehearsal artifact")
    package_inventory = _sha(
        plan.get("package_inventory_sha256"), "private rehearsal package inventory"
    )
    source_revision = _revision(
        plan.get("source_revision"), "private rehearsal source revision"
    )
    body = {
        "schema_version": PRIVATE_VERIFICATION_SCHEMA,
        "status": "PASS_PRIVATE_REHEARSAL",
        "private_rehearsal_sha256": plan_sha,
        "publication_gate_sha256": plan["publication_gate_sha256"],
        "artifact_sha256": artifact,
        "github_verification_sha256": _validate_remote(
            github,
            service="github",
            repository=GITHUB_REPOSITORY,
            artifact_sha256=artifact,
            source_revision=source_revision,
            package_inventory_sha256=package_inventory,
        ),
        "huggingface_verification_sha256": _validate_remote(
            huggingface,
            service="huggingface",
            repository=HF_REPOSITORY,
            artifact_sha256=artifact,
            source_revision=source_revision,
            package_inventory_sha256=package_inventory,
        ),
        "publication_performed": False,
    }
    return {**body, "receipt_sha256": _json_sha256(body)}


def build_public_plan(
    gate: Mapping[str, Any], private: Mapping[str, Any]
) -> dict[str, Any]:
    gate_sha = _validate_gate(gate)
    private_sha = _validate_private_verification(private)
    if gate.get("artifact_sha256") != private.get("artifact_sha256"):
        raise V3PublicationError("private verification targets another artifact")
    if private.get("publication_gate_sha256") != gate_sha:
        raise V3PublicationError(
            "private verification targets another publication gate"
        )
    body = {
        "schema_version": PUBLIC_PLAN_SCHEMA,
        "status": "PUBLIC_TRANSITION_AUTHORIZED_NOT_PERFORMED",
        "owner": "infercrane",
        "model_id": MODEL_ID,
        "artifact_sha256": gate["artifact_sha256"],
        "publication_gate_sha256": gate_sha,
        "private_verification_sha256": private_sha,
        "github_repository": GITHUB_REPOSITORY,
        "huggingface_repository": HF_REPOSITORY,
        "network_used": False,
        "repositories_created": False,
        "publication_performed": False,
    }
    result = {**body, "receipt_sha256": _json_sha256(body)}
    _validate_public_plan(result, gate=gate, private=private)
    return result


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists():
        raise V3PublicationError("output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    authorize = commands.add_parser("authorize")
    for name in (
        "package",
        "public-repo",
        "qualification",
        "output",
    ):
        authorize.add_argument(f"--{name}", type=Path, required=True)
    authorize.add_argument("--public-v03", type=Path)
    authorize.add_argument("--official-v03", type=Path)
    authorize.add_argument("--confirmation", required=True)
    authorize.add_argument("--authorized-at-utc", required=True)
    gate = commands.add_parser("gate")
    for name in (
        "package",
        "public-repo",
        "qualification",
        "authorization",
        "output",
    ):
        gate.add_argument(f"--{name}", type=Path, required=True)
    gate.add_argument("--public-v03", type=Path)
    gate.add_argument("--official-v03", type=Path)
    private = commands.add_parser("private-plan")
    private.add_argument("--gate", type=Path, required=True)
    private.add_argument("--source-revision", required=True)
    private.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify-private")
    for name in ("plan", "github", "huggingface", "output"):
        verify.add_argument(f"--{name}", type=Path, required=True)
    public = commands.add_parser("public-plan")
    public.add_argument("--gate", type=Path, required=True)
    public.add_argument("--private-verification", type=Path, required=True)
    public.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "authorize":
            result = build_owner_authorization(
                package=args.package,
                public_repo=args.public_repo,
                qualification=args.qualification,
                public_v03=args.public_v03,
                official_v03=args.official_v03,
                confirmation=args.confirmation,
                authorized_at_utc=args.authorized_at_utc,
            )
        elif args.command == "gate":
            result = build_evidence_gate(
                package=args.package,
                public_repo=args.public_repo,
                qualification=args.qualification,
                public_v03=args.public_v03,
                official_v03=args.official_v03,
                authorization=args.authorization,
            )
        elif args.command == "private-plan":
            result = build_private_plan(
                _object(args.gate, "publication gate"),
                source_revision=args.source_revision,
            )
        elif args.command == "verify-private":
            result = build_private_verification(
                _object(args.plan, "private rehearsal"),
                _object(args.github, "GitHub verification"),
                _object(args.huggingface, "Hugging Face verification"),
            )
        else:
            result = build_public_plan(
                _object(args.gate, "publication gate"),
                _object(args.private_verification, "private verification"),
            )
        _write_once(args.output, result)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
