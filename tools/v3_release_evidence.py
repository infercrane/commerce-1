#!/usr/bin/env python3
"""Validate the public, content-free Commerce-1 evidence snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools import v3_publication as publication
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    import v3_publication as publication

SCHEMA_VERSION = "infercrane-commerce-1-public-evidence-snapshot/v4"
STATUS = "PUBLIC_EVALUATIONS_COMPLETE_RUNTIME_REQUALIFICATION_PENDING"
QUALIFIED_STATUS = "PUBLIC_EVALUATION_AND_RUNTIME_QUALIFICATION_COMPLETE"
EVIDENCE_PATH = Path("release/commerce-1-evidence.json")
CANONICAL_RELEASE_ARTIFACT_SHA256 = (
    "f97262c59b102a39ae37826ffe1e1db8ddabaa1ec95b4e73bcb44328420ace57"
)
CANONICAL_ARTIFACT_TRANSFORM_SHA256 = (
    "ac7ae1330b4e912697b11ddb41770121f37ff88047e5f2046e3a338f9307d561"
)
CANONICAL_PUBLIC_QUALIFICATION_ATTESTATION_SHA256 = (
    "2b01ab21b12c1d1c555688f205717708ade4a1b0a5b415aec08d36148c8e3639"
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class V3ReleaseEvidenceError(ValueError):
    """The public evidence snapshot is incomplete, overstated, or changed."""


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


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise V3ReleaseEvidenceError(f"{label} must be a lowercase SHA-256")
    return value


def validate_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact public evidence without implying publication or rank."""

    expected_fields = {
        "schema_version",
        "status",
        "model_id",
        "contract_sha256",
        "evaluated_artifact_sha256",
        "release_artifact_sha256",
        "artifact_transform_sha256",
        "calibration_file_sha256",
        "public_evaluation",
        "current_leaderboard",
        "native_runtime_qualification",
        "release_gates",
        "post_launch_evidence",
        "publication_performed",
        "receipt_sha256",
    }
    evidence = dict(value)
    if set(evidence) != expected_fields:
        raise V3ReleaseEvidenceError("public evidence snapshot fields changed")
    supplied_receipt = _sha(evidence.pop("receipt_sha256", None), "receipt")
    if supplied_receipt != _json_sha256(evidence):
        raise V3ReleaseEvidenceError("public evidence snapshot self-hash changed")

    evaluation = evidence.get("public_evaluation")
    leaderboard = evidence.get("current_leaderboard")
    qualification = evidence.get("native_runtime_qualification")
    gates = evidence.get("release_gates")
    post_launch = evidence.get("post_launch_evidence")
    if (
        evidence.get("schema_version") != SCHEMA_VERSION
        or evidence.get("status") not in {STATUS, QUALIFIED_STATUS}
        or evidence.get("model_id") != publication.MODEL_ID
        or evidence.get("contract_sha256") != publication.CANONICAL_CONTRACT_SHA256
        or evidence.get("evaluated_artifact_sha256")
        != publication.CANONICAL_EVALUATED_ARTIFACT_SHA256
        or evidence.get("calibration_file_sha256")
        != publication.CANONICAL_CALIBRATION_FILE_SHA256
        or evidence.get("publication_performed") is not False
        or not isinstance(evaluation, Mapping)
        or not isinstance(leaderboard, Mapping)
        or not isinstance(qualification, Mapping)
        or not isinstance(gates, Mapping)
        or not isinstance(post_launch, Mapping)
    ):
        raise V3ReleaseEvidenceError("public evidence snapshot identity changed")

    release_artifact = evidence.get("release_artifact_sha256")
    artifact_transform = evidence.get("artifact_transform_sha256")
    if (release_artifact is None) != (artifact_transform is None):
        raise V3ReleaseEvidenceError(
            "release artifact and transform must become available together"
        )
    if release_artifact is not None:
        _sha(release_artifact, "release artifact")
        _sha(artifact_transform, "artifact transform")

    expected_evaluation = {
        "edition": "0.2.1",
        "scope": "complete_public_suite",
        "score": "60.99",
        "historical_release_baseline": "52.55",
        "margin_over_historical_baseline": "8.44",
        "status": "PASS_COMPLETE_LOCAL_REPRODUCTION",
        "evaluation_authority": "self_hosted",
        "selected_in_edition": 120340,
        "selected_scoreable": 119898,
        "excluded_selected": 442,
        "added": 30419,
        "merged": 150317,
        "counts": {"ok": 150317},
        "results_sha256": publication.CANONICAL_V021_RESULTS_SHA256,
        "scores_sha256": publication.CANONICAL_V021_SCORES_SHA256,
    }
    if dict(evaluation) != expected_evaluation:
        raise V3ReleaseEvidenceError("public Decision Index 0.2.1 evidence changed")
    if not math.isclose(
        float(evaluation["score"]) - float(evaluation["historical_release_baseline"]),
        float(evaluation["margin_over_historical_baseline"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise V3ReleaseEvidenceError("public evaluation margin does not reproduce")

    if dict(leaderboard) != {
        "edition": "0.3",
        "official_full_score": None,
        "official_rank": None,
        "rank_claimed": False,
        "note": (
            "Commerce-1 has no official Decision Index 0.3 Full Score or current rank."
        ),
    }:
        raise V3ReleaseEvidenceError("leaderboard scope was overstated")
    pending_gates = {
        "exact_public_v0_2_1": "complete",
        "native_runtime_qualification": (
            "pending_requalification_after_runtime_contract_hardening"
        ),
        "private_stage_and_byte_verification": "pending",
        "public_transition": "pending",
    }
    pending_qualification = {
        "status": "REQUALIFICATION_REQUIRED",
        "public_projection_schema": publication.PUBLIC_QUALIFICATION_SCHEMA,
        "public_projection_attestation_sha256": None,
        "maximum_input_tokens": 65_536,
        "question_microbatch_size": 1,
        "publication_performed": False,
        "current_for_release": False,
        "reason": (
            "Runtime contract hardening changed release bytes; a new native "
            "qualification must bind the final public package."
        ),
    }
    if dict(post_launch) != {
        "public_v0_3_evaluation": "pending_content_bound_public_artifact",
        "official_v0_3_maintainer_evaluation": "pending",
    }:
        raise V3ReleaseEvidenceError("post-launch evidence snapshot changed")
    if evidence.get("status") == STATUS:
        if (
            release_artifact is not None
            or artifact_transform is not None
            or dict(qualification) != pending_qualification
            or dict(gates) != pending_gates
            or evidence.get("publication_performed") is not False
        ):
            raise V3ReleaseEvidenceError(
                "prequalification evidence cannot pass a release gate"
            )
    else:
        if release_artifact is None or artifact_transform is None:
            raise V3ReleaseEvidenceError(
                "qualified evidence requires release and transform identities"
            )
        projection_attestation = _sha(
            qualification.get("public_projection_attestation_sha256"),
            "public qualification attestation",
        )
        if (
            release_artifact != CANONICAL_RELEASE_ARTIFACT_SHA256
            or artifact_transform != CANONICAL_ARTIFACT_TRANSFORM_SHA256
            or projection_attestation
            != CANONICAL_PUBLIC_QUALIFICATION_ATTESTATION_SHA256
        ):
            raise V3ReleaseEvidenceError(
                "qualified release bindings differ from the final receipt"
            )
        qualified_summary = {
            "status": "PASS_NATIVE_RUNTIME_QUALIFICATION",
            "public_projection_schema": publication.PUBLIC_QUALIFICATION_SCHEMA,
            "public_projection_attestation_sha256": projection_attestation,
            "maximum_input_tokens": 65_536,
            "question_microbatch_size": 1,
            "publication_performed": False,
            "current_for_release": True,
        }
        if dict(qualification) != qualified_summary:
            raise V3ReleaseEvidenceError(
                "public qualification summary changed or exposes protected fields"
            )
        qualified_gates = {
            **pending_gates,
            "native_runtime_qualification": "complete",
        }
        if (
            dict(gates) != qualified_gates
            or evidence.get("publication_performed") is not False
        ):
            raise V3ReleaseEvidenceError("qualified release gates changed")
    return {**evidence, "receipt_sha256": supplied_receipt}


def load_and_validate(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise V3ReleaseEvidenceError("evidence path must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V3ReleaseEvidenceError("cannot read evidence JSON") from error
    if not isinstance(value, Mapping):
        raise V3ReleaseEvidenceError("evidence JSON must be an object")
    return validate_evidence(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=EVIDENCE_PATH)
    arguments = parser.parse_args()
    validated = load_and_validate(arguments.path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "model_id": validated["model_id"],
                "public_decision_index_0_2_1": validated["public_evaluation"]["score"],
                "official_v0_3_rank": None,
                "receipt_sha256": validated["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
