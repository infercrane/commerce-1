from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools import v3_release_evidence as evidence

REPOSITORY = Path(__file__).resolve().parents[1]


def _value() -> dict[str, object]:
    return json.loads((REPOSITORY / evidence.EVIDENCE_PATH).read_text())


def test_checked_in_v3_evidence_is_exact_and_rank_safe() -> None:
    validated = evidence.load_and_validate(REPOSITORY / evidence.EVIDENCE_PATH)
    assert validated["public_evaluation"]["score"] == "60.99"
    assert validated["current_leaderboard"]["official_full_score"] is None
    assert validated["current_leaderboard"]["official_rank"] is None
    assert validated["current_leaderboard"]["rank_claimed"] is False
    assert validated["native_runtime_qualification"]["status"] == (
        "PASS_NATIVE_RUNTIME_QUALIFICATION"
    )
    assert validated["native_runtime_qualification"]["current_for_release"] is True
    assert validated["release_artifact_sha256"] == (
        "f97262c59b102a39ae37826ffe1e1db8ddabaa1ec95b4e73bcb44328420ace57"
    )
    assert validated["artifact_transform_sha256"] == (
        "ac7ae1330b4e912697b11ddb41770121f37ff88047e5f2046e3a338f9307d561"
    )
    assert validated["native_runtime_qualification"]["public_projection_schema"] == (
        "infercrane-commerce-1-public-runtime-qualification/v2"
    )
    assert validated["release_gates"]["native_runtime_qualification"] == "complete"
    assert validated["publication_performed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("public_evaluation", "score"), "61.00"),
        (("public_evaluation", "results_sha256"), "0" * 64),
        (("current_leaderboard", "official_rank"), 2),
        (("current_leaderboard", "rank_claimed"), True),
        (("release_gates", "native_runtime_qualification"), "pending"),
        (("native_runtime_qualification", "question_microbatch_size"), 8),
        (
            (
                "native_runtime_qualification",
                "public_projection_attestation_sha256",
            ),
            "0" * 64,
        ),
        (("native_runtime_qualification", "attempt_id"), "1" * 32),
        (("release_artifact_sha256",), "0" * 64),
        (("publication_performed",), True),
    ],
)
def test_v3_evidence_rejects_changed_or_overstated_claims(
    path: tuple[str, ...], value: object
) -> None:
    candidate = copy.deepcopy(_value())
    current: dict[str, object] = candidate
    for key in path[:-1]:
        current = current[key]  # type: ignore[assignment]
    current[path[-1]] = value
    body = {key: item for key, item in candidate.items() if key != "receipt_sha256"}
    candidate["receipt_sha256"] = evidence._json_sha256(body)
    with pytest.raises(evidence.V3ReleaseEvidenceError):
        evidence.validate_evidence(candidate)


def test_v3_publication_pins_exact_canonical_evaluation() -> None:
    from tools import v3_publication as publication

    assert publication.CANONICAL_V021_DECISION_INDEX == 60.99
    assert publication.CANONICAL_V021_RESULTS_SHA256 == (
        "a8e6241e86614c95d224090bb361c8ec92153256fe2e114ed6e9f6041bc69ad4"
    )
    checked_in = _value()
    assert "attempt_id" not in checked_in["public_evaluation"]
    assert "final_receipt_sha256" not in checked_in["public_evaluation"]


def test_qualified_snapshot_keeps_only_safe_projection_summary() -> None:
    candidate = _value()
    candidate["status"] = evidence.QUALIFIED_STATUS
    candidate["release_artifact_sha256"] = evidence.CANONICAL_RELEASE_ARTIFACT_SHA256
    candidate["artifact_transform_sha256"] = (
        evidence.CANONICAL_ARTIFACT_TRANSFORM_SHA256
    )
    candidate["native_runtime_qualification"] = {
        "status": "PASS_NATIVE_RUNTIME_QUALIFICATION",
        "public_projection_schema": (
            "infercrane-commerce-1-public-runtime-qualification/v2"
        ),
        "public_projection_attestation_sha256": (
            evidence.CANONICAL_PUBLIC_QUALIFICATION_ATTESTATION_SHA256
        ),
        "maximum_input_tokens": 65_536,
        "question_microbatch_size": 1,
        "publication_performed": False,
        "current_for_release": True,
    }
    candidate["release_gates"]["native_runtime_qualification"] = "complete"
    body = {key: item for key, item in candidate.items() if key != "receipt_sha256"}
    candidate["receipt_sha256"] = evidence._json_sha256(body)
    validated = evidence.validate_evidence(candidate)
    assert (
        validated["native_runtime_qualification"][
            "public_projection_attestation_sha256"
        ]
        == evidence.CANONICAL_PUBLIC_QUALIFICATION_ATTESTATION_SHA256
    )

    candidate["native_runtime_qualification"]["attempt_id"] = "1" * 32
    body = {key: item for key, item in candidate.items() if key != "receipt_sha256"}
    candidate["receipt_sha256"] = evidence._json_sha256(body)
    with pytest.raises(evidence.V3ReleaseEvidenceError, match="protected fields"):
        evidence.validate_evidence(candidate)
