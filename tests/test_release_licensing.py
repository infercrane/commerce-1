from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_third_party_notice_covers_only_the_current_launch_lineage() -> None:
    notice = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")

    assert "## Training sources" in notice
    assert "perplexity-ai/pplx-decider-v1.1-27b" in notice
    assert "nvidia/HelpSteer2" in notice
    assert "Anthropic/hh-rlhf" in notice
    assert "allenai/qasc" in notice
    assert "kubernetes/kubernetes" in notice
    assert "denis-pplx/autojev-27b" not in notice
    assert "effective-v4" not in notice


def test_third_party_notice_binds_current_upstream_hash_evidence() -> None:
    notice = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")

    for value in (
        "3b45dead91dfa6d95aad6b95764a606fab2bf7a6",
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
        "3792cbe9e964a7ed65fff010d3f94bb27c6e726e2940ab6b0e1db5d30df92510",
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b",
        "f75f40db0268656ba07736ec8e59a9720c1910ce554c85354cec74b1c8bda175",
        "c90ae4776b98cd6fc0a22558784c43b07e1f0b00a090dda99797d6b6efab4a3e",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "d4fb18db48757e273261a5597b8a09381de8073da060474ce33d0643e06c875e",
    ):
        assert value in notice


def test_runtime_pins_are_not_presented_as_training_license_evidence() -> None:
    notice = (ROOT / "THIRD_PARTY.md").read_text(encoding="utf-8")
    normalized = " ".join(notice.split())

    assert "dependency versions are pinned" in notice
    assert "retain their own upstream licenses" in notice
    assert "runtime dependencies, not training-data sources" in normalized


def test_current_release_licenses_are_public_and_legacy_notices_are_absent() -> None:
    current = ROOT / "release/licenses/commerce-1"
    assert (current / "models/pplx-decider-LICENSE").is_file()
    assert (current / "models/qwen-LICENSE").is_file()
    assert (current / "training/kubernetes-LICENSE").is_file()
    assert not list((ROOT / "licenses/upstream").glob("*"))
