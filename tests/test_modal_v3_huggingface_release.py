from __future__ import annotations

import hashlib
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tools import modal_v3_huggingface_release as release


def test_remote_verification_disk_matches_current_modal_minimum() -> None:
    assert release.REMOTE_VERIFICATION_EPHEMERAL_DISK_MIB == 512 * 1024


def test_modal_hub_transport_uses_only_mounted_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []

    def api_factory(*, token: str) -> object:
        calls.append({"api_token": token})
        return sentinel

    def download_factory(*, token: str, **kwargs: object) -> str:
        calls.append({"download_token": token, **kwargs})
        return "/tmp/downloaded"

    monkeypatch.setenv("HF_TOKEN", "fixture-secret")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=api_factory,
            hf_hub_download=download_factory,
        ),
    )

    api, downloader = release._modal_hub_transport()

    assert api is sentinel
    assert downloader(repo_id="infercrane/Commerce-1", filename="README.md") == (
        "/tmp/downloaded"
    )
    assert calls == [
        {"api_token": "fixture-secret"},
        {
            "download_token": "fixture-secret",
            "repo_id": "infercrane/Commerce-1",
            "filename": "README.md",
        },
    ]


def test_modal_hub_transport_fails_without_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="unavailable"):
        release._modal_hub_transport()


def _write(root: Path, name: str, payload: bytes) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _item(payload: bytes) -> dict[str, object]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}


def _card() -> bytes:
    return (
        b"---\ninference: false\n---\n"
        b"# InferCrane Commerce-1\n\nModel: infercrane/Commerce-1\n"
    )


def _write_json(root: Path, name: str, value: object) -> None:
    _write(root, name, (release._canonical_json(value) + "\n").encode())


def _public_package() -> dict[str, object]:
    return {
        "manifest_sha256": "a" * 64,
        "inventory_sha256": "b" * 64,
        "release_artifact_sha256": "c" * 64,
        "calibration_file_sha256": "d" * 64,
        "runtime_manifest_sha256": "e" * 64,
        "runtime_identity_sha256": "f" * 64,
        "public_evaluation_file_sha256": "1" * 64,
        "public_evaluation": {
            "attestation_sha256": "2" * 64,
            "decision_index": 60.99,
            "merged": 150_317,
        },
        "training_data_lineage": {"lineage_sha256": "3" * 64},
    }


def _materialize_public_assets(root: Path, package: dict[str, object]) -> None:
    qualification_sha = "4" * 64
    protected_sha = "5" * 64
    _write(root, "README.md", _card())
    for name in (
        "ARCHITECTURE.md",
        "USAGE.md",
        "LOCAL_RUN.md",
        "API.md",
        "REPRODUCIBILITY.md",
        "GITHUB_RELEASE.md",
        "LICENSE",
        "THIRD_PARTY.md",
    ):
        _write(root, name, f"public {name}\n".encode())
    _write_json(root, "PROVENANCE.json", {"public": True})
    _write_json(
        root,
        "QUALIFICATION.json",
        {
            "attestation_sha256": qualification_sha,
            "protected_qualification_receipt_sha256": protected_sha,
        },
    )
    public_evaluation = package["public_evaluation"]
    training_lineage = package["training_data_lineage"]
    assert isinstance(public_evaluation, dict)
    assert isinstance(training_lineage, dict)
    evidence_body = {
        "schema_version": release.PUBLIC_EVIDENCE_SCHEMA,
        "status": release.PUBLIC_ASSET_STATUS,
        "model_id": release.publication.MODEL_ID,
        "release_manifest_sha256": package["manifest_sha256"],
        "package_files_sha256": package["inventory_sha256"],
        "artifact_sha256": package["release_artifact_sha256"],
        "calibration_file_sha256": package["calibration_file_sha256"],
        "runtime_manifest_sha256": package["runtime_manifest_sha256"],
        "runtime_identity_sha256": package["runtime_identity_sha256"],
        "public_evaluation_file_sha256": package["public_evaluation_file_sha256"],
        "public_evaluation_attestation_sha256": public_evaluation["attestation_sha256"],
        "training_data_lineage_sha256": training_lineage["lineage_sha256"],
        "public_qualification_attestation_sha256": qualification_sha,
        "protected_qualification_receipt_sha256": protected_sha,
        "qualification_passed": True,
        "results": {
            "decision_index_0_2_1": public_evaluation["decision_index"],
            "evaluated_rows": 150_317,
            "official_decision_index_0_3_full_score": None,
            "official_rank": None,
        },
        "release_state": release.PUBLIC_RELEASE_STATE,
        "release_evidence_ready": True,
        "publication_gate": release.PUBLIC_RELEASE_STATE,
        "publication_ready": False,
        "upload_performed": False,
        "publication_performed": False,
    }
    evidence = {
        **evidence_body,
        "evidence_sha256": release._json_sha256(evidence_body),
    }
    _write_json(root, "verified-evidence.json", evidence)
    payload_hashes = {
        path.name: release._file_sha256(path)
        for path in root.iterdir()
        if path.is_file()
    }
    checksums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(payload_hashes.items())
    ).encode()
    _write(root, "SHA256SUMS", checksums)
    file_hashes = {
        path.name: release._file_sha256(path)
        for path in root.iterdir()
        if path.is_file()
    }
    manifest_body = {
        "schema_version": release.PUBLIC_ASSET_SCHEMA,
        "status": release.PUBLIC_ASSET_STATUS,
        "organization": "infercrane",
        "model_id": release.publication.MODEL_ID,
        "package_manifest_sha256": package["manifest_sha256"],
        "verified_evidence_sha256": evidence["evidence_sha256"],
        "public_qualification_attestation_sha256": qualification_sha,
        "unresolved_placeholders": [],
        "files": file_hashes,
        "checksums": {
            "path": "SHA256SUMS",
            "sha256": file_hashes["SHA256SUMS"],
        },
        "upload_performed": False,
        "publication_performed": False,
    }
    _write_json(
        root,
        "publication-manifest.json",
        {**manifest_body, "manifest_sha256": release._json_sha256(manifest_body)},
    )


def test_assets_supply_card_without_copying_package(tmp_path: Path) -> None:
    package = tmp_path / "package"
    assets = tmp_path / "assets"
    model = b"weights"
    _write(package, "model/model.safetensors", model)
    _write(assets, "README.md", _card())
    asset_inventory = release._scan_public_asset_inventory(assets)
    combined, sources = release._merge_sources(
        package_root=package,
        package_inventory={"model/model.safetensors": _item(model)},
        asset_root=assets,
        asset_inventory=asset_inventory,
    )
    assert set(combined) == {
        "README.md",
        "release-assets/README.md",
        "model/model.safetensors",
    }
    assert sources["README.md"] == assets / "README.md"
    assert sources["release-assets/README.md"] == assets / "README.md"
    assert sources["model/model.safetensors"] == package / "model/model.safetensors"
    assert list(package.rglob("README.md")) == []


def test_public_card_allows_non_token_wildcard_notation(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    _write(
        assets,
        "README.md",
        _card() + b"Unresolved fields look like `{{VERIFIED_*}}`.\n",
    )

    inventory = release._scan_public_asset_inventory(assets)

    assert set(inventory) == {"README.md"}


def test_overlay_rejects_changed_package_path(tmp_path: Path) -> None:
    package = tmp_path / "package"
    assets = tmp_path / "assets"
    _write(package, "README.md", b"immutable\n")
    _write(assets, "README.md", _card())
    with pytest.raises(release.ModalHuggingFaceReleaseError, match="conflicts"):
        release._merge_sources(
            package_root=package,
            package_inventory={"README.md": _item(b"immutable\n")},
            asset_root=assets,
            asset_inventory=release._scan_public_asset_inventory(assets),
        )


def test_overlay_accepts_only_byte_identical_collision(tmp_path: Path) -> None:
    package = tmp_path / "package"
    assets = tmp_path / "assets"
    card = _card()
    _write(package, "README.md", card)
    _write(assets, "README.md", card)
    combined, sources = release._merge_sources(
        package_root=package,
        package_inventory={"README.md": _item(card)},
        asset_root=assets,
        asset_inventory=release._scan_public_asset_inventory(assets),
    )
    assert combined == {
        "README.md": _item(card),
        "release-assets/README.md": _item(card),
    }
    assert sources["README.md"] == assets / "README.md"
    assert sources["release-assets/README.md"] == assets / "README.md"


def test_asset_and_package_checksum_manifests_remain_independently_verifiable(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    assets = tmp_path / "assets"
    package_checksums = b"package checksum inventory\n"
    asset_checksums = b"public asset checksum inventory\n"
    _write(package, "SHA256SUMS", package_checksums)
    _write(assets, "SHA256SUMS", asset_checksums)

    combined, sources = release._merge_sources(
        package_root=package,
        package_inventory={"SHA256SUMS": _item(package_checksums)},
        asset_root=assets,
        asset_inventory={"SHA256SUMS": _item(asset_checksums)},
    )

    assert combined == {
        "SHA256SUMS": _item(package_checksums),
        "release-assets/SHA256SUMS": _item(asset_checksums),
    }
    assert sources["SHA256SUMS"] == package / "SHA256SUMS"
    assert sources["release-assets/SHA256SUMS"] == assets / "SHA256SUMS"


@pytest.mark.parametrize(
    "name,payload,add_card,error",
    [
        ("notes.md", b"no card\n", False, "README"),
        ("README.md", b"# {{MODEL}}\n", False, "wrong model"),
        (
            "README.md",
            _card() + b"hf_" + b"abcdefghijklmnopqrstuvwxyz\n",
            False,
            "credential",
        ),
        ("secret-token.txt", b"safe\n", True, "sensitive filename"),
    ],
)
def test_public_assets_fail_closed(
    tmp_path: Path, name: str, payload: bytes, add_card: bool, error: str
) -> None:
    assets = tmp_path / "assets"
    _write(assets, name, payload)
    if add_card:
        _write(assets, "README.md", _card())
    with pytest.raises(release.ModalHuggingFaceReleaseError, match=error):
        release._scan_public_asset_inventory(assets)


def test_public_asset_stream_scan_covers_large_files_and_boundaries(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    _write(assets, "README.md", _card())
    prefix = b"x" * (release._CREDENTIAL_SCAN_CHUNK_BYTES - 3) + b" "
    _write(
        assets,
        "large-evidence.bin",
        prefix + b"hf_" + b"a" * 24 + b"x" * (4 * 1024 * 1024),
    )
    with pytest.raises(
        release.ModalHuggingFaceReleaseError,
        match="credential material",
    ):
        release._scan_public_asset_inventory(assets)


@pytest.mark.parametrize(
    "credential",
    [
        b"github_pat_" + b"a" * 24,
        b"ghp_" + b"b" * 24,
        b"hf_" + b"c" * 24,
        b"sk-or-v1-" + b"d" * 24,
        b"sk-" + b"e" * 24,
        b"AKIA" + b"F" * 16,
        b"ASIA" + b"G" * 16,
        b"-----BEGIN " + b"RSA" + b" PRIVATE " + b"KEY-----",
        b"-----BEGIN " + b"EC" + b" PRIVATE " + b"KEY-----",
        b"-----BEGIN " + b"OPENSSH" + b" PRIVATE " + b"KEY-----",
    ],
)
def test_public_asset_stream_scan_rejects_all_credential_classes_at_boundary(
    tmp_path: Path, credential: bytes
) -> None:
    assets = tmp_path / "assets"
    _write(assets, "README.md", _card())
    split = min(10, len(credential) - 1)
    prefix = b"x" * (release._CREDENTIAL_SCAN_CHUNK_BYTES - split - 1) + b" "
    _write(assets, "boundary.bin", prefix + credential + b"\n")
    with pytest.raises(
        release.ModalHuggingFaceReleaseError,
        match="credential material",
    ):
        release._scan_public_asset_inventory(assets)


def test_public_assets_require_release_ready_package_bound_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "assets"
    package = _public_package()
    _materialize_public_assets(assets, package)
    monkeypatch.setattr(
        release.publication,
        "_validate_public_qualification",
        lambda value, observed: value["attestation_sha256"]
        if observed is package
        else (_ for _ in ()).throw(AssertionError("wrong package")),
    )

    inventory = release._public_asset_inventory(assets, package=package)

    assert set(inventory) == release.PUBLIC_ASSET_REQUIRED_FILES


def test_public_assets_reject_extra_raw_receipt_even_when_inventory_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "assets"
    package = _public_package()
    _materialize_public_assets(assets, package)
    _write_json(assets, "raw-receipt.json", {"attempt_id": "synthetic-private"})
    monkeypatch.setattr(
        release.publication,
        "_validate_public_qualification",
        lambda value, observed: value["attestation_sha256"],
    )

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="asset set"):
        release._public_asset_inventory(assets, package=package)


def test_public_assets_reject_evidence_detached_from_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = tmp_path / "assets"
    package = _public_package()
    _materialize_public_assets(assets, package)
    monkeypatch.setattr(
        release.publication,
        "_validate_public_qualification",
        lambda value, observed: value["attestation_sha256"],
    )
    detached = {**package, "release_artifact_sha256": "9" * 64}

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="detached"):
        release._public_asset_inventory(assets, package=detached)


def test_safe_relative_rejects_escape() -> None:
    for value in ("../package", "/package", "a/./b", ""):
        with pytest.raises(release.ModalHuggingFaceReleaseError):
            release._safe_relative(value, "path")


def test_modal_paths_resolve_assets_from_result_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "results"
    checkout = tmp_path / "checkout"
    package = result / "runs/final/package"
    assets = result / "runs/final/runtime-qualification/attempt/public-assets"
    package.mkdir(parents=True)
    assets.mkdir(parents=True)
    checkout.mkdir()
    monkeypatch.setattr(release, "RESULT_MOUNT", result)
    monkeypatch.setattr(release, "REMOTE_PUBLIC_REPO", checkout)

    observed_package, observed_assets, control = release._modal_paths(
        "runs/final/package",
        "runs/final/runtime-qualification/attempt/public-assets",
        "runs/final/publication/control",
    )

    assert observed_package == package
    assert observed_assets == assets
    assert observed_assets.is_relative_to(result)
    assert control == (result / "runs/final/publication/control").resolve()


def test_modal_paths_reject_control_symlink_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "results"
    (result / "package").mkdir(parents=True)
    (result / "assets").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (result / "control-link").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(release, "RESULT_MOUNT", result)

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="symlink"):
        release._modal_paths("package", "assets", "control-link/release")


def test_expected_plan_is_self_hashed() -> None:
    inputs = {
        "package": {
            "artifact_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "inventory_sha256": "c" * 64,
        },
        "asset_inventory_sha256": "d" * 64,
        "upload_inventory_sha256": "e" * 64,
        "model_card_sha256": "f" * 64,
        "authorization_sha256": "1" * 64,
        "gate_sha256": "2" * 64,
        "private_plan_sha256": "3" * 64,
        "private_plan": {"source_revision": "4" * 40},
    }
    plan = release._expected_plan(inputs)
    receipt = plan.pop("receipt_sha256")
    assert receipt == release._json_sha256(plan)
    assert plan["publication_performed"] is False
    assert plan["repository"] == "infercrane/Commerce-1"


def test_stage_operation_must_bind_combined_inventory() -> None:
    inputs = {
        "package": {"artifact_sha256": "a" * 64, "inventory_sha256": "b" * 64},
        "asset_inventory_sha256": "c" * 64,
        "upload_inventory_sha256": "d" * 64,
        "model_card_sha256": "e" * 64,
    }
    remote = {"revision": "1" * 40, "receipt_sha256": "f" * 64}
    body = {
        "schema_version": release.STAGE_OPERATION_SCHEMA,
        "status": "PRIVATE_UPLOAD_AND_COMBINED_INVENTORY_VERIFICATION_PASSED",
        "repository": "infercrane/Commerce-1",
        "visibility": "private",
        "revision": "1" * 40,
        "artifact_sha256": "a" * 64,
        "package_inventory_sha256": "b" * 64,
        "public_asset_inventory_sha256": "c" * 64,
        "upload_inventory_sha256": "d" * 64,
        "model_card_sha256": "e" * 64,
        "private_stage_plan_sha256": "2" * 64,
        "remote_verification_sha256": "f" * 64,
        "authenticated_identity_sha256": "3" * 64,
        "authenticated_org_role": "admin",
        "repository_created": True,
        "uploaded_file_count": 10,
        "network_used": True,
        "upload_performed": True,
        "publication_performed": False,
        "completed_at_utc": "2026-10-08T00:00:00+00:00",
    }
    operation = release._receipt(body)
    assert (
        release._validate_stage_operation(
            operation,
            inputs=inputs,
            stage_plan_sha256="2" * 64,
            remote_receipt=remote,
        )
        == operation["receipt_sha256"]
    )
    operation["upload_inventory_sha256"] = "9" * 64
    operation["receipt_sha256"] = release._json_sha256(
        {key: value for key, value in operation.items() if key != "receipt_sha256"}
    )
    with pytest.raises(release.ModalHuggingFaceReleaseError, match="detached"):
        release._validate_stage_operation(
            operation,
            inputs=inputs,
            stage_plan_sha256="2" * 64,
            remote_receipt=remote,
        )


def _validated_control_package(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "package"
    root.mkdir()
    return {
        "root": root,
        "manifest_sha256": "1" * 64,
        "inventory_sha256": "2" * 64,
        "artifact_sha256": "3" * 64,
        "contract_sha256": "4" * 64,
        "calibration_file_sha256": "5" * 64,
        "runtime_identity_sha256": "6" * 64,
        "public_evaluation": {"attestation_sha256": "7" * 64},
        "provenance_sha256": "8" * 64,
        "provenance_file_sha256": "9" * 64,
        "mandatory_notices": [{"subject_id": "public"}],
        "mandatory_notices_receipt_sha256": "a" * 64,
        "license_shas": {"project": "b" * 64, "third_party": "c" * 64},
    }


def test_create_release_controls_is_create_once_and_returns_hashes_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = _validated_control_package(tmp_path)
    qualification = tmp_path / "qualification.json"
    assets = tmp_path / "assets"
    control = tmp_path / "control"
    assets.mkdir()
    control.mkdir()
    _write_json(
        tmp_path,
        "qualification.json",
        {"schema_version": release.publication.CURRENT_QUALIFICATION_SCHEMA},
    )
    monkeypatch.setattr(
        release.publication,
        "_validated_release_materials",
        lambda **_kwargs: (
            package,
            "d" * 64,
            None,
            None,
            release.publication.INITIAL_RELEASE_STATE,
            None,
        ),
    )
    monkeypatch.setattr(
        release,
        "_validate_qualification_projection",
        lambda **_kwargs: {"README.md": {"sha256": "e" * 64, "size_bytes": 123}},
    )
    now = datetime(2026, 10, 8, 10, 0, tzinfo=UTC)

    summary = release.create_release_controls(
        package=Path(package["root"]),
        public_repo=tmp_path,
        qualification=qualification,
        assets=assets,
        control=control,
        confirmation=release.publication.OWNER_RELEASE_CONFIRMATION,
        source_revision="f" * 40,
        authorized_at_utc="2026-10-08T10:00:00Z",
        now=now,
    )

    assert {path.name for path in control.iterdir()} == set(release.CONTROL_FILENAMES)
    body = {key: value for key, value in summary.items() if key != "receipt_sha256"}
    assert summary["receipt_sha256"] == release._json_sha256(body)
    assert summary["credentials_used"] is False
    assert summary["gpu_used"] is False
    assert summary["publication_performed"] is False
    assert "authorized_at_utc" not in summary
    assert "owner" not in summary
    with pytest.raises(release.ModalHuggingFaceReleaseError, match="already exists"):
        release.create_release_controls(
            package=Path(package["root"]),
            public_repo=tmp_path,
            qualification=qualification,
            assets=assets,
            control=control,
            confirmation=release.publication.OWNER_RELEASE_CONFIRMATION,
            source_revision="f" * 40,
            authorized_at_utc="2026-10-08T10:00:00Z",
            now=now,
        )


@pytest.mark.parametrize(
    "confirmation,revision,authorized_at,error",
    [
        ("yes", "f" * 40, "2026-10-08T10:00:00Z", "exact owner"),
        (
            release.publication.OWNER_RELEASE_CONFIRMATION,
            "main",
            "2026-10-08T10:00:00Z",
            "immutable lowercase commit",
        ),
        (
            release.publication.OWNER_RELEASE_CONFIRMATION,
            "f" * 40,
            "2026-10-08T09:00:00Z",
            "real current invocation time",
        ),
        (
            release.publication.OWNER_RELEASE_CONFIRMATION,
            "f" * 40,
            "2026-10-08T10:00:00+01:00",
            "explicit ISO-8601 UTC",
        ),
    ],
)
def test_release_control_inputs_fail_before_package_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    confirmation: str,
    revision: str,
    authorized_at: str,
    error: str,
) -> None:
    monkeypatch.setattr(
        release.publication,
        "_validated_release_materials",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("package read")),
    )
    with pytest.raises(ValueError, match=error):
        release.build_release_controls(
            package=tmp_path / "package",
            public_repo=tmp_path,
            qualification=tmp_path / "qualification.json",
            assets=tmp_path / "assets",
            confirmation=confirmation,
            source_revision=revision,
            authorized_at_utc=authorized_at,
            now=datetime(2026, 10, 8, 10, 0, tzinfo=UTC),
        )


def test_qualification_projection_must_bind_protected_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = tmp_path / "qualification.json"
    assets = tmp_path / "assets"
    assets.mkdir()
    qualification.write_text(
        json.dumps(
            {"schema_version": release.publication.CURRENT_QUALIFICATION_SCHEMA}
        ),
        encoding="utf-8",
    )
    (assets / "QUALIFICATION.json").write_text(
        json.dumps(
            {
                "schema_version": release.publication.PUBLIC_QUALIFICATION_SCHEMA,
                "protected_qualification_receipt_sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "_public_asset_inventory", lambda *_args, **_kw: {})

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="detached"):
        release._validate_qualification_projection(
            qualification_path=qualification,
            assets=assets,
            package={},
        )


def test_qualification_projection_binds_exact_protected_receipt_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qualification = tmp_path / "qualification.json"
    assets = tmp_path / "assets"
    assets.mkdir()
    qualification.write_text(
        json.dumps(
            {"schema_version": release.publication.CURRENT_QUALIFICATION_SCHEMA}
        ),
        encoding="utf-8",
    )
    protected_sha = release.publication._file_sha256(qualification)
    (assets / "QUALIFICATION.json").write_text(
        json.dumps(
            {
                "schema_version": release.publication.PUBLIC_QUALIFICATION_SCHEMA,
                "protected_qualification_receipt_sha256": protected_sha,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(release, "_public_asset_inventory", lambda *_args, **_kw: {})

    assert (
        release._validate_qualification_projection(
            qualification_path=qualification,
            assets=assets,
            package={},
        )
        == {}
    )


def test_modal_control_path_must_be_disjoint_from_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "results"
    package = result / "run/package"
    qualification = result / "run/qualification.json"
    assets = result / "run/assets"
    package.mkdir(parents=True)
    assets.mkdir()
    qualification.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(release, "RESULT_MOUNT", result)

    with pytest.raises(release.ModalHuggingFaceReleaseError, match="disjoint"):
        release._modal_control_paths(
            "run/package",
            "run/qualification.json",
            "run/assets",
            "run/package/controls",
        )


def test_modal_release_runbook_uses_the_durable_protected_receipt() -> None:
    runbook = (
        Path(__file__).parents[1] / "release/MODAL_HUGGINGFACE_PUBLICATION.md"
    ).read_text(encoding="utf-8")

    assert "runtime-qualification/<attempt>/FINAL.json" in runbook
    assert "native-runtime-qualification.json" not in runbook
