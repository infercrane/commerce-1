from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_v3_publication import (
    _authorization_file,
    _evidence_files,
    _package,
    _receipt,
    _write,
)
from tools import v3_huggingface_release as hf_release
from tools import v3_publication as publication


class FakeHub:
    def __init__(self, root: Path, *, role: str = "admin") -> None:
        self.root = root
        self.role = role
        self.exists = False
        self.private = True
        self.revision = "1" * 40
        self.created = 0
        self.uploaded = 0
        self.updated = 0
        self.tamper_after_upload: str | None = None
        self.tamper_after_public: str | None = None

    def whoami(self) -> dict[str, object]:
        return {
            "name": "release-operator",
            "orgs": [{"name": "infercrane", "roleInOrg": self.role}],
        }

    def repo_exists(self, *, repo_id: str, repo_type: str) -> bool:
        assert repo_id == publication.HF_REPOSITORY
        assert repo_type == "model"
        return self.exists

    def create_repo(
        self, *, repo_id: str, repo_type: str, private: bool, exist_ok: bool
    ) -> None:
        assert repo_id == publication.HF_REPOSITORY
        assert repo_type == "model"
        assert private is True
        assert exist_ok is False
        self.root.mkdir(parents=True)
        (self.root / ".gitattributes").write_text("*.safetensors filter=lfs\n")
        self.exists = True
        self.private = True
        self.created += 1

    def model_info(self, repo_id: str, *, revision: str) -> SimpleNamespace:
        assert repo_id == publication.HF_REPOSITORY
        assert revision == "main"
        if not self.exists:
            raise RuntimeError("missing")
        return SimpleNamespace(private=self.private, sha=self.revision)

    def list_repo_files(
        self, *, repo_id: str, repo_type: str, revision: str
    ) -> list[str]:
        assert repo_id == publication.HF_REPOSITORY
        assert repo_type == "model"
        assert revision == "main" or len(revision) == 40
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def upload_large_folder(
        self, *, repo_id: str, repo_type: str, folder_path: str
    ) -> None:
        assert repo_id == publication.HF_REPOSITORY
        assert repo_type == "model"
        for source in Path(folder_path).rglob("*"):
            if source.is_file():
                destination = self.root / source.relative_to(folder_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
        if self.tamper_after_upload is not None:
            (self.root / self.tamper_after_upload).write_text("tampered\n")
        self.revision = "2" * 40
        self.uploaded += 1

    def update_repo_settings(
        self, *, repo_id: str, repo_type: str, private: bool
    ) -> None:
        assert repo_id == publication.HF_REPOSITORY
        assert repo_type == "model"
        self.private = private
        if private is False and self.tamper_after_public is not None:
            (self.root / self.tamper_after_public).write_text("tampered\n")
        self.updated += 1

    def download(self, **kwargs: str) -> str:
        assert kwargs["repo_id"] == publication.HF_REPOSITORY
        assert kwargs["repo_type"] == "model"
        return str(self.root / kwargs["filename"])


def _chain(
    tmp_path: Path, *, official_v03: bool = True
) -> dict[str, Path | dict[str, object]]:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    if official_v03:
        authorization = _authorization_file(
            tmp_path,
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public=public,
            official=official,
        )
    else:
        authorization_value = publication.build_owner_authorization(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=None,
            official_v03=None,
            confirmation=publication.OWNER_RELEASE_CONFIRMATION,
            authorized_at_utc="2026-10-07T00:00:00+00:00",
        )
        authorization = tmp_path / "owner-authorization.json"
        _write(authorization, authorization_value)
    gate_value = publication.build_evidence_gate(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public if official_v03 else None,
        official_v03=official if official_v03 else None,
        authorization=authorization,
    )
    gate = tmp_path / "gate.json"
    _write(gate, gate_value)
    private_value = publication.build_private_plan(gate_value, source_revision="a" * 40)
    private_plan = tmp_path / "private-plan.json"
    _write(private_plan, private_value)
    stage_value = hf_release.build_private_stage_plan(
        package=package,
        public_repo=public_repo,
        authorization_path=authorization,
        gate_path=gate,
        private_plan_path=private_plan,
    )
    stage_plan = tmp_path / "hf-private-plan.json"
    _write(stage_plan, stage_value)
    return {
        "package": package,
        "public_repo": public_repo,
        "authorization": authorization,
        "gate": gate,
        "gate_value": gate_value,
        "private_plan": private_plan,
        "private_value": private_value,
        "stage_plan": stage_plan,
        "stage_value": stage_value,
    }


def _stage(
    chain: dict[str, Path | dict[str, object]], hub: FakeHub
) -> tuple[dict[str, object], dict[str, object]]:
    return hf_release.stage_private(
        package=chain["package"],  # type: ignore[arg-type]
        public_repo=chain["public_repo"],  # type: ignore[arg-type]
        authorization_path=chain["authorization"],  # type: ignore[arg-type]
        gate_path=chain["gate"],  # type: ignore[arg-type]
        private_plan_path=chain["private_plan"],  # type: ignore[arg-type]
        stage_plan_path=chain["stage_plan"],  # type: ignore[arg-type]
        confirmation=hf_release.PRIVATE_STAGE_CONFIRMATION,
        api=hub,
        download_file=hub.download,
        now=datetime(2026, 10, 7, tzinfo=UTC),
    )


def test_local_private_staging_is_disabled_for_real_clients(tmp_path: Path) -> None:
    with pytest.raises(
        hf_release.V3HuggingFaceReleaseError,
        match="local private staging is disabled",
    ):
        hf_release.stage_private(
            package=tmp_path / "package",
            public_repo=tmp_path / "public",
            authorization_path=tmp_path / "authorization.json",
            gate_path=tmp_path / "gate.json",
            private_plan_path=tmp_path / "private-plan.json",
            stage_plan_path=tmp_path / "stage-plan.json",
            confirmation=hf_release.PRIVATE_STAGE_CONFIRMATION,
        )


def test_local_public_transition_is_disabled_for_real_clients(tmp_path: Path) -> None:
    with pytest.raises(
        hf_release.V3HuggingFaceReleaseError,
        match="local public transition is disabled",
    ):
        hf_release.publish_public(
            package=tmp_path / "package",
            public_repo=tmp_path / "public",
            authorization_path=tmp_path / "authorization.json",
            gate_path=tmp_path / "gate.json",
            private_plan_path=tmp_path / "private-plan.json",
            github_receipt_path=tmp_path / "github.json",
            huggingface_receipt_path=tmp_path / "huggingface.json",
            private_verification_path=tmp_path / "private-verification.json",
            public_plan_path=tmp_path / "public-plan.json",
            confirmation=publication.OWNER_RELEASE_CONFIRMATION,
        )


def test_local_cli_exposes_plan_only() -> None:
    help_text = hf_release._parser().format_help()
    assert "{plan-private}" in help_text
    assert "stage-private" not in help_text
    assert "publish-public" not in help_text


def _public_chain(
    tmp_path: Path,
    chain: dict[str, Path | dict[str, object]],
    huggingface: dict[str, object],
) -> dict[str, Path]:
    private_value = chain["private_value"]
    assert isinstance(private_value, dict)
    artifact = str(private_value["artifact_sha256"])
    github = _receipt(
        {
            "schema_version": publication.REMOTE_RECEIPT_SCHEMA,
            "status": "IMMUTABLE_PRIVATE_REVISION_VERIFIED",
            "service": "github",
            "repository": publication.GITHUB_REPOSITORY,
            "visibility": "private",
            "revision": "a" * 40,
            "source_revision": "a" * 40,
            "artifact_sha256": artifact,
            "inventory_sha256": "b" * 64,
            "verified": True,
            "verified_at_utc": "2026-10-07T00:00:00+00:00",
        }
    )
    github_path = tmp_path / "github.json"
    huggingface_path = tmp_path / "huggingface.json"
    _write(github_path, github)
    _write(huggingface_path, huggingface)
    verification = publication.build_private_verification(
        private_value, github, huggingface
    )
    verification_path = tmp_path / "private-verification.json"
    _write(verification_path, verification)
    gate_value = chain["gate_value"]
    assert isinstance(gate_value, dict)
    public = publication.build_public_plan(gate_value, verification)
    public_path = tmp_path / "public-plan.json"
    _write(public_path, public)
    return {
        "github": github_path,
        "huggingface": huggingface_path,
        "verification": verification_path,
        "public": public_path,
    }


def test_private_plan_is_local_infercrane_only_and_unpublished(tmp_path: Path) -> None:
    chain = _chain(tmp_path)
    plan = chain["stage_value"]
    assert isinstance(plan, dict)
    assert plan["owner"] == "infercrane"
    assert plan["repository"] == "infercrane/Commerce-1"
    assert plan["visibility"] == "private"
    assert plan["network_used"] is False
    assert plan["upload_performed"] is False
    assert plan["publication_performed"] is False


def test_private_stage_authenticates_uploads_and_verifies_bytes(tmp_path: Path) -> None:
    chain = _chain(tmp_path / "chain")
    hub = FakeHub(tmp_path / "remote")
    remote, operation = _stage(chain, hub)
    assert hub.created == 1
    assert hub.uploaded == 1
    assert hub.private is True
    assert remote["status"] == "IMMUTABLE_PRIVATE_REVISION_VERIFIED"
    assert remote["visibility"] == "private"
    assert operation["publication_performed"] is False
    assert operation["authenticated_org_role"] == "admin"


def test_initial_private_stage_keeps_v03_score_and_rank_null(tmp_path: Path) -> None:
    chain = _chain(tmp_path / "chain", official_v03=False)
    gate = chain["gate_value"]
    assert isinstance(gate, dict)
    assert gate["release_state"] == publication.INITIAL_RELEASE_STATE
    assert gate["official_v03_full_score"] is None
    assert gate["official_v03_rank"] is None

    hub = FakeHub(tmp_path / "remote")
    remote, operation = _stage(chain, hub)
    assert remote["status"] == "IMMUTABLE_PRIVATE_REVISION_VERIFIED"
    assert operation["publication_performed"] is False


def test_private_stage_refuses_wrong_identity_or_public_target(tmp_path: Path) -> None:
    chain = _chain(tmp_path / "identity")
    wrong = FakeHub(tmp_path / "wrong", role="read")
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="access"):
        _stage(chain, wrong)
    assert wrong.created == 0

    public = FakeHub(tmp_path / "public")
    public.create_repo(
        repo_id=publication.HF_REPOSITORY,
        repo_type="model",
        private=True,
        exist_ok=False,
    )
    public.private = False
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="not private"):
        _stage(chain, public)
    assert public.uploaded == 0


def test_private_stage_rejects_remote_byte_drift(tmp_path: Path) -> None:
    chain = _chain(tmp_path / "chain")
    hub = FakeHub(tmp_path / "remote")
    hub.tamper_after_upload = "README.md"
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="bytes changed"):
        _stage(chain, hub)
    assert hub.private is True


def test_model_card_must_target_infercrane_commerce_one_without_placeholders(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    package = chain["package"]
    assert isinstance(package, Path)
    (package / "README.md").write_text(
        "# InferCrane Commerce-1\n\n{{UNRESOLVED}}\n", encoding="utf-8"
    )
    with pytest.raises(hf_release.V3HuggingFaceReleaseError):
        hf_release._validate_model_card(package)

    (package / "README.md").write_text(
        "# InferCrane Commerce-1 V3\n\nCanonical model: `infercrane/Commerce-1`.\n",
        encoding="utf-8",
    )
    with pytest.raises(hf_release.V3HuggingFaceReleaseError):
        hf_release._validate_model_card(package)


def test_remote_inventory_accepts_huggingface_snapshot_symlinks(
    tmp_path: Path,
) -> None:
    blob = tmp_path / "cache" / "blobs" / "abc"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"verified model bytes")
    snapshot = tmp_path / "cache" / "snapshots" / ("a" * 40) / "model.bin"
    snapshot.parent.mkdir(parents=True)
    snapshot.symlink_to(blob)

    class SnapshotHub:
        def list_repo_files(self, **_: str) -> list[str]:
            return ["model.bin"]

    hf_release._verify_remote_inventory(
        api=SnapshotHub(),
        download_file=lambda **_: str(snapshot),
        revision="a" * 40,
        expected={
            "model.bin": {
                "size_bytes": blob.stat().st_size,
                "sha256": hf_release._file_sha256(blob),
            }
        },
    )


def test_remote_inventory_accepts_only_inert_lfs_metadata(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"verified model bytes")
    attributes = tmp_path / ".gitattributes"
    attributes.write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )

    class MetadataHub:
        def list_repo_files(self, **_: str) -> list[str]:
            return [".gitattributes", "model.safetensors"]

    paths = {".gitattributes": attributes, "model.safetensors": model}
    hf_release._verify_remote_inventory(
        api=MetadataHub(),
        download_file=lambda **kwargs: str(paths[kwargs["filename"]]),
        revision="a" * 40,
        expected={
            "model.safetensors": {
                "size_bytes": model.stat().st_size,
                "sha256": hf_release._file_sha256(model),
            }
        },
    )

    attributes.write_text(
        "*.safetensors filter=malicious diff=lfs\n",
        encoding="utf-8",
    )
    with pytest.raises(
        hf_release.V3HuggingFaceReleaseError,
        match="unsafe declaration",
    ):
        hf_release._verify_remote_inventory(
            api=MetadataHub(),
            download_file=lambda **kwargs: str(paths[kwargs["filename"]]),
            revision="a" * 40,
            expected={
                "model.safetensors": {
                    "size_bytes": model.stat().st_size,
                    "sha256": hf_release._file_sha256(model),
                }
            },
        )


def test_remote_inventory_accepts_current_hub_default_lfs_metadata(
    tmp_path: Path,
) -> None:
    attributes = tmp_path / ".gitattributes"
    attributes.write_text(
        "\n".join(
            f"{pattern} filter=lfs diff=lfs merge=lfs -text"
            for pattern in sorted(hf_release.ALLOWED_LFS_PATTERNS)
        )
        + "\n",
        encoding="utf-8",
    )

    assert hf_release._validate_hub_metadata(attributes, ".gitattributes") == (
        hf_release._file_sha256(attributes)
    )


def test_public_transition_revalidates_private_chain_and_keeps_revision(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    hub = FakeHub(tmp_path / "remote")
    remote, _ = _stage(chain, hub)
    public = _public_chain(tmp_path, chain, remote)
    receipt = hf_release.publish_public(
        package=chain["package"],  # type: ignore[arg-type]
        public_repo=chain["public_repo"],  # type: ignore[arg-type]
        authorization_path=chain["authorization"],  # type: ignore[arg-type]
        gate_path=chain["gate"],  # type: ignore[arg-type]
        private_plan_path=chain["private_plan"],  # type: ignore[arg-type]
        github_receipt_path=public["github"],
        huggingface_receipt_path=public["huggingface"],
        private_verification_path=public["verification"],
        public_plan_path=public["public"],
        confirmation=publication.OWNER_RELEASE_CONFIRMATION,
        api=hub,
        download_file=hub.download,
        now=datetime(2026, 10, 7, tzinfo=UTC),
    )
    assert hub.updated == 1
    assert hub.private is False
    assert receipt["revision"] == remote["revision"]
    assert receipt["publication_performed"] is True
    assert receipt["legal_conclusion"] is False


def test_public_transition_requires_confirmation_admin_and_pinned_revision(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    hub = FakeHub(tmp_path / "remote")
    remote, _ = _stage(chain, hub)
    public = _public_chain(tmp_path, chain, remote)
    arguments = {
        "package": chain["package"],
        "public_repo": chain["public_repo"],
        "authorization_path": chain["authorization"],
        "gate_path": chain["gate"],
        "private_plan_path": chain["private_plan"],
        "github_receipt_path": public["github"],
        "huggingface_receipt_path": public["huggingface"],
        "private_verification_path": public["verification"],
        "public_plan_path": public["public"],
        "api": hub,
        "download_file": hub.download,
    }
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="confirmation"):
        hf_release.publish_public(**arguments, confirmation="yes")  # type: ignore[arg-type]
    assert hub.updated == 0

    hub.role = "write"
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="admin"):
        hf_release.publish_public(  # type: ignore[arg-type]
            **arguments, confirmation=publication.OWNER_RELEASE_CONFIRMATION
        )
    assert hub.updated == 0

    hub.role = "admin"
    hub.revision = "3" * 40
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="revision"):
        hf_release.publish_public(  # type: ignore[arg-type]
            **arguments, confirmation=publication.OWNER_RELEASE_CONFIRMATION
        )
    assert hub.updated == 0


def test_public_transition_rolls_back_when_post_publish_bytes_change(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path / "chain")
    hub = FakeHub(tmp_path / "remote")
    remote, _ = _stage(chain, hub)
    public = _public_chain(tmp_path, chain, remote)
    hub.tamper_after_public = "README.md"
    with pytest.raises(hf_release.V3HuggingFaceReleaseError, match="rollback failed"):
        hf_release.publish_public(
            package=chain["package"],  # type: ignore[arg-type]
            public_repo=chain["public_repo"],  # type: ignore[arg-type]
            authorization_path=chain["authorization"],  # type: ignore[arg-type]
            gate_path=chain["gate"],  # type: ignore[arg-type]
            private_plan_path=chain["private_plan"],  # type: ignore[arg-type]
            github_receipt_path=public["github"],
            huggingface_receipt_path=public["huggingface"],
            private_verification_path=public["verification"],
            public_plan_path=public["public"],
            confirmation=publication.OWNER_RELEASE_CONFIRMATION,
            api=hub,
            download_file=hub.download,
        )
    assert hub.private is True
    assert hub.updated == 2


def test_v3_publisher_does_not_import_legacy_uploader() -> None:
    source = Path(hf_release.__file__).read_text(encoding="utf-8")
    assert "effective_v4_hf_bundle" not in source
    assert "tools.huggingface_release" not in source
