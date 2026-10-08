#!/usr/bin/env python3
"""Build a local, non-mutating Commerce-1 Hugging Face verification plan.

The command-line interface exposes only ``plan-private``.  The transport
helpers remain importable solely for dependency-injected hermetic tests and for
the server-side Modal release implementation.  Without an injected API and
downloader they fail closed before any network access.  The sole operator
mutation path is ``tools/modal_v3_huggingface_release.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools import v3_publication as publication

PRIVATE_STAGE_PLAN_SCHEMA = "infercrane-commerce-1-v3-hf-private-stage-plan/v1"
PRIVATE_OPERATION_SCHEMA = "infercrane-commerce-1-v3-hf-private-operation/v1"
PUBLIC_OPERATION_SCHEMA = "infercrane-commerce-1-v3-hf-public-operation/v1"
PRIVATE_STAGE_CONFIRMATION = "STAGE_INFERCRANE_COMMERCE_1_V3_PRIVATE"
ALLOWED_HUB_METADATA_FILES = frozenset({".gitattributes"})
ALLOWED_LFS_PATTERNS = frozenset(
    {
        "*.7z",
        "*.arrow",
        "*.bin",
        "*.bz2",
        "*.ckpt",
        "*.ftz",
        "*.gz",
        "*.h5",
        "*.joblib",
        "*.lfs.*",
        "*.mlmodel",
        "*.model",
        "*.msgpack",
        "*.npy",
        "*.npz",
        "*.onnx",
        "*.ot",
        "*.parquet",
        "*.pb",
        "*.pickle",
        "*.pkl",
        "*.pt",
        "*.pth",
        "*.rar",
        "*.safetensors",
        "*.tar",
        "*.tar.*",
        "*.tflite",
        "*.tgz",
        "*.wasm",
        "*.xz",
        "*.zip",
        "*.zst",
        "*tfevents*",
        "model/tokenizer.json",
        "saved_model/**/*",
    }
)
ALLOWED_LFS_ATTRIBUTES = frozenset({"filter=lfs", "diff=lfs", "merge=lfs", "-text"})
_PLACEHOLDER = re.compile(r"\{\{[^{}]+\}\}")


class V3HuggingFaceReleaseError(ValueError):
    """V3 publication inputs, identity, or remote bytes are unsafe."""


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


def _object(path: Path, label: str) -> dict[str, Any]:
    return publication._object(path, label)


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    publication._write_once(path, value)


def _require_new_output(path: Path, *, package: Path, label: str) -> None:
    destination = path.resolve()
    package_root = package.resolve(strict=True)
    if destination.exists():
        raise V3HuggingFaceReleaseError(f"{label} already exists")
    if destination == package_root or destination.is_relative_to(package_root):
        raise V3HuggingFaceReleaseError(
            f"{label} must remain outside the immutable package"
        )


def _receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": _json_sha256(body)}


def _validate_model_card(package: Path) -> tuple[str, str]:
    card = publication._packaged_file(package, "README.md", "V3 model card")
    try:
        text = card.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise V3HuggingFaceReleaseError("V3 model card is not valid UTF-8") from error
    if (
        "# InferCrane Commerce-1\n" not in text
        or "# InferCrane Commerce-1 V3" in text
        or "inference: false" not in text
        or "pipeline_tag:" in text
        or publication.MODEL_ID not in text
        or "yasin" + "toy" in text.casefold()
        or _PLACEHOLDER.search(text) is not None
    ):
        raise V3HuggingFaceReleaseError(
            "V3 model card target is missing, misleading, personal, or unresolved"
        )
    return "README.md", _file_sha256(card)


def _validate_private_inputs(
    *,
    package: Path,
    public_repo: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
) -> dict[str, Any]:
    validated = publication._validate_package(package, public_repo)
    authorization = _object(authorization_path, "owner publication authorization")
    gate = _object(gate_path, "publication gate")
    private_plan = _object(private_plan_path, "private rehearsal")
    gate_sha = publication._validate_gate(gate)
    authorization_sha = publication._validate_owner_authorization(
        authorization,
        package=validated,
        qualification_sha256=gate["runtime_qualification_sha256"],
        public_v03_sha256=gate["public_v03_final_sha256"],
        official_v03_sha256=gate["official_v03_attestation_sha256"],
        release_state=gate["release_state"],
        official_v03_full_score=gate["official_v03_full_score"],
    )
    if (
        gate.get("owner_publication_authorization_sha256") != authorization_sha
        or gate.get("artifact_sha256") != validated["artifact_sha256"]
        or gate.get("release_manifest_sha256") != validated["manifest_sha256"]
        or gate.get("package_inventory_sha256") != validated["inventory_sha256"]
    ):
        raise V3HuggingFaceReleaseError(
            "publication gate is detached from the package or owner authorization"
        )
    private_plan_sha = publication._validate_private_plan(private_plan)
    if (
        private_plan.get("publication_gate_sha256") != gate_sha
        or private_plan.get("artifact_sha256") != validated["artifact_sha256"]
        or private_plan.get("package_inventory_sha256") != validated["inventory_sha256"]
        or private_plan.get("huggingface")
        != {"repository": publication.HF_REPOSITORY, "visibility": "private"}
    ):
        raise V3HuggingFaceReleaseError(
            "private rehearsal is detached from the exact V3 package"
        )
    card_path, card_sha = _validate_model_card(validated["root"])
    full_inventory = publication._regular_inventory(validated["root"])
    return {
        "package": validated,
        "authorization": authorization,
        "authorization_sha256": authorization_sha,
        "gate": gate,
        "gate_sha256": gate_sha,
        "private_plan": private_plan,
        "private_plan_sha256": private_plan_sha,
        "model_card_path": card_path,
        "model_card_sha256": card_sha,
        "full_inventory": full_inventory,
        "full_inventory_sha256": _json_sha256(full_inventory),
    }


def build_private_stage_plan(
    *,
    package: Path,
    public_repo: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
) -> dict[str, Any]:
    """Create a zero-network plan for the exact private upload."""

    inputs = _validate_private_inputs(
        package=package,
        public_repo=public_repo,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    return _private_stage_plan_from_inputs(inputs)


def _private_stage_plan_from_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package_value = inputs["package"]
    body = {
        "schema_version": PRIVATE_STAGE_PLAN_SCHEMA,
        "status": "PRIVATE_STAGE_PLANNED_NOT_PERFORMED",
        "owner": "infercrane",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "artifact_sha256": package_value["artifact_sha256"],
        "release_manifest_sha256": package_value["manifest_sha256"],
        "package_inventory_sha256": package_value["inventory_sha256"],
        "full_inventory_sha256": inputs["full_inventory_sha256"],
        "model_card_path": inputs["model_card_path"],
        "model_card_sha256": inputs["model_card_sha256"],
        "owner_authorization_sha256": inputs["authorization_sha256"],
        "publication_gate_sha256": inputs["gate_sha256"],
        "private_rehearsal_sha256": inputs["private_plan_sha256"],
        "source_revision": inputs["private_plan"]["source_revision"],
        "network_used": False,
        "repository_created": False,
        "upload_performed": False,
        "publication_performed": False,
    }
    return _receipt(body)


def _validate_stage_plan(plan: Mapping[str, Any], expected: Mapping[str, Any]) -> str:
    fields = set(expected)
    publication._require_fields(plan, fields, "Hugging Face private stage plan")
    supplied = publication._self_hash(
        plan, "receipt_sha256", "Hugging Face private stage plan"
    )
    if dict(plan) != dict(expected):
        raise V3HuggingFaceReleaseError(
            "Hugging Face private stage plan changed from validated inputs"
        )
    return supplied


def _authenticated_identity(api: Any, *, require_admin: bool) -> dict[str, str]:
    try:
        value = api.whoami()
    except Exception as error:
        raise V3HuggingFaceReleaseError(
            "Hugging Face authentication could not be verified"
        ) from error
    if not isinstance(value, Mapping) or not isinstance(value.get("name"), str):
        raise V3HuggingFaceReleaseError("Hugging Face identity response is invalid")
    organization: Mapping[str, Any] | None = None
    orgs = value.get("orgs")
    if isinstance(orgs, list):
        for item in orgs:
            if isinstance(item, Mapping) and item.get("name") == "infercrane":
                organization = item
                break
    role = organization.get("roleInOrg") if organization is not None else None
    allowed = {"admin"} if require_admin else {"admin", "write"}
    if not isinstance(role, str) or role.casefold() not in allowed:
        required = "admin" if require_admin else "write or admin"
        raise V3HuggingFaceReleaseError(
            f"authenticated identity lacks InferCrane {required} access"
        )
    body = {
        "actor": value["name"],
        "organization": "infercrane",
        "role": role.casefold(),
    }
    return {**body, "identity_sha256": _json_sha256(body)}


def _info_private(info: Any) -> bool:
    value = getattr(info, "private", None)
    if not isinstance(value, bool):
        raise V3HuggingFaceReleaseError("remote visibility could not be verified")
    return value


def _info_revision(info: Any) -> str:
    return publication._revision(
        str(getattr(info, "sha", "")), "Hugging Face immutable revision"
    )


def _repo_exists(api: Any) -> bool:
    try:
        return bool(
            api.repo_exists(repo_id=publication.HF_REPOSITORY, repo_type="model")
        )
    except Exception as error:
        raise V3HuggingFaceReleaseError(
            "Hugging Face repository existence check failed"
        ) from error


def _download_matches(downloaded: Path, item: Mapping[str, Any]) -> bool:
    """Verify Hub bytes while accepting its snapshot-to-blob cache symlink.

    ``hf_hub_download`` normally returns a symlink inside the local snapshot
    cache.  The symlink is an implementation detail, not remote evidence; the
    resolved regular file's size and SHA-256 are the evidence we need.
    """

    try:
        resolved = downloaded.resolve(strict=True)
        return (
            resolved.is_file()
            and resolved.stat().st_size == item["size_bytes"]
            and _file_sha256(resolved) == item["sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def _validate_hub_metadata(path: Path, name: str) -> str:
    """Allow only inert, standard LFS declarations created by the Hub."""

    if name != ".gitattributes":
        raise V3HuggingFaceReleaseError("unsupported Hub metadata file")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > 16 * 1024:
            raise V3HuggingFaceReleaseError("unsafe Hub .gitattributes file")
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise V3HuggingFaceReleaseError(
            "cannot read Hub .gitattributes metadata"
        ) from error
    declarations = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        pattern, attributes = fields[0], frozenset(fields[1:])
        if (
            pattern not in ALLOWED_LFS_PATTERNS
            or "filter=lfs" not in attributes
            or not attributes <= ALLOWED_LFS_ATTRIBUTES
        ):
            raise V3HuggingFaceReleaseError(
                "Hub .gitattributes contains an unsafe declaration"
            )
        declarations += 1
    if declarations == 0:
        raise V3HuggingFaceReleaseError("Hub .gitattributes has no LFS declarations")
    return _file_sha256(resolved)


def _download_and_validate_hub_metadata(
    *,
    download_file: Callable[..., str],
    revision: str,
    name: str,
) -> str:
    try:
        downloaded = Path(
            download_file(
                repo_id=publication.HF_REPOSITORY,
                filename=name,
                repo_type="model",
                revision=revision,
            )
        )
    except Exception as error:
        raise V3HuggingFaceReleaseError(
            f"remote metadata verification download failed: {name}"
        ) from error
    return _validate_hub_metadata(downloaded, name)


def _verify_remote_inventory(
    *,
    api: Any,
    download_file: Callable[..., str],
    revision: str,
    expected: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    try:
        remote_files = set(
            api.list_repo_files(
                repo_id=publication.HF_REPOSITORY,
                repo_type="model",
                revision=revision,
            )
        )
    except Exception as error:
        raise V3HuggingFaceReleaseError("remote file inventory failed") from error
    expected_files = set(expected)
    if remote_files - expected_files - ALLOWED_HUB_METADATA_FILES:
        raise V3HuggingFaceReleaseError("remote repository has undeclared files")
    if expected_files - remote_files:
        raise V3HuggingFaceReleaseError("remote repository is missing package files")
    metadata = {
        name: _download_and_validate_hub_metadata(
            download_file=download_file,
            revision=revision,
            name=name,
        )
        for name in sorted(remote_files & ALLOWED_HUB_METADATA_FILES)
    }
    for name, item in expected.items():
        try:
            downloaded = Path(
                download_file(
                    repo_id=publication.HF_REPOSITORY,
                    filename=name,
                    repo_type="model",
                    revision=revision,
                )
            )
        except Exception as error:
            raise V3HuggingFaceReleaseError(
                f"remote verification download failed: {name}"
            ) from error
        if not _download_matches(downloaded, item):
            raise V3HuggingFaceReleaseError(f"remote file bytes changed: {name}")
    return metadata


def _verify_existing_files_before_resume(
    *,
    api: Any,
    download_file: Callable[..., str],
    expected: Mapping[str, Mapping[str, Any]],
) -> None:
    try:
        names = set(
            api.list_repo_files(
                repo_id=publication.HF_REPOSITORY,
                repo_type="model",
                revision="main",
            )
        )
    except Exception as error:
        raise V3HuggingFaceReleaseError("existing private inventory failed") from error
    if names - set(expected) - ALLOWED_HUB_METADATA_FILES:
        raise V3HuggingFaceReleaseError(
            "existing private repository contains undeclared files"
        )
    for name in sorted(names & ALLOWED_HUB_METADATA_FILES):
        _download_and_validate_hub_metadata(
            download_file=download_file,
            revision="main",
            name=name,
        )
    for name in sorted(names & set(expected)):
        try:
            downloaded = Path(
                download_file(
                    repo_id=publication.HF_REPOSITORY,
                    filename=name,
                    repo_type="model",
                    revision="main",
                )
            )
        except Exception as error:
            raise V3HuggingFaceReleaseError(
                f"existing private file could not be verified: {name}"
            ) from error
        item = expected[name]
        if not _download_matches(downloaded, item):
            raise V3HuggingFaceReleaseError(
                f"existing private file differs from the package: {name}"
            )


def stage_private(
    *,
    package: Path,
    public_repo: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    stage_plan_path: Path,
    confirmation: str,
    api: Any | None = None,
    download_file: Callable[..., str] | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exercise the legacy transport only with an injected test client.

    Real private staging is deliberately Modal-only: the local Hub helper
    writes resume state into the 50+ GiB package and can exhaust local disk.
    """

    if api is None or download_file is None:
        raise V3HuggingFaceReleaseError(
            "local private staging is disabled; use "
            "tools/modal_v3_huggingface_release.py::stage_private_modal"
        )

    if confirmation != PRIVATE_STAGE_CONFIRMATION:
        raise V3HuggingFaceReleaseError("exact private-stage confirmation is required")
    inputs = _validate_private_inputs(
        package=package,
        public_repo=public_repo,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    expected_plan = _private_stage_plan_from_inputs(inputs)
    stage_plan = _object(stage_plan_path, "Hugging Face private stage plan")
    stage_plan_sha = _validate_stage_plan(stage_plan, expected_plan)
    client = api
    downloader = download_file
    identity = _authenticated_identity(client, require_admin=False)
    created = False
    if not _repo_exists(client):
        try:
            client.create_repo(
                repo_id=publication.HF_REPOSITORY,
                repo_type="model",
                private=True,
                exist_ok=False,
            )
        except Exception as error:
            raise V3HuggingFaceReleaseError(
                "private Hugging Face repository creation failed"
            ) from error
        created = True
    try:
        before = client.model_info(publication.HF_REPOSITORY, revision="main")
    except Exception as error:
        raise V3HuggingFaceReleaseError("private repository lookup failed") from error
    if _info_private(before) is not True:
        raise V3HuggingFaceReleaseError(
            "refusing to stage V3 into a repository that is not private"
        )
    _verify_existing_files_before_resume(
        api=client,
        download_file=downloader,
        expected=inputs["full_inventory"],
    )
    try:
        client.upload_large_folder(
            repo_id=publication.HF_REPOSITORY,
            repo_type="model",
            folder_path=str(inputs["package"]["root"]),
        )
        after = client.model_info(publication.HF_REPOSITORY, revision="main")
    except Exception as error:
        raise V3HuggingFaceReleaseError("private package upload failed") from error
    if _info_private(after) is not True:
        raise V3HuggingFaceReleaseError("private upload changed repository visibility")
    revision = _info_revision(after)
    hub_metadata = _verify_remote_inventory(
        api=client,
        download_file=downloader,
        revision=revision,
        expected=inputs["full_inventory"],
    )
    if set(hub_metadata) != {".gitattributes"}:
        raise V3HuggingFaceReleaseError(
            "private repository lacks canonical Hub metadata"
        )
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    remote_body = {
        "schema_version": publication.REMOTE_RECEIPT_SCHEMA,
        "status": "IMMUTABLE_PRIVATE_REVISION_VERIFIED",
        "service": "huggingface",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "revision": revision,
        "source_revision": inputs["private_plan"]["source_revision"],
        "artifact_sha256": inputs["package"]["artifact_sha256"],
        "inventory_sha256": inputs["package"]["inventory_sha256"],
        "hub_metadata": hub_metadata,
        "verified": True,
        "verified_at_utc": timestamp,
    }
    remote_receipt = _receipt(remote_body)
    publication._validate_remote(
        remote_receipt,
        service="huggingface",
        repository=publication.HF_REPOSITORY,
        artifact_sha256=inputs["package"]["artifact_sha256"],
        source_revision=inputs["private_plan"]["source_revision"],
        package_inventory_sha256=inputs["package"]["inventory_sha256"],
    )
    operation_body = {
        "schema_version": PRIVATE_OPERATION_SCHEMA,
        "status": "PRIVATE_UPLOAD_AND_IMMUTABLE_VERIFICATION_PASSED",
        "owner": "infercrane",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "revision": revision,
        "artifact_sha256": inputs["package"]["artifact_sha256"],
        "full_inventory_sha256": inputs["full_inventory_sha256"],
        "model_card_sha256": inputs["model_card_sha256"],
        "owner_authorization_sha256": inputs["authorization_sha256"],
        "publication_gate_sha256": inputs["gate_sha256"],
        "private_rehearsal_sha256": inputs["private_plan_sha256"],
        "private_stage_plan_sha256": stage_plan_sha,
        "remote_verification_sha256": remote_receipt["receipt_sha256"],
        "authenticated_identity_sha256": identity["identity_sha256"],
        "authenticated_org_role": identity["role"],
        "repository_created": created,
        "network_used": True,
        "upload_performed": True,
        "publication_performed": False,
        "completed_at_utc": timestamp,
    }
    return remote_receipt, _receipt(operation_body)


def _validate_public_inputs(
    *,
    package: Path,
    public_repo: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    github_receipt_path: Path,
    huggingface_receipt_path: Path,
    private_verification_path: Path,
    public_plan_path: Path,
) -> dict[str, Any]:
    inputs = _validate_private_inputs(
        package=package,
        public_repo=public_repo,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    github = _object(github_receipt_path, "GitHub private verification")
    huggingface = _object(huggingface_receipt_path, "Hugging Face private verification")
    expected_private = publication.build_private_verification(
        inputs["private_plan"], github, huggingface
    )
    private_verification = _object(private_verification_path, "private verification")
    if private_verification != expected_private:
        raise V3HuggingFaceReleaseError(
            "private verification does not reproduce from remote receipts"
        )
    private_sha = publication._validate_private_verification(private_verification)
    public_plan = _object(public_plan_path, "public transition plan")
    public_plan_sha = publication._validate_public_plan(
        public_plan,
        gate=inputs["gate"],
        private=private_verification,
    )
    if (
        public_plan.get("private_verification_sha256") != private_sha
        or public_plan.get("artifact_sha256") != inputs["package"]["artifact_sha256"]
    ):
        raise V3HuggingFaceReleaseError(
            "public transition plan is detached from the private release"
        )
    return {
        **inputs,
        "github_receipt": github,
        "huggingface_receipt": huggingface,
        "private_verification": private_verification,
        "private_verification_sha256": private_sha,
        "public_plan": public_plan,
        "public_plan_sha256": public_plan_sha,
    }


def publish_public(
    *,
    package: Path,
    public_repo: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    github_receipt_path: Path,
    huggingface_receipt_path: Path,
    private_verification_path: Path,
    public_plan_path: Path,
    confirmation: str,
    api: Any | None = None,
    download_file: Callable[..., str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Exercise public-transition logic only with an injected test client."""

    if api is None or download_file is None:
        raise V3HuggingFaceReleaseError(
            "local public transition is disabled; use "
            "tools/modal_v3_huggingface_release.py::publish_public_modal"
        )

    if confirmation != publication.OWNER_RELEASE_CONFIRMATION:
        raise V3HuggingFaceReleaseError("exact owner release confirmation is required")
    inputs = _validate_public_inputs(
        package=package,
        public_repo=public_repo,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
        github_receipt_path=github_receipt_path,
        huggingface_receipt_path=huggingface_receipt_path,
        private_verification_path=private_verification_path,
        public_plan_path=public_plan_path,
    )
    client = api
    downloader = download_file
    identity = _authenticated_identity(client, require_admin=True)
    if not _repo_exists(client):
        raise V3HuggingFaceReleaseError("verified private repository no longer exists")
    try:
        before = client.model_info(publication.HF_REPOSITORY, revision="main")
    except Exception as error:
        raise V3HuggingFaceReleaseError("private repository lookup failed") from error
    expected_revision = publication._revision(
        inputs["huggingface_receipt"].get("revision"),
        "Hugging Face verified private revision",
    )
    if _info_private(before) is not True or _info_revision(before) != expected_revision:
        raise V3HuggingFaceReleaseError(
            "private repository visibility or revision changed before publication"
        )
    before_metadata = _verify_remote_inventory(
        api=client,
        download_file=downloader,
        revision=expected_revision,
        expected=inputs["full_inventory"],
    )
    if before_metadata != inputs["huggingface_receipt"].get("hub_metadata"):
        raise V3HuggingFaceReleaseError("private Hub metadata changed")
    transition_attempted = False
    try:
        transition_attempted = True
        client.update_repo_settings(
            repo_id=publication.HF_REPOSITORY,
            repo_type="model",
            private=False,
        )
        after = client.model_info(publication.HF_REPOSITORY, revision="main")
        if (
            _info_private(after) is not False
            or _info_revision(after) != expected_revision
        ):
            raise V3HuggingFaceReleaseError(
                "public visibility or immutable revision verification failed"
            )
        after_metadata = _verify_remote_inventory(
            api=client,
            download_file=downloader,
            revision=expected_revision,
            expected=inputs["full_inventory"],
        )
        if after_metadata != before_metadata:
            raise V3HuggingFaceReleaseError("public Hub metadata changed")
    except Exception as error:
        if transition_attempted:
            try:
                client.update_repo_settings(
                    repo_id=publication.HF_REPOSITORY,
                    repo_type="model",
                    private=True,
                )
                rolled_back = client.model_info(
                    publication.HF_REPOSITORY, revision="main"
                )
                if (
                    _info_private(rolled_back) is not True
                    or _info_revision(rolled_back) != expected_revision
                ):
                    raise V3HuggingFaceReleaseError(
                        "private rollback visibility could not be verified"
                    )
                rollback_metadata = _verify_remote_inventory(
                    api=client,
                    download_file=downloader,
                    revision=expected_revision,
                    expected=inputs["full_inventory"],
                )
                if rollback_metadata != before_metadata:
                    raise V3HuggingFaceReleaseError(
                        "private rollback metadata could not be verified"
                    )
            except Exception as rollback_error:
                raise V3HuggingFaceReleaseError(
                    "public verification failed and private rollback failed"
                ) from rollback_error
            raise V3HuggingFaceReleaseError(
                "public verification failed; repository reverted to private"
            ) from error
        raise V3HuggingFaceReleaseError(
            "public visibility transition failed before visibility changed"
        ) from error
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    body = {
        "schema_version": PUBLIC_OPERATION_SCHEMA,
        "status": "PUBLIC_VISIBILITY_AND_IMMUTABLE_VERIFICATION_PASSED",
        "owner": "infercrane",
        "repository": publication.HF_REPOSITORY,
        "visibility": "public",
        "revision": expected_revision,
        "artifact_sha256": inputs["package"]["artifact_sha256"],
        "full_inventory_sha256": inputs["full_inventory_sha256"],
        "model_card_sha256": inputs["model_card_sha256"],
        "hub_metadata": after_metadata,
        "owner_authorization_sha256": inputs["authorization_sha256"],
        "publication_gate_sha256": inputs["gate_sha256"],
        "private_verification_sha256": inputs["private_verification_sha256"],
        "public_transition_plan_sha256": inputs["public_plan_sha256"],
        "authenticated_identity_sha256": identity["identity_sha256"],
        "authenticated_org_role": identity["role"],
        "network_used": True,
        "upload_performed": False,
        "publication_performed": True,
        "legal_conclusion": False,
        "completed_at_utc": timestamp,
    }
    return _receipt(body)


def _private_arguments(parser: argparse.ArgumentParser) -> None:
    for name in (
        "package",
        "public-repo",
        "authorization",
        "gate",
        "private-plan",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan-private")
    _private_arguments(plan)
    plan.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan-private":
            result = build_private_stage_plan(
                package=args.package,
                public_repo=args.public_repo,
                authorization_path=args.authorization,
                gate_path=args.gate,
                private_plan_path=args.private_plan,
            )
            _write_once(args.output, result)
        else:
            raise V3HuggingFaceReleaseError("unsupported local operation")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
