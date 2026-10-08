#!/usr/bin/env python3
"""Stage Commerce-1 from a Modal volume without copying the checkpoint.

The qualified model package is immutable and can be tens of gigabytes.  This
module keeps it on the release volume, validates it with ``v3_publication``,
combines it with a separately rendered public-assets directory, and uploads the
two roots in one private Hugging Face commit.  It never uses
``upload_large_folder`` because that helper writes resume metadata inside its
input directory.

``create_release_controls_modal`` is network-blocked and writes the owner
authorization, publication gate, and private plan directly beside the volume
package. ``plan_private_modal`` is network read-only and writes a self-hashed
stage plan. ``stage_private_modal`` requires that plan, authenticates as an
InferCrane organization writer, creates or resumes a private repository, and
verifies the immutable remote revision byte for byte. Public visibility remains
a separate fully gated operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from tools import v3_huggingface_release as local_release
from tools import v3_publication as publication

ROOT = Path(__file__).parents[1]
APP_NAME = "infercrane-commerce-1-hf-private-stage"
RESULT_VOLUME_NAME = "infercrane-commerce-v3-clean-results-v1"
RESULT_MOUNT = Path("/release-results")
REMOTE_PUBLIC_REPO = Path("/workspace/public-repo")
HF_SECRET_NAME = "infercrane-huggingface"
REMOTE_VERIFICATION_EPHEMERAL_DISK_MIB = 512 * 1024
STAGE_PLAN_SCHEMA = "infercrane-commerce-1-modal-hf-private-stage-plan/v1"
STAGE_OPERATION_SCHEMA = "infercrane-commerce-1-modal-hf-private-operation/v1"
PUBLIC_OPERATION_SCHEMA = "infercrane-commerce-1-modal-hf-public-operation/v1"
CONTROL_CREATION_SCHEMA = "infercrane-commerce-1-modal-release-controls/v1"
STAGE_CONFIRMATION = "STAGE_INFERCRANE_COMMERCE_1_PRIVATE_FROM_MODAL"
AUTHORIZATION_TIME_TOLERANCE = timedelta(minutes=15)
CONTROL_FILENAMES = (
    "owner-authorization.json",
    "publication-gate.json",
    "private-plan.json",
)
MAX_PUBLIC_ASSET_FILE_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_ASSET_TOTAL_BYTES = 128 * 1024 * 1024
PUBLIC_ASSET_SCHEMA = "infercrane-commerce-1-public-release-assets/v3"
PUBLIC_EVIDENCE_SCHEMA = "infercrane-commerce-1-public-verified-evidence/v2"
PUBLIC_ASSET_STATUS = "READY_FOR_OWNER_RELEASE_GATE"
PUBLIC_RELEASE_STATE = "INITIAL_PUBLIC_RELEASE_PENDING_OFFICIAL_V0_3"
PUBLIC_ASSET_ARCHIVE_PREFIX = "release-assets"
PUBLIC_ASSET_ARCHIVE_ONLY = frozenset({"publication-manifest.json", "SHA256SUMS"})
PUBLIC_ASSET_REQUIRED_FILES = frozenset(
    {
        "README.md",
        "ARCHITECTURE.md",
        "USAGE.md",
        "LOCAL_RUN.md",
        "API.md",
        "REPRODUCIBILITY.md",
        "GITHUB_RELEASE.md",
        "LICENSE",
        "THIRD_PARTY.md",
        "PROVENANCE.json",
        "publication-manifest.json",
        "verified-evidence.json",
        "QUALIFICATION.json",
        "SHA256SUMS",
    }
)
# Keep this identical to the canonical public-release renderer. Wildcard
# notation such as ``{{VERIFIED_*}}`` is explanatory prose, while real template
# fields contain only uppercase letters, digits, and underscores.
_PLACEHOLDER = re.compile(r"\{\{[A-Z0-9_]+\}\}")
_SENSITIVE_NAME = re.compile(
    r"(^|[._-])(secret|token|credential|private[._-]?key)([._-]|$)", re.I
)
_CREDENTIAL_RULES = (
    (
        "GitHub credential",
        re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"),
    ),
    (
        "Hugging Face credential",
        re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{20,}"),
    ),
    (
        "OpenAI or OpenRouter credential",
        re.compile(rb"(?<![A-Za-z0-9])sk-(?:or-v1-)?[A-Za-z0-9_-]{20,}"),
    ),
    (
        "AWS access key",
        re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "private key",
        re.compile(b"-----BEGIN " + b"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)
_CREDENTIAL_SCAN_CHUNK_BYTES = 1024 * 1024
_CREDENTIAL_SCAN_OVERLAP_BYTES = 64


class ModalHuggingFaceReleaseError(ValueError):
    """The package, assets, controls, or remote state are unsafe."""


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
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_credential_material(path: Path) -> bool:
    """Stream-scan a bounded public asset, including chunk boundaries."""

    overlap = b""
    with path.open("rb") as source:
        while chunk := source.read(_CREDENTIAL_SCAN_CHUNK_BYTES):
            payload = overlap + chunk
            if any(pattern.search(payload) for _label, pattern in _CREDENTIAL_RULES):
                return True
            overlap = payload[-_CREDENTIAL_SCAN_OVERLAP_BYTES:]
    return False


def _receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": _json_sha256(body)}


def _authorization_time(value: str, *, now: datetime | None = None) -> str:
    """Require an explicit UTC invocation time, not a generated placeholder."""

    if not isinstance(value, str):
        raise ModalHuggingFaceReleaseError(
            "authorized-at must be an explicit ISO-8601 UTC timestamp"
        )
    try:
        publication._utc_timestamp(value, "owner authorization time")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError, publication.V3PublicationError) as error:
        raise ModalHuggingFaceReleaseError(
            "authorized-at must be an explicit ISO-8601 UTC timestamp"
        ) from error
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if abs(current - parsed.astimezone(UTC)) > AUTHORIZATION_TIME_TOLERANCE:
        raise ModalHuggingFaceReleaseError(
            "authorized-at must be the real current invocation time"
        )
    return value


def _safe_relative(value: str, label: str) -> Path:
    path = PurePosixPath(value)
    raw_parts = value.split("/")
    if (
        not value
        or path.is_absolute()
        or "." in raw_parts
        or ".." in raw_parts
        or "" in raw_parts
    ):
        raise ModalHuggingFaceReleaseError(f"{label} must be a safe relative path")
    return Path(*path.parts)


def _inside(root: Path, relative: str, label: str) -> Path:
    candidate = root / _safe_relative(relative, label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ModalHuggingFaceReleaseError(f"{label} does not exist") from error
    if not resolved.is_relative_to(resolved_root):
        raise ModalHuggingFaceReleaseError(f"{label} escapes its trusted root")
    return candidate


def _package_upload_inventory(
    validated: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Reuse the already-verified manifest inventory without rehashing 51 GiB."""

    root = Path(validated["root"])
    manifest = validated["manifest"]
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ModalHuggingFaceReleaseError("qualified package inventory is absent")
    inventory = {
        str(name): {"sha256": item["sha256"], "size_bytes": item["size_bytes"]}
        for name, item in files.items()
        if isinstance(name, str) and isinstance(item, Mapping)
    }
    if len(inventory) != len(files):
        raise ModalHuggingFaceReleaseError("qualified package inventory is invalid")
    manifest_path = root / "release-manifest.json"
    inventory["release-manifest.json"] = {
        "sha256": _file_sha256(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
    }
    return dict(sorted(inventory.items()))


def _scan_public_asset_inventory(root: Path) -> dict[str, dict[str, Any]]:
    """Inventory and content-scan one regular, bounded public-assets tree."""

    if root.is_symlink() or not root.is_dir():
        raise ModalHuggingFaceReleaseError("public assets must be a regular directory")
    inventory = publication._regular_inventory(root)
    if "README.md" not in inventory:
        raise ModalHuggingFaceReleaseError("public assets must contain README.md")
    total = 0
    for name, item in inventory.items():
        size = item["size_bytes"]
        total += size
        if size > MAX_PUBLIC_ASSET_FILE_BYTES:
            raise ModalHuggingFaceReleaseError(f"public asset is too large: {name}")
        if _SENSITIVE_NAME.search(name):
            raise ModalHuggingFaceReleaseError(
                f"public asset has a sensitive filename: {name}"
            )
        path = root / name
        if _contains_credential_material(path):
            raise ModalHuggingFaceReleaseError(
                f"public asset contains credential material: {name}"
            )
    if total > MAX_PUBLIC_ASSET_TOTAL_BYTES:
        raise ModalHuggingFaceReleaseError("public assets exceed the size ceiling")
    card = (root / "README.md").read_text(encoding="utf-8")
    if (
        "# InferCrane Commerce-1\n" not in card
        or publication.MODEL_ID not in card
        or "# InferCrane Commerce-1 V3" in card
        or "inference: false" not in card
        or "pipeline_tag:" in card
        or "yasin" + "toy" in card.casefold()
        or _PLACEHOLDER.search(card) is not None
    ):
        raise ModalHuggingFaceReleaseError(
            "public README is personal, unresolved, or targets the wrong model"
        )
    return inventory


def _public_asset_inventory(
    root: Path, *, package: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate materialized release assets against the exact qualified package."""

    inventory = _scan_public_asset_inventory(root)
    if set(inventory) != PUBLIC_ASSET_REQUIRED_FILES:
        missing = sorted(PUBLIC_ASSET_REQUIRED_FILES - set(inventory))
        extra = sorted(set(inventory) - PUBLIC_ASSET_REQUIRED_FILES)
        raise ModalHuggingFaceReleaseError(
            f"public release asset set changed: missing={missing}, extra={extra}"
        )

    manifest = publication._object(
        root / "publication-manifest.json", "public asset publication manifest"
    )
    evidence = publication._object(
        root / "verified-evidence.json", "public asset verified evidence"
    )
    qualification = publication._object(
        root / "QUALIFICATION.json", "public asset runtime qualification"
    )
    try:
        manifest_sha256 = publication._self_hash(
            manifest, "manifest_sha256", "public asset publication manifest"
        )
        evidence_sha256 = publication._self_hash(
            evidence, "evidence_sha256", "public asset verified evidence"
        )
        qualification_sha256 = publication._validate_public_qualification(
            qualification, package
        )
    except publication.V3PublicationError as error:
        raise ModalHuggingFaceReleaseError(
            "public release evidence is invalid or detached"
        ) from error

    expected_manifest_fields = {
        "schema_version",
        "status",
        "organization",
        "model_id",
        "package_manifest_sha256",
        "verified_evidence_sha256",
        "public_qualification_attestation_sha256",
        "unresolved_placeholders",
        "files",
        "checksums",
        "upload_performed",
        "publication_performed",
        "manifest_sha256",
    }
    expected_evidence_fields = {
        "schema_version",
        "status",
        "model_id",
        "release_manifest_sha256",
        "package_files_sha256",
        "artifact_sha256",
        "calibration_file_sha256",
        "runtime_manifest_sha256",
        "runtime_identity_sha256",
        "public_evaluation_file_sha256",
        "public_evaluation_attestation_sha256",
        "training_data_lineage_sha256",
        "public_qualification_attestation_sha256",
        "protected_qualification_receipt_sha256",
        "qualification_passed",
        "results",
        "release_state",
        "release_evidence_ready",
        "publication_gate",
        "publication_ready",
        "upload_performed",
        "publication_performed",
        "evidence_sha256",
    }
    public_evaluation = package.get("public_evaluation")
    training_lineage = package.get("training_data_lineage")
    if (
        set(manifest) != expected_manifest_fields
        or set(evidence) != expected_evidence_fields
        or not isinstance(public_evaluation, Mapping)
        or not isinstance(training_lineage, Mapping)
    ):
        raise ModalHuggingFaceReleaseError("public release evidence fields changed")

    file_hashes = {
        name: item["sha256"]
        for name, item in inventory.items()
        if name != "publication-manifest.json"
    }
    checksums_without_self = {
        name: digest for name, digest in file_hashes.items() if name != "SHA256SUMS"
    }
    expected_checksums = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(checksums_without_self.items())
    ).encode()
    checksum_path = root / "SHA256SUMS"
    if (
        manifest.get("schema_version") != PUBLIC_ASSET_SCHEMA
        or manifest.get("status") != PUBLIC_ASSET_STATUS
        or manifest.get("organization") != "infercrane"
        or manifest.get("model_id") != publication.MODEL_ID
        or manifest.get("package_manifest_sha256") != package["manifest_sha256"]
        or manifest.get("verified_evidence_sha256") != evidence_sha256
        or manifest.get("public_qualification_attestation_sha256")
        != qualification_sha256
        or manifest.get("unresolved_placeholders") != []
        or manifest.get("files") != file_hashes
        or manifest.get("checksums")
        != {"path": "SHA256SUMS", "sha256": _file_sha256(checksum_path)}
        or manifest.get("upload_performed") is not False
        or manifest.get("publication_performed") is not False
        or checksum_path.read_bytes() != expected_checksums
    ):
        raise ModalHuggingFaceReleaseError(
            "public asset manifest is detached, incomplete, or not release-ready"
        )

    results = evidence.get("results")
    score = public_evaluation.get("decision_index")
    if (
        evidence.get("schema_version") != PUBLIC_EVIDENCE_SCHEMA
        or evidence.get("status") != PUBLIC_ASSET_STATUS
        or evidence.get("model_id") != publication.MODEL_ID
        or evidence.get("release_manifest_sha256") != package["manifest_sha256"]
        or evidence.get("package_files_sha256") != package["inventory_sha256"]
        or evidence.get("artifact_sha256") != package["release_artifact_sha256"]
        or evidence.get("calibration_file_sha256") != package["calibration_file_sha256"]
        or evidence.get("runtime_manifest_sha256") != package["runtime_manifest_sha256"]
        or evidence.get("runtime_identity_sha256") != package["runtime_identity_sha256"]
        or evidence.get("public_evaluation_file_sha256")
        != package["public_evaluation_file_sha256"]
        or evidence.get("public_evaluation_attestation_sha256")
        != public_evaluation.get("attestation_sha256")
        or evidence.get("training_data_lineage_sha256")
        != training_lineage.get("lineage_sha256")
        or evidence.get("public_qualification_attestation_sha256")
        != qualification_sha256
        or evidence.get("protected_qualification_receipt_sha256")
        != qualification.get("protected_qualification_receipt_sha256")
        or evidence.get("qualification_passed") is not True
        or results
        != {
            "decision_index_0_2_1": score,
            "evaluated_rows": 150_317,
            "official_decision_index_0_3_full_score": None,
            "official_rank": None,
        }
        or isinstance(score, bool)
        or not isinstance(score, int | float)
        or float(score) <= publication.RELEASE_BASELINE_DECISION_INDEX
        or public_evaluation.get("merged") != 150_317
        or evidence.get("release_state") != PUBLIC_RELEASE_STATE
        or evidence.get("release_evidence_ready") is not True
        or evidence.get("publication_gate") != PUBLIC_RELEASE_STATE
        or evidence.get("publication_ready") is not False
        or evidence.get("upload_performed") is not False
        or evidence.get("publication_performed") is not False
    ):
        raise ModalHuggingFaceReleaseError(
            "verified public evidence is detached or not release-ready"
        )
    if manifest_sha256 != manifest["manifest_sha256"]:
        raise ModalHuggingFaceReleaseError("public asset manifest hash changed")
    return inventory


def _validate_qualification_projection(
    *,
    qualification_path: Path,
    assets: Path,
    package: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind the supplied qualification to the sanitized public asset projection."""

    asset_inventory = _public_asset_inventory(assets, package=package)
    supplied = publication._object(qualification_path, "runtime qualification")
    projected = publication._object(
        assets / "QUALIFICATION.json", "public asset runtime qualification"
    )
    if supplied.get("schema_version") == publication.PUBLIC_QUALIFICATION_SCHEMA:
        if supplied != projected:
            raise ModalHuggingFaceReleaseError(
                "public qualification differs from the release asset projection"
            )
    elif projected.get("protected_qualification_receipt_sha256") != (
        publication._file_sha256(qualification_path)
    ):
        raise ModalHuggingFaceReleaseError(
            "public qualification is detached from the protected qualification"
        )
    return asset_inventory


def build_release_controls(
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    assets: Path,
    confirmation: str,
    source_revision: str,
    authorized_at_utc: str,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate once and build the three initial-release controls in memory."""

    if confirmation != publication.OWNER_RELEASE_CONFIRMATION:
        raise ModalHuggingFaceReleaseError(
            "exact owner release confirmation is required"
        )
    publication._revision(source_revision, "source revision")
    _authorization_time(authorized_at_utc, now=now)
    (
        validated,
        qualification_sha,
        public_sha,
        official_sha,
        release_state,
        official_full_score,
    ) = publication._validated_release_materials(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=None,
        official_v03=None,
    )
    if (
        public_sha is not None
        or official_sha is not None
        or release_state != publication.INITIAL_RELEASE_STATE
        or official_full_score is not None
    ):
        raise ModalHuggingFaceReleaseError(
            "server-side control creation is restricted to the initial release state"
        )
    asset_inventory = _validate_qualification_projection(
        qualification_path=qualification,
        assets=assets,
        package=validated,
    )
    authorization = publication._owner_authorization_from_validated_materials(
        validated=validated,
        qualification_sha=qualification_sha,
        public_sha=public_sha,
        official_sha=official_sha,
        release_state=release_state,
        official_full_score=official_full_score,
        authorized_at_utc=authorized_at_utc,
    )
    gate = publication._evidence_gate_from_validated_materials(
        validated=validated,
        qualification_sha=qualification_sha,
        public_sha=public_sha,
        official_sha=official_sha,
        release_state=release_state,
        official_full_score=official_full_score,
        authorization=authorization,
    )
    private_plan = publication.build_private_plan(gate, source_revision=source_revision)
    authorization_sha = publication._validate_owner_authorization(
        authorization,
        package=validated,
        qualification_sha256=qualification_sha,
        public_v03_sha256=public_sha,
        official_v03_sha256=official_sha,
        release_state=release_state,
        official_v03_full_score=official_full_score,
    )
    gate_sha = publication._validate_gate(gate)
    private_plan_sha = publication._validate_private_plan(private_plan)
    if private_plan.get("publication_gate_sha256") != gate_sha:
        raise ModalHuggingFaceReleaseError(
            "private rehearsal is detached from the generated publication gate"
        )
    summary = _receipt(
        {
            "schema_version": CONTROL_CREATION_SCHEMA,
            "status": "INITIAL_RELEASE_CONTROLS_CREATED",
            "release_manifest_sha256": validated["manifest_sha256"],
            "package_inventory_sha256": validated["inventory_sha256"],
            "runtime_qualification_sha256": qualification_sha,
            "public_asset_inventory_sha256": _json_sha256(asset_inventory),
            "owner_authorization_sha256": authorization_sha,
            "publication_gate_sha256": gate_sha,
            "private_plan_sha256": private_plan_sha,
            "source_revision": source_revision,
            "network_used": False,
            "credentials_used": False,
            "gpu_used": False,
            "repositories_created": False,
            "upload_performed": False,
            "publication_performed": False,
        }
    )
    return authorization, gate, private_plan, summary


def create_release_controls(
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    assets: Path,
    control: Path,
    confirmation: str,
    source_revision: str,
    authorized_at_utc: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Create the validated control set once, returning only a hash summary."""

    if control.is_symlink() or not control.is_dir():
        raise ModalHuggingFaceReleaseError(
            "control path must be a regular non-symlink directory"
        )
    outputs = [control / name for name in CONTROL_FILENAMES]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ModalHuggingFaceReleaseError("release control already exists")
    authorization, gate, private_plan, summary = build_release_controls(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        assets=assets,
        confirmation=confirmation,
        source_revision=source_revision,
        authorized_at_utc=authorized_at_utc,
        now=now,
    )
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ModalHuggingFaceReleaseError("release control appeared during validation")
    for path, value in zip(outputs, (authorization, gate, private_plan), strict=True):
        _write_once(path, value)
    observed_authorization = publication._object(
        outputs[0], "created owner publication authorization"
    )
    observed_gate = publication._object(outputs[1], "created publication gate")
    observed_plan = publication._object(outputs[2], "created private rehearsal")
    if (
        observed_authorization != authorization
        or observed_gate != gate
        or observed_plan != private_plan
    ):
        raise ModalHuggingFaceReleaseError(
            "created release control bytes failed verification"
        )
    return summary


def _merge_sources(
    *,
    package_root: Path,
    package_inventory: Mapping[str, Mapping[str, Any]],
    asset_root: Path,
    asset_inventory: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    """Build a collision-free Hub projection of both evidence domains.

    The immutable checkpoint package retains its original root paths, including
    its own ``SHA256SUMS``. The complete public-assets bundle is preserved under
    ``release-assets/`` so its manifest and checksum file remain self-contained.
    User-facing public assets are additionally projected at repository root;
    archive-only governance files stay namespaced to avoid checksum ambiguity.
    """

    inventory = {name: dict(item) for name, item in package_inventory.items()}
    sources = {name: package_root / name for name in package_inventory}
    for name, item in asset_inventory.items():
        archived_name = f"{PUBLIC_ASSET_ARCHIVE_PREFIX}/{name}"
        if archived_name in inventory:
            raise ModalHuggingFaceReleaseError(
                f"public asset archive conflicts with immutable package path: {name}"
            )
        inventory[archived_name] = dict(item)
        sources[archived_name] = asset_root / name
        if name in PUBLIC_ASSET_ARCHIVE_ONLY:
            continue
        if name in inventory and inventory[name] != dict(item):
            raise ModalHuggingFaceReleaseError(
                f"public asset conflicts with immutable package bytes: {name}"
            )
        inventory[name] = dict(item)
        sources[name] = asset_root / name
    return dict(sorted(inventory.items())), dict(sorted(sources.items()))


def _validate_controls(
    *,
    package: Path,
    public_repo: Path,
    assets: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
) -> dict[str, Any]:
    validated = publication._validate_package(package, public_repo)
    authorization = publication._object(
        authorization_path, "owner publication authorization"
    )
    gate = publication._object(gate_path, "publication gate")
    private_plan = publication._object(private_plan_path, "private rehearsal")
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
        raise ModalHuggingFaceReleaseError(
            "publication gate is detached from package or authorization"
        )
    private_sha = publication._validate_private_plan(private_plan)
    if (
        private_plan.get("publication_gate_sha256") != gate_sha
        or private_plan.get("artifact_sha256") != validated["artifact_sha256"]
        or private_plan.get("package_inventory_sha256") != validated["inventory_sha256"]
        or private_plan.get("huggingface")
        != {"repository": publication.HF_REPOSITORY, "visibility": "private"}
    ):
        raise ModalHuggingFaceReleaseError(
            "private rehearsal is detached from the package"
        )
    package_inventory = _package_upload_inventory(validated)
    asset_inventory = _public_asset_inventory(assets, package=validated)
    upload_inventory, upload_sources = _merge_sources(
        package_root=validated["root"],
        package_inventory=package_inventory,
        asset_root=assets,
        asset_inventory=asset_inventory,
    )
    return {
        "package": validated,
        "authorization_sha256": authorization_sha,
        "gate_sha256": gate_sha,
        "private_plan_sha256": private_sha,
        "private_plan": private_plan,
        "asset_inventory": asset_inventory,
        "asset_inventory_sha256": _json_sha256(asset_inventory),
        "upload_inventory": upload_inventory,
        "upload_inventory_sha256": _json_sha256(upload_inventory),
        "upload_sources": upload_sources,
        "model_card_sha256": asset_inventory["README.md"]["sha256"],
    }


def build_stage_plan(
    *,
    package: Path,
    public_repo: Path,
    assets: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
) -> dict[str, Any]:
    inputs = _validate_controls(
        package=package,
        public_repo=public_repo,
        assets=assets,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    package_value = inputs["package"]
    body = {
        "schema_version": STAGE_PLAN_SCHEMA,
        "status": "PRIVATE_STAGE_PLANNED_NOT_PERFORMED",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "artifact_sha256": package_value["artifact_sha256"],
        "release_manifest_sha256": package_value["manifest_sha256"],
        "package_inventory_sha256": package_value["inventory_sha256"],
        "public_asset_inventory_sha256": inputs["asset_inventory_sha256"],
        "upload_inventory_sha256": inputs["upload_inventory_sha256"],
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


def _expected_plan(inputs: Mapping[str, Any]) -> dict[str, Any]:
    package_value = inputs["package"]
    body = {
        "schema_version": STAGE_PLAN_SCHEMA,
        "status": "PRIVATE_STAGE_PLANNED_NOT_PERFORMED",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "artifact_sha256": package_value["artifact_sha256"],
        "release_manifest_sha256": package_value["manifest_sha256"],
        "package_inventory_sha256": package_value["inventory_sha256"],
        "public_asset_inventory_sha256": inputs["asset_inventory_sha256"],
        "upload_inventory_sha256": inputs["upload_inventory_sha256"],
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


def _missing_remote_files(
    *,
    api: Any,
    downloader: Callable[..., str],
    expected: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    local_release._verify_existing_files_before_resume(
        api=api, download_file=downloader, expected=expected
    )
    names = set(
        api.list_repo_files(
            repo_id=publication.HF_REPOSITORY,
            repo_type="model",
            revision="main",
        )
    )
    return sorted(set(expected) - names)


def _modal_hub_transport() -> tuple[Any, Callable[..., str]]:
    """Create the only production Hub transport from the mounted Modal secret."""

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ModalHuggingFaceReleaseError("HF_TOKEN is unavailable")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise ModalHuggingFaceReleaseError("huggingface-hub is unavailable") from error
    api = HfApi(token=token)

    def download_file(**kwargs: Any) -> str:
        return hf_hub_download(token=token, **kwargs)

    return api, download_file


def _commit_missing(
    *,
    api: Any,
    names: list[str],
    sources: Mapping[str, Path],
    parent_commit: str,
) -> None:
    if not names:
        return
    try:
        from huggingface_hub import CommitOperationAdd
    except ImportError as error:
        raise ModalHuggingFaceReleaseError("huggingface-hub is unavailable") from error
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=sources[name])
        for name in names
    ]
    api.create_commit(
        repo_id=publication.HF_REPOSITORY,
        repo_type="model",
        operations=operations,
        commit_message="Stage immutable Commerce-1 release privately",
        parent_commit=parent_commit,
        num_threads=8,
    )


def stage_private(
    *,
    package: Path,
    public_repo: Path,
    assets: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    stage_plan_path: Path,
    confirmation: str,
    api: Any | None = None,
    downloader: Callable[..., str] | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if confirmation != STAGE_CONFIRMATION:
        raise ModalHuggingFaceReleaseError(
            "exact private-stage confirmation is required"
        )
    inputs = _validate_controls(
        package=package,
        public_repo=public_repo,
        assets=assets,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    supplied_plan = publication._object(stage_plan_path, "Modal private stage plan")
    expected_plan = _expected_plan(inputs)
    if supplied_plan != expected_plan:
        raise ModalHuggingFaceReleaseError("private stage plan changed from inputs")
    plan_sha = publication._self_hash(
        supplied_plan, "receipt_sha256", "Modal private stage plan"
    )
    if api is None or downloader is None:
        raise ModalHuggingFaceReleaseError(
            "server-side Hugging Face transport must be injected"
        )
    client = api
    download_file = downloader
    identity = local_release._authenticated_identity(client, require_admin=False)
    created = False
    if not local_release._repo_exists(client):
        client.create_repo(
            repo_id=publication.HF_REPOSITORY,
            repo_type="model",
            private=True,
            exist_ok=False,
        )
        created = True
    before = client.model_info(publication.HF_REPOSITORY, revision="main")
    if local_release._info_private(before) is not True:
        raise ModalHuggingFaceReleaseError("target repository is not private")
    parent = local_release._info_revision(before)
    missing = _missing_remote_files(
        api=client,
        downloader=download_file,
        expected=inputs["upload_inventory"],
    )
    _commit_missing(
        api=client,
        names=missing,
        sources=inputs["upload_sources"],
        parent_commit=parent,
    )
    after = client.model_info(publication.HF_REPOSITORY, revision="main")
    if local_release._info_private(after) is not True:
        raise ModalHuggingFaceReleaseError("private upload changed visibility")
    revision = local_release._info_revision(after)
    hub_metadata = local_release._verify_remote_inventory(
        api=client,
        download_file=download_file,
        revision=revision,
        expected=inputs["upload_inventory"],
    )
    if set(hub_metadata) != {".gitattributes"}:
        raise ModalHuggingFaceReleaseError(
            "private repository lacks canonical Hub metadata"
        )
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    package_value = inputs["package"]
    remote_body = {
        "schema_version": publication.REMOTE_RECEIPT_SCHEMA,
        "status": "IMMUTABLE_PRIVATE_REVISION_VERIFIED",
        "service": "huggingface",
        "repository": publication.HF_REPOSITORY,
        "visibility": "private",
        "revision": revision,
        "source_revision": inputs["private_plan"]["source_revision"],
        "artifact_sha256": package_value["artifact_sha256"],
        "inventory_sha256": package_value["inventory_sha256"],
        "hub_metadata": hub_metadata,
        "verified": True,
        "verified_at_utc": timestamp,
    }
    remote = _receipt(remote_body)
    publication._validate_remote(
        remote,
        service="huggingface",
        repository=publication.HF_REPOSITORY,
        artifact_sha256=package_value["artifact_sha256"],
        source_revision=inputs["private_plan"]["source_revision"],
        package_inventory_sha256=package_value["inventory_sha256"],
    )
    operation = _receipt(
        {
            "schema_version": STAGE_OPERATION_SCHEMA,
            "status": "PRIVATE_UPLOAD_AND_COMBINED_INVENTORY_VERIFICATION_PASSED",
            "repository": publication.HF_REPOSITORY,
            "visibility": "private",
            "revision": revision,
            "artifact_sha256": package_value["artifact_sha256"],
            "package_inventory_sha256": package_value["inventory_sha256"],
            "public_asset_inventory_sha256": inputs["asset_inventory_sha256"],
            "upload_inventory_sha256": inputs["upload_inventory_sha256"],
            "model_card_sha256": inputs["model_card_sha256"],
            "private_stage_plan_sha256": plan_sha,
            "remote_verification_sha256": remote["receipt_sha256"],
            "authenticated_identity_sha256": identity["identity_sha256"],
            "authenticated_org_role": identity["role"],
            "repository_created": created,
            "uploaded_file_count": len(missing),
            "network_used": True,
            "upload_performed": bool(missing),
            "publication_performed": False,
            "completed_at_utc": timestamp,
        }
    )
    return remote, operation


def _validate_stage_operation(
    value: Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    stage_plan_sha256: str,
    remote_receipt: Mapping[str, Any],
) -> str:
    supplied = publication._self_hash(
        value, "receipt_sha256", "Modal private stage operation"
    )
    package_value = inputs["package"]
    if (
        value.get("schema_version") != STAGE_OPERATION_SCHEMA
        or value.get("status")
        != "PRIVATE_UPLOAD_AND_COMBINED_INVENTORY_VERIFICATION_PASSED"
        or value.get("repository") != publication.HF_REPOSITORY
        or value.get("visibility") != "private"
        or value.get("revision") != remote_receipt.get("revision")
        or value.get("artifact_sha256") != package_value["artifact_sha256"]
        or value.get("package_inventory_sha256") != package_value["inventory_sha256"]
        or value.get("public_asset_inventory_sha256")
        != inputs["asset_inventory_sha256"]
        or value.get("upload_inventory_sha256") != inputs["upload_inventory_sha256"]
        or value.get("model_card_sha256") != inputs["model_card_sha256"]
        or value.get("private_stage_plan_sha256") != stage_plan_sha256
        or value.get("remote_verification_sha256")
        != remote_receipt.get("receipt_sha256")
        or value.get("network_used") is not True
        or value.get("publication_performed") is not False
    ):
        raise ModalHuggingFaceReleaseError(
            "private stage operation is detached from the combined upload"
        )
    return supplied


def _validate_public_controls(
    *,
    package: Path,
    public_repo: Path,
    assets: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    stage_plan_path: Path,
    stage_operation_path: Path,
    github_receipt_path: Path,
    huggingface_receipt_path: Path,
    private_verification_path: Path,
    public_plan_path: Path,
) -> dict[str, Any]:
    inputs = _validate_controls(
        package=package,
        public_repo=public_repo,
        assets=assets,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
    )
    stage_plan = publication._object(stage_plan_path, "Modal private stage plan")
    if stage_plan != _expected_plan(inputs):
        raise ModalHuggingFaceReleaseError("private stage plan changed from inputs")
    stage_plan_sha = publication._self_hash(
        stage_plan, "receipt_sha256", "Modal private stage plan"
    )
    github = publication._object(github_receipt_path, "GitHub private verification")
    huggingface = publication._object(
        huggingface_receipt_path, "Hugging Face private verification"
    )
    expected_private = publication.build_private_verification(
        inputs["private_plan"], github, huggingface
    )
    private_verification = publication._object(
        private_verification_path, "private verification"
    )
    if private_verification != expected_private:
        raise ModalHuggingFaceReleaseError(
            "private verification does not reproduce from remote receipts"
        )
    private_sha = publication._validate_private_verification(private_verification)
    public_plan = publication._object(public_plan_path, "public transition plan")
    public_plan_sha = publication._validate_public_plan(
        public_plan,
        gate=publication._object(gate_path, "publication gate"),
        private=private_verification,
    )
    if (
        public_plan.get("private_verification_sha256") != private_sha
        or public_plan.get("artifact_sha256") != inputs["package"]["artifact_sha256"]
    ):
        raise ModalHuggingFaceReleaseError(
            "public transition plan is detached from the private release"
        )
    operation = publication._object(
        stage_operation_path, "Modal private stage operation"
    )
    operation_sha = _validate_stage_operation(
        operation,
        inputs=inputs,
        stage_plan_sha256=stage_plan_sha,
        remote_receipt=huggingface,
    )
    return {
        **inputs,
        "github_receipt": github,
        "huggingface_receipt": huggingface,
        "private_verification_sha256": private_sha,
        "public_plan_sha256": public_plan_sha,
        "stage_operation_sha256": operation_sha,
    }


def publish_public(
    *,
    package: Path,
    public_repo: Path,
    assets: Path,
    authorization_path: Path,
    gate_path: Path,
    private_plan_path: Path,
    stage_plan_path: Path,
    stage_operation_path: Path,
    github_receipt_path: Path,
    huggingface_receipt_path: Path,
    private_verification_path: Path,
    public_plan_path: Path,
    confirmation: str,
    api: Any | None = None,
    downloader: Callable[..., str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if confirmation != publication.OWNER_RELEASE_CONFIRMATION:
        raise ModalHuggingFaceReleaseError(
            "exact owner release confirmation is required"
        )
    inputs = _validate_public_controls(
        package=package,
        public_repo=public_repo,
        assets=assets,
        authorization_path=authorization_path,
        gate_path=gate_path,
        private_plan_path=private_plan_path,
        stage_plan_path=stage_plan_path,
        stage_operation_path=stage_operation_path,
        github_receipt_path=github_receipt_path,
        huggingface_receipt_path=huggingface_receipt_path,
        private_verification_path=private_verification_path,
        public_plan_path=public_plan_path,
    )
    if api is None or downloader is None:
        raise ModalHuggingFaceReleaseError(
            "server-side Hugging Face transport must be injected"
        )
    client = api
    download_file = downloader
    identity = local_release._authenticated_identity(client, require_admin=True)
    if not local_release._repo_exists(client):
        raise ModalHuggingFaceReleaseError("verified private repository is absent")
    before = client.model_info(publication.HF_REPOSITORY, revision="main")
    expected_revision = publication._revision(
        inputs["huggingface_receipt"].get("revision"),
        "Hugging Face verified private revision",
    )
    if (
        local_release._info_private(before) is not True
        or local_release._info_revision(before) != expected_revision
    ):
        raise ModalHuggingFaceReleaseError(
            "private repository visibility or revision changed"
        )
    before_metadata = local_release._verify_remote_inventory(
        api=client,
        download_file=download_file,
        revision=expected_revision,
        expected=inputs["upload_inventory"],
    )
    if before_metadata != inputs["huggingface_receipt"].get("hub_metadata"):
        raise ModalHuggingFaceReleaseError("private Hub metadata changed")
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
            local_release._info_private(after) is not False
            or local_release._info_revision(after) != expected_revision
        ):
            raise ModalHuggingFaceReleaseError(
                "public visibility or revision verification failed"
            )
        after_metadata = local_release._verify_remote_inventory(
            api=client,
            download_file=download_file,
            revision=expected_revision,
            expected=inputs["upload_inventory"],
        )
        if after_metadata != before_metadata:
            raise ModalHuggingFaceReleaseError("public Hub metadata changed")
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
                    local_release._info_private(rolled_back) is not True
                    or local_release._info_revision(rolled_back) != expected_revision
                ):
                    raise ModalHuggingFaceReleaseError(
                        "private rollback could not be verified"
                    )
                rollback_metadata = local_release._verify_remote_inventory(
                    api=client,
                    download_file=download_file,
                    revision=expected_revision,
                    expected=inputs["upload_inventory"],
                )
                if rollback_metadata != before_metadata:
                    raise ModalHuggingFaceReleaseError(
                        "private rollback metadata could not be verified"
                    )
            except Exception as rollback_error:
                raise ModalHuggingFaceReleaseError(
                    "public verification and private rollback both failed"
                ) from rollback_error
            raise ModalHuggingFaceReleaseError(
                "public verification failed; repository reverted to private"
            ) from error
        raise
    timestamp = (now or datetime.now(UTC)).astimezone(UTC).isoformat()
    return _receipt(
        {
            "schema_version": PUBLIC_OPERATION_SCHEMA,
            "status": "PUBLIC_VISIBILITY_AND_COMBINED_INVENTORY_VERIFICATION_PASSED",
            "repository": publication.HF_REPOSITORY,
            "visibility": "public",
            "revision": expected_revision,
            "artifact_sha256": inputs["package"]["artifact_sha256"],
            "package_inventory_sha256": inputs["package"]["inventory_sha256"],
            "public_asset_inventory_sha256": inputs["asset_inventory_sha256"],
            "upload_inventory_sha256": inputs["upload_inventory_sha256"],
            "model_card_sha256": inputs["model_card_sha256"],
            "hub_metadata": after_metadata,
            "private_stage_operation_sha256": inputs["stage_operation_sha256"],
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
    )


def _control_directory(
    control_relative_path: str, *, protected_inputs: tuple[Path, ...]
) -> Path:
    relative_control = _safe_relative(control_relative_path, "control path")
    resolved_root = RESULT_MOUNT.resolve(strict=True)
    control = RESULT_MOUNT
    for part in relative_control.parts:
        control /= part
        if control.is_symlink():
            raise ModalHuggingFaceReleaseError("control path crosses a symlink")
        if control.exists() and not control.is_dir():
            raise ModalHuggingFaceReleaseError(
                "control path must be a regular directory"
            )
    candidate = control.resolve(strict=False)
    for protected in protected_inputs:
        resolved = protected.resolve(strict=True)
        if (
            candidate == resolved
            or candidate.is_relative_to(resolved)
            or resolved.is_relative_to(candidate)
        ):
            raise ModalHuggingFaceReleaseError(
                "control path must be disjoint from immutable release inputs"
            )
    control.mkdir(parents=True, exist_ok=True)
    try:
        control = control.resolve(strict=True)
    except OSError as error:
        raise ModalHuggingFaceReleaseError("control path is unavailable") from error
    if not control.is_relative_to(resolved_root):
        raise ModalHuggingFaceReleaseError("control path escapes its trusted root")
    return control


def _modal_paths(
    package_relative_path: str,
    assets_relative_path: str,
    control_relative_path: str,
) -> tuple[Path, Path, Path]:
    package = _inside(RESULT_MOUNT, package_relative_path, "package path")
    assets = _inside(RESULT_MOUNT, assets_relative_path, "public assets path")
    control = _control_directory(
        control_relative_path, protected_inputs=(package, assets)
    )
    return package, assets, control


def _modal_control_paths(
    package_relative_path: str,
    qualification_relative_path: str,
    assets_relative_path: str,
    control_relative_path: str,
) -> tuple[Path, Path, Path, Path]:
    package = _inside(RESULT_MOUNT, package_relative_path, "package path")
    qualification = _inside(
        RESULT_MOUNT, qualification_relative_path, "qualification path"
    )
    assets = _inside(RESULT_MOUNT, assets_relative_path, "public assets path")
    if qualification.is_symlink() or not qualification.is_file():
        raise ModalHuggingFaceReleaseError("qualification path must be a regular file")
    control = _control_directory(
        control_relative_path,
        protected_inputs=(package, qualification, assets),
    )
    return package, qualification, assets, control


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    publication._write_once(path, value)


try:
    import modal
except ImportError:  # pragma: no cover - pure helpers remain locally testable
    modal = None


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("huggingface-hub==1.31.0")
        .add_local_dir(ROOT / "tools", "/workspace/public-repo/tools", copy=True)
        .add_local_dir(
            ROOT / "src/infercrane_commerce_1",
            "/workspace/public-repo/src/infercrane_commerce_1",
            copy=True,
        )
        .add_local_dir(ROOT / "release", "/workspace/public-repo/release", copy=True)
        .env(
            {
                "PYTHONPATH": "/workspace/public-repo",
                "HF_HOME": "/tmp/huggingface",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
    )
    app = modal.App(APP_NAME)
    result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=False)
    hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])

    @app.function(
        image=image,
        volumes={RESULT_MOUNT: result_volume},
        cpu=4.0,
        memory=16384,
        timeout=24 * 60 * 60,
        max_containers=1,
        block_network=True,
    )
    def create_release_controls_modal(
        package_relative_path: str,
        qualification_relative_path: str,
        assets_relative_path: str,
        control_relative_path: str,
        confirmation: str,
        source_revision: str,
        authorized_at_utc: str,
    ) -> dict[str, Any]:
        package, qualification, assets, control = _modal_control_paths(
            package_relative_path,
            qualification_relative_path,
            assets_relative_path,
            control_relative_path,
        )
        summary = create_release_controls(
            package=package,
            public_repo=REMOTE_PUBLIC_REPO,
            qualification=qualification,
            assets=assets,
            control=control,
            confirmation=confirmation,
            source_revision=source_revision,
            authorized_at_utc=authorized_at_utc,
        )
        result_volume.commit()
        return summary

    @app.function(
        image=image,
        volumes={RESULT_MOUNT: result_volume},
        cpu=4.0,
        memory=16384,
        timeout=24 * 60 * 60,
        max_containers=1,
    )
    def plan_private_modal(
        package_relative_path: str,
        assets_relative_path: str,
        control_relative_path: str,
    ) -> dict[str, Any]:
        package, assets, control = _modal_paths(
            package_relative_path, assets_relative_path, control_relative_path
        )
        result = build_stage_plan(
            package=package,
            public_repo=REMOTE_PUBLIC_REPO,
            assets=assets,
            authorization_path=control / "owner-authorization.json",
            gate_path=control / "publication-gate.json",
            private_plan_path=control / "private-plan.json",
        )
        _write_once(control / "hf-private-stage-plan.json", result)
        result_volume.commit()
        return result

    @app.function(
        image=image,
        volumes={RESULT_MOUNT: result_volume},
        secrets=[hf_secret],
        cpu=8.0,
        memory=32768,
        ephemeral_disk=REMOTE_VERIFICATION_EPHEMERAL_DISK_MIB,
        timeout=24 * 60 * 60,
        max_containers=1,
    )
    def stage_private_modal(
        package_relative_path: str,
        assets_relative_path: str,
        control_relative_path: str,
        confirmation: str,
    ) -> dict[str, Any]:
        if not os.environ.get("HF_TOKEN"):
            raise ModalHuggingFaceReleaseError("HF_TOKEN is unavailable")
        package, assets, control = _modal_paths(
            package_relative_path, assets_relative_path, control_relative_path
        )
        remote_path = control / "huggingface-private-verification.json"
        operation_path = control / "huggingface-private-operation.json"
        if remote_path.exists() or operation_path.exists():
            raise ModalHuggingFaceReleaseError("private-stage receipts already exist")
        api, downloader = _modal_hub_transport()
        remote, operation = stage_private(
            package=package,
            public_repo=REMOTE_PUBLIC_REPO,
            assets=assets,
            authorization_path=control / "owner-authorization.json",
            gate_path=control / "publication-gate.json",
            private_plan_path=control / "private-plan.json",
            stage_plan_path=control / "hf-private-stage-plan.json",
            confirmation=confirmation,
            api=api,
            downloader=downloader,
        )
        _write_once(remote_path, remote)
        _write_once(operation_path, operation)
        result_volume.commit()
        return operation

    @app.function(
        image=image,
        volumes={RESULT_MOUNT: result_volume},
        secrets=[hf_secret],
        cpu=8.0,
        memory=32768,
        ephemeral_disk=REMOTE_VERIFICATION_EPHEMERAL_DISK_MIB,
        timeout=24 * 60 * 60,
        max_containers=1,
    )
    def publish_public_modal(
        package_relative_path: str,
        assets_relative_path: str,
        control_relative_path: str,
        confirmation: str,
    ) -> dict[str, Any]:
        if not os.environ.get("HF_TOKEN"):
            raise ModalHuggingFaceReleaseError("HF_TOKEN is unavailable")
        package, assets, control = _modal_paths(
            package_relative_path, assets_relative_path, control_relative_path
        )
        output = control / "huggingface-publication.json"
        if output.exists():
            raise ModalHuggingFaceReleaseError(
                "public operation receipt already exists"
            )
        api, downloader = _modal_hub_transport()
        receipt = publish_public(
            package=package,
            public_repo=REMOTE_PUBLIC_REPO,
            assets=assets,
            authorization_path=control / "owner-authorization.json",
            gate_path=control / "publication-gate.json",
            private_plan_path=control / "private-plan.json",
            stage_plan_path=control / "hf-private-stage-plan.json",
            stage_operation_path=control / "huggingface-private-operation.json",
            github_receipt_path=control / "github-private-verification.json",
            huggingface_receipt_path=control / "huggingface-private-verification.json",
            private_verification_path=control / "private-verification.json",
            public_plan_path=control / "public-plan.json",
            confirmation=confirmation,
            api=api,
            downloader=downloader,
        )
        _write_once(output, receipt)
        result_volume.commit()
        return receipt
