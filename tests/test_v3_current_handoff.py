from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.test_v3_publication import _evidence_files, _package, _receipt, _write
from tools import v3_publication as release

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_dependency_constants_match_publishable_source() -> None:
    source_root = REPOSITORY_ROOT / "src" / "infercrane_commerce_1"
    observed = {
        relative_path: release._file_sha256(source_root / relative_path)
        for relative_path in release.PUBLIC_RUNTIME_SOURCE_FILES
    }
    assert observed == release.PUBLIC_RUNTIME_SOURCE_FILES
    assert release._json_sha256(observed) == release.PUBLIC_RUNTIME_SOURCE_SHA256


def _runtime_dependencies() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": release.RUNTIME_DEPENDENCY_SCHEMA,
        "pyproject_sha256": release.PUBLIC_PYPROJECT_SHA256,
        "uv_lock_sha256": release.PUBLIC_UV_LOCK_SHA256,
        "requirements_sha256": release.PUBLIC_REQUIREMENTS_SHA256,
        "runtime_source_files": release.PUBLIC_RUNTIME_SOURCE_FILES,
        "runtime_source_sha256": release.PUBLIC_RUNTIME_SOURCE_SHA256,
        "python_version": "3.12.12",
        "python_implementation": "CPython",
        "libc": "glibc-2.36",
        "packages": release.RUNTIME_PACKAGE_VERSIONS,
    }
    return {**body, "identity_sha256": release._json_sha256(body)}


def _rehash_package(package: Path, manifest_body: dict[str, object]) -> None:
    checksums = package / "SHA256SUMS"
    inventory = release._regular_inventory(
        package,
        exclude=frozenset({"release-manifest.json", "SHA256SUMS"}),
    )
    checksums.write_text(
        "".join(f"{item['sha256']}  {name}\n" for name, item in inventory.items()),
        encoding="utf-8",
    )
    manifest_body["checksums"] = {
        "path": "SHA256SUMS",
        "sha256": release._file_sha256(checksums),
    }
    manifest_body["files"] = release._regular_inventory(
        package,
        exclude=frozenset({"release-manifest.json"}),
    )
    _write(
        package / "release-manifest.json",
        {
            **manifest_body,
            "manifest_sha256": release._json_sha256(manifest_body),
        },
    )


def _current_package(tmp_path: Path) -> tuple[Path, Path, str, str]:
    package, public_repo, artifact, calibration = _package(tmp_path)
    manifest = json.loads((package / "release-manifest.json").read_text())
    contract = manifest["contract_sha256"]
    evidence = manifest["evidence_receipts"]

    obsolete = package / evidence["training_evaluation_bridge"]["packaged_path"]
    obsolete.unlink()
    del evidence["training_evaluation_bridge"]
    shutil.rmtree(package / release.PUBLIC_RUNTIME_ROOT)
    (package / "NOTICE.inventory.json").unlink()
    manifest.pop("public_runtime")
    manifest.pop("mandatory_notices")

    # V4 binds the qualified runtime source tree directly to the publishable
    # package.  The legacy fixture used a separate one-file runtime marker, so
    # make the transformed fixture mirror the current producer contract.
    runtime_source = package / "runtime-source"
    shutil.rmtree(runtime_source)
    shutil.copytree(public_repo / "src/infercrane_commerce_1", runtime_source)

    request_body: dict[str, object] = {
        "schema_version": "infercrane-commerce-1-v3-sharded-evidence-request/v1",
        "attempt_id": release.EVALUATOR_ATTEMPT,
        "artifact_sha256": artifact,
        "calibration_file_sha256": calibration,
    }
    request = {
        **request_body,
        "request_sha256": release._json_sha256(request_body),
    }
    request_path = package / "evidence/REQUEST.json"
    _write(request_path, request)
    evidence["evaluation_request"] = {
        "packaged_path": "evidence/REQUEST.json",
        "sha256": release._file_sha256(request_path),
        "request_sha256": request["request_sha256"],
    }
    promotion_gate_path = package / "evidence/promotion_gate.json"
    _write(
        promotion_gate_path,
        {
            "schema_version": "promotion/v2",
            "status": "PASS",
            "artifact": artifact,
        },
    )
    evidence["promotion_gate"] = {
        "packaged_path": "evidence/promotion_gate.json",
        "sha256": release._file_sha256(promotion_gate_path),
        "schema_version": "promotion/v2",
        "status": "PASS",
    }

    sharded_path = package / evidence["sharded_final"]["packaged_path"]
    sharded = json.loads(sharded_path.read_text())
    sharded_body = {
        **{key: value for key, value in sharded.items() if key != "receipt_sha256"},
        "request_sha256": request["request_sha256"],
        "decision_index": 53.0,
    }
    sharded = _receipt(sharded_body)
    _write(sharded_path, sharded)
    evidence["sharded_final"].update(
        {
            "sha256": release._file_sha256(sharded_path),
            "receipt_sha256": sharded["receipt_sha256"],
        }
    )

    spend_path = package / evidence["spend_upper_bound"]["packaged_path"]
    spend = json.loads(spend_path.read_text())
    spend_body = {
        **{key: value for key, value in spend.items() if key != "receipt_sha256"},
        "request_sha256": request["request_sha256"],
    }
    spend = _receipt(spend_body)
    _write(spend_path, spend)
    evidence["spend_upper_bound"].update(
        {
            "sha256": release._file_sha256(spend_path),
            "receipt_sha256": spend["receipt_sha256"],
        }
    )
    sharded_body = {
        **{key: value for key, value in sharded.items() if key != "receipt_sha256"},
        "spend_upper_bound_receipt_sha256": spend["receipt_sha256"],
    }
    sharded = _receipt(sharded_body)
    _write(sharded_path, sharded)
    evidence["sharded_final"].update(
        {
            "sha256": release._file_sha256(sharded_path),
            "receipt_sha256": sharded["receipt_sha256"],
        }
    )

    calibration_value = json.loads(
        (package / manifest["calibration"]["path"]).read_text()
    )
    publication = {
        "artifact_sha256": artifact,
        "reproducible_score": "53.0",
        "publication_performed": False,
    }
    promoted = _receipt(
        {
            "schema_version": release.PROMOTED_FINAL_SCHEMA,
            "status": release.PROMOTED_FINAL_STATUS,
            "contract_sha256": contract,
            "artifact_sha256": artifact,
            "calibration": calibration_value,
            "training_lineage": {"legacy_serial_evaluation_adopted": False},
            "sharded_evidence": {
                "attempt_id": release.EVALUATOR_ATTEMPT,
                "request_sha256": request["request_sha256"],
                "receipt_sha256": sharded["receipt_sha256"],
                "spend_receipt_sha256": spend["receipt_sha256"],
                "decision_index": 53.0,
                "rows": 150317,
                "official_complete": True,
            },
            "publication": publication,
            "legacy_serial_evaluation_used": False,
            "publication_performed": False,
        }
    )
    final_path = package / evidence["final"]["packaged_path"]
    _write(final_path, promoted)
    evidence["final"].update(
        {
            "sha256": release._file_sha256(final_path),
            "receipt_sha256": promoted["receipt_sha256"],
        }
    )
    audit = _receipt(
        {
            "schema_version": release.PROMOTED_AUDIT_SCHEMA,
            "status": release.PROMOTED_AUDIT_STATUS,
            "contract_sha256": contract,
            "artifact_sha256": artifact,
            "promoted_final_receipt_sha256": promoted["receipt_sha256"],
            "sharded_final_receipt_sha256": sharded["receipt_sha256"],
            "spend_receipt_sha256": spend["receipt_sha256"],
            "request_sha256": request["request_sha256"],
            "decision_index": 53.0,
            "publication": publication,
            "publication_performed": False,
        },
        "audit_sha256",
    )
    audit_path = package / evidence["final_audit"]["packaged_path"]
    _write(audit_path, audit)
    evidence["final_audit"].update(
        {
            "sha256": release._file_sha256(audit_path),
            "audit_sha256": audit["audit_sha256"],
        }
    )

    runtime_identity_files: dict[str, str] = {}
    for kind in ("source", "runtime"):
        binding = manifest[f"{kind}_identity"]
        identity_path = package / binding["path"]
        tree_path = binding["packaged_path"]
        files = release._identity_files(package / tree_path, f"fixture {kind}")
        identity_body = {
            "schema_version": release.IDENTITY_SCHEMA_V2,
            "kind": kind,
            "packaged_path": tree_path,
            "files": files,
            "identity_sha256": release._json_sha256({"kind": kind, "files": files}),
        }
        _write(identity_path, identity_body)
        manifest[f"{kind}_identity"] = {
            "path": binding["path"],
            "sha256": release._file_sha256(identity_path),
            "identity_sha256": identity_body["identity_sha256"],
            "tree_path": tree_path,
        }
        if kind == "runtime":
            runtime_identity_files = dict(files)

    public_runtime = package / release.PUBLIC_RUNTIME_ROOT
    public_runtime_files: dict[str, str] = {}
    for name, digest in runtime_identity_files.items():
        relative = f"src/infercrane_commerce_1/{name}"
        target = public_runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(public_repo / relative, target)
        public_runtime_files[relative] = digest
    manifest["public_runtime"] = {
        "path": release.PUBLIC_RUNTIME_ROOT,
        "files": public_runtime_files,
        "inventory_sha256": release._json_sha256(public_runtime_files),
    }

    notice_root = package / release.CURRENT_NOTICE_ROOT
    for name in release.CURRENT_NOTICE_FILES:
        target = notice_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPOSITORY_ROOT / "release/licenses/commerce-1" / name,
            target,
        )
    manifest["mandatory_notices"] = {
        "path": release.CURRENT_NOTICE_ROOT,
        "files": release.CURRENT_NOTICE_FILES,
        "inventory_sha256": release._json_sha256(release.CURRENT_NOTICE_FILES),
        "upstream_model_license_raw_sha256": (
            "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
        ),
        "upstream_model_license_normalization": "CRLF_TO_LF_ONLY",
    }

    # V5 validates the protected raw controls above during assembly, then
    # distributes only content-free attestations.
    for relative in ("orchestration-source", "runtime-source", "identity"):
        shutil.rmtree(package / relative)
    shutil.rmtree(package / "evidence")
    source_root = package / "source/autojev"
    source_root.mkdir(parents=True)
    (source_root / "model.py").write_text("MODEL = True\n", encoding="utf-8")
    source_files = {
        name: item["sha256"]
        for name, item in release._regular_inventory(package / "source").items()
    }

    contract = release.CANONICAL_CONTRACT_SHA256
    provenance_value = json.loads((package / "PROVENANCE.json").read_text())
    sources = provenance_value["required_training_sources"]
    for index, source in enumerate(sources):
        source.update(
            {
                "repository": f"https://example.invalid/source-{index}",
                "revision": f"{index + 1:x}" * 40,
                "requested_roles": ["train", "calibration"],
            }
        )
    provenance_body = {
        key: value
        for key, value in provenance_value.items()
        if key != "provenance_sha256"
    }
    provenance_value = {
        **provenance_body,
        "provenance_sha256": release._json_sha256(provenance_body),
    }
    _write(package / "PROVENANCE.json", provenance_value)
    manifest["provenance"] = {
        "path": "PROVENANCE.json",
        "sha256": release._file_sha256(package / "PROVENANCE.json"),
        "provenance_sha256": provenance_value["provenance_sha256"],
    }

    checkpoint_data = {
        "train": "1" * 64,
        "development": "2" * 64,
        "temperature": "3" * 64,
        "git_commit": "4" * 40,
    }
    release_config = {
        "format_version": 1,
        "base_model": "Qwen/example",
        "revision": "4" * 40,
        "codes": ["A", "B"],
        "token_ids": [1, 2],
        "temperature": 1.0,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
    }
    original_config = {**release_config, "provenance": checkpoint_data}
    _write(package / "model/decision_config.json", release_config)
    release_records = release._regular_inventory(package / "model")
    release_files = {name: item["sha256"] for name, item in release_records.items()}
    release_sizes = {name: item["size_bytes"] for name, item in release_records.items()}
    release_artifact = release._json_sha256(release_files)
    original_config_bytes = (
        json.dumps(original_config, sort_keys=True) + "\n"
    ).encode()
    original_records = {name: dict(item) for name, item in release_records.items()}
    original_records["decision_config.json"] = {
        "sha256": release.hashlib.sha256(original_config_bytes).hexdigest(),
        "size_bytes": len(original_config_bytes),
    }
    evaluated_artifact = release._json_sha256(
        {name: item["sha256"] for name, item in original_records.items()}
    )
    transform_body = {
        "schema_version": release.ARTIFACT_TRANSFORM_SCHEMA,
        "status": release.ARTIFACT_TRANSFORM_STATUS,
        "decision_config_file": "decision_config.json",
        "retained_decision_config_fields": list(release.DECISION_CONFIG_FIELDS),
        "removed_decision_config_keys": ["provenance"],
        "original": {
            "artifact_sha256": evaluated_artifact,
            "decision_config_sha256": original_records["decision_config.json"][
                "sha256"
            ],
            "files": original_records,
        },
        "release": {
            "artifact_sha256": release_artifact,
            "decision_config_sha256": release_records["decision_config.json"]["sha256"],
            "files": release_records,
        },
        "unchanged_files": {
            name: item
            for name, item in release_records.items()
            if name != "decision_config.json"
        },
    }
    transform = {
        **transform_body,
        "attestation_sha256": release._json_sha256(transform_body),
    }
    transform_path = package / release.ARTIFACT_TRANSFORM_PATH
    _write(transform_path, transform)
    transform_binding = {
        "path": release.ARTIFACT_TRANSFORM_PATH,
        "sha256": release._file_sha256(transform_path),
        "transform_sha256": transform["attestation_sha256"],
        "evaluated_artifact_sha256": evaluated_artifact,
        "release_artifact_sha256": release_artifact,
    }
    checkpoint_body = {
        "schema_version": "infercrane-commerce-1-v3-native-checkpoint/v1",
        "format": "native-transformers-sharded-full-checkpoint",
        "artifact_sha256": release_artifact,
        "files": release_files,
        "sizes_bytes": release_sizes,
        "model_index": "model.safetensors.index.json",
        "model_shards": [],
        "readout": "readout.safetensors",
        "base_model": "Qwen/example",
        "base_model_revision": "4" * 40,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
    }
    checkpoint = {
        **checkpoint_body,
        "inventory_sha256": release._json_sha256(checkpoint_body),
    }
    lineage_sources = [
        {
            "source_id": source["source_id"],
            "repository": source["repository"],
            "revision": source["revision"],
            "data_sha256": f"{index + 7:x}" * 64,
            "license": source["license_declaration"],
            "license_evidence_sha256": source["license_evidence_sha256"],
            "approved_roles": source["requested_roles"],
            "commercial_use_permitted": True,
            "redistribution_permitted": True,
            "review_file_sha256": f"{index + 9:x}" * 64,
            "review_receipt_sha256": f"{index + 10:x}" * 64,
        }
        for index, source in enumerate(sources)
    ]
    lineage_body = {
        "schema_version": release.TRAINING_DATA_LINEAGE_SCHEMA,
        "status": "VERIFIED_PROTECTED_INPUTS_NOT_DISTRIBUTED",
        "model_id": release.MODEL_ID,
        "contract_sha256": contract,
        "protected_inputs": {
            "prepared_manifest": {
                "file_sha256": "5" * 64,
                "manifest_sha256": "6" * 64,
            },
            "merge_input_bundle": {
                "file_sha256": "7" * 64,
                "bundle_sha256": "8" * 64,
            },
        },
        "checkpoint_data_provenance": checkpoint_data,
        "protected_owner_authority_sha256": (release.OWNER_AUTHORITY_STATEMENTS_SHA256),
        "protected_authorization_records": {
            "authorization_record_sha256": "c" * 64,
            "approval_record_sha256": "d" * 64,
            "record_set_sha256": release._json_sha256(
                {
                    "authorization_record_sha256": "c" * 64,
                    "approval_record_sha256": "d" * 64,
                }
            ),
        },
        "sources": lineage_sources,
    }
    lineage = {
        **lineage_body,
        "lineage_sha256": release._json_sha256(lineage_body),
    }
    _write(package / release.TRAINING_DATA_LINEAGE_PATH, lineage)

    evaluation_body = {
        "schema_version": release.PUBLIC_EVALUATION_SCHEMA,
        "status": "PASS_COMPLETE_LOCAL_REPRODUCTION",
        "model_id": release.MODEL_ID,
        "edition": "0.2.1",
        "scope": "complete_public_suite",
        "evaluation_authority": "self_hosted",
        "contract_sha256": contract,
        "evaluated_artifact_sha256": evaluated_artifact,
        "release_artifact_sha256": release_artifact,
        "artifact_transform_sha256": transform["attestation_sha256"],
        "calibration_file_sha256": calibration,
        "suite_sha256": "9" * 64,
        "selected_in_edition": 120_340,
        "excluded_selected": 442,
        "selected_scoreable": 119_898,
        "added": 30_419,
        "merged": 150_317,
        "counts": {"ok": 150_317},
        "results_sha256": "a" * 64,
        "scores_sha256": "b" * 64,
        "decision_index": 53.0,
        "reproduction_complete": True,
    }
    evaluation = {
        **evaluation_body,
        "attestation_sha256": release._json_sha256(evaluation_body),
    }
    _write(package / release.PUBLIC_EVALUATION_PATH, evaluation)
    evaluation_file_sha = release._file_sha256(package / release.PUBLIC_EVALUATION_PATH)

    runtime_body = {
        "schema_version": "infercrane-commerce-1-native-autojev-runtime/v1",
        "model": "infercrane/commerce-1",
        "backend": "autojev-native-v1",
        "checkpoint": {
            "path": "model",
            "artifact_sha256": release_artifact,
            "files": release_files,
        },
        "artifact_transform": transform_binding,
        "calibration": {
            "path": "calibration/calibration.json",
            "file_sha256": calibration,
            "calibration_sha256": "f" * 64,
        },
        "source": {
            "path": "source",
            "files": source_files,
            "files_sha256": release._json_sha256(source_files),
        },
        "evidence": {
            "path": release.PUBLIC_EVALUATION_PATH,
            "sha256": evaluation_file_sha,
            "attestation_sha256": evaluation["attestation_sha256"],
        },
    }
    runtime = {
        **runtime_body,
        "runtime_identity_sha256": release._json_sha256(runtime_body),
    }
    runtime_path = package / "runtime.json"
    _write(runtime_path, runtime)

    for field in (
        "evaluator_attempt_id",
        "evidence_receipts",
        "source_identity",
        "runtime_identity",
    ):
        manifest.pop(field, None)
    manifest.update(
        {
            "schema_version": release.CURRENT_PACKAGE_SCHEMA,
            "contract_sha256": contract,
            "artifact_sha256": release_artifact,
            "evaluated_artifact_sha256": evaluated_artifact,
            "artifact_transform": transform_binding,
            "checkpoint": checkpoint,
            "autojev_source": {
                "path": "source",
                "files": source_files,
                "files_sha256": release._json_sha256(source_files),
            },
            "training_data_lineage": lineage,
            "public_evaluation": {
                "path": release.PUBLIC_EVALUATION_PATH,
                "sha256": evaluation_file_sha,
                "attestation_sha256": evaluation["attestation_sha256"],
            },
            "native_runtime": {
                "manifest_path": "runtime.json",
                "manifest_sha256": release._file_sha256(runtime_path),
                "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                "evidence_role": "public_evaluation",
                "evidence_path": release.PUBLIC_EVALUATION_PATH,
            },
        }
    )
    manifest.pop("manifest_sha256", None)
    manifest.pop("checksums", None)
    manifest.pop("files", None)
    _rehash_package(package, manifest)
    return package, public_repo, release_artifact, calibration


def _current_official(artifact: str, public_score: float) -> dict[str, object]:
    board_revision = "5" * 40
    body: dict[str, object] = {
        "schema_version": release.CURRENT_OFFICIAL_V03_SCHEMA,
        "status": "OFFICIAL_BOARD_V0_3_EVIDENCE",
        "edition": "0.3",
        "panel_id": "decision-index-0.3",
        "artifact_sha256": artifact,
        "candidate_repository_id": release.HF_REPOSITORY,
        "candidate_revision": "3" * 40,
        "official_repository_id": release.OFFICIAL_REPOSITORY,
        "official_submission_pr_url": "https://github.com/apolinario/decision-index/pull/1",
        "official_submission_pr_head_revision": "4" * 40,
        "official_submission_pr_merged": True,
        "official_board_space_id": release.OFFICIAL_BOARD,
        "official_board_space_revision": board_revision,
        "official_board_index_url": (
            f"https://huggingface.co/spaces/{release.OFFICIAL_BOARD}/resolve/"
            f"{board_revision}/data/index.json"
        ),
        "official_board_v03_url": (
            f"https://huggingface.co/spaces/{release.OFFICIAL_BOARD}/resolve/"
            f"{board_revision}/data/v03.json"
        ),
        "official_board_index_sha256": "6" * 64,
        "official_board_v03_sha256": "7" * 64,
        "official_board_entry_sha256": "8" * 64,
        "results_dataset_receipt_sha256": "9" * 64,
        "public_complete": True,
        "private_same_skills_complete": True,
        "private_new_domains_complete": True,
        "decision_index": 64.68,
        "public_score_raw": public_score,
        "private_same_skills_score_raw": 65.0,
        "private_new_domains_score_raw": 65.0,
        "private_same_skills_score_equated": 65.0,
        "private_new_domains_score_equated": 65.0,
        "equating_provenance": release.AUDITED_V03_EQUATING,
        "weights": release.CURRENT_V03_WEIGHTS,
        "capacity": {
            "complete": True,
            "total": 100,
            "answered": 100,
            "errors": 0,
            "unsupported": 0,
            "coverage": 1.0,
            "unsupported_counted_wrong": True,
            "declared_limits": [],
        },
        "latency_hardware": "NVIDIA RTX PRO 6000",
        "latency_median_ms": 100.0,
        "latency_mean_ms": 110.0,
        "latency_p80_ms": 120.0,
        "latency_sample_count": 100,
        "observed_at_utc": "2026-10-07T00:00:00+00:00",
    }
    return _receipt(body, "attestation_sha256")


def _upgrade_qualification(path: Path, validated: dict[str, object]) -> None:
    current = json.loads(path.read_text())
    body = {key: value for key, value in current.items() if key != "receipt_sha256"}
    body.pop("artifact_sha256")
    body.update(
        {
            "schema_version": release.CURRENT_QUALIFICATION_SCHEMA,
            "evaluated_artifact_sha256": validated["evaluated_artifact_sha256"],
            "release_artifact_sha256": validated["release_artifact_sha256"],
            "artifact_transform_sha256": validated["artifact_transform_sha256"],
            "sharded_spend_receipt_sha256": "e" * 64,
            "maximum_input_tokens": 65_536,
            "question_microbatch_size": 1,
            "artifact_transform_parity": {
                "status": "PASS_EVALUATED_RELEASE_PARITY",
                "absolute_tolerance": 1e-7,
                "maximum_absolute_difference": 0.0,
                "comparisons": 3,
                "evaluated_outputs_sha256": "c" * 64,
                "release_outputs_sha256": "d" * 64,
            },
            "runtime_dependencies": _runtime_dependencies(),
        }
    )
    body["runtime_identity"]["revision"] = validated["release_artifact_sha256"]
    body["runtime_identity"]["evidence_sha256"] = validated[
        "public_evaluation_file_sha256"
    ]
    body["runtime_identity"]["runtime_identity_sha256"] = validated[
        "runtime_identity_sha256"
    ]
    _write(path, _receipt(body))


def _public_qualification(validated: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": release.PUBLIC_QUALIFICATION_SCHEMA,
        "status": "PASS_NATIVE_RUNTIME_QUALIFICATION",
        "model_id": release.MODEL_ID,
        "contract_sha256": validated["contract_sha256"],
        "evaluated_artifact_sha256": validated["evaluated_artifact_sha256"],
        "release_artifact_sha256": validated["release_artifact_sha256"],
        "artifact_transform_sha256": validated["artifact_transform_sha256"],
        "calibration_file_sha256": validated["calibration_file_sha256"],
        "release_manifest_sha256": validated["manifest_sha256"],
        "package_files_sha256": validated["inventory_sha256"],
        "runtime_manifest_sha256": validated["runtime_manifest_sha256"],
        "runtime_identity_sha256": validated["runtime_identity_sha256"],
        "public_evaluation_file_sha256": validated["public_evaluation_file_sha256"],
        "public_evaluation_attestation_sha256": validated["public_evaluation"][
            "attestation_sha256"
        ],
        "maximum_input_tokens": 65_536,
        "question_microbatch_size": 1,
        "fixture_manifest_sha256": "1" * 64,
        "fixture_count": 3,
        "question_types": ["choice", "noul", "score"],
        "parity": {
            "status": "PASS_NATIVE_DIRECT_AUTOJEV_PARITY",
            "absolute_tolerance": 1e-7,
            "maximum_absolute_difference": 0.0,
            "comparisons": 3,
            "native_outputs_sha256": "2" * 64,
            "direct_outputs_sha256": "3" * 64,
        },
        "artifact_transform_parity": {
            "status": "PASS_EVALUATED_RELEASE_PARITY",
            "absolute_tolerance": 1e-7,
            "maximum_absolute_difference": 0.0,
            "comparisons": 3,
            "evaluated_outputs_sha256": "4" * 64,
            "release_outputs_sha256": "5" * 64,
        },
        "runtime_dependencies": _runtime_dependencies(),
        "performance": {
            "content_free": True,
            "fixture_manifest_sha256": "1" * 64,
            "warmup_iterations": 1,
            "measured_iterations": 3,
            "decisions_per_iteration": 3,
            "sample_count": 3,
            "latency_unit": "milliseconds_per_complete_fixture_set",
            "latency_mean_ms": 10.0,
            "latency_p50_ms": 9.0,
            "latency_p95_ms": 12.0,
            "latency_max_ms": 13.0,
            "throughput_decisions_per_second": 100.0,
        },
        "hardware": {
            "accelerator_count": 1,
            "accelerator": "NVIDIA H200",
            "torch": "2.14.0",
            "cuda": "13.0",
            "precision": "bfloat16",
        },
        "offline": True,
        "network_required": False,
        "publication_performed": False,
        "benchmark_evidence_mutated": False,
        "qualified_at_utc": "2026-10-07T00:00:00+00:00",
        "protected_qualification_receipt_sha256": "6" * 64,
    }
    return _receipt(body, "attestation_sha256")


def test_current_v5_handoff_validates_end_to_end(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _current_package(tmp_path)
    validated = release._validate_package(package, public_repo)
    assert validated["artifact_sha256"] == artifact
    assert validated["public_evaluation"]["decision_index"] == 53.0
    assert validated["training_data_lineage"]["sources"]
    assert "evidence_receipts" not in validated["manifest"]

    qualification, public, legacy_official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    legacy_official.unlink()
    _upgrade_qualification(qualification, validated)

    official = tmp_path / "official-current-v03.json"
    _write(
        official,
        _current_official(validated["evaluated_artifact_sha256"], 63.4),
    )
    materials = release._validated_release_materials(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public,
        official_v03=official,
    )
    assert materials[0]["manifest"]["schema_version"] == release.CURRENT_PACKAGE_SCHEMA


def test_current_v5_accepts_only_bound_public_qualification_v2(
    tmp_path: Path,
) -> None:
    package, public_repo, _artifact, _calibration = _current_package(tmp_path)
    validated = release._validate_package(package, public_repo)
    projection = _public_qualification(validated)
    assert (
        release._validate_qualification(projection, validated)
        == projection["attestation_sha256"]
    )

    detached = json.loads(json.dumps(projection))
    detached["release_artifact_sha256"] = "f" * 64
    body = {
        key: value for key, value in detached.items() if key != "attestation_sha256"
    }
    detached["attestation_sha256"] = release._json_sha256(body)
    with pytest.raises(release.V3PublicationError, match="detached"):
        release._validate_qualification(detached, validated)

    changed_runtime = json.loads(json.dumps(projection))
    changed_runtime["runtime_dependencies"]["packages"]["torch"] = "2.13.0"
    dependency_body = {
        key: value
        for key, value in changed_runtime["runtime_dependencies"].items()
        if key != "identity_sha256"
    }
    changed_runtime["runtime_dependencies"]["identity_sha256"] = release._json_sha256(
        dependency_body
    )
    body = {
        key: value
        for key, value in changed_runtime.items()
        if key != "attestation_sha256"
    }
    changed_runtime["attestation_sha256"] = release._json_sha256(body)
    with pytest.raises(
        release.V3PublicationError, match="runtime dependency identity changed"
    ):
        release._validate_qualification(changed_runtime, validated)


def test_current_v4_qualification_requires_transform_parity(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _current_package(tmp_path)
    qualification, _public, _official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    validated = release._validate_package(package, public_repo)
    _upgrade_qualification(qualification, validated)
    value = json.loads(qualification.read_text())
    value.pop("artifact_transform_parity")
    body = {key: item for key, item in value.items() if key != "receipt_sha256"}
    value["receipt_sha256"] = release._json_sha256(body)
    with pytest.raises(release.V3PublicationError, match="fields changed"):
        release._validate_qualification(value, validated)


def test_canonical_score_is_keyed_to_evaluated_not_release_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, public_repo, _artifact, calibration = _current_package(tmp_path)
    manifest = json.loads((package / "release-manifest.json").read_text())
    assert manifest["evaluated_artifact_sha256"] != manifest["artifact_sha256"]
    evaluation_path = package / release.PUBLIC_EVALUATION_PATH
    evaluation = json.loads(evaluation_path.read_text())
    evaluation_body = {
        key: value for key, value in evaluation.items() if key != "attestation_sha256"
    }
    evaluation_body["decision_index"] = 54.0
    evaluation = {
        **evaluation_body,
        "attestation_sha256": release._json_sha256(evaluation_body),
    }
    _write(evaluation_path, evaluation)

    runtime_path = package / "runtime.json"
    runtime = json.loads(runtime_path.read_text())
    runtime_body = {
        key: value for key, value in runtime.items() if key != "runtime_identity_sha256"
    }
    runtime_body["evidence"] = {
        "path": release.PUBLIC_EVALUATION_PATH,
        "sha256": release._file_sha256(evaluation_path),
        "attestation_sha256": evaluation["attestation_sha256"],
    }
    runtime = {
        **runtime_body,
        "runtime_identity_sha256": release._json_sha256(runtime_body),
    }
    _write(runtime_path, runtime)
    manifest["public_evaluation"] = {
        "path": release.PUBLIC_EVALUATION_PATH,
        "sha256": release._file_sha256(evaluation_path),
        "attestation_sha256": evaluation["attestation_sha256"],
    }
    manifest["native_runtime"]["manifest_sha256"] = release._file_sha256(runtime_path)
    manifest["native_runtime"]["runtime_identity_sha256"] = runtime[
        "runtime_identity_sha256"
    ]
    manifest.pop("manifest_sha256")
    manifest.pop("checksums")
    manifest.pop("files")
    _rehash_package(package, manifest)

    monkeypatch.setattr(
        release,
        "CANONICAL_EVALUATED_ARTIFACT_SHA256",
        manifest["evaluated_artifact_sha256"],
    )
    monkeypatch.setattr(release, "CANONICAL_CALIBRATION_FILE_SHA256", calibration)
    monkeypatch.setattr(release, "CANONICAL_V021_RESULTS_SHA256", "a" * 64)
    monkeypatch.setattr(release, "CANONICAL_V021_SCORES_SHA256", "b" * 64)
    with pytest.raises(
        release.V3PublicationError,
        match="canonical Commerce-1 public evaluation changed",
    ):
        release._validate_package(package, public_repo)


def test_current_official_score_must_strictly_beat_baseline(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _current_package(tmp_path)
    qualification, public, _legacy_official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    validated = release._validate_package(package, public_repo)
    _upgrade_qualification(qualification, validated)

    public_value = json.loads(public.read_text())
    public_body = {
        key: value for key, value in public_value.items() if key != "receipt_sha256"
    }
    public_body["public_decision_index"] = release.RELEASE_BASELINE_DECISION_INDEX
    _write(public, _receipt(public_body))
    official = tmp_path / "official-at-baseline.json"
    official_value = _current_official(
        validated["evaluated_artifact_sha256"],
        release.RELEASE_BASELINE_DECISION_INDEX,
    )
    official_body = {
        key: value
        for key, value in official_value.items()
        if key != "attestation_sha256"
    }
    baseline = release.RELEASE_BASELINE_DECISION_INDEX
    official_body.update(
        {
            "decision_index": baseline,
            "private_same_skills_score_raw": baseline,
            "private_new_domains_score_raw": baseline,
            "private_same_skills_score_equated": baseline,
            "private_new_domains_score_equated": baseline,
        }
    )
    _write(official, _receipt(official_body, "attestation_sha256"))

    try:
        release._validated_release_materials(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
        )
    except release.V3PublicationError as error:
        assert "does not beat the 52.55 release baseline" in str(error)
    else:
        raise AssertionError("a score equal to the baseline must fail closed")


def test_current_official_evidence_cannot_substitute_another_repository(
    tmp_path: Path,
) -> None:
    package, public_repo, _artifact, _calibration = _current_package(tmp_path)
    validated = release._validate_package(package, public_repo)
    official = _current_official(validated["evaluated_artifact_sha256"], 63.4)
    body = {
        key: value for key, value in official.items() if key != "attestation_sha256"
    }
    body["candidate_repository_id"] = "infercrane/not-commerce-1-v3"
    substituted = _receipt(body, "attestation_sha256")

    with pytest.raises(release.V3PublicationError, match="incomplete or detached"):
        release._validate_official_v03(substituted, validated)
