from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import v3_publication as release


def _receipt(
    body: dict[str, object], field: str = "receipt_sha256"
) -> dict[str, object]:
    return {**body, field: release._json_sha256(body)}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _package(tmp_path: Path) -> tuple[Path, Path, str, str]:
    package = tmp_path / "package"
    public_repo = tmp_path / "public"
    artifact = "a" * 64
    contract = "b" * 64
    calibration_bytes = b'{"calibration":"bound"}\n'
    calibration_sha = release.hashlib.sha256(calibration_bytes).hexdigest()
    public_name = "src/infercrane_commerce_1/runtime.py"
    public_bytes = b"MODEL_ID = 'infercrane/commerce-1'\n"
    public_digest = release.hashlib.sha256(public_bytes).hexdigest()
    (public_repo / public_name).parent.mkdir(parents=True)
    (public_repo / public_name).write_bytes(public_bytes)

    final = _receipt(
        {
            "schema_version": "infercrane-commerce-1-v3-training-evaluation/v3",
            "status": "SUCCEEDED",
            "contract_sha256": contract,
            "artifact_sha256": artifact,
            "calibration": {"sha256": calibration_sha},
        }
    )
    audit = _receipt(
        {
            "schema_version": "infercrane-commerce-1-v3-final-audit/v1",
            "status": "PASS_LOCAL_EVIDENCE_PENDING_OFFICIAL_V0_3",
            "contract_sha256": contract,
            "artifact_sha256": artifact,
            "calibration_sha256": "c" * 64,
            "final_receipt_sha256": final["receipt_sha256"],
        },
        "audit_sha256",
    )
    spend = _receipt(
        {
            "schema_version": "infercrane-commerce-1-v3-spend-upper-bound/v1",
            "request_sha256": "d" * 64,
            "within_campaign_hard_cap": True,
        }
    )
    sharded = _receipt(
        {
            "schema_version": "infercrane-commerce-1-v3-sharded-evidence/v1",
            "status": "PASS_COMPLETE_OFFICIAL_0_2_1",
            "contract_sha256": contract,
            "artifact_sha256": artifact,
            "calibration_file_sha256": calibration_sha,
            "attempt_id": release.EVALUATOR_ATTEMPT,
            "request_sha256": "d" * 64,
            "spend_upper_bound_receipt_sha256": spend["receipt_sha256"],
            "official_complete": True,
            "selected_in_edition": 120340,
            "excluded_selected": 442,
            "selected_scoreable": 119898,
            "added": 30419,
            "merged": 150317,
            "decision_index": 53.0,
        }
    )
    bridge = _receipt(
        {
            "schema_version": release.TRAINING_EVALUATION_BRIDGE_SCHEMA,
            "status": "PASS_TRAINING_TO_EXACT_EVALUATION_LINEAGE",
            "artifact_sha256": artifact,
            "contract_sha256": contract,
            "calibration_file_sha256": calibration_sha,
            "training_final_receipt_sha256": final["receipt_sha256"],
            "final_audit_sha256": audit["audit_sha256"],
            "sharded_final_receipt_sha256": sharded["receipt_sha256"],
        }
    )
    evidence_values = {
        "final": final,
        "final_audit": audit,
        "sharded_final": sharded,
        "spend_upper_bound": spend,
        "training_evaluation_bridge": bridge,
    }
    evidence: dict[str, dict[str, object]] = {}
    for role, value in evidence_values.items():
        relative = f"evidence/{role}.json"
        path = package / relative
        _write(path, value)
        evidence[role] = {
            "packaged_path": relative,
            "sha256": release._file_sha256(path),
        }
        if role == "final_audit":
            evidence[role]["audit_sha256"] = value["audit_sha256"]
        else:
            evidence[role]["receipt_sha256"] = value["receipt_sha256"]

    source_file = package / "orchestration-source/runner.py"
    runtime_file = package / "runtime-source/runtime.py"
    source_file.parent.mkdir(parents=True)
    runtime_file.parent.mkdir(parents=True)
    source_file.write_text("SOURCE = True\n", encoding="utf-8")
    runtime_file.write_text("RUNTIME = True\n", encoding="utf-8")
    identities: dict[str, dict[str, object]] = {}
    for kind, relative_root, file in (
        ("source", "orchestration-source", source_file),
        ("runtime", "runtime-source", runtime_file),
    ):
        name = file.name
        identity = {
            "schema_version": "infercrane-commerce-1-v3-source-runtime-identity/v1",
            "kind": kind,
            "files": {name: release._file_sha256(file)},
        }
        identity_path = package / f"identity/{kind}.json"
        _write(identity_path, identity)
        identities[kind] = {
            "path": f"identity/{kind}.json",
            "sha256": release._file_sha256(identity_path),
            "identity_sha256": "e" * 64,
            "packaged_path": relative_root,
        }

    (package / public_name).parent.mkdir(parents=True, exist_ok=True)
    public_packaged = package / release.PUBLIC_RUNTIME_ROOT / public_name
    public_packaged.parent.mkdir(parents=True)
    public_packaged.write_bytes(public_bytes)
    calibration = package / "calibration/calibration.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_bytes(calibration_bytes)
    runtime = package / "runtime.json"
    _write(runtime, {"runtime_identity_sha256": "f" * 64})
    provenance = package / "PROVENANCE.json"
    training_sources = [
        {
            "source_id": f"source-{index}",
            "license_declaration": "Apache-2.0",
            "license_evidence_sha256": f"{index + 1:x}" * 64,
            "notice_requirement": f"Retain source {index} notice.",
        }
        for index in range(6)
    ]
    model_lineage = {
        "base": {
            "model_id": "Qwen/example",
            "revision": "a" * 40,
            "license_declaration": "Apache-2.0",
            "notice_requirement": "Retain base notice.",
        },
        "initialization": {
            "model_id": "infercrane/example",
            "revision": "b" * 40,
            "license_declaration": "Apache-2.0",
            "notice_requirement": "Retain initialization notice.",
        },
    }
    provenance_body = {
        "schema_version": "infercrane-commerce-1-v3-provenance-notices/v1",
        "model_id": release.MODEL_ID,
        "legal_conclusion": False,
        "publication_approval": False,
        "required_source_count": 6,
        "required_training_sources": training_sources,
        "model_lineage": model_lineage,
    }
    _write(
        provenance,
        {
            **provenance_body,
            "provenance_sha256": release._json_sha256(provenance_body),
        },
    )
    notices = [
        {
            "subject_type": "training_source",
            "subject_id": item["source_id"],
            "license_declaration": item["license_declaration"],
            "license_evidence_sha256": item["license_evidence_sha256"],
            "notice_requirement": item["notice_requirement"],
        }
        for item in training_sources
    ]
    notices.extend(
        {
            "subject_type": "model_lineage",
            "subject_id": role,
            "model_id": model_lineage[role]["model_id"],
            "revision": model_lineage[role]["revision"],
            "license_declaration": model_lineage[role]["license_declaration"],
            "notice_requirement": model_lineage[role]["notice_requirement"],
        }
        for role in ("base", "initialization")
    )
    notices.sort(key=lambda item: (item["subject_type"], item["subject_id"]))
    notice_body = {
        "schema_version": "infercrane-commerce-1-v3-mandatory-notices/v1",
        "model_id": release.MODEL_ID,
        "notices": notices,
    }
    notice_value = _receipt(notice_body)
    notice_path = package / "NOTICE.inventory.json"
    _write(notice_path, notice_value)
    (package / "LICENSE").write_text("Apache-2.0\n", encoding="utf-8")
    (package / "THIRD_PARTY.md").write_text("Required notices.\n", encoding="utf-8")
    (package / "README.md").write_text(
        "---\ninference: false\n---\n"
        f"# InferCrane Commerce-1\n\nCanonical model: `{release.MODEL_ID}`.\n",
        encoding="utf-8",
    )
    (package / "model").mkdir()
    (package / "model/model.safetensors").write_bytes(b"weights")
    (package / "SHA256SUMS").write_bytes(b"")

    inventory_without_checksums = release._regular_inventory(
        package, exclude=frozenset({"release-manifest.json", "SHA256SUMS"})
    )
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{item['sha256']}  {name}\n"
            for name, item in inventory_without_checksums.items()
        ),
        encoding="utf-8",
    )
    inventory = release._regular_inventory(
        package, exclude=frozenset({"release-manifest.json"})
    )
    public_files = {public_name: public_digest}
    manifest_body = {
        "schema_version": release.PACKAGE_SCHEMA,
        "status": release.PACKAGE_STATUS,
        "model_id": release.MODEL_ID,
        "contract_sha256": contract,
        "artifact_sha256": artifact,
        "upload_performed": False,
        "publication_performed": False,
        "files": inventory,
        "evidence_receipts": evidence,
        "source_identity": identities["source"],
        "runtime_identity": identities["runtime"],
        "public_runtime": {
            "path": release.PUBLIC_RUNTIME_ROOT,
            "files": public_files,
            "inventory_sha256": release._json_sha256(public_files),
        },
        "provenance": {
            "path": "PROVENANCE.json",
            "sha256": release._file_sha256(provenance),
        },
        "mandatory_notices": {
            "path": "NOTICE.inventory.json",
            "sha256": release._file_sha256(notice_path),
            "receipt_sha256": notice_value["receipt_sha256"],
        },
        "licenses": {
            "project": {
                "path": "LICENSE",
                "sha256": release._file_sha256(package / "LICENSE"),
            },
            "third_party": {
                "path": "THIRD_PARTY.md",
                "sha256": release._file_sha256(package / "THIRD_PARTY.md"),
            },
        },
        "calibration": {
            "path": "calibration/calibration.json",
            "sha256": calibration_sha,
        },
        "native_runtime": {
            "manifest_path": "runtime.json",
            "manifest_sha256": release._file_sha256(runtime),
            "runtime_identity_sha256": "f" * 64,
        },
    }
    _write(
        package / "release-manifest.json",
        {**manifest_body, "manifest_sha256": release._json_sha256(manifest_body)},
    )
    return package, public_repo, artifact, calibration_sha


def _evidence_files(
    tmp_path: Path, package: Path, artifact: str, calibration: str
) -> tuple[Path, Path, Path]:
    manifest = json.loads((package / "release-manifest.json").read_text())
    evaluated_artifact = manifest.get("evaluated_artifact_sha256", artifact)
    evidence = manifest.get("evidence_receipts")
    if isinstance(evidence, dict):
        source_lineage_sha256 = evidence["sharded_final"]["receipt_sha256"]
        protected_spend_sha256 = evidence["spend_upper_bound"]["receipt_sha256"]
        runtime_evidence_sha256 = "3" * 64
    else:
        source_lineage_sha256 = manifest["public_evaluation"]["attestation_sha256"]
        protected_spend_sha256 = "e" * 64
        runtime_evidence_sha256 = manifest["public_evaluation"]["sha256"]
    qualification_body: dict[str, object] = {
        "schema_version": release.QUALIFICATION_SCHEMA,
        "status": "PASS_NATIVE_RUNTIME_QUALIFICATION",
        "attempt_id": "1" * 32,
        "request_sha256": "2" * 64,
        "contract_sha256": manifest["contract_sha256"],
        "artifact_sha256": evaluated_artifact,
        "calibration_file_sha256": calibration,
        "evaluator_attempt_id": release.EVALUATOR_ATTEMPT,
        "evaluation_budget_authorization_sha256": release.EVALUATION_AUTHORIZATION,
        "exact_evaluation_reservation_upper_bound_usd": release.EVALUATION_RESERVATION,
        "evaluation_campaign_hard_cap_usd": release.EVALUATION_CAP,
        "release_manifest_sha256": manifest["manifest_sha256"],
        "sharded_final_receipt_sha256": source_lineage_sha256,
        "spend_receipt_sha256": protected_spend_sha256,
        "runtime_identity": {
            "model": "infercrane/commerce-1",
            "revision": artifact,
            "runtime_identity_sha256": "f" * 64,
            "evidence_sha256": runtime_evidence_sha256,
            "backend": "autojev-native-v1",
        },
        "fixture_manifest_sha256": "4" * 64,
        "fixture_count": 3,
        "question_types": ["choice", "noul", "score"],
        "parity": {
            "status": "PASS_NATIVE_DIRECT_AUTOJEV_PARITY",
            "absolute_tolerance": 1e-7,
            "maximum_absolute_difference": 0.0,
            "comparisons": 3,
            "native_outputs_sha256": "a" * 64,
            "direct_outputs_sha256": "b" * 64,
        },
        "performance": {
            "content_free": True,
            "sample_count": 12,
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
            "precision": "bfloat16",
        },
        "offline": True,
        "network_required": False,
        "publication_performed": False,
        "benchmark_evidence_mutated": False,
        "qualified_at_utc": "2026-10-07T00:00:00+00:00",
    }
    qualification = tmp_path / "qualification.json"
    _write(qualification, _receipt(qualification_body))

    merge = {
        "source_v021_rows": 150317,
        "source_v021_reused_rows": 137540,
        "new_rebuilt_gsm8k_rows": 2638,
        "merged_rows": 140178,
        "counts": {"ok": 140178},
        "results_sha256": "5" * 64,
    }
    public_files = {
        "results.jsonl": "5" * 64,
        "environment.json": "6" * 64,
        "status.json": "7" * 64,
        "benchmark-summary.json": "8" * 64,
        "index.json": "9" * 64,
        "scores.json": "a" * 64,
    }
    public_body: dict[str, object] = {
        "schema_version": release.PUBLIC_V03_SCHEMA,
        "status": "PASS_COMPLETE_PUBLIC_DECISION_INDEX_0_3",
        "attempt_id": "b" * 32,
        "request_sha256": "c" * 64,
        "contract_sha256": manifest["contract_sha256"],
        "source_edition": "0.2.1",
        "target_edition": "0.3",
        "kit_revision": release.KIT_REVISION,
        "artifact_sha256": evaluated_artifact,
        "calibration_file_sha256": calibration,
        "runtime_sha256": "d" * 64,
        "source_lineage_receipt_sha256": source_lineage_sha256,
        "spend_receipt_sha256": "e" * 64,
        "inventory_sha256": "f" * 64,
        "context_receipt_sha256": "1" * 64,
        "dispatch_sha256": "2" * 64,
        "worker_receipt_sha256": "3" * 64,
        "source_was_relabelled": False,
        "reused_v021": 137540,
        "new_rebuilt_gsm8k": 2638,
        "merged": 140178,
        "merge": merge,
        "official_public_complete": True,
        "public_decision_index": 63.4,
        "official_full_score_claimed": False,
        "public_run_files": public_files,
    }
    public = tmp_path / "public-v03.json"
    _write(public, _receipt(public_body))

    official_body: dict[str, object] = {
        "schema_version": release.OFFICIAL_V03_SCHEMA,
        "status": "OFFICIAL_MAINTAINER_V0_3_ATTESTED",
        "edition": "0.3",
        "panel_id": "decision-index-0.3",
        "artifact_sha256": evaluated_artifact,
        "official_repository_id": release.OFFICIAL_REPOSITORY,
        "official_submission_pr_url": "https://github.com/apolinario/decision-index/pull/1",
        "official_submission_pr_revision": "4" * 40,
        "official_board_space_id": release.OFFICIAL_BOARD,
        "official_board_space_revision": "5" * 40,
        "maintainer_operator_identity_sha256": "6" * 64,
        "self_attestation": False,
        "public_results_dataset_sha256": "7" * 64,
        "maintainer_rescore_sha256": "8" * 64,
        "public_complete": True,
        "private_same_skills_complete": True,
        "private_new_domains_complete": True,
        "decision_index": 64.68,
        "public_score": 63.4,
        "private_same_skills_score": 65.0,
        "private_new_domains_score": 65.0,
        "weights": {
            "public": "0.20",
            "private_same_skills": "0.50",
            "private_new_domains": "0.30",
        },
        "coverage": 1,
        "errors": 0,
        "unsupported": 0,
        "latency_hardware": "NVIDIA RTX PRO 6000",
        "latency_median_ms": 100.0,
        "latency_mean_ms": 110.0,
        "latency_p80_ms": 120.0,
        "latency_sample_count": 100,
        "held_out_answer_agreement_sha256": "9" * 64,
        "evaluated_at_utc": "2026-10-07T00:00:00+00:00",
    }
    official = tmp_path / "official-v03.json"
    _write(official, _receipt(official_body, "attestation_sha256"))
    return qualification, public, official


def _authorization_file(
    tmp_path: Path,
    *,
    package: Path,
    public_repo: Path,
    qualification: Path,
    public: Path,
    official: Path,
) -> Path:
    value = release.build_owner_authorization(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public,
        official_v03=official,
        confirmation=release.OWNER_RELEASE_CONFIRMATION,
        authorized_at_utc="2026-10-07T00:00:00+00:00",
    )
    path = tmp_path / "owner-authorization.json"
    _write(path, value)
    return path


def test_complete_gate_and_private_first_sequence(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    authorization = _authorization_file(
        tmp_path,
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public=public,
        official=official,
    )
    gate = release.build_evidence_gate(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public,
        official_v03=official,
        authorization=authorization,
    )
    assert gate["owner"] == "infercrane"
    plan = release.build_private_plan(gate, source_revision="a" * 40)
    assert plan["github"] == {
        "repository": release.GITHUB_REPOSITORY,
        "visibility": "private",
    }
    assert plan["huggingface"] == {
        "repository": release.HF_REPOSITORY,
        "visibility": "private",
    }
    assert plan["publication_performed"] is False

    def remote(service: str, repository: str, revision: str) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": release.REMOTE_RECEIPT_SCHEMA,
            "status": "IMMUTABLE_PRIVATE_REVISION_VERIFIED",
            "service": service,
            "repository": repository,
            "visibility": "private",
            "revision": revision,
            "source_revision": "a" * 40,
            "artifact_sha256": artifact,
            "inventory_sha256": (
                plan["package_inventory_sha256"]
                if service == "huggingface"
                else "b" * 64
            ),
            "verified": True,
            "verified_at_utc": "2026-10-07T00:00:00+00:00",
        }
        if service == "huggingface":
            body["hub_metadata"] = {".gitattributes": "c" * 64}
        return _receipt(body)

    private = release.build_private_verification(
        plan,
        remote("github", release.GITHUB_REPOSITORY, "a" * 40),
        remote("huggingface", release.HF_REPOSITORY, "d" * 40),
    )
    public_plan = release.build_public_plan(gate, private)
    assert public_plan["status"] == "PUBLIC_TRANSITION_AUTHORIZED_NOT_PERFORMED"
    assert public_plan["repositories_created"] is False
    assert public_plan["publication_performed"] is False


def test_initial_publication_does_not_require_circular_v03_evidence(
    tmp_path: Path,
) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, _public, _official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    authorization_value = release.build_owner_authorization(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=None,
        official_v03=None,
        confirmation=release.OWNER_RELEASE_CONFIRMATION,
        authorized_at_utc="2026-10-07T00:00:00+00:00",
    )
    authorization = tmp_path / "initial-owner-authorization.json"
    _write(authorization, authorization_value)
    gate = release.build_evidence_gate(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=None,
        official_v03=None,
        authorization=authorization,
    )

    assert gate["status"] == "PASS_INITIAL_PUBLICATION_EVIDENCE"
    assert gate["release_state"] == release.INITIAL_RELEASE_STATE
    assert gate["public_v03_final_sha256"] is None
    assert gate["official_v03_attestation_sha256"] is None
    assert gate["official_v03_full_score"] is None
    assert gate["official_v03_rank"] is None
    assert (
        authorization_value["blocker_dispositions"]["official-v0.3-private-evaluation"]
        == "DEFERRED_UNTIL_PUBLIC_IMMUTABLE_ENDPOINTS_EXIST"
    )
    plan = release.build_private_plan(gate, source_revision="a" * 40)
    assert plan["huggingface"]["repository"] == release.HF_REPOSITORY


def test_initial_publication_requires_v03_inputs_as_an_exact_pair(
    tmp_path: Path,
) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, _official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    with pytest.raises(release.V3PublicationError, match="supplied together"):
        release.build_owner_authorization(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=None,
            confirmation=release.OWNER_RELEASE_CONFIRMATION,
            authorized_at_utc="2026-10-07T00:00:00+00:00",
        )


def test_owner_authorization_preserves_training_disclosure_and_binds_authority(
    tmp_path: Path,
) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    with pytest.raises(release.V3PublicationError, match="exact owner"):
        release.build_owner_authorization(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            confirmation="yes",
            authorized_at_utc="2026-10-07T00:00:00+00:00",
        )
    authorization = release.build_owner_authorization(
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public_v03=public,
        official_v03=official,
        confirmation=release.OWNER_RELEASE_CONFIRMATION,
        authorized_at_utc="2026-10-07T00:00:00+00:00",
    )
    assert authorization["owner"] == "infercrane"
    assert authorization["owner_authorized_release"] is True
    assert authorization["legal_conclusion"] is False
    assert "authority_statements" not in authorization
    assert (
        authorization["authority_statements_sha256"]
        == release.OWNER_AUTHORITY_STATEMENTS_SHA256
    )
    assert authorization["accepted_known_disclosures"] == {
        "package_provenance_is_training_time_disclosure": True,
        "package_legal_conclusion": False,
        "package_publication_approval": False,
        "owner_accepts_no_legal_conclusion_claim": True,
    }
    provenance = json.loads((package / "PROVENANCE.json").read_text())
    assert provenance["legal_conclusion"] is False
    assert provenance["publication_approval"] is False


def test_gate_rejects_detached_owner_authorization(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    authorization = _authorization_file(
        tmp_path,
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public=public,
        official=official,
    )
    detached = json.loads(authorization.read_text())
    detached["artifact_sha256"] = "0" * 64
    body = {key: value for key, value in detached.items() if key != "receipt_sha256"}
    detached["receipt_sha256"] = release._json_sha256(body)
    _write(tmp_path / "detached.json", detached)
    with pytest.raises(release.V3PublicationError, match="detached"):
        release.build_evidence_gate(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            authorization=tmp_path / "detached.json",
        )


def test_gate_rejects_runtime_drift_and_nonhermetic_evidence(tmp_path: Path) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path)
    qualification, public, official = _evidence_files(
        tmp_path, package, artifact, calibration
    )
    authorization = _authorization_file(
        tmp_path,
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public=public,
        official=official,
    )
    (public_repo / "src/infercrane_commerce_1/runtime.py").write_text("DRIFT = True\n")
    with pytest.raises(release.V3PublicationError, match="public runtime differs"):
        release.build_evidence_gate(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            authorization=authorization,
        )

    package, public_repo, artifact, calibration = _package(tmp_path / "second")
    qualification, public, official = _evidence_files(
        tmp_path / "second", package, artifact, calibration
    )
    authorization = _authorization_file(
        tmp_path / "second",
        package=package,
        public_repo=public_repo,
        qualification=qualification,
        public=public,
        official=official,
    )
    manifest_path = package / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["evidence_receipts"]["final"]["packaged_path"]
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = release._json_sha256(body)
    _write(manifest_path, manifest)
    with pytest.raises(release.V3PublicationError, match="not hermetic"):
        release.build_evidence_gate(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            authorization=authorization,
        )


def test_public_plan_cannot_skip_private_verification() -> None:
    gate_body = {
        "schema_version": release.GATE_SCHEMA,
        "status": "PASS_V3_PUBLICATION_EVIDENCE",
        "artifact_sha256": "a" * 64,
    }
    gate = _receipt(gate_body)
    with pytest.raises(release.V3PublicationError):
        release.build_public_plan(gate, {})


def test_release_rejects_public_score_drift_and_baseline_regression(
    tmp_path: Path,
) -> None:
    package, public_repo, artifact, calibration = _package(tmp_path / "drift")
    qualification, public, official = _evidence_files(
        tmp_path / "drift", package, artifact, calibration
    )
    official_value = json.loads(official.read_text())
    official_value["public_score"] = 60.0
    official_value["decision_index"] = 64.0
    official_body = {
        key: value
        for key, value in official_value.items()
        if key != "attestation_sha256"
    }
    _write(
        official,
        {
            **official_body,
            "attestation_sha256": release._json_sha256(official_body),
        },
    )
    with pytest.raises(release.V3PublicationError, match="does not match"):
        release.build_owner_authorization(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            confirmation=release.OWNER_RELEASE_CONFIRMATION,
            authorized_at_utc="2026-10-07T00:00:00+00:00",
        )

    package, public_repo, artifact, calibration = _package(tmp_path / "baseline")
    qualification, public, official = _evidence_files(
        tmp_path / "baseline", package, artifact, calibration
    )
    public_value = json.loads(public.read_text())
    public_value["public_decision_index"] = release.RELEASE_BASELINE_DECISION_INDEX
    public_body = {
        key: value for key, value in public_value.items() if key != "receipt_sha256"
    }
    _write(public, _receipt(public_body))
    official_value = json.loads(official.read_text())
    official_value.update(
        {
            "decision_index": release.RELEASE_BASELINE_DECISION_INDEX,
            "public_score": release.RELEASE_BASELINE_DECISION_INDEX,
            "private_same_skills_score": release.RELEASE_BASELINE_DECISION_INDEX,
            "private_new_domains_score": release.RELEASE_BASELINE_DECISION_INDEX,
        }
    )
    official_body = {
        key: value
        for key, value in official_value.items()
        if key != "attestation_sha256"
    }
    _write(
        official,
        {
            **official_body,
            "attestation_sha256": release._json_sha256(official_body),
        },
    )
    with pytest.raises(release.V3PublicationError, match=r"52\.55 release baseline"):
        release.build_owner_authorization(
            package=package,
            public_repo=public_repo,
            qualification=qualification,
            public_v03=public,
            official_v03=official,
            confirmation=release.OWNER_RELEASE_CONFIRMATION,
            authorized_at_utc="2026-10-07T00:00:00+00:00",
        )


def test_private_plan_rejects_partial_self_hashed_gate() -> None:
    partial_gate = _receipt(
        {
            "schema_version": release.GATE_SCHEMA,
            "status": "PASS_V3_PUBLICATION_EVIDENCE",
            "artifact_sha256": "a" * 64,
        }
    )
    with pytest.raises(release.V3PublicationError, match="fields changed"):
        release.build_private_plan(partial_gate, source_revision="b" * 40)


def test_packaged_file_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    package = tmp_path / "package"
    outside = tmp_path / "outside"
    package.mkdir()
    outside.mkdir()
    (outside / "evidence.json").write_text("{}\n", encoding="utf-8")
    (package / "evidence").symlink_to(outside, target_is_directory=True)
    with pytest.raises(release.V3PublicationError, match="crosses a symlink"):
        release._packaged_file(package, "evidence/evidence.json", "evidence")
