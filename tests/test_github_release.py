from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tools import github_release as release


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "InferCrane Release")
    _git(root, "config", "user.email", "release@infercrane.com")
    (root / "README.md").write_text("# Commerce-1\n", encoding="utf-8")
    (root / "runtime.py").write_text("print('ready')\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Release Commerce-1")
    return root


class FakeGitHub:
    def __init__(self, local: dict[str, Any]) -> None:
        self.local = local
        self.repo: dict[str, Any] | None = None
        self.visibility_changes: list[str] = []
        self.pushed = False
        self.corrupt_public = False
        self.fail_rollback = False
        self.extra_tag = False

    def identity(self) -> dict[str, Any]:
        return {"login": "operator"}

    def organization_membership(self, owner: str) -> dict[str, Any]:
        return {
            "state": "active",
            "role": "admin",
            "organization": {"login": owner},
        }

    def _metadata(self, visibility: str) -> dict[str, Any]:
        return {
            "full_name": release.FULL_NAME,
            "name": release.REPOSITORY,
            "owner": {"login": release.OWNER},
            "visibility": visibility,
            "private": visibility == "private",
            "fork": False,
            "archived": False,
            "default_branch": release.DEFAULT_BRANCH,
        }

    def repository(self, full_name: str) -> dict[str, Any] | None:
        assert full_name == release.FULL_NAME
        return self.repo

    def create_private_repository(self, owner: str, name: str) -> dict[str, Any]:
        assert (owner, name) == (release.OWNER, release.REPOSITORY)
        self.repo = self._metadata("private")
        return self.repo

    def push_main(self, repo: Path, remote_url: str) -> None:
        assert repo == Path(self.local["root"])
        assert remote_url == release.REMOTE_URL
        self.pushed = True

    def commit(self, full_name: str, revision: str) -> dict[str, Any]:
        assert self.pushed
        return {
            "sha": self.local["revision"],
            "commit": {"tree": {"sha": self.local["tree"]}},
        }

    def tree(self, full_name: str, tree_sha: str) -> dict[str, Any]:
        entries = [
            {"path": path, "type": "blob", **item}
            for path, item in self.local["inventory"].items()
        ]
        for entry in entries:
            entry["size"] = entry.pop("size_bytes")
        if self.corrupt_public and self.repo and self.repo["visibility"] == "public":
            entries[0]["sha"] = "f" * 40
        return {"sha": tree_sha, "truncated": False, "tree": entries}

    def references(self, full_name: str, namespace: str) -> list[dict[str, Any]]:
        if namespace == "tags":
            if not self.extra_tag:
                return []
            return [
                {
                    "ref": "refs/tags/unverified",
                    "object": {
                        "type": "commit",
                        "sha": self.local["revision"],
                    },
                }
            ]
        return [
            {
                "ref": "refs/heads/main",
                "object": {
                    "type": "commit",
                    "sha": self.local["revision"],
                },
            }
        ]

    def set_visibility(self, full_name: str, visibility: str) -> dict[str, Any]:
        self.visibility_changes.append(visibility)
        if visibility == "private" and self.fail_rollback:
            raise release.GitHubReleaseError("rollback failed")
        self.repo = self._metadata(visibility)
        return self.repo


def test_plan_is_local_only_and_binds_exact_inventory(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    plan = release.build_plan(repo)
    assert plan["repository"] == "infercrane/commerce-1"
    assert plan["network_used"] is False
    assert plan["commit_count"] == 1
    assert sorted(plan["inventory"]) == ["README.md", "runtime.py"]
    assert plan["security_scan"]["status"] == "PASS_PUBLIC_OBJECT_CLOSURE"
    assert plan["security_scan"]["commit_identity"] == {
        "author_name": "InferCrane Release",
        "author_email": "release@infercrane.com",
        "committer_name": "InferCrane Release",
        "committer_email": "release@infercrane.com",
    }
    assert plan["security_scan"]["scanned_blobs"] == 2
    assert plan["security_scan"]["scanned_commits"] == 1
    assert plan["security_scan"]["scanned_commit_bytes"] > 0
    assert plan["receipt_sha256"] == release._json_sha256(
        {key: value for key, value in plan.items() if key != "receipt_sha256"}
    )


def test_plan_rejects_dirty_untracked_and_multi_commit_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "untracked.txt").write_text("not published\n", encoding="utf-8")
    with pytest.raises(release.GitHubReleaseError, match="clean"):
        release.build_plan(repo)
    (repo / "untracked.txt").unlink()
    (repo / "README.md").write_text("changed\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "Second commit")
    with pytest.raises(release.GitHubReleaseError, match="squashed root"):
        release.build_plan(repo)


def test_plan_rejects_symlink_even_when_committed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "alias").symlink_to("README.md")
    _git(repo, "add", "alias")
    _git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(release.GitHubReleaseError, match="symlink"):
        release.build_plan(repo)


def test_plan_rejects_personal_commit_metadata(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "config", "user.name", "Personal Developer")
    _git(repo, "config", "user.email", "developer@example.invalid")
    _git(repo, "commit", "--amend", "--no-edit", "--reset-author")
    with pytest.raises(release.GitHubReleaseError, match="organization identity"):
        release.build_plan(repo)


def test_plan_rejects_sensitive_blob_in_reachable_git_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    private_namespace = "yasin" + "toy"
    (repo / "README.md").write_text(
        f"# Commerce-1\nprivate owner: {private_namespace}\n",
        encoding="utf-8",
    )
    _git(repo, "add", "README.md")
    _git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(release.GitHubReleaseError, match="private owner namespace"):
        release.build_plan(repo)


@pytest.mark.parametrize(
    "message,error",
    [
        ("Release by " + "yasin" + "toy", "private owner namespace"),
        ("Release " + "github_" + "pat_" + "a" * 24, "GitHub credential"),
    ],
)
def test_plan_rejects_sensitive_commit_message_with_clean_blobs(
    tmp_path: Path, message: str, error: str
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "commit", "--amend", "-m", message)

    with pytest.raises(release.GitHubReleaseError, match=error):
        release.build_plan(repo)


def test_plan_rejects_internal_modal_path_and_resource_id(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    internal_root = "/__" + "modal/volumes/"
    resource = "vo-" + "A" * 24
    (repo / "evidence.json").write_text(
        json.dumps({"path": internal_root + resource + "/artifact"}),
        encoding="utf-8",
    )
    _git(repo, "add", "evidence.json")
    _git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(release.GitHubReleaseError, match="Modal internal"):
        release.build_plan(repo)


def test_plan_rejects_internal_modal_volume_uri(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    uri = "modal-" + "volume:private-checkpoint"
    (repo / "evidence.json").write_text(
        json.dumps({"source": uri}),
        encoding="utf-8",
    )
    _git(repo, "add", "evidence.json")
    _git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(release.GitHubReleaseError, match="Modal internal volume URI"):
        release.build_plan(repo)


def test_plan_rejects_raw_historical_release_tree(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    historical = repo / "release/training-data/manifest.json"
    historical.parent.mkdir(parents=True)
    historical.write_text(
        json.dumps({"attempt_id": "a" * 32}) + "\n",
        encoding="utf-8",
    )
    _git(repo, "add", str(historical.relative_to(repo)))
    _git(repo, "commit", "--amend", "--no-edit")
    with pytest.raises(release.GitHubReleaseError, match="private or generated path"):
        release.build_plan(repo)


def test_plan_rejects_extra_reachable_ref_even_with_one_head_commit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    _git(repo, "tag", "candidate")
    with pytest.raises(release.GitHubReleaseError, match="only refs/heads/main"):
        release.build_plan(repo)


def test_stage_private_creates_pushes_and_verifies_exact_revision(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    receipt_path = tmp_path / "private.json"
    receipt = release.stage_private(
        repo=repo,
        receipt_path=receipt_path,
        confirmation=release.PRIVATE_CONFIRMATION,
        api=api,
    )
    assert api.pushed is True
    assert receipt["status"] == "PRIVATE_REVISION_VERIFIED"
    assert receipt["visibility"] == "private"
    assert receipt["source_revision"] == local["revision"]
    assert receipt["inventory"] == local["inventory"]
    assert receipt_path.stat().st_mode & 0o777 == 0o600


def test_stage_private_refuses_existing_repository_and_wrong_operator(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    api.repo = api._metadata("private")
    with pytest.raises(release.GitHubReleaseError, match="already exists"):
        release.stage_private(
            repo=repo,
            receipt_path=tmp_path / "private.json",
            confirmation=release.PRIVATE_CONFIRMATION,
            api=api,
        )
    api.repo = None
    api.organization_membership = lambda owner: {  # type: ignore[method-assign]
        "state": "active",
        "role": "member",
        "organization": {"login": owner},
    }
    with pytest.raises(release.GitHubReleaseError, match="admin"):
        release.stage_private(
            repo=repo,
            receipt_path=tmp_path / "private.json",
            confirmation=release.PRIVATE_CONFIRMATION,
            api=api,
        )


def test_stage_private_rejects_unverified_remote_refs(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    api.extra_tag = True
    with pytest.raises(release.GitHubReleaseError, match="remains private"):
        release.stage_private(
            repo=repo,
            receipt_path=tmp_path / "private.json",
            confirmation=release.PRIVATE_CONFIRMATION,
            api=api,
        )
    assert api.repo is not None
    assert api.repo["visibility"] == "private"


def test_publish_reverifies_private_then_public_and_binds_receipts(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    private_path = tmp_path / "private.json"
    private = release.stage_private(
        repo=repo,
        receipt_path=private_path,
        confirmation=release.PRIVATE_CONFIRMATION,
        api=api,
    )
    public_path = tmp_path / "public.json"
    public = release.publish_public(
        repo=repo,
        private_receipt_path=private_path,
        public_receipt_path=public_path,
        confirmation=release.PUBLIC_CONFIRMATION,
        api=api,
    )
    assert api.visibility_changes == ["public"]
    assert public["status"] == "PUBLIC_REVISION_VERIFIED"
    assert public["visibility"] == "public"
    assert public["private_stage_receipt_sha256"] == private["receipt_sha256"]
    assert public["remote_inventory_sha256"] == local["inventory_sha256"]


def test_publish_rejects_stale_or_tampered_private_receipt(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    private_path = tmp_path / "private.json"
    release.stage_private(
        repo=repo,
        receipt_path=private_path,
        confirmation=release.PRIVATE_CONFIRMATION,
        api=api,
    )
    receipt = json.loads(private_path.read_text(encoding="utf-8"))
    receipt["inventory_sha256"] = "0" * 64
    private_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(release.GitHubReleaseError, match="hash does not match"):
        release.publish_public(
            repo=repo,
            private_receipt_path=private_path,
            public_receipt_path=tmp_path / "public.json",
            confirmation=release.PUBLIC_CONFIRMATION,
            api=api,
        )
    assert api.visibility_changes == []


def test_public_verification_failure_rolls_back_and_records_failure(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    private_path = tmp_path / "private.json"
    release.stage_private(
        repo=repo,
        receipt_path=private_path,
        confirmation=release.PRIVATE_CONFIRMATION,
        api=api,
    )
    api.corrupt_public = True
    failure_path = tmp_path / "failed-public.json"
    with pytest.raises(release.GitHubReleaseError, match="rolled back"):
        release.publish_public(
            repo=repo,
            private_receipt_path=private_path,
            public_receipt_path=failure_path,
            confirmation=release.PUBLIC_CONFIRMATION,
            api=api,
        )
    assert api.visibility_changes == ["public", "private"]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "PUBLIC_VERIFICATION_FAILED_ROLLED_BACK"
    assert failure["rollback_confirmed"] is True


def test_public_verification_records_unconfirmed_rollback(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    local = release._local_snapshot(repo)
    api = FakeGitHub(local)
    private_path = tmp_path / "private.json"
    release.stage_private(
        repo=repo,
        receipt_path=private_path,
        confirmation=release.PRIVATE_CONFIRMATION,
        api=api,
    )
    api.corrupt_public = True
    api.fail_rollback = True
    failure_path = tmp_path / "failed-public.json"
    with pytest.raises(release.GitHubReleaseError, match="rollback is unconfirmed"):
        release.publish_public(
            repo=repo,
            private_receipt_path=private_path,
            public_receipt_path=failure_path,
            confirmation=release.PUBLIC_CONFIRMATION,
            api=api,
        )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "PUBLIC_VERIFICATION_FAILED_ROLLBACK_UNCONFIRMED"
    assert failure["rollback_attempted"] is True
    assert failure["rollback_confirmed"] is False


def test_confirmations_and_receipt_location_fail_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    api = FakeGitHub(release._local_snapshot(repo))
    with pytest.raises(release.GitHubReleaseError, match="confirmation"):
        release.stage_private(
            repo=repo,
            receipt_path=tmp_path / "private.json",
            confirmation="yes",
            api=api,
        )
    with pytest.raises(release.GitHubReleaseError, match="outside"):
        release.stage_private(
            repo=repo,
            receipt_path=repo / "private.json",
            confirmation=release.PRIVATE_CONFIRMATION,
            api=api,
        )
    occupied = tmp_path / "occupied.json"
    occupied.write_text("do not overwrite\n", encoding="utf-8")
    with pytest.raises(release.GitHubReleaseError, match="already exists"):
        release.stage_private(
            repo=repo,
            receipt_path=occupied,
            confirmation=release.PRIVATE_CONFIRMATION,
            api=api,
        )
    assert api.repo is None


def test_gh_cli_push_uses_credential_store_without_token_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs.get("env")))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(release.subprocess, "run", run)
    release.GhCLI().push_main(tmp_path, release.REMOTE_URL)
    command, environment = calls[0]
    assert command == [
        "git",
        "-C",
        str(tmp_path),
        "push",
        "--porcelain",
        "https://github.com/infercrane/commerce-1.git",
        "HEAD:refs/heads/main",
    ]
    assert environment is not None
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert all("token" not in argument.casefold() for argument in command)
