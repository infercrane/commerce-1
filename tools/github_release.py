#!/usr/bin/env python3
"""Publish the Commerce-1 source repository through a private-first gate.

The default ``plan`` command is local-only.  Network mutation is split into
``stage-private`` and ``publish-public``.  Both operations bind the exact Git
commit, tree, and regular-file inventory.  Authentication is delegated to the
GitHub CLI credential store; tokens are never accepted as arguments or written
to receipts.

The source history is deliberately required to contain exactly one root
commit.  That makes the verified current inventory the complete public Git
history, rather than exposing superseded release-candidate bytes through old
commits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

OWNER = "infercrane"
REPOSITORY = "commerce-1"
FULL_NAME = f"{OWNER}/{REPOSITORY}"
REMOTE_URL = f"https://github.com/{FULL_NAME}.git"
DEFAULT_BRANCH = "main"
PLAN_SCHEMA = "infercrane-commerce-1-github-plan/v1"
PRIVATE_RECEIPT_SCHEMA = "infercrane-commerce-1-github-private-stage/v1"
PUBLIC_RECEIPT_SCHEMA = "infercrane-commerce-1-github-publication/v1"
PRIVATE_CONFIRMATION = "STAGE_INFERCRANE_COMMERCE_1_GITHUB_PRIVATE"
PUBLIC_CONFIRMATION = "PUBLISH_INFERCRANE_COMMERCE_1_GITHUB_PUBLIC"
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SECURITY_SCAN_SCHEMA = "infercrane-commerce-1-github-security-scan/v1"
ORG_COMMIT_NAME_PATTERN = re.compile(r"^InferCrane(?:[ -][A-Za-z0-9 ._-]+)?$")
ORG_COMMIT_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@infercrane[.]com$"
)

# Public source is an allowlist, not an archive.  These locations contain
# local build products or superseded raw release evidence and must never enter
# the one-commit public history.  The checks use Git paths, so they cannot be
# bypassed by .gitignore rules.
FORBIDDEN_PUBLIC_PATHS = frozenset(
    {
        ".DS_Store",
        ".dockerignore",
        "Dockerfile",
        "docs/BUNDLE.md",
        "licenses/upstream/autojev-model-LICENSE",
        "licenses/upstream/autojev-model-NOTICE",
        "licenses/upstream/autojev-public-provenance.json",
        "licenses/upstream/autojev-source-LICENSE",
        "release/.DS_Store",
        "release/FINAL_IMAGE_QUALIFICATION.md",
        # Superseded effective-v4 publication controls.  These files are
        # retained only in protected operator archives; including any one of
        # them would make the purported V3 source release historically mixed.
        "release/PROVENANCE_REVIEW.md",
        "release/PUBLICATION_CHECKLIST.md",
        "release/causal-conv1d-wheel.json",
        "release/final-image-authoritative-license-evidence.json",
        "release/prebuild-license-inventory.json",
        "release/prebuild-runtime.spdx.json",
        "release/prebuild-vulnerability-audit.json",
        "release/provenance-lock.json",
        "release/qualification-tools.json",
        "release/runtime-bom.lock.json",
        "release/upstream-evidence-index.json",
        "tests/conftest.py",
        "tests/test_api.py",
        "tests/test_bundle.py",
        "tests/test_cli.py",
        "tests/test_distribution.py",
        "tests/test_effective_v4_hf_bundle.py",
        "tests/test_final_image_qualification.py",
        "tests/test_huggingface_release.py",
        "tests/test_modal_cuda_abi_smoke.py",
        "tests/test_modal_private_oci_qualification.py",
        "tests/test_openrouter_decisions.py",
        "tests/test_qualification_tools.py",
        "tests/test_typed_decisions_v2.py",
        "tools/effective_v4_hf_bundle.py",
        "tools/final_image_qualification.py",
        "tools/huggingface_release.py",
        "tools/install_qualification_tools.py",
        "tools/modal_cuda_abi_smoke.py",
        "tools/modal_private_oci_qualification.py",
        "tools/runtime_supply_chain.py",
    }
)
FORBIDDEN_PUBLIC_PREFIXES = (
    ".commerce-v3-recovery/",
    ".release-tools/",
    "artifacts/",
    "build/",
    "dist/",
    "huggingface/",
    "licenses/upstream/",
    "release/internal/",
    "release/private/",
    "release/training-data/",
    "release/upstream/",
    "runtime/",
    "src/commerce_one/",
    "vendor/",
)

# Construct private identifiers in pieces so the scanner does not have to
# exempt its own source.  Every rule is applied to the bytes of every blob and
# commit object reachable from every local branch or tag.
_CONTENT_RULES = (
    (
        "private owner namespace",
        re.compile(b"yasin" + b"toy", re.IGNORECASE),
    ),
    (
        "private owner name",
        re.compile(b"Yasin" + b"[ ]+Toy", re.IGNORECASE),
    ),
    (
        "private owner email",
        re.compile(
            b"(?:yasin[.]toy@|@" + b"yandex[.]com)",
            re.IGNORECASE,
        ),
    ),
    (
        "Modal internal path",
        re.compile(b"/__" + b"modal/", re.IGNORECASE),
    ),
    (
        "Modal internal volume URI",
        re.compile(b"modal-" + b"volume:", re.IGNORECASE),
    ),
    (
        "Modal internal resource identifier",
        re.compile(rb"(?<![A-Za-z0-9])(?:vo|ap|fc)-[A-Za-z0-9]{12,}"),
    ),
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
        re.compile(rb"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
    ),
    (
        "private key",
        re.compile(b"-----BEGIN " + b"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)


class GitHubReleaseError(ValueError):
    """The local repository, GitHub state, or publication control is unsafe."""


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


def _receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    return {**body, "receipt_sha256": _json_sha256(body)}


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists():
        raise GitHubReleaseError("receipt path already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GitHubReleaseError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise GitHubReleaseError(f"{label} must be a JSON object")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> str:
    supplied = value.get("receipt_sha256")
    if not isinstance(supplied, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied):
        raise GitHubReleaseError(f"{label} receipt hash is invalid")
    body = dict(value)
    body.pop("receipt_sha256", None)
    if _json_sha256(body) != supplied:
        raise GitHubReleaseError(f"{label} receipt hash does not match")
    return supplied


def _run_git(repo: Path, arguments: Sequence[str], *, binary: bool = False) -> Any:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        operation = arguments[0] if arguments else "command"
        raise GitHubReleaseError(f"git {operation} failed")
    return result.stdout


def _inventory(repo: Path, revision: str) -> dict[str, dict[str, Any]]:
    raw = _run_git(repo, ["ls-tree", "-r", "-z", "-l", revision], binary=True)
    inventory: dict[str, dict[str, Any]] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_sha, raw_size = metadata.split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise GitHubReleaseError(
                "Git inventory contains an invalid entry"
            ) from error
        mode_text = mode.decode("ascii")
        type_text = object_type.decode("ascii")
        sha_text = object_sha.decode("ascii")
        if type_text != "blob" or mode_text not in {"100644", "100755"}:
            raise GitHubReleaseError(
                f"Git inventory contains a symlink, submodule, or special file: {path}"
            )
        if not OBJECT_PATTERN.fullmatch(sha_text):
            raise GitHubReleaseError("Git inventory contains an invalid object ID")
        if path in inventory or path.startswith("/") or "\x00" in path:
            raise GitHubReleaseError("Git inventory contains an unsafe path")
        inventory[path] = {
            "mode": mode_text,
            "sha": sha_text,
            "size_bytes": int(raw_size),
        }
    if not inventory:
        raise GitHubReleaseError("Git inventory is empty")
    return dict(sorted(inventory.items()))


def _validate_public_paths(inventory: Mapping[str, Any]) -> None:
    for path in inventory:
        if path in FORBIDDEN_PUBLIC_PATHS or path.startswith(FORBIDDEN_PUBLIC_PREFIXES):
            raise GitHubReleaseError(
                f"Git public inventory contains a private or generated path: {path}"
            )


def _commit_identity(repo: Path, revision: str) -> dict[str, str]:
    raw = _run_git(
        repo,
        [
            "show",
            "-s",
            "--format=%an%x00%ae%x00%cn%x00%ce",
            revision,
        ],
        binary=True,
    )
    values = raw.rstrip(b"\n").split(b"\0")
    if len(values) != 4:
        raise GitHubReleaseError("Git commit identity is malformed")
    try:
        author_name, author_email, committer_name, committer_email = (
            item.decode("utf-8") for item in values
        )
    except UnicodeDecodeError as error:
        raise GitHubReleaseError("Git commit identity is not UTF-8") from error
    for role, name, email in (
        ("author", author_name, author_email),
        ("committer", committer_name, committer_email),
    ):
        if (
            ORG_COMMIT_NAME_PATTERN.fullmatch(name) is None
            or ORG_COMMIT_EMAIL_PATTERN.fullmatch(email) is None
        ):
            raise GitHubReleaseError(
                f"Git {role} must use an InferCrane organization identity"
            )
    return {
        "author_name": author_name,
        "author_email": author_email,
        "committer_name": committer_name,
        "committer_email": committer_email,
    }


def _reachable_objects(repo: Path, revision: str) -> dict[str, list[str]]:
    refs = str(
        _run_git(
            repo,
            [
                "for-each-ref",
                "--format=%(refname)",
                "refs/heads",
                "refs/tags",
            ],
        )
    ).splitlines()
    if refs != [f"refs/heads/{DEFAULT_BRANCH}"]:
        raise GitHubReleaseError(
            "Git public object closure must contain only refs/heads/main"
        )
    revisions = str(_run_git(repo, ["rev-list", "--all"])).splitlines()
    if revisions != [revision]:
        raise GitHubReleaseError(
            "Git public object closure contains an unexpected commit"
        )
    records = str(_run_git(repo, ["rev-list", "--objects", "--all"])).splitlines()
    objects: dict[str, list[str]] = {}
    for record in records:
        object_id, separator, path = record.partition(" ")
        if OBJECT_PATTERN.fullmatch(object_id) is None:
            raise GitHubReleaseError("Git public object closure is malformed")
        objects.setdefault(object_id, [])
        if separator:
            objects[object_id].append(path)
    if revision not in objects:
        raise GitHubReleaseError("Git public object closure omits HEAD")
    return dict(sorted(objects.items()))


def _scan_public_objects(
    repo: Path,
    *,
    revision: str,
    inventory: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_public_paths(inventory)
    identity = _commit_identity(repo, revision)
    objects = _reachable_objects(repo, revision)
    expected_blobs = {str(item["sha"]): path for path, item in inventory.items()}
    observed_blobs: set[str] = set()
    observed_commits: set[str] = set()
    scanned_bytes = 0
    scanned_commit_bytes = 0
    object_types: dict[str, int] = {}
    for object_id, paths in objects.items():
        object_type = str(_run_git(repo, ["cat-file", "-t", object_id])).strip()
        object_types[object_type] = object_types.get(object_type, 0) + 1
        if object_type not in {"blob", "commit"}:
            continue
        payload = _run_git(repo, ["cat-file", object_type, object_id], binary=True)
        scanned_bytes += len(payload)
        if object_type == "blob":
            observed_blobs.add(object_id)
            display_path = expected_blobs.get(object_id) or (
                paths[0] if paths else object_id
            )
        else:
            observed_commits.add(object_id)
            scanned_commit_bytes += len(payload)
            display_path = f"commit {object_id}"
        for label, pattern in _CONTENT_RULES:
            if pattern.search(payload) is not None:
                raise GitHubReleaseError(
                    f"Git public object contains {label}: {display_path}"
                )
    if observed_blobs != set(expected_blobs):
        raise GitHubReleaseError(
            "Git public object closure differs from the committed file inventory"
        )
    if observed_commits != {revision}:
        raise GitHubReleaseError(
            "Git public object closure differs from the sole release commit"
        )
    body = {
        "schema_version": SECURITY_SCAN_SCHEMA,
        "status": "PASS_PUBLIC_OBJECT_CLOSURE",
        "revision": revision,
        "commit_identity": identity,
        "reachable_objects": len(objects),
        "object_types": dict(sorted(object_types.items())),
        "scanned_blobs": len(observed_blobs),
        "scanned_commits": len(observed_commits),
        "scanned_commit_bytes": scanned_commit_bytes,
        "scanned_bytes": scanned_bytes,
        "path_policy": {
            "forbidden_paths": sorted(FORBIDDEN_PUBLIC_PATHS),
            "forbidden_prefixes": list(FORBIDDEN_PUBLIC_PREFIXES),
        },
    }
    return {**body, "security_scan_sha256": _json_sha256(body)}


def _ensure_receipt_outside_repo(path: Path, repo: Path) -> None:
    destination = path.resolve()
    root = repo.resolve(strict=True)
    if destination == root or destination.is_relative_to(root):
        raise GitHubReleaseError("receipts must remain outside the public repository")


def _require_new_receipt(path: Path, repo: Path) -> None:
    _ensure_receipt_outside_repo(path, repo)
    if path.resolve().exists():
        raise GitHubReleaseError("receipt path already exists")


def _local_snapshot(repo: Path) -> dict[str, Any]:
    root = repo.resolve(strict=True)
    top = Path(str(_run_git(root, ["rev-parse", "--show-toplevel"])).strip()).resolve(
        strict=True
    )
    if top != root:
        raise GitHubReleaseError("repository path must be the Git worktree root")
    dirty = _run_git(
        root, ["status", "--porcelain=v1", "-z", "--untracked-files=all"], binary=True
    )
    if dirty:
        raise GitHubReleaseError("repository must be clean, including untracked files")
    branch = str(_run_git(root, ["symbolic-ref", "--short", "HEAD"])).strip()
    if branch != DEFAULT_BRANCH:
        raise GitHubReleaseError(f"release branch must be {DEFAULT_BRANCH}")
    revision = str(_run_git(root, ["rev-parse", "HEAD"])).strip()
    tree = str(_run_git(root, ["rev-parse", "HEAD^{tree}"])).strip()
    if not COMMIT_PATTERN.fullmatch(revision) or not OBJECT_PATTERN.fullmatch(tree):
        raise GitHubReleaseError("Git revision or tree is not a full SHA-1 object ID")
    commit_count = int(str(_run_git(root, ["rev-list", "--count", "HEAD"])).strip())
    roots = str(_run_git(root, ["rev-list", "--max-parents=0", "HEAD"])).splitlines()
    if commit_count != 1 or roots != [revision]:
        raise GitHubReleaseError(
            "public release history must be one squashed root commit"
        )
    inventory = _inventory(root, revision)
    security_scan = _scan_public_objects(
        root,
        revision=revision,
        inventory=inventory,
    )
    return {
        "root": str(root),
        "branch": branch,
        "revision": revision,
        "tree": tree,
        "commit_count": commit_count,
        "inventory": inventory,
        "inventory_sha256": _json_sha256(inventory),
        "files": len(inventory),
        "bytes": sum(item["size_bytes"] for item in inventory.values()),
        "security_scan": security_scan,
        "security_scan_sha256": security_scan["security_scan_sha256"],
    }


def build_plan(repo: Path) -> dict[str, Any]:
    """Build a zero-network plan for the exact committed source revision."""

    local = _local_snapshot(repo)
    body = {
        "schema_version": PLAN_SCHEMA,
        "status": "PRIVATE_STAGE_PLANNED_NOT_PERFORMED",
        "repository": FULL_NAME,
        "visibility_sequence": ["private", "public"],
        "default_branch": DEFAULT_BRANCH,
        "source_revision": local["revision"],
        "source_tree": local["tree"],
        "commit_count": local["commit_count"],
        "inventory": local["inventory"],
        "inventory_sha256": local["inventory_sha256"],
        "files": local["files"],
        "bytes": local["bytes"],
        "security_scan": local["security_scan"],
        "security_scan_sha256": local["security_scan_sha256"],
        "network_used": False,
        "repository_created": False,
        "push_performed": False,
        "publication_performed": False,
    }
    return _receipt(body)


class GitHubAPI(Protocol):
    def identity(self) -> Mapping[str, Any]: ...

    def organization_membership(self, owner: str) -> Mapping[str, Any]: ...

    def repository(self, full_name: str) -> Mapping[str, Any] | None: ...

    def create_private_repository(self, owner: str, name: str) -> Mapping[str, Any]: ...

    def push_main(self, repo: Path, remote_url: str) -> None: ...

    def commit(self, full_name: str, revision: str) -> Mapping[str, Any]: ...

    def tree(self, full_name: str, tree_sha: str) -> Mapping[str, Any]: ...

    def references(
        self, full_name: str, namespace: str
    ) -> Sequence[Mapping[str, Any]]: ...

    def set_visibility(self, full_name: str, visibility: str) -> Mapping[str, Any]: ...


@dataclass
class GhCLI:
    """Small GitHub CLI adapter that never accepts or emits a token."""

    def _api_value(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        fields: Sequence[tuple[str, str, bool]] = (),
        optional: bool = False,
    ) -> Any:
        command = ["gh", "api", "--method", method, endpoint]
        for name, value, typed in fields:
            command.extend(["-F" if typed else "-f", f"{name}={value}"])
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            if optional and (
                "HTTP 404" in result.stderr or "Not Found" in result.stderr
            ):
                return None
            raise GitHubReleaseError(f"GitHub API {method} {endpoint} failed")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GitHubReleaseError("GitHub API returned invalid JSON") from error
        return value

    def _api(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        fields: Sequence[tuple[str, str, bool]] = (),
        optional: bool = False,
    ) -> Mapping[str, Any] | None:
        value = self._api_value(
            endpoint, method=method, fields=fields, optional=optional
        )
        if value is None and optional:
            return None
        if not isinstance(value, Mapping):
            raise GitHubReleaseError("GitHub API response is not an object")
        return value

    def identity(self) -> Mapping[str, Any]:
        value = self._api("user")
        assert value is not None
        return value

    def organization_membership(self, owner: str) -> Mapping[str, Any]:
        value = self._api(f"user/memberships/orgs/{owner}")
        assert value is not None
        return value

    def repository(self, full_name: str) -> Mapping[str, Any] | None:
        return self._api(f"repos/{full_name}", optional=True)

    def create_private_repository(self, owner: str, name: str) -> Mapping[str, Any]:
        value = self._api(
            f"orgs/{owner}/repos",
            method="POST",
            fields=(
                ("name", name, False),
                ("private", "true", True),
                ("visibility", "private", False),
                ("auto_init", "false", True),
                ("description", "Commerce-1 source and local serving runtime", False),
            ),
        )
        assert value is not None
        return value

    def push_main(self, repo: Path, remote_url: str) -> None:
        environment = dict(os.environ)
        environment["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "push",
                "--porcelain",
                remote_url,
                f"HEAD:refs/heads/{DEFAULT_BRANCH}",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if result.returncode != 0:
            raise GitHubReleaseError("GitHub push failed; target remains private")

    def commit(self, full_name: str, revision: str) -> Mapping[str, Any]:
        value = self._api(f"repos/{full_name}/commits/{revision}")
        assert value is not None
        return value

    def tree(self, full_name: str, tree_sha: str) -> Mapping[str, Any]:
        value = self._api(f"repos/{full_name}/git/trees/{tree_sha}?recursive=1")
        assert value is not None
        return value

    def references(self, full_name: str, namespace: str) -> Sequence[Mapping[str, Any]]:
        if namespace not in {"heads", "tags"}:
            raise GitHubReleaseError("GitHub reference namespace is invalid")
        value = self._api_value(f"repos/{full_name}/git/matching-refs/{namespace}/")
        if not isinstance(value, list) or any(
            not isinstance(item, Mapping) for item in value
        ):
            raise GitHubReleaseError("GitHub references response is invalid")
        return value

    def set_visibility(self, full_name: str, visibility: str) -> Mapping[str, Any]:
        value = self._api(
            f"repos/{full_name}",
            method="PATCH",
            fields=(("visibility", visibility, False),),
        )
        assert value is not None
        return value


def _validate_operator(api: GitHubAPI) -> dict[str, str]:
    identity = api.identity()
    login = identity.get("login")
    membership = api.organization_membership(OWNER)
    state = membership.get("state")
    role = membership.get("role")
    organization = membership.get("organization")
    organization_login = (
        organization.get("login") if isinstance(organization, Mapping) else None
    )
    if not isinstance(login, str) or not login:
        raise GitHubReleaseError("authenticated GitHub identity is invalid")
    if state != "active" or role != "admin" or organization_login != OWNER:
        raise GitHubReleaseError(
            "authenticated GitHub identity must be an active InferCrane admin"
        )
    return {"actor": login, "organization": OWNER, "role": "admin"}


def _validate_repository_metadata(value: Mapping[str, Any], *, visibility: str) -> None:
    owner = value.get("owner")
    owner_login = owner.get("login") if isinstance(owner, Mapping) else None
    expected_private = visibility == "private"
    if (
        value.get("full_name") != FULL_NAME
        or owner_login != OWNER
        or value.get("name") != REPOSITORY
        or value.get("visibility") != visibility
        or value.get("private") is not expected_private
        or value.get("fork") is not False
        or value.get("archived") is not False
        or value.get("default_branch") != DEFAULT_BRANCH
    ):
        raise GitHubReleaseError(
            f"GitHub repository identity or {visibility} visibility is invalid"
        )


def _remote_inventory(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise GitHubReleaseError("GitHub recursive tree is missing or truncated")
    inventory: dict[str, dict[str, Any]] = {}
    for entry in tree["tree"]:
        if not isinstance(entry, Mapping):
            raise GitHubReleaseError("GitHub tree contains an invalid entry")
        entry_type = entry.get("type")
        if entry_type == "tree":
            continue
        path = entry.get("path")
        mode = entry.get("mode")
        sha = entry.get("sha")
        size = entry.get("size")
        if (
            entry_type != "blob"
            or mode not in {"100644", "100755"}
            or not isinstance(path, str)
            or not OBJECT_PATTERN.fullmatch(str(sha))
            or not isinstance(size, int)
            or size < 0
            or path in inventory
        ):
            raise GitHubReleaseError(
                "GitHub tree contains a symlink, submodule, or invalid blob"
            )
        inventory[path] = {"mode": mode, "sha": sha, "size_bytes": size}
    return dict(sorted(inventory.items()))


def _verify_remote_references(api: GitHubAPI, revision: str) -> None:
    heads = api.references(FULL_NAME, "heads")
    tags = api.references(FULL_NAME, "tags")
    if len(heads) != 1 or tags:
        raise GitHubReleaseError("GitHub repository must expose only main and no tags")
    head = heads[0]
    target = head.get("object")
    if (
        head.get("ref") != f"refs/heads/{DEFAULT_BRANCH}"
        or not isinstance(target, Mapping)
        or target.get("type") != "commit"
        or target.get("sha") != revision
    ):
        raise GitHubReleaseError("GitHub main reference differs from local source")


def _verify_remote(
    api: GitHubAPI, local: Mapping[str, Any], *, visibility: str
) -> dict[str, Any]:
    repository = api.repository(FULL_NAME)
    if repository is None:
        raise GitHubReleaseError("GitHub repository does not exist")
    _validate_repository_metadata(repository, visibility=visibility)
    _verify_remote_references(api, str(local["revision"]))
    commit = api.commit(FULL_NAME, DEFAULT_BRANCH)
    commit_sha = commit.get("sha")
    commit_body = commit.get("commit")
    remote_tree = commit_body.get("tree") if isinstance(commit_body, Mapping) else None
    tree_sha = remote_tree.get("sha") if isinstance(remote_tree, Mapping) else None
    if commit_sha != local["revision"] or tree_sha != local["tree"]:
        raise GitHubReleaseError("GitHub revision or tree differs from local source")
    tree_document = api.tree(FULL_NAME, str(tree_sha))
    if tree_document.get("sha") != local["tree"]:
        raise GitHubReleaseError("GitHub tree response differs from local source")
    inventory = _remote_inventory(tree_document)
    if inventory != local["inventory"]:
        raise GitHubReleaseError("GitHub inventory differs from local source")
    return {
        "visibility": visibility,
        "revision": commit_sha,
        "tree": tree_sha,
        "inventory": inventory,
        "inventory_sha256": _json_sha256(inventory),
        "files": len(inventory),
        "bytes": sum(item["size_bytes"] for item in inventory.values()),
    }


def _assert_snapshot_unchanged(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    fields = (
        "revision",
        "tree",
        "inventory_sha256",
        "security_scan_sha256",
        "files",
        "bytes",
    )
    if any(before[field] != after[field] for field in fields):
        raise GitHubReleaseError("local source changed during the release operation")


def stage_private(
    *,
    repo: Path,
    receipt_path: Path,
    confirmation: str,
    api: GitHubAPI | None = None,
) -> dict[str, Any]:
    """Create an empty private repository, push, and verify exact remote bytes."""

    if confirmation != PRIVATE_CONFIRMATION:
        raise GitHubReleaseError("private-stage confirmation is invalid")
    local = _local_snapshot(repo)
    _require_new_receipt(receipt_path, repo)
    client = api or GhCLI()
    operator = _validate_operator(client)
    if client.repository(FULL_NAME) is not None:
        raise GitHubReleaseError(
            "target repository already exists; refuse to inherit unverified history"
        )
    created = client.create_private_repository(OWNER, REPOSITORY)
    _validate_repository_metadata(created, visibility="private")
    try:
        client.push_main(Path(local["root"]), REMOTE_URL)
        after_push = _local_snapshot(repo)
        _assert_snapshot_unchanged(local, after_push)
        remote = _verify_remote(client, local, visibility="private")
    except Exception as error:
        try:
            current = client.repository(FULL_NAME)
            if current is not None and current.get("visibility") != "private":
                client.set_visibility(FULL_NAME, "private")
        except Exception:
            raise GitHubReleaseError(
                "private staging failed and private visibility could not be reconfirmed"
            ) from error
        raise GitHubReleaseError(
            "private staging failed; the created repository remains private"
        ) from error
    body = {
        "schema_version": PRIVATE_RECEIPT_SCHEMA,
        "status": "PRIVATE_REVISION_VERIFIED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository": FULL_NAME,
        "visibility": "private",
        "default_branch": DEFAULT_BRANCH,
        "actor": operator["actor"],
        "source_revision": local["revision"],
        "source_tree": local["tree"],
        "commit_count": local["commit_count"],
        "inventory": local["inventory"],
        "inventory_sha256": local["inventory_sha256"],
        "files": local["files"],
        "bytes": local["bytes"],
        "security_scan": local["security_scan"],
        "security_scan_sha256": local["security_scan_sha256"],
        "remote_revision": remote["revision"],
        "remote_tree": remote["tree"],
        "remote_inventory_sha256": remote["inventory_sha256"],
        "network_used": True,
        "repository_created": True,
        "push_performed": True,
        "immutable_revision_verified": True,
        "publication_performed": False,
    }
    receipt = _receipt(body)
    _write_once(receipt_path, receipt)
    return receipt


def _validate_private_receipt(
    value: Mapping[str, Any], local: Mapping[str, Any]
) -> str:
    receipt_sha = _validate_self_hash(value, "GitHub private-stage")
    expected = {
        "schema_version": PRIVATE_RECEIPT_SCHEMA,
        "status": "PRIVATE_REVISION_VERIFIED",
        "repository": FULL_NAME,
        "visibility": "private",
        "default_branch": DEFAULT_BRANCH,
        "source_revision": local["revision"],
        "source_tree": local["tree"],
        "commit_count": 1,
        "inventory": local["inventory"],
        "inventory_sha256": local["inventory_sha256"],
        "files": local["files"],
        "bytes": local["bytes"],
        "security_scan": local["security_scan"],
        "security_scan_sha256": local["security_scan_sha256"],
        "remote_revision": local["revision"],
        "remote_tree": local["tree"],
        "remote_inventory_sha256": local["inventory_sha256"],
        "network_used": True,
        "repository_created": True,
        "push_performed": True,
        "immutable_revision_verified": True,
        "publication_performed": False,
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise GitHubReleaseError(
                f"GitHub private-stage receipt has invalid {field}"
            )
    if not isinstance(value.get("created_at_utc"), str) or not isinstance(
        value.get("actor"), str
    ):
        raise GitHubReleaseError("GitHub private-stage receipt metadata is invalid")
    return receipt_sha


def publish_public(
    *,
    repo: Path,
    private_receipt_path: Path,
    public_receipt_path: Path,
    confirmation: str,
    api: GitHubAPI | None = None,
) -> dict[str, Any]:
    """Reverify the private stage, publish, then verify the public copy exactly."""

    if confirmation != PUBLIC_CONFIRMATION:
        raise GitHubReleaseError("public-transition confirmation is invalid")
    local = _local_snapshot(repo)
    _ensure_receipt_outside_repo(private_receipt_path, repo)
    _require_new_receipt(public_receipt_path, repo)
    if private_receipt_path.resolve() == public_receipt_path.resolve():
        raise GitHubReleaseError("private and public receipts must use different paths")
    private_receipt = _load_object(private_receipt_path, "GitHub private-stage receipt")
    private_receipt_sha = _validate_private_receipt(private_receipt, local)
    client = api or GhCLI()
    operator = _validate_operator(client)
    _verify_remote(client, local, visibility="private")
    _assert_snapshot_unchanged(local, _local_snapshot(repo))
    try:
        client.set_visibility(FULL_NAME, "public")
        public_remote = _verify_remote(client, local, visibility="public")
        _assert_snapshot_unchanged(local, _local_snapshot(repo))
    except Exception as error:
        rollback_confirmed = False
        try:
            client.set_visibility(FULL_NAME, "private")
            _verify_remote(client, local, visibility="private")
            rollback_confirmed = True
        except Exception:
            rollback_confirmed = False
        failure_body = {
            "schema_version": PUBLIC_RECEIPT_SCHEMA,
            "status": (
                "PUBLIC_VERIFICATION_FAILED_ROLLED_BACK"
                if rollback_confirmed
                else "PUBLIC_VERIFICATION_FAILED_ROLLBACK_UNCONFIRMED"
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "repository": FULL_NAME,
            "source_revision": local["revision"],
            "source_tree": local["tree"],
            "inventory_sha256": local["inventory_sha256"],
            "private_stage_receipt_sha256": private_receipt_sha,
            "publication_performed": True,
            "public_verification_passed": False,
            "rollback_attempted": True,
            "rollback_confirmed": rollback_confirmed,
        }
        _write_once(public_receipt_path, _receipt(failure_body))
        if rollback_confirmed:
            raise GitHubReleaseError(
                "public verification failed; visibility was rolled back to private"
            ) from error
        raise GitHubReleaseError(
            "public verification failed and rollback is unconfirmed; "
            "inspect immediately"
        ) from error
    body = {
        "schema_version": PUBLIC_RECEIPT_SCHEMA,
        "status": "PUBLIC_REVISION_VERIFIED",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repository": FULL_NAME,
        "visibility": "public",
        "default_branch": DEFAULT_BRANCH,
        "actor": operator["actor"],
        "source_revision": local["revision"],
        "source_tree": local["tree"],
        "commit_count": local["commit_count"],
        "inventory": local["inventory"],
        "inventory_sha256": local["inventory_sha256"],
        "files": local["files"],
        "bytes": local["bytes"],
        "security_scan": local["security_scan"],
        "security_scan_sha256": local["security_scan_sha256"],
        "remote_revision": public_remote["revision"],
        "remote_tree": public_remote["tree"],
        "remote_inventory_sha256": public_remote["inventory_sha256"],
        "private_stage_receipt_sha256": private_receipt_sha,
        "network_used": True,
        "publication_performed": True,
        "public_verification_passed": True,
        "rollback_attempted": False,
        "rollback_confirmed": False,
    }
    receipt = _receipt(body)
    _write_once(public_receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="validate locally without network")
    plan.add_argument("--repo", type=Path, default=Path("."))
    plan.add_argument("--output", type=Path)

    private = subparsers.add_parser(
        "stage-private", help="create, push, and verify a private repository"
    )
    private.add_argument("--repo", type=Path, default=Path("."))
    private.add_argument("--receipt", type=Path, required=True)
    private.add_argument("--confirm", required=True)

    public = subparsers.add_parser(
        "publish-public", help="reverify private state, publish, and reverify"
    )
    public.add_argument("--repo", type=Path, default=Path("."))
    public.add_argument("--private-receipt", type=Path, required=True)
    public.add_argument("--public-receipt", type=Path, required=True)
    public.add_argument("--confirm", required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.command == "plan":
        result = build_plan(arguments.repo)
        if arguments.output:
            _ensure_receipt_outside_repo(arguments.output, arguments.repo)
            _write_once(arguments.output, result)
    elif arguments.command == "stage-private":
        result = stage_private(
            repo=arguments.repo,
            receipt_path=arguments.receipt,
            confirmation=arguments.confirm,
        )
    else:
        result = publish_public(
            repo=arguments.repo,
            private_receipt_path=arguments.private_receipt,
            public_receipt_path=arguments.public_receipt,
            confirmation=arguments.confirm,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
