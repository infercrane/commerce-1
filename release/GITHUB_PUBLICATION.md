# GitHub publication

Commerce-1 source is published to exactly `infercrane/commerce-1`. The workflow
is private first: create an empty private repository, push one squashed root
commit, verify the remote commit/tree/file inventory, and only then change
visibility. After the visibility change, the same immutable revision and every
Git blob are verified again.

`tools/github_release.py` never accepts a token. Authenticate `gh` through its
normal credential store before running these commands. Receipts must be placed
outside this repository and should be retained with the release evidence.

## 1. Freeze one public commit

Finish the model package, runtime qualification, documentation, notices, and
tests before freezing source. The release repository must be clean, on `main`,
and contain exactly one root commit. Amend the existing release-candidate
commit instead of adding public history. Both the author and committer must use
an `InferCrane...` name and an `@infercrane.com` email address; personal commit
metadata is a hard publication failure:

```bash
git status --short
git config user.name "InferCrane Release"
git config user.email "release@infercrane.com"
git add --all
git commit --amend --no-edit --reset-author
uv run --python 3.12 --frozen --extra test pytest -q
uv run --python 3.12 --frozen --extra test ruff check tools/github_release.py tests/test_github_release.py
```

The single-commit rule prevents previous candidate bytes, removed credentials,
or stale legal claims from becoming reachable through public Git history.

## 2. Local plan (no network)

```bash
uv run --python 3.12 --frozen python tools/github_release.py plan \
  --repo . \
  --output ../commerce-1-github-plan.json
```

This fails on dirty/untracked files, another branch, multiple commits, any
branch or tag besides `main`, symlinks, submodules, special files, or an
in-repository receipt path. It also scans every blob reachable from every
local public ref, not just the checked-out worktree. The scan rejects known
credential forms, personal owner identity, internal Modal paths/resource IDs,
generated artifacts, and raw historical training-data evidence. Its exact
commit identity, object counts, scanned byte count, and policy are bound into
the plan receipt.

### Safe public source inventory

The source projection is intentionally smaller than the protected release
workspace. Its allowed content is:

- root project metadata and deterministic dependency locks;
- `.github/` workflows and `deploy/` manifests that contain no credentials;
- public API, runtime, and publication documentation under `docs/` and
  `release/`;
- redistributable notices under `licenses/` and `THIRD_PARTY.md`;
- native runtime and server source under `src/infercrane_commerce_1/`;
- current qualification/publication utilities under `tools/`; and
- their content-free tests under `tests/`.

The plan receipt's sorted `inventory` is the canonical file-by-file public
inventory. A reviewer must compare that list to these categories before any
private stage. Top-level categories do not override the forbidden paths below,
and any unexpected file is reason to stop and review rather than widen the
projection.

Raw or superseded evidence belongs outside this source repository. In
particular, `release/private/`, `release/internal/`, `release/training-data/`,
`release/upstream/`, `huggingface/`, `artifacts/`, `dist/`, `vendor/`, and local
recovery/tool-state directories can never enter the public inventory. The
effective-v4 publication scripts, templates, provenance lock, and historical
AutoJev notices are also rejected by exact path. The superseded Dockerfile,
`runtime/`, `src/commerce_one/`, and their qualification receipts are rejected
until a fresh V3 container is built and qualified. Current V3 training lineage
and measured results must be represented only by the content-free summaries
validated by the V3 publication tools. Protected authorization, spend,
provider-operation, evaluator-attempt, and filesystem-location receipts
remain in the private release archive.

## 3. Private stage and exact verification

The target must not already exist. This prevents inheriting unverified Git
history. The authenticated GitHub account must be an active InferCrane admin.

```bash
uv run --python 3.12 --frozen python tools/github_release.py stage-private \
  --repo . \
  --receipt ../commerce-1-github-private.json \
  --confirm STAGE_INFERCRANE_COMMERCE_1_GITHUB_PRIVATE
```

The repository remains private if push or verification fails. Do not continue
without a self-hashed receipt whose status is `PRIVATE_REVISION_VERIFIED`.

## 4. Public transition and exact re-verification

```bash
uv run --python 3.12 --frozen python tools/github_release.py publish-public \
  --repo . \
  --private-receipt ../commerce-1-github-private.json \
  --public-receipt ../commerce-1-github-public.json \
  --confirm PUBLISH_INFERCRANE_COMMERCE_1_GITHUB_PUBLIC
```

The command rechecks private visibility and exact bytes before mutation. It
then changes visibility and verifies repository identity, public visibility,
commit, tree, modes, sizes, and blob IDs again. If public verification fails,
it attempts to restore private visibility and writes a failure receipt. A
rollback-unconfirmed receipt is an immediate operator incident; do not announce
the release or delete private release evidence.

Publication is complete only when the public receipt reports
`PUBLIC_REVISION_VERIFIED` and the public revision equals the source revision
used by the Hugging Face and benchmark release controls.
