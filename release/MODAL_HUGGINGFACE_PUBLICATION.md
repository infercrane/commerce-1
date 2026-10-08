# Commerce-1 server-side Hugging Face release

`tools/modal_v3_huggingface_release.py` publishes directly from the qualified
package on `infercrane-commerce-v3-clean-results-v1`. It does not download or
copy the checkpoint to the operator laptop and does not write upload metadata
inside the immutable package.

The fixed destination is `infercrane/Commerce-1`. Planning performs no Hub
mutation. Private staging can only create or resume a private repository.
Public visibility is a separate operation that requires the complete GitHub +
Hugging Face private-verification chain and rolls back to private if the pinned
revision cannot be verified after the transition.

## Inputs

Set these after the final native qualification succeeds:

```bash
PACKAGE_RELATIVE='runs/<contract>/runtime-qualification/<attempt>/package'
QUALIFICATION_RELATIVE='runs/<contract>/runtime-qualification/<attempt>/FINAL.json'
CONTROL_RELATIVE='runs/<contract>/publication/<artifact>'
ASSETS_RELATIVE='runs/<contract>/runtime-qualification/<attempt>/public-assets'
SOURCE_REVISION='<exact-40-character-lowercase-public-repository-commit>'
AUTHORIZED_AT_UTC='<real-current-UTC-time, for example 2026-10-08T12:34:56Z>'
```

The server-side release materializer writes the public assets beside the
qualification as
`runs/<contract>/runtime-qualification/<attempt>/public-assets` on
`infercrane-commerce-v3-clean-results-v1`. The uploader reads that exact result
volume path; it does not expect or embed a `huggingface/public` checkout tree.
The assets must be the exact 14-file release projection: seven rendered model
documents, `LICENSE`, `THIRD_PARTY.md`, `PROVENANCE.json`,
`publication-manifest.json`, `verified-evidence.json`, `QUALIFICATION.json`,
and `SHA256SUMS`. Extra files, including raw receipts, are rejected. Their
inventories, self-hashes, qualification, package bindings, and
`READY_FOR_OWNER_RELEASE_GATE` state are revalidated before planning. The card
must identify `# InferCrane Commerce-1` and `infercrane/Commerce-1`; unresolved
template markers and credential-shaped content are rejected. An asset may
overlap a package path only when its bytes are identical.

## 1. Create the owner controls on the result volume

Create the authorization, publication gate, and private plan beside the
qualified package. This CPU-only function has network access blocked, mounts
no secret, requests no GPU, creates no repository, uploads nothing, and cannot
publish. It validates the V5 package, baked public repository source,
protected qualification, and exact public-assets projection before writing:

- `owner-authorization.json`
- `publication-gate.json`
- `private-plan.json`

Use the exact checked-in public repository commit. Set `AUTHORIZED_AT_UTC` to
the real invocation time in UTC immediately before the command; non-UTC,
inferred, default, or timestamps outside the 15-minute clock-skew window are
rejected. The three outputs are mode `0600` and create-once. If any already
exists, the operation fails before reading the large package. The control
directory must be a safe, non-symlink path disjoint from the immutable
package, qualification, and public assets.

```bash
modal run tools/modal_v3_huggingface_release.py::create_release_controls_modal \
  --package-relative-path "$PACKAGE_RELATIVE" \
  --qualification-relative-path "$QUALIFICATION_RELATIVE" \
  --assets-relative-path "$ASSETS_RELATIVE" \
  --control-relative-path "$CONTROL_RELATIVE" \
  --confirmation AUTHORIZE_INFERCRANE_COMMERCE_1_V3_RELEASE \
  --source-revision "$SOURCE_REVISION" \
  --authorized-at-utc "$AUTHORIZED_AT_UTC"
```

The command returns only a self-hashed, content-free summary of the package,
qualification, asset inventory, and three receipt hashes. It does not return
authorization content, operator identity, credentials, internal paths, or
model bytes. Do not put credentials or tokens in any release-control file.

The Hub upload preserves the immutable checkpoint package at its original
paths. It also preserves the complete 14-file public projection under
`release-assets/`, where its `publication-manifest.json` and `SHA256SUMS`
remain a self-contained verification domain. User-facing documents are
additionally exposed at repository root. The package's different root
`SHA256SUMS` is never overwritten or silently dropped.

## 2. Bind the package and public assets

This reads and validates the release volume but does not contact Hugging Face:

```bash
modal run tools/modal_v3_huggingface_release.py::plan_private_modal \
  --package-relative-path "$PACKAGE_RELATIVE" \
  --assets-relative-path "$ASSETS_RELATIVE" \
  --control-relative-path "$CONTROL_RELATIVE"
```

It writes `hf-private-stage-plan.json` into the control directory. The plan
binds the package inventory, public-assets inventory, combined upload
inventory, model card, owner authorization, evidence gate, private rehearsal,
and source revision.

## 3. Stage privately and verify

```bash
modal run --detach tools/modal_v3_huggingface_release.py::stage_private_modal \
  --package-relative-path "$PACKAGE_RELATIVE" \
  --assets-relative-path "$ASSETS_RELATIVE" \
  --control-relative-path "$CONTROL_RELATIVE" \
  --confirmation STAGE_INFERCRANE_COMMERCE_1_PRIVATE_FROM_MODAL
```

The function uses the `infercrane-huggingface` Modal secret. It passes package
files directly to a Hub commit, so no second 51+ GiB tree is created. LFS
pre-upload uses eight threads. A retry verifies already-present files and
uploads only missing paths. The repository remains private throughout.

Success writes:

- `huggingface-private-verification.json`
- `huggingface-private-operation.json`

Both are written only after every remote file at the immutable revision passes
size and SHA-256 verification.

## 4. Complete private-first control evidence

Independently create and verify the private GitHub repository. Then add these
files to the same control directory:

- `github-private-verification.json`
- `private-verification.json`, reproduced with
  `tools/v3_publication.py verify-private`
- `public-plan.json`, produced with `tools/v3_publication.py public-plan`

The Hugging Face receipt can be retrieved without model bytes:

```bash
modal volume get infercrane-commerce-v3-clean-results-v1 \
  "$CONTROL_RELATIVE/huggingface-private-verification.json" \
  /protected/huggingface-private-verification.json
```

Upload only these completed post-stage control files with `modal volume put`.
Do not put credentials or tokens in any control file.

## 5. Make the verified revision public

```bash
modal run tools/modal_v3_huggingface_release.py::publish_public_modal \
  --package-relative-path "$PACKAGE_RELATIVE" \
  --assets-relative-path "$ASSETS_RELATIVE" \
  --control-relative-path "$CONTROL_RELATIVE" \
  --confirmation AUTHORIZE_INFERCRANE_COMMERCE_1_V3_RELEASE
```

This requires InferCrane organization admin access. It verifies the complete
private revision, changes only repository visibility, verifies that the same
revision and combined inventory are public, and writes
`huggingface-publication.json`. If post-transition verification fails, it
attempts and verifies a rollback to private visibility.

The Modal functions use CPU only. The private and public verification steps
reserve 512 GiB of ephemeral disk for byte-level remote verification, matching
the current Modal workspace minimum; no GPU is requested.
