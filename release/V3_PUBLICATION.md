# Commerce-1 publication

## Current evidence snapshot

The frozen public Decision Index 0.2.1 evaluation is complete: **60.99**, with
150,317/150,317 merged rows successful. Public result and score hashes are
recorded without publishing the protected evaluator receipt.
This public reproduction is not an official Decision Index 0.3 Full Score or a
current leaderboard rank. Decision Index 0.3 evidence remains pending a
sanitized, content-bound attestation. Validate the checked-in content-free
snapshot with:

```bash
uv run --frozen --extra test python tools/v3_release_evidence.py
```

The final package was rebuilt after runtime-contract hardening and passed a
fresh native runtime qualification with `maximum_input_tokens=65536` and
`question_microbatch_size=1`. The checked-in evidence snapshot binds only the
content-free release artifact, transform, and public qualification attestation
hashes; provider and attempt identities remain protected. Private remote
verification and the public visibility transition remain separate fail-closed
launch gates below. Official private evaluation remains post-launch evidence
because the maintainers require immutable public model, runtime, and result
links.

Commerce-1 is published only by the `infercrane` organization. The fixed
targets are:

- GitHub: `infercrane/commerce-1`
- Hugging Face: `infercrane/Commerce-1`

The V3 release tool never creates a repository, uploads bytes, changes
visibility, or publishes a release. It produces owner-only, self-hashed plans
after checking evidence. Remote mutation belongs in a separately approved CI
job that consumes those plans and returns byte-verification receipts.

A self-hash proves byte integrity, not operator identity. Generate and retain
these receipts only inside an InferCrane-controlled release environment. The
mutation job must authenticate the operator and verify remote identity through
its own protected CI/OIDC boundary; a locally constructed receipt is not
publication authority.

## Initial public-release evidence

The gate rejects a candidate unless all of the following bind to the same
artifact:

- a hermetic V3 release package, including content-free training and evaluation
  attestations, their explicit lineage bridge, the native runtime, and every
  model/runtime byte needed for local execution;
- the v3 native runtime qualification receipt;
- a separate V3 owner publication authorization with every known blocker
  either resolved or explicitly deferred until immutable public URLs exist; and
- a public-runtime inventory byte-identical to this repository.

The package's complete public Decision Index 0.2.1 score must be strictly above
the frozen `52.55` baseline. For this artifact it is exactly `60.99`, bound to
the hashes in `release/commerce-1-evidence.json`. Initial authorization records
`public_v03_final_sha256`, `official_v03_attestation_sha256`, v0.3 Full Score,
and rank as `null`; it cannot imply an official board result.

After initial publication, run the public v0.3 continuation and submit the
immutable public assets for maintainer evaluation. Re-running `authorize` and
`gate` with both `--public-v03` and `--official-v03` produces the stricter
`OFFICIAL_V0_3_VERIFIED` upgrade state. Supplying only one is rejected. The
official Full Score must reproduce from its components and remain above the
same baseline. `63.3` is a target, not a manufactured claim.

Absolute build-machine references are not accepted as substitutes for
packaged bytes. Protected governance records, raw authorization text, spend
records, provider identifiers, and orchestration source are validated during
assembly but are never copied into the public package. Its immutable
provenance must continue to say
`legal_conclusion: false` and `publication_approval: false`: it is a
training-time disclosure, not a mutable release switch.

## 1. Create the exact owner authorization

Only an owner-intended invocation may create this receipt. For the initial
release, the generator validates the complete package, exact 0.2.1 result, and
runtime qualification. It then binds the exact contract, artifact,
calibration, evidence, runtime identity, mandatory notices, and InferCrane
ownership. It records `legal_conclusion: false`; authorization is not a legal
opinion.

The receipt binds a canonical SHA-256 of the protected owner authorities. The
verbatim authorization statements and operator identity stay in protected
release-control storage and are not copied into the public repository,
package, or receipts.

```bash
uv run --frozen --extra test python tools/v3_publication.py authorize \
  --package /release/commerce-1-v3 \
  --public-repo "$PWD" \
  --qualification /evidence/native-runtime-qualification.json \
  --confirmation AUTHORIZE_INFERCRANE_COMMERCE_1_V3_RELEASE \
  --authorized-at-utc 2026-10-07T00:00:00Z \
  --output /protected/commerce-1-v3-owner-authorization.json
```

The timestamp must be the real invocation time. The authorization contains a
fixed blocker-disposition map. Quality, runtime, lineage, and notice evidence
must pass before it can be created. Official v0.3 is recorded as deferred—not
passed—until public immutable endpoints exist.

## 2. Build the evidence gate

```bash
uv run --frozen --extra test python tools/v3_publication.py gate \
  --package /release/commerce-1-v3 \
  --public-repo "$PWD" \
  --qualification /evidence/native-runtime-qualification.json \
  --authorization /protected/commerce-1-v3-owner-authorization.json \
  --output /protected/commerce-1-v3-publication-gate.json
```

Do not mutate immutable training provenance or edit unknown facts merely to
make the command pass.

## 3. Plan a private rehearsal

Use the exact committed GitHub revision. A branch name or tag is rejected.

```bash
uv run --frozen --extra test python tools/v3_publication.py private-plan \
  --gate /protected/commerce-1-v3-publication-gate.json \
  --source-revision "$(git rev-parse HEAD)" \
  --output /protected/commerce-1-v3-private-plan.json
```

The approved CI job must stage both targets as **private**, then independently
download/clone immutable revisions and compare the complete inventories. It
must emit one self-hashed
`infercrane-commerce-1-v3-private-remote-verification/v1` receipt per target.
The Hugging Face inventory must equal the package inventory from the plan; the
GitHub revision must equal the planned source revision.

## 4. Verify private remotes

```bash
uv run --frozen --extra test python tools/v3_publication.py verify-private \
  --plan /protected/commerce-1-v3-private-plan.json \
  --github /protected/github-private-verification.json \
  --huggingface /protected/huggingface-private-verification.json \
  --output /protected/commerce-1-v3-private-verification.json
```

The remote receipts must record `visibility: private`, immutable 40-character
revisions, full inventory hashes, and the exact planned artifact. Tokens must
come from owner-only files or CI secret stores, never arguments, logs, or the
release package.

## 5. Authorize—not perform—the public transition

```bash
uv run --frozen --extra test python tools/v3_publication.py public-plan \
  --gate /protected/commerce-1-v3-publication-gate.json \
  --private-verification /protected/commerce-1-v3-private-verification.json \
  --output /protected/commerce-1-v3-public-plan.json
```

The result remains `PUBLIC_TRANSITION_AUTHORIZED_NOT_PERFORMED`. A separate
human-approved job may change visibility only after reviewing that plan. After
the transition, users must install by the two immutable revisions described in
the repository README; `main`, `latest`, and unpinned model downloads are not
release instructions.

For the authenticated, private-first Hugging Face mutation and immutable-byte
verification sequence, follow
[`MODAL_HUGGINGFACE_PUBLICATION.md`](MODAL_HUGGINGFACE_PUBLICATION.md). It is
the sole operator mutation path, consumes these receipts, and verifies the
combined package plus exact public-assets inventory.
