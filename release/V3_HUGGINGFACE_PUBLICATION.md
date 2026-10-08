# Local Hugging Face verification helper

This page is retained only to explain the non-mutating compatibility helper
in `tools/v3_huggingface_release.py`. It is **not** an operator publication
runbook.

The sole supported private-stage and public-transition workflow is
[`MODAL_HUGGINGFACE_PUBLICATION.md`](MODAL_HUGGINGFACE_PUBLICATION.md). That
workflow uploads the exact qualified package plus the exact 14-file,
release-ready public projection materialized on the result volume. Mixing it
with the older package-only transport would produce a different inventory and
is rejected.

## Local command boundary

The local CLI exposes one command: `plan-private`. It performs no network
access and no repository mutation:

```bash
uv run --frozen --extra release python tools/v3_huggingface_release.py \
  plan-private \
  --package /release/commerce-1-v3 \
  --public-repo "$PWD" \
  --authorization /protected/commerce-1-v3-owner-authorization.json \
  --gate /protected/commerce-1-v3-publication-gate.json \
  --private-plan /protected/commerce-1-v3-private-plan.json \
  --output /protected/commerce-1-v3-hf-private-stage-plan.json
```

The local CLI has no `stage-private` or `publish-public` command. The
underlying transport functions require explicitly injected API and downloader
objects so hermetic tests can exercise verification and rollback behavior;
normal callers fail before authentication or network access.

For an actual release, follow the Modal runbook from its first step through
`publish_public_modal`. Do not substitute this local helper for any mutation
step.
