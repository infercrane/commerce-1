# Commerce-1 production serving plan

This runbook is a deployment boundary, not proof that a deployment exists.
Do not publish a URL or submit a provider application until every launch gate
at the end is backed by a receipt for the same artifact and runtime identities.

## Topology

```text
client -> HTTPS edge -> private Commerce-1 process -> one pinned GPU
                              |
                              +-> read-only immutable release package
                              +-> owner-only API token file
```

Use a single model process per GPU. The server serializes access to the model
and returns `429` with `Retry-After` instead of creating an unbounded queue.
Scale with additional identical processes behind the edge only after the exact
replica tuple has passed runtime qualification.

The edge must:

- terminate TLS with a currently valid certificate;
- reject bodies over 1 MiB before forwarding;
- preserve SSE without buffering or compression;
- set connection and request deadlines, rate limits, and an in-flight cap;
- forward neither client certificates nor internal routing headers;
- never log authorization values, request bodies, response bodies, states,
  questions, or answers;
- expose only the intended API routes, not metrics or an administration port.

Load the model from a read-only, content-addressed package. Keep Hugging Face
and Transformers offline. The serving network policy needs no direct internet
egress after the image and package are staged. Keep writable compiler/cache
paths separate from the package and ephemeral; do not snapshot them as model
evidence.

## Secret boundary

Create a distinct random bearer token for this service in the deployment
secret manager. Mount it as a regular file owned by the unprivileged process
user with mode `0400` or `0600`. Never put it in:

- command arguments or environment variables;
- the image, model repository, package, logs, metrics, receipts, or crash dumps;
- a browser bundle or `NEXT_PUBLIC_*` configuration.

Non-loopback startup fails unless both `--allow-non-loopback` and an acceptable
`--api-key-file` are supplied. `/healthz`, `/readyz`, and `/health` are the only
unauthenticated routes; readiness exposes content hashes, never credentials or
local paths.

## Probes

- Liveness: `GET /healthz`, success only when the process can answer HTTP.
- Readiness: `GET /readyz`, success only after the exact model is loaded.
- Drain a replica before replacement; do not use liveness to infer readiness,
  benchmark quality, price, or GPU health.

After deployment, run the contract smoke from a trusted operator host. Supply
the three hashes from the signed release/qualification receipts, not from the
live endpoint:

```bash
uv run --frozen --extra test python tools/v3_api_smoke.py \
  --base-url https://commerce-api.infercrane.com \
  --api-key-file /run/secrets/commerce-1-smoke-token \
  --model-revision "$EXPECTED_ARTIFACT_SHA256" \
  --runtime-sha256 "$EXPECTED_RUNTIME_SHA256" \
  --evidence-sha256 "$EXPECTED_EVIDENCE_SHA256"
```

The smoke checks liveness, readiness, model discovery, OpenRouter Decisions,
OpenAI non-streaming, OpenAI SSE termination, exact identity continuity, and
measured usage. Its `PASS_API_CONTRACT_SMOKE` receipt explicitly sets quality,
performance, and production qualification claims to false.

## OpenRouter boundary

This repository now supplies the provider-local mechanics required for
OpenAI clients: `/v1/chat/completions`, streaming, usage on streaming and
non-streaming responses, and `/v1/models`. The primary typed-decision API is
also available at `/api/alpha/decisions`.

The local `/models` response is discovery metadata, not an OpenRouter
commercial provider catalog. Pricing, context limit, max-output declaration,
datacenter locations, public privacy/data-retention policy, invoicing, and
operator contact details belong in the InferCrane gateway/control plane after
they are real. Do not add placeholders or zero-price claims to this runtime.

## Launch gates

All are required for a public production claim:

1. The final frozen evaluation passes the release threshold and its exact
   evidence file is bound by `runtime.json`.
2. The exact public package and runtime bytes pass GPU runtime qualification;
   no private/local substitute is accepted.
3. The public GitHub and Hugging Face revisions are immutable and independently
   downloaded, hashed, and matched to the release receipt.
4. The HTTPS deployment passes `tools/v3_api_smoke.py` with those hashes.
5. Load, sustained streaming, error, restart, drain, and failover tests pass on
   the named production hardware and region.
6. Monitoring covers availability, latency, saturation, GPU memory, 429/5xx
   rates, restarts, and identity drift without customer content.
7. Pricing, capacity, monthly invoicing, privacy, retention, abuse handling,
   and incident ownership are approved and published.
8. OpenRouter runs its own integration test and accepts the provider/model.

Until those gates pass, use the terms “local candidate runtime” and “API
contract smoke,” not “production qualified,” “world class,” or “available on
OpenRouter.”
