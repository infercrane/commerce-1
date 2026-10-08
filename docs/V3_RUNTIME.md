# Commerce-1 native runtime

This is the InferCrane-owned local-serving path for a self-contained
Commerce-1 release package. Superseded serving implementations and their
runtime tuples are intentionally excluded from the public source projection.

## Required package layout

Pass the same immutable package root as both `--model-dir` and `--base-dir`.
The runtime accepts only this content-bound layout:

```text
runtime.json
model/
calibration/calibration.json
source/autojev/__init__.py
source/autojev/model.py
source/autojev/types.py
evidence/public-evaluation.json
evidence/checkpoint-transform-attestation.json
```

`runtime.json` must use
`infercrane-commerce-1-native-autojev-runtime/v1`. Its semantic
`runtime_identity_sha256` binds the exact checkpoint file map and artifact
identity, calibration file and semantic identity, AutoJev source closure, and
the evaluated-to-release checkpoint transform plus content-free public
evaluation attestation. Protected evaluator, spend, and
provider records are not distributed. Unknown fields, unsafe paths, symlinks,
changed bytes, mutable
model revisions, missing checkpoint files, calibration regressions, and source
closure drift fail before model construction.

The native runtime requires Python 3.12 and the exact `v3-runtime` dependency
pins in `pyproject.toml` and `uv.lock`. It sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before importing AutoJev and initializes the model
from the verified local `model/` directory only. There is no network fallback.

The model is 27B BF16: weights alone require roughly 54 GiB before activation
and runtime overhead. The qualified configuration is Linux, the locked CUDA
dependency tuple, and one NVIDIA H200. Plan additional GPU memory plus enough
disk and host RAM to stage the checkpoint. Macs, CPU-only systems, and 24/48
GiB GPUs are not supported by the qualified path. A different NVIDIA GPU is an
unqualified configuration until its own native qualification receipt exists.

## HTTP contract

- `GET /healthz` returns liveness only.
- `GET /readyz` (and `GET /health`) returns readiness plus the exact artifact,
  runtime, and evidence identities loaded in memory.
- `GET /v1/model` exposes immutable artifact, runtime, and evidence identities
  without local paths, credentials, or environment values.
- `GET /v1/models` (and `GET /models`) exposes OpenAI-style model discovery
  with the same exact identities. `created: 0` means the runtime manifest does
  not make a trusted creation-time claim.
- `POST /v1/systemone` accepts ordered Choice, Noul, and Score questions.
- `POST /api/alpha/decisions` is an alias over the same service and scorer for
  the OpenRouter Decisions wire shape.
- `POST /v1/chat/completions` (and `POST /chat/completions`) is a strict
  OpenAI envelope: one user message contains a JSON object with exactly
  `state` and `questions`; the assistant content is JSON containing `answers`.
  Both streaming and non-streaming responses expose the exact identities and
  measured input-token usage. Completion tokens are zero because the model
  scores options and does not generate text.

The response schema is `infercrane-commerce-1-systemone/v1`. Choice returns a
selected option and its ordered probability map. Noul returns `P(true)` as a
number. Score returns the probability-weighted zero-based ordinal, legend,
and probability map. The API reports measured input tokens but leaves cost
`null`; it does not invent billing or model-quality results.

Runtime qualification is external, immutable release evidence; the server does
not infer it from installed files or advertise an unbound qualification claim.
Installing or running this server is not release, publication, benchmark,
latency, or hardware-qualification evidence.

## Local command

```bash
uv sync --frozen --python 3.12 --extra v3-runtime
uv run --python 3.12 --frozen --extra v3-runtime commerce-1-v3-serve \
  --model-dir /absolute/path/to/immutable-v3-package \
  --base-dir /absolute/path/to/immutable-v3-package \
  --offline
```

Loopback is the default and non-loopback binding requires the explicit
`--allow-non-loopback` flag plus `--api-key-file`. The token file must be a
regular non-symlink owned by the process user, with no group or other access.
The CLI never accepts the token value in an argument or environment variable.

```bash
install -m 0600 /dev/null /run/secrets/commerce-1-api-key
# Write the opaque token using the deployment secret manager, not shell history.

uv run --python 3.12 --frozen --extra v3-runtime commerce-1-v3-serve \
  --model-dir "$COMMERCE_ONE_V3_BUNDLE" \
  --base-dir "$COMMERCE_ONE_V3_BUNDLE" \
  --offline \
  --host 0.0.0.0 \
  --port 8000 \
  --allow-non-loopback \
  --api-key-file /run/secrets/commerce-1-api-key
```

The primary integration remains `/v1/systemone` or
`/api/alpha/decisions`. Chat compatibility does not turn Commerce-1 into a
prose model and rejects ordinary chat text, system messages, sampling
parameters, and unknown fields instead of silently changing semantics.

For production topology, controls, exact-identity smoke commands, and the
remaining external launch gates, see
[`deploy/commerce-1-v3/README.md`](../deploy/commerce-1-v3/README.md).
