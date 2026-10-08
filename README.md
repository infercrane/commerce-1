# Commerce-1

A 27B, self-hosted decision model for commerce agents. Give Commerce-1 one
structured state and ordered `Choice`, `Noul`, or `Score` questions; it returns
calibrated probability distributions and typed answers—not prose—through an
offline, content-verified runtime.

[![CI](https://github.com/infercrane/commerce-1/actions/workflows/ci.yml/badge.svg)](https://github.com/infercrane/commerce-1/actions/workflows/ci.yml)
[![Weights](https://img.shields.io/badge/weights-Hugging%20Face-yellow)](https://huggingface.co/infercrane/Commerce-1)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

## Highlights

- **Typed decisions, not generated prose.** Choice selection, binary
  probability (`Noul`), and ordinal scoring share one request contract.
- **Many questions, one state.** Score several agent actions against the same
  structured context through one request and one shared-state contract.
- **Probabilities included.** Route uncertain actions to a person, policy
  engine, or a more expensive model instead of treating every answer equally.
- **Offline and self-hosted.** The native runtime verifies every package byte
  before loading and rejects network fallback.
- **Evidence before claims.** Evaluation and runtime qualification are bound to
  immutable artifacts; missing evidence remains unknown.

## At a glance

| | Commerce-1 |
|---|---|
| Model | Full-checkpoint 27B BF16 typed decision model |
| Input | JSON state plus ordered Choice, Noul, and Score questions |
| Output | Typed answers plus raw ordered probability distributions |
| Input limit | 65,536 tokens, with fail-closed overflow handling |
| APIs | `/v1/systemone`, `/api/alpha/decisions`, strict JSON-only `/v1/chat/completions` |
| Qualified path | Linux, one NVIDIA H200, exact locked runtime tuple |
| Weights | [`infercrane/Commerce-1`](https://huggingface.co/infercrane/Commerce-1) |

## Quick start

Use the immutable 40-character source and model revisions from the verified
release receipts. Floating branches and model revisions are not part of the
qualified path.

Commerce-1's BF16 weights are approximately 54 GiB. Allow additional GPU
memory for activations and enough disk and RAM to stage the checkpoint. CPU,
macOS, and 24/48 GiB GPU paths are not qualified by this release.

```bash
export COMMERCE_ONE_SOURCE_REVISION=<verified-github-revision>
export COMMERCE_ONE_MODEL_REVISION=<verified-huggingface-revision>

git clone https://github.com/infercrane/commerce-1.git
cd commerce-1
git checkout --detach "$COMMERCE_ONE_SOURCE_REVISION"
test "$(git rev-parse HEAD)" = "$COMMERCE_ONE_SOURCE_REVISION"

uv sync --frozen --python 3.12 --extra v3-runtime
uv run --frozen --python 3.12 --extra v3-runtime hf download infercrane/Commerce-1 \
  --revision "$COMMERCE_ONE_MODEL_REVISION" \
  --local-dir ../commerce-1-model

export COMMERCE_ONE_MODEL_DIR="$(cd ../commerce-1-model && pwd)"
uv run --frozen --python 3.12 --extra v3-runtime commerce-1-v3-serve \
  --model-dir "$COMMERCE_ONE_MODEL_DIR" \
  --base-dir "$COMMERCE_ONE_MODEL_DIR" \
  --offline \
  --host 127.0.0.1 \
  --port 8000
```

## Make a decision

```bash
curl --fail-with-body --silent --show-error \
  -H 'content-type: application/json' \
  --data '{
    "model":"infercrane/commerce-1",
    "state":{"cart_total":117,"approved_total":94},
    "questions":{
      "route":{
        "type":"choice",
        "instructions":"Choose the safest next action.",
        "criteria":{
          "continue":"Continue.",
          "review":"Ask for review.",
          "block":"Stop."
        }
      },
      "review":{
        "type":"noul",
        "instructions":"Is review required?"
      },
      "risk":{
        "type":"score",
        "instructions":"Score commerce risk.",
        "criteria":["low","guarded","high","block"]
      }
    }
  }' \
  http://127.0.0.1:8000/v1/systemone
```

| Primitive | You provide | Commerce-1 returns |
|---|---|---|
| `Choice` | Named candidate actions and their criteria | Selected action plus a probability for every action |
| `Noul` | One yes/no decision instruction | Boolean answer plus `P(true)` |
| `Score` | Ordered labels from low to high | Expected ordinal score plus a probability for every label |

Probabilities help callers set review thresholds, but they are not guarantees
of correctness. Recalibrate on representative traffic before using confidence
for consequential automation.

## Measured evidence

| Evidence | Result | Boundary |
|---|---:|---|
| Decision Index 0.2.1 | **60.99** | Self-hosted complete public-suite reproduction |
| Coverage | **150,317 / 150,317** | Every merged row completed |
| Native/runtime parity | **Exact** | 9 comparisons, maximum absolute difference `0.0` |
| Release/evaluated parity | **Exact** | 9 comparisons, maximum absolute difference `0.0` |
| Qualified hardware | **1× NVIDIA H200** | BF16, CUDA 13.0, PyTorch 2.14.0 |
| Qualified throughput | **8.836 decisions/s** | Three-decision fixture set, one-question microbatch |
| Qualified latency | **340.146 ms p50 / 349.026 ms p95** | Per complete three-decision fixture set |

The Decision Index result is not an official 0.3 Full Score or leaderboard
rank. Those remain pending maintainer evaluation against the immutable public
release. Exact content-free bindings live in
[`release/commerce-1-evidence.json`](release/commerce-1-evidence.json); the
public model package includes `QUALIFICATION.json` for the runtime measurement.

Validate the checked-in evidence snapshot:

```bash
uv run --frozen --python 3.12 --extra test \
  python tools/v3_release_evidence.py
```

## Where it fits

Commerce-1 is designed for structured decisions inside commerce-agent
workflows, including:

- proceed, review, or block gates;
- tool and workflow routing;
- approval-policy drift checks;
- product, fulfillment, and support action selection;
- calibrated risk or priority scoring.

It is not a chat model, search engine, live-fact retriever, payment processor,
or autonomous authority for refunds, disputes, fraud, credit, safety, or legal
decisions. A benchmark result does not establish fitness for a production
policy. Validate the model on your own distribution and keep consequential
actions behind explicit controls.

## How it works

Commerce-1 starts from
[`perplexity-ai/pplx-decider-v1.1-27b`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and uses a noncausal full-attention decision backbone with last-position
pooling. A decision readout scores all supplied answer options directly,
without decoding an explanation token by token. Choice-order augmentation and
per-primitive held-out temperature calibration make the typed probabilities
usable by downstream policy code.

The runtime inventories and hashes every required package file before model
load. It rejects symlinks, undeclared files, mutable repository identities,
changed checkpoint bytes, calibration drift, source-closure drift, unknown
manifest fields, and network fallback. See
[`docs/V3_RUNTIME.md`](docs/V3_RUNTIME.md) for the exact contract.

The same scorer is available at `POST /api/alpha/decisions`. A strict
OpenAI-compatible JSON bridge is available at `POST /v1/chat/completions` for
clients that cannot call the native route. Commerce-1 does not generate text:
the user message contains one JSON decision object and the assistant content
contains the typed answers as JSON.

## Serving boundary

Loopback is the default. Non-loopback serving requires both
`--allow-non-loopback` and `--api-key-file`. The key file must be a regular,
non-symlink file owned by the process user with mode `0600`; token values are
not accepted in command arguments or environment variables.

Useful routes:

- `GET /healthz` — process liveness only.
- `GET /readyz` — readiness and immutable loaded identities.
- `GET /v1/model` — content-free model, runtime, and evidence identity.
- `GET /v1/models` — OpenAI-style discovery with the same identities.

Responses report measured input tokens and zero generated tokens. Cost remains
`null` until backed by real metering. See [`docs/API.md`](docs/API.md).

## Development

```bash
uv sync --frozen --python 3.12 --extra test
uv run --frozen --python 3.12 --extra test pytest -q
uv run --frozen --python 3.12 --extra test ruff check .
uv run --frozen --python 3.12 --extra test ruff format --check .
```

Publication is private-first and exact-byte verified. The operator contracts
are documented in [`release/V3_PUBLICATION.md`](release/V3_PUBLICATION.md),
[`release/MODAL_HUGGINGFACE_PUBLICATION.md`](release/MODAL_HUGGINGFACE_PUBLICATION.md),
and [`release/GITHUB_PUBLICATION.md`](release/GITHUB_PUBLICATION.md).

## License

The source code in this repository is licensed under Apache-2.0. Model and
training-source notices are documented separately in
[`THIRD_PARTY.md`](THIRD_PARTY.md). The source-code license does not override
upstream model or dataset terms.
