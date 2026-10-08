# Commerce-1 third-party notices

Audit date: 2026-10-08.

This file covers the exact Commerce-1 launch checkpoint. It is a provenance
and attribution record, not legal advice. The release package includes this
file, the repository Apache-2.0 `LICENSE`, the byte-exact evidence listed
below, and full license terms under `licenses/`.

## Model lineage

Commerce-1 continues training
`perplexity-ai/pplx-decider-v1.1-27b@3b45dead91dfa6d95aad6b95764a606fab2bf7a6`,
which derives from
`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Both upstream
repositories declare Apache-2.0.

Commerce-1 updates the governed checkpoint weights on the six-source mixture
below, retains the noncausal decision architecture and 255-option readout,
and fits a new calibration temperature. The upstream Apache-2.0 license and
the exact pplx-decider NOTICE are retained in `licenses/models/`. The upstream
license file uses CRLF line endings at both pinned revisions (SHA-256
`bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`);
the bundled LF-normalized copy has SHA-256
`689f0c220c9b4e857f35ca7d0dca51397e5a545dc4ad3987438c023e24ff0ab6`.
The exact pplx-decider NOTICE has SHA-256
`3792cbe9e964a7ed65fff010d3f94bb27c6e726e2940ab6b0e1db5d30df92510`.

## Training sources

The authoritative machine-readable six-source list is `PROVENANCE.json` in
the release package. During assembly, protected prepared-data, merge, and
source-rights records are validated against the checkpoint. The public package
retains only their content-free, hash-bound lineage attestation; operator
identities, authorization text, provider operation IDs, and budgets remain in
protected release control storage.

### InferCrane procedural data

- `infercrane-broad-procedural-v1`
- `infercrane-commerce-procedural-v3`

These are first-party generated datasets licensed under Apache-2.0. The
repository `LICENSE` is the retained license byte (SHA-256
`8a2ffa9d3338c1bdc3100a804dc669b97db139e674be10bfb642c020c482d9c6`).
They were generated, split, rendered as typed Choice/Noul/Score decisions,
and combined for Commerce-1 continued training.

### NVIDIA HelpSteer2

Source: `nvidia/HelpSteer2@990b2711a36180dd19d9c94b8627844866f8982a`.
The source declares CC BY 4.0 and is attributed to NVIDIA, Scale AI, Zhilin
Wang, Alexander Bukharin, Olivier Delalleau, Daniel Egert, Gerald Shen, Jiaqi
Zeng, Oleksii Kuchaiev, Yi Dong, Jimmy J. Zhang, and Makesh Narsimhan
Sreedhar. Commerce-1 uses a selected subset transformed from ratings and
preferences into typed-decision rows, changing the original format and task.

The exact pinned dataset card is retained at
`licenses/training/nvidia-helpsteer2-README.md` (SHA-256
`835effb9e7d9cd0e8b7b8c1816d1d97a8961a036543108e4a0b6a5e22712ff7b`).
The CC BY 4.0 legal code is retained at `licenses/terms/CC-BY-4.0.txt`.

### Anthropic HH-RLHF preference-model data

Source: `Anthropic/hh-rlhf@09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa`.
The source declares MIT and is attributed to Anthropic. Commerce-1 uses
selected preference-model examples transformed into typed decisions. It does
not use the dataset as dialogue supervised fine-tuning and does not use the
red-team transcripts as preference-model data.

The exact pinned dataset card is retained at
`licenses/training/anthropic-hh-rlhf-README.md` (SHA-256
`f75f40db0268656ba07736ec8e59a9720c1910ce554c85354cec74b1c8bda175`).
The MIT license from the primary source repository at immutable revision
`anthropics/hh-rlhf@c72f5cee8eb7b4d2ea5617657f4430d5e333af07` is retained at
`licenses/terms/anthropic-hh-rlhf-MIT-LICENSE` (SHA-256
`d4fb18db48757e273261a5597b8a09381de8073da060474ce33d0643e06c875e`).

### AllenAI QASC

Source: `allenai/qasc@a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`.
The source declares CC BY 4.0 and is attributed to Tushar Khot, Peter Clark,
Michal Guerquin, Peter Jansen, and Ashish Sabharwal. Commerce-1 uses selected
question-answer material transformed into typed decisions, a modification of
the original QASC task.

The exact pinned dataset card is retained at
`licenses/training/allenai-qasc-README.md` (SHA-256
`c90ae4776b98cd6fc0a22558784c43b07e1f0b00a090dda99797d6b6efab4a3e`).
The CC BY 4.0 legal code is retained at `licenses/terms/CC-BY-4.0.txt`.

### Kubernetes OpenAPI material

Source: `kubernetes/kubernetes@e7967bb9b76d43a6388abb7a4e90d2f899ef97ff`.
The source is Apache-2.0 and is attributed to The Kubernetes Authors.
Commerce-1 uses selected API descriptions transformed into typed tool
decisions. The exact upstream license is retained at
`licenses/training/kubernetes-LICENSE` (SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

## Runtime and tooling

The release package includes the exact native serving source used for
qualification. Its dependency versions are pinned by the public runtime lock.
Torch, Transformers, Accelerate, Flash Linear Attention, Triton, FastAPI,
Uvicorn, Safetensors, NumPy, Pillow, Pydantic, Hugging Face Hub, and their
transitive dependencies retain their own upstream licenses. They are runtime
dependencies, not training-data sources.

Jebadiah and Kev were protocol, evaluation, or earlier experiment references.
They are not launch-checkpoint training sources. Public benchmark and
independent commerce evaluation datasets are evaluation-only and are not
ingredients of the released weights.

No affiliation with or endorsement by NVIDIA, Scale AI, Anthropic, AllenAI,
Kubernetes, Perplexity AI, Qwen, TypeSafe, or the Decision Index maintainers
is implied.
