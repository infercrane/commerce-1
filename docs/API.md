# Commerce-1 API

Commerce-1 is a typed decision model, not a prose generator. All inference
routes use the same loaded checkpoint, calibration, and scorer.

## Native decision request

`POST /v1/systemone` and `POST /api/alpha/decisions` accept the same request:

```json
{
  "model": "infercrane/commerce-1",
  "state": {"cart_total": 117, "approved_total": 94},
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the safest next action.",
      "criteria": {
        "continue": "Continue.",
        "review": "Ask for review.",
        "block": "Stop."
      }
    },
    "review": {
      "type": "noul",
      "instructions": "Is review required?"
    },
    "risk": {
      "type": "score",
      "instructions": "Score commerce risk.",
      "criteria": ["low", "guarded", "high", "block"]
    }
  }
}
```

The request preserves question and option order. Unknown fields, unknown model
IDs, empty criteria, unsupported question types, and requests past the byte or
token limit fail closed. The embedded JSON used by the OpenAI compatibility
route additionally rejects duplicate keys before typed validation.

## Native response

Successful responses use `infercrane-commerce-1-systemone/v1` and contain:

- immutable model, artifact, revision, runtime, and evidence identities;
- `answers` in request order;
- raw ordered probability distributions;
- measured input-token usage and zero output tokens; and
- `cost: null` until real metering exists.

`Choice` selects one named option and returns its ordered probability map.
`Noul` returns `P(true)`. `Score` returns the probability-weighted zero-based
ordinal, the criteria legend, and the ordered probability map. The server does
not replace or renormalize model probabilities.

## OpenAI compatibility

`POST /v1/chat/completions` and `POST /chat/completions` provide a strict JSON
bridge for clients that require the OpenAI envelope. Exactly one `user`
message is accepted. Its content must be a JSON object with exactly `state`
and `questions`; ordinary chat text, system messages, sampling parameters, and
unknown fields are rejected.

The assistant content is compact JSON containing `answers`. Both buffered and
streaming responses expose the exact runtime and evidence identities. The SSE
route ends with `data: [DONE]` and disables proxy buffering. Token usage is
measured; generated-token count is zero because Commerce-1 scores candidates
instead of decoding prose.

## Discovery and health

- `GET /healthz` returns process liveness only.
- `GET /readyz` and `GET /health` return readiness plus loaded identities.
- `GET /v1/model` returns the content-free immutable model identity.
- `GET /v1/models` and `GET /models` return OpenAI-style discovery.

`created: 0` in model discovery is an explicit unknown sentinel, not a claimed
artifact timestamp.

## Errors

The server uses stable JSON error codes and never includes exception text,
local paths, environment values, or credentials:

- `400 invalid_request` — malformed JSON or unsupported chat bridge input;
- `401 unauthorized` — missing or incorrect key on protected deployments;
- `413 request_too_large` or `context_limit_exceeded`;
- `415 invalid_request` — unsupported media type or content encoding;
- `422 unsupported_question` — typed request validation failed;
- `429 overloaded` with `Retry-After: 1`; and
- `503 model_unavailable` — inference could not complete.

Every response carries a request ID and defensive cache/content headers. See
[`V3_RUNTIME.md`](V3_RUNTIME.md) for loading, offline, and network controls.
