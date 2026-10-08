"""Strict OpenAI chat envelope for the Commerce-1 decision runtime.

Commerce-1 does not generate prose. The compatibility route accepts one user
message containing a JSON decision request and returns the typed answers as
compact JSON in the assistant message. The native ``/v1/systemone`` route is
the preferred interface; this module exists so OpenAI clients and provider
health checks can exercise the exact same model without an alternate scorer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .api import EvaluationRequest
from .runtime import MODEL_ID, ModelIdentity


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user"]
    content: str = Field(min_length=2, max_length=1_048_576)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: Literal[True] = True


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage] = Field(min_length=1, max_length=1)
    stream: bool = False
    stream_options: StreamOptions | None = None

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        if value != MODEL_ID:
            raise ValueError(f"model must be {MODEL_ID}")
        return value


class ChatDecisionError(ValueError):
    """A chat request does not contain the strict decision payload."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ChatDecisionError("chat decision JSON contains duplicate keys")
        result[key] = value
    return result


def decision_from_chat(request: ChatCompletionRequest) -> EvaluationRequest:
    """Parse the only supported chat payload into the native request model."""

    content = request.messages[0].content
    try:
        value = json.loads(content, object_pairs_hook=_no_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ChatDecisionError(
            "the user message must contain one JSON decision object"
        ) from error
    if not isinstance(value, dict) or set(value) != {"state", "questions"}:
        raise ChatDecisionError(
            "the user message must contain exactly state and questions"
        )
    try:
        return EvaluationRequest.model_validate({"model": request.model, **value})
    except ValidationError as error:
        raise ChatDecisionError("the embedded decision request is invalid") from error


def _usage(result: Mapping[str, Any]) -> dict[str, int]:
    usage = result.get("usage")
    if not isinstance(usage, Mapping):
        raise ChatDecisionError("decision response did not report measured usage")
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens < 0
        or isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens != 0
    ):
        raise ChatDecisionError("decision response reported invalid usage")
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _content(result: Mapping[str, Any]) -> str:
    answers = result.get("answers")
    if not isinstance(answers, Mapping):
        raise ChatDecisionError("decision response did not contain typed answers")
    return json.dumps(
        {"answers": answers},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def completion_response(
    *,
    request_id: str,
    identity: ModelIdentity,
    result: Mapping[str, Any],
    created: int | None = None,
) -> dict[str, Any]:
    """Build a non-streaming OpenAI-compatible response without fake tokens."""

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()) if created is None else created,
        "model": identity.model,
        "system_fingerprint": f"rt_{identity.runtime_identity_sha256}",
        "model_revision": identity.revision,
        "runtime_identity_sha256": identity.runtime_identity_sha256,
        "evidence_sha256": identity.evidence_sha256,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _content(result)},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(result),
    }


def completion_stream(
    *,
    request_id: str,
    identity: ModelIdentity,
    result: Mapping[str, Any],
    include_usage: bool,
    created: int | None = None,
) -> Iterator[bytes]:
    """Emit a finite OpenAI SSE stream for one non-generative decision pass."""

    timestamp = int(time.time()) if created is None else created
    common = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": timestamp,
        "model": identity.model,
        "system_fingerprint": f"rt_{identity.runtime_identity_sha256}",
        "model_revision": identity.revision,
        "runtime_identity_sha256": identity.runtime_identity_sha256,
        "evidence_sha256": identity.evidence_sha256,
    }

    def event(value: Mapping[str, Any]) -> bytes:
        return (
            "data: "
            + json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n\n"
        ).encode("utf-8")

    yield event(
        {
            **common,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": _content(result)},
                    "finish_reason": None,
                }
            ],
        }
    )
    yield event(
        {
            **common,
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
            ],
        }
    )
    if include_usage:
        yield event({**common, "choices": [], "usage": _usage(result)})
    yield b"data: [DONE]\n\n"


def model_list(identity: ModelIdentity) -> dict[str, Any]:
    """Return the stable OpenAI discovery shape plus immutable identities."""

    return {
        "object": "list",
        "data": [
            {
                "id": identity.model,
                "object": "model",
                # The artifact has no trusted wall-clock creation claim in its
                # manifest. Zero is an explicit unknown sentinel.
                "created": 0,
                "owned_by": "infercrane",
                "model_revision": identity.revision,
                "runtime_identity_sha256": identity.runtime_identity_sha256,
                "evidence_sha256": identity.evidence_sha256,
                "capabilities": {
                    "systemone": True,
                    "chat_completions_json_bridge": True,
                    "streaming": True,
                    "generated_tokens": False,
                },
            }
        ],
    }
