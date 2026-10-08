from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from infercrane_commerce_1 import server
from infercrane_commerce_1.runtime import (
    MAX_INPUT_TOKENS,
    MODEL_ID,
    QUESTION_MICROBATCH_SIZE,
    ContextLimitExceededError,
    ModelIdentity,
    RuntimePrediction,
    RuntimeUnavailableError,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.identity = ModelIdentity(
            model=MODEL_ID,
            revision="a" * 64,
            runtime_identity_sha256="b" * 64,
            evidence_sha256="c" * 64,
            backend="test-runtime-v1",
        )
        self.closed = False

    def predict(self, *, state: Any, questions: dict[str, dict[str, Any]]):
        del state
        values = {
            "choice": [0.1, 0.8, 0.1],
            "noul": [0.2, 0.8],
            "score": [0.02, 0.08, 0.2, 0.7],
        }
        return RuntimePrediction(
            distributions={
                name: values[question["type"]] for name, question in questions.items()
            },
            input_tokens=137,
        )

    def close(self) -> None:
        self.closed = True


class OverContextRuntime(FakeRuntime):
    def predict(self, *, state: Any, questions: dict[str, dict[str, Any]]):
        del state, questions
        raise ContextLimitExceededError("request exceeds the token limit")


def _decision() -> dict[str, Any]:
    return {
        "state": {"cart_total": 117, "approved_total": 94},
        "questions": {
            "route": {
                "type": "choice",
                "instructions": "Choose the safest next action.",
                "criteria": {
                    "continue": "Continue.",
                    "review": "Review.",
                    "block": "Stop.",
                },
            },
            "review": {
                "type": "noul",
                "instructions": "Is review required?",
            },
            "risk": {
                "type": "score",
                "instructions": "Score commerce risk.",
                "criteria": ["low", "guarded", "high", "block"],
            },
        },
    }


def _systemone() -> dict[str, Any]:
    return {"model": MODEL_ID, **_decision()}


def _chat(*, stream: bool = False) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(_decision(), separators=(",", ":")),
            }
        ],
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def test_readiness_discovery_and_decisions_expose_exact_identity() -> None:
    runtime = FakeRuntime()
    with TestClient(server.create_app(runtime)) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        ready = client.get("/readyz")
        health_alias = client.get("/health")
        models = client.get("/v1/models")
        models_alias = client.get("/models")
        native = client.post("/v1/systemone", json=_systemone())
        decisions = client.post("/api/alpha/decisions", json=_systemone())

    expected_identity = {
        "status": "ready",
        "model": MODEL_ID,
        "model_revision": "a" * 64,
        "runtime_identity_sha256": "b" * 64,
        "evidence_sha256": "c" * 64,
        "limits": {
            "maximum_input_tokens": MAX_INPUT_TOKENS,
            "maximum_questions": 64,
            "maximum_choice_options": 255,
            "maximum_score_levels": 10,
            "question_microbatch_size": QUESTION_MICROBATCH_SIZE,
        },
    }
    assert ready.status_code == 200
    assert ready.json() == expected_identity
    assert health_alias.json() == expected_identity
    assert models.json() == models_alias.json()
    listed = models.json()["data"][0]
    assert listed["id"] == MODEL_ID
    assert listed["model_revision"] == "a" * 64
    assert listed["runtime_identity_sha256"] == "b" * 64
    assert listed["evidence_sha256"] == "c" * 64
    assert listed["capabilities"]["generated_tokens"] is False
    assert native.status_code == decisions.status_code == 200
    assert native.json()["answers"] == decisions.json()["answers"]
    assert native.json()["runtime_identity_sha256"] == "b" * 64
    assert runtime.closed is True


def test_openai_nonstreaming_bridge_is_typed_measured_and_content_bound() -> None:
    with TestClient(server.create_app(FakeRuntime())) as client:
        response = client.post("/v1/chat/completions", json=_chat())
        alias = client.post("/chat/completions", json=_chat())

    assert response.status_code == alias.status_code == 200
    value = response.json()
    assert value["object"] == "chat.completion"
    assert value["model"] == MODEL_ID
    assert value["model_revision"] == "a" * 64
    assert value["runtime_identity_sha256"] == "b" * 64
    assert value["evidence_sha256"] == "c" * 64
    assert value["system_fingerprint"] == "rt_" + "b" * 64
    assert value["usage"] == {
        "prompt_tokens": 137,
        "completion_tokens": 0,
        "total_tokens": 137,
    }
    content = json.loads(value["choices"][0]["message"]["content"])
    assert content["answers"]["route"]["choice"] == "review"
    assert content["answers"]["review"]["noul"] == 0.8
    assert content["answers"]["risk"]["score"] == pytest.approx(2.58)


def test_openai_stream_reports_usage_and_terminates() -> None:
    with TestClient(server.create_app(FakeRuntime())) as client:
        response = client.post("/v1/chat/completions", json=_chat(stream=True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    payloads = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads[-1] == "[DONE]"
    events = [json.loads(value) for value in payloads[:-1]]
    assert json.loads(events[0]["choices"][0]["delta"]["content"])["answers"]
    assert events[1]["choices"][0]["finish_reason"] == "stop"
    assert events[2]["choices"] == []
    assert events[2]["usage"] == {
        "prompt_tokens": 137,
        "completion_tokens": 0,
        "total_tokens": 137,
    }
    assert all(event["runtime_identity_sha256"] == "b" * 64 for event in events)


def test_chat_bridge_rejects_ordinary_text_unknown_fields_and_duplicate_keys() -> None:
    requests = [
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "Which route?"}],
        },
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps({**_decision(), "prompt": "ignored?"}),
                }
            ],
        },
        {
            "model": MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": '{"state":"one","state":"two","questions":{}}',
                }
            ],
        },
    ]
    with TestClient(server.create_app(FakeRuntime())) as client:
        responses = [
            client.post("/v1/chat/completions", json=value) for value in requests
        ]

    assert [response.status_code for response in responses] == [400, 400, 400]
    assert all(
        response.json()["error"]["class"] == "invalid_request" for response in responses
    )


def test_file_backed_auth_protects_metadata_and_inference_without_leaking_key() -> None:
    token = "ic_test_opaque_secret"
    with TestClient(server.create_app(FakeRuntime(), api_key=token)) as client:
        ready = client.get("/readyz")
        missing = client.get("/v1/model")
        wrong = client.post(
            "/v1/systemone",
            headers={"Authorization": "Bearer wrong"},
            json=_systemone(),
        )
        allowed = client.post(
            "/v1/systemone",
            headers={"Authorization": f"Bearer {token}"},
            json=_systemone(),
        )

    assert ready.status_code == 200
    assert missing.status_code == wrong.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert allowed.status_code == 200
    serialized = "\n".join(
        [missing.text, wrong.text, allowed.text, repr(dict(allowed.headers))]
    )
    assert token not in serialized


def test_request_boundary_rejects_untyped_or_encoded_posts_and_sets_headers() -> None:
    with TestClient(server.create_app(FakeRuntime())) as client:
        untyped = client.post("/v1/systemone", content=b"{}")
        encoded = client.post(
            "/v1/systemone",
            headers={
                "content-type": "application/json",
                "content-encoding": "gzip",
            },
            content=b"not-really-gzip",
        )
        valid = client.post("/v1/systemone", json=_systemone())

    assert untyped.status_code == encoded.status_code == 415
    assert valid.status_code == 200
    assert valid.headers["cache-control"] == "no-store"
    assert valid.headers["x-content-type-options"] == "nosniff"
    assert valid.headers["referrer-policy"] == "no-referrer"
    assert valid.headers["x-request-id"]


def test_context_limit_fails_closed_for_native_and_chat_apis() -> None:
    with TestClient(server.create_app(OverContextRuntime())) as client:
        native = client.post("/v1/systemone", json=_systemone())
        chat = client.post("/v1/chat/completions", json=_chat())

    for response in (native, chat):
        assert response.status_code == 413
        assert response.json()["error"] == {
            "class": "context_limit_exceeded",
            "message": "request exceeds the model context limit",
        }


def test_api_key_file_requires_owner_only_regular_file(tmp_path: Path) -> None:
    secret = tmp_path / "token"
    secret.write_text("ic_file_secret\n", encoding="utf-8")
    secret.chmod(0o600)
    assert server._read_api_key(secret) == "ic_file_secret"

    secret.chmod(0o640)
    with pytest.raises(RuntimeUnavailableError, match="0400 or 0600"):
        server._read_api_key(secret)

    secret.chmod(0o600)
    link = tmp_path / "token-link"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(RuntimeUnavailableError, match="symlink"):
        server._read_api_key(link)


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership test")
def test_non_loopback_start_requires_explicit_key_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeUnavailableError, match="api-key-file"):
        server.main(
            [
                "--model-dir",
                str(tmp_path),
                "--base-dir",
                str(tmp_path),
                "--host",
                "0.0.0.0",
                "--allow-non-loopback",
                "--offline",
            ]
        )
