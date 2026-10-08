from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from tools import v3_api_smoke as smoke


def test_smoke_url_requires_https_or_loopback() -> None:
    assert (
        smoke._base_url("https://model.example.com:443", allow_http_loopback=False)
        == "https://model.example.com:443"
    )
    assert (
        smoke._base_url("http://127.0.0.1:8000", allow_http_loopback=True)
        == "http://127.0.0.1:8000"
    )
    for value in (
        "http://model.example.com",
        "https://user:secret@model.example.com",
        "https://model.example.com/v1",
        "https://model.example.com?token=secret",
    ):
        with pytest.raises(smoke.SmokeError):
            smoke._base_url(value, allow_http_loopback=False)


def test_smoke_never_forwards_authorization_across_redirects() -> None:
    requests: list[tuple[str, str | None]] = []

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            if self.path == "/start":
                self.send_response(302)
                self.send_header(
                    "Location",
                    f"http://127.0.0.1:{self.server.server_port}/capture",
                )
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(smoke.SmokeError, match="HTTP 302"):
            smoke._request(
                base_url=f"http://127.0.0.1:{server.server_port}",
                path="/start",
                token="operator-secret",
                body=None,
                accept="application/json",
                timeout=1,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)

    assert requests == [("/start", "Bearer operator-secret")]


def test_smoke_checks_every_surface_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision, runtime, evidence = "a" * 64, "b" * 64, "c" * 64
    calls: list[str] = []

    def json_response(**kwargs: Any):
        path = kwargs["path"]
        calls.append(path)
        if path == "/healthz":
            return {"status": "ok"}, {"x-request-id": "r"}
        if path == "/readyz":
            return {
                "status": "ready",
                "model": "infercrane/commerce-1",
                "model_revision": revision,
                "runtime_identity_sha256": runtime,
                "evidence_sha256": evidence,
            }, {"x-request-id": "r"}
        if path == "/v1/models":
            return {
                "data": [
                    {
                        "id": "infercrane/commerce-1",
                        "model_revision": revision,
                        "runtime_identity_sha256": runtime,
                        "evidence_sha256": evidence,
                    }
                ]
            }, {"x-request-id": "r"}
        usage = (
            {"prompt_tokens": 10, "completion_tokens": 0}
            if path == "/v1/chat/completions"
            else {"input_tokens": 10, "output_tokens": 0}
        )
        return {
            "model": "infercrane/commerce-1",
            "model_revision": revision,
            "runtime_identity_sha256": runtime,
            "evidence_sha256": evidence,
            "answers": {"route": {"type": "noul", "noul": 0.5}},
            "usage": usage,
        }, {"x-request-id": "r"}

    stream_event = {
        "model": "infercrane/commerce-1",
        "model_revision": revision,
        "runtime_identity_sha256": runtime,
        "evidence_sha256": evidence,
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
        },
    }

    def request(**kwargs: Any):
        calls.append(kwargs["path"] + "#stream")
        body = (
            "data: "
            + json.dumps(stream_event, separators=(",", ":"))
            + "\n\ndata: [DONE]\n\n"
        ).encode()
        return "text/event-stream", body, {"x-request-id": "r"}

    monkeypatch.setattr(smoke, "_json_response", json_response)
    monkeypatch.setattr(smoke, "_request", request)
    receipt = smoke.run_smoke(
        base_url="https://model.example.com",
        token="opaque",
        model_revision=revision,
        runtime_sha256=runtime,
        evidence_sha256=evidence,
        timeout=1,
    )

    assert receipt["status"] == "PASS_API_CONTRACT_SMOKE"
    assert receipt["production_qualification"] is False
    assert calls == [
        "/healthz",
        "/readyz",
        "/v1/models",
        "/api/alpha/decisions",
        "/v1/chat/completions",
        "/v1/chat/completions#stream",
    ]
