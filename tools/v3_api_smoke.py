"""Fail-closed remote smoke for one exact Commerce-1 V3 deployment.

This probe never prints credentials or response bodies. It establishes wire,
identity, authentication, and measured-usage continuity only; it is not model
quality, performance, uptime, billing, or hardware qualification evidence.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from infercrane_commerce_1.runtime import MODEL_ID
from infercrane_commerce_1.server import _read_api_key

MAX_RESPONSE_BYTES = 1_048_576


class SmokeError(RuntimeError):
    """A public contract or identity check failed."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so bearer credentials never cross request boundaries."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _digest(value: str, description: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise SmokeError(f"{description} must be a lowercase SHA-256")
    return value


def _base_url(value: str, *, allow_http_loopback: bool) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SmokeError("base URL cannot contain credentials, query, or fragment")
    if parsed.path not in {"", "/"}:
        raise SmokeError("base URL cannot contain a path")
    try:
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise SmokeError("base URL has an invalid port") from error
    if not host:
        raise SmokeError("base URL must contain a host")
    if parsed.scheme == "https":
        pass
    elif parsed.scheme == "http" and allow_http_loopback:
        try:
            addresses = {
                item[4][0]
                for item in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            }
        except OSError as error:
            raise SmokeError("loopback host cannot be resolved") from error
        if not addresses or any(
            address not in {"127.0.0.1", "::1"} for address in addresses
        ):
            raise SmokeError("plain HTTP is allowed only on loopback")
    else:
        raise SmokeError("production smoke requires HTTPS")
    authority = host
    if ":" in host:
        authority = f"[{host}]"
    if port is not None:
        authority = f"{authority}:{port}"
    return f"{parsed.scheme}://{authority}"


def _request(
    *,
    base_url: str,
    path: str,
    token: str | None,
    body: dict[str, Any] | None,
    accept: str,
    timeout: float,
) -> tuple[str, bytes, dict[str, str]]:
    headers = {"Accept": accept}
    payload = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        payload = json.dumps(
            body,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=payload,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    opener = urllib.request.build_opener(
        _NoRedirectHandler(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    try:
        with opener.open(
            request,
            timeout=timeout,
        ) as response:
            value = response.read(MAX_RESPONSE_BYTES + 1)
            content_type = response.headers.get_content_type()
            response_headers = {
                key.lower(): item for key, item in response.headers.items()
            }
    except urllib.error.HTTPError as error:
        raise SmokeError(f"{path} returned HTTP {error.code}") from error
    except (OSError, TimeoutError) as error:
        raise SmokeError(f"{path} could not be reached") from error
    if len(value) > MAX_RESPONSE_BYTES:
        raise SmokeError(f"{path} response exceeded the byte limit")
    return content_type, value, response_headers


def _json_response(
    *,
    base_url: str,
    path: str,
    token: str | None,
    body: dict[str, Any] | None = None,
    timeout: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    content_type, raw, headers = _request(
        base_url=base_url,
        path=path,
        token=token,
        body=body,
        accept="application/json",
        timeout=timeout,
    )
    if content_type != "application/json":
        raise SmokeError(f"{path} returned an unexpected content type")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError(f"{path} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise SmokeError(f"{path} did not return an object")
    if not headers.get("x-request-id"):
        raise SmokeError(f"{path} did not return a request identity")
    if headers.get("cache-control") != "no-store":
        raise SmokeError(f"{path} did not disable response caching")
    return value, headers


def _identity(
    value: dict[str, Any],
    *,
    model_revision: str,
    runtime_sha256: str,
    evidence_sha256: str,
    model_key: str = "model",
) -> None:
    expected = {
        model_key: MODEL_ID,
        "model_revision": model_revision,
        "runtime_identity_sha256": runtime_sha256,
        "evidence_sha256": evidence_sha256,
    }
    if any(value.get(key) != item for key, item in expected.items()):
        raise SmokeError("deployment identity does not match the expected release")


def _decision() -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "state": {
            "order_total": 117,
            "approved_total": 94,
            "inventory_available": True,
        },
        "questions": {
            "next_action": {
                "type": "choice",
                "instructions": "Choose the safest next action.",
                "criteria": {
                    "continue": "Continue the order.",
                    "review": "Ask for human review.",
                    "block": "Stop the order.",
                },
            },
            "needs_review": {
                "type": "noul",
                "instructions": "Does this order require review?",
            },
        },
    }


def _usage(value: dict[str, Any], *, chat: bool) -> None:
    usage = value.get("usage")
    if not isinstance(usage, dict):
        raise SmokeError("inference did not return measured usage")
    prompt = usage.get("prompt_tokens" if chat else "input_tokens")
    completion = usage.get("completion_tokens" if chat else "output_tokens")
    if (
        isinstance(prompt, bool)
        or not isinstance(prompt, int)
        or prompt <= 0
        or isinstance(completion, bool)
        or completion != 0
    ):
        raise SmokeError("inference usage is invalid")


def run_smoke(
    *,
    base_url: str,
    token: str,
    model_revision: str,
    runtime_sha256: str,
    evidence_sha256: str,
    timeout: float,
) -> dict[str, Any]:
    checks: list[str] = []
    health, _ = _json_response(
        base_url=base_url,
        path="/healthz",
        token=None,
        timeout=timeout,
    )
    if health != {"status": "ok"}:
        raise SmokeError("liveness contract changed")
    checks.append("liveness")

    ready, _ = _json_response(
        base_url=base_url,
        path="/readyz",
        token=None,
        timeout=timeout,
    )
    if ready.get("status") != "ready":
        raise SmokeError("deployment is not ready")
    _identity(
        ready,
        model_revision=model_revision,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    checks.append("readiness_identity")

    models, _ = _json_response(
        base_url=base_url,
        path="/v1/models",
        token=token,
        timeout=timeout,
    )
    listed = models.get("data")
    invalid_models = (
        not isinstance(listed, list)
        or len(listed) != 1
        or not isinstance(listed[0], dict)
    )
    if invalid_models:
        raise SmokeError("model discovery contract changed")
    _identity(
        listed[0],
        model_revision=model_revision,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
        model_key="id",
    )
    if listed[0].get("id") != MODEL_ID:
        raise SmokeError("model discovery returned the wrong model")
    checks.append("model_discovery_identity")

    decision = _decision()
    native, _ = _json_response(
        base_url=base_url,
        path="/api/alpha/decisions",
        token=token,
        body=decision,
        timeout=timeout,
    )
    _identity(
        native,
        model_revision=model_revision,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    _usage(native, chat=False)
    if not isinstance(native.get("answers"), dict):
        raise SmokeError("decision endpoint returned no typed answers")
    checks.append("decisions_with_measured_usage")

    embedded = {key: value for key, value in decision.items() if key != "model"}
    chat_request = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": json.dumps(embedded, separators=(",", ":")),
            }
        ],
    }
    chat, _ = _json_response(
        base_url=base_url,
        path="/v1/chat/completions",
        token=token,
        body=chat_request,
        timeout=timeout,
    )
    _identity(
        chat,
        model_revision=model_revision,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    _usage(chat, chat=True)
    checks.append("openai_nonstream_with_measured_usage")

    content_type, raw, headers = _request(
        base_url=base_url,
        path="/v1/chat/completions",
        token=token,
        body={
            **chat_request,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        accept="text/event-stream",
        timeout=timeout,
    )
    if content_type != "text/event-stream" or not headers.get("x-request-id"):
        raise SmokeError("streaming response headers changed")
    lines = [
        line.removeprefix("data: ")
        for line in raw.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]
    if not lines or lines[-1] != "[DONE]":
        raise SmokeError("streaming response did not terminate")
    try:
        events = [json.loads(line) for line in lines[:-1]]
    except json.JSONDecodeError as error:
        raise SmokeError("streaming response contained invalid JSON") from error
    usage_events = [event for event in events if event.get("usage") is not None]
    if len(usage_events) != 1:
        raise SmokeError("streaming response did not report usage exactly once")
    _identity(
        usage_events[0],
        model_revision=model_revision,
        runtime_sha256=runtime_sha256,
        evidence_sha256=evidence_sha256,
    )
    _usage(usage_events[0], chat=True)
    checks.append("openai_stream_with_measured_usage")

    return {
        "schema_version": "infercrane-commerce-1-v3-api-smoke/v1",
        "status": "PASS_API_CONTRACT_SMOKE",
        "model": MODEL_ID,
        "model_revision": model_revision,
        "runtime_identity_sha256": runtime_sha256,
        "evidence_sha256": evidence_sha256,
        "checks": checks,
        "quality_claim": False,
        "performance_claim": False,
        "production_qualification": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--allow-http-loopback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        base_url = _base_url(
            args.base_url,
            allow_http_loopback=args.allow_http_loopback,
        )
        if not 0 < args.timeout <= 600:
            raise SmokeError("timeout must be between 0 and 600 seconds")
        receipt = run_smoke(
            base_url=base_url,
            token=_read_api_key(args.api_key_file),
            model_revision=_digest(args.model_revision, "model revision"),
            runtime_sha256=_digest(args.runtime_sha256, "runtime identity"),
            evidence_sha256=_digest(args.evidence_sha256, "evidence identity"),
            timeout=args.timeout,
        )
    except RuntimeError as error:
        print(f"v3 API smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
