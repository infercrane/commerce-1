"""Typed FastAPI server for a content-identified Commerce-1 runtime."""

from __future__ import annotations

import argparse
import importlib
import os
import secrets
import stat
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .api import EvaluationRequest, answer_from_distribution
from .openai_compat import (
    ChatCompletionRequest,
    ChatDecisionError,
    completion_response,
    completion_stream,
    decision_from_chat,
    model_list,
)
from .runtime import (
    API_SCHEMA_VERSION,
    MODEL_ID,
    RELEASE_STATUS,
    ContextLimitExceededError,
    DecisionRuntime,
    ModelIdentity,
    RuntimeContractError,
    RuntimePrediction,
    RuntimeUnavailableError,
)

MAX_REQUEST_BYTES = 1_048_576
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RUNTIME_FACTORY = "infercrane_commerce_1.backend:create_runtime"

RuntimeFactory = Callable[..., DecisionRuntime]


class ServingError(RuntimeError):
    """Base class for stable HTTP errors that do not expose internals."""


class OverloadedError(ServingError):
    """The single local model worker is already processing another request."""


def _error(error_class: str, message: str, request_id: str) -> dict[str, Any]:
    return {
        "object": "error",
        "request_id": request_id,
        "error": {"class": error_class, "message": message},
    }


class CommerceOneService:
    """Serialize model access and convert raw runtime distributions to answers."""

    def __init__(self, runtime: DecisionRuntime) -> None:
        identity = runtime.identity
        if not isinstance(identity, ModelIdentity):
            raise RuntimeContractError("runtime.identity must be a ModelIdentity")
        self.runtime = runtime
        self.identity = identity
        self._lock = threading.Lock()
        self._ready = True

    def model_info(self) -> dict[str, Any]:
        return self.identity.public_payload()

    def readiness(self) -> dict[str, Any]:
        return {
            "status": "ready" if self._ready else "unavailable",
            "model": self.identity.model,
            "model_revision": self.identity.revision,
            "runtime_identity_sha256": self.identity.runtime_identity_sha256,
            "evidence_sha256": self.identity.evidence_sha256,
            "limits": self.identity.public_payload()["limits"],
        }

    @property
    def ready(self) -> bool:
        return self._ready

    def decide(self, request: EvaluationRequest) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise OverloadedError("the local model is busy")
        try:
            questions = {
                question_id: question.model_dump(exclude_none=True)
                for question_id, question in request.questions.items()
            }
            prediction = self.runtime.predict(
                state=request.state,
                questions=questions,
            )
        finally:
            self._lock.release()
        if not isinstance(prediction, RuntimePrediction):
            raise RuntimeContractError(
                "runtime.predict must return a RuntimePrediction"
            )
        if list(prediction.distributions) != list(request.questions):
            raise RuntimeContractError("runtime changed question coverage or order")

        answers: dict[str, Any] = {}
        raw: dict[str, dict[str, float]] = {}
        for question_id, question in request.questions.items():
            answer, distribution = answer_from_distribution(
                question,
                list(prediction.distributions[question_id]),
                question_id=question_id,
            )
            answers[question_id] = answer
            raw[question_id] = distribution
        return {
            "schema_version": API_SCHEMA_VERSION,
            "id": f"dec_{uuid.uuid4().hex}",
            "object": "decision.response",
            "model": MODEL_ID,
            "model_revision": self.identity.revision,
            "runtime_identity_sha256": self.identity.runtime_identity_sha256,
            "evidence_sha256": self.identity.evidence_sha256,
            "provider": "InferCrane",
            "release_status": RELEASE_STATUS,
            "answers": answers,
            "raw": raw,
            "usage": {
                "input_tokens": prediction.input_tokens,
                "output_tokens": 0,
                "cost": None,
            },
        }

    def close(self) -> None:
        self._ready = False
        self.runtime.close()


def create_app(
    runtime: DecisionRuntime,
    *,
    close_runtime: bool = True,
    api_key: str | None = None,
) -> FastAPI:
    """Create an app around an already loaded, content-identified runtime."""

    service = CommerceOneService(runtime)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if close_runtime:
                service.close()

    app = FastAPI(
        title="InferCrane Commerce-1",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service

    if api_key is not None and (not api_key or len(api_key) > 4096):
        raise RuntimeContractError("API key must be a bounded non-empty value")

    @app.middleware("http")
    async def request_metadata(request: Request, call_next):
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()

        def finish(response):
            response.headers["x-request-id"] = request_id
            response.headers["cache-control"] = "no-store"
            response.headers["x-content-type-options"] = "nosniff"
            response.headers["referrer-policy"] = "no-referrer"
            response.headers["server-timing"] = (
                f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
            )
            return response

        path = request.url.path
        if api_key is not None and path not in {"/health", "/healthz", "/readyz"}:
            supplied = request.headers.get("authorization", "")
            prefix = "Bearer "
            authorized = supplied.startswith(prefix) and secrets.compare_digest(
                supplied[len(prefix) :], api_key
            )
            if not authorized:
                return finish(
                    JSONResponse(
                        status_code=401,
                        content=_error(
                            "unauthorized",
                            "a valid bearer token is required",
                            request_id,
                        ),
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                )

        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if content_type.partition(";")[0].strip().lower() != "application/json":
                return finish(
                    JSONResponse(
                        status_code=415,
                        content=_error(
                            "invalid_request",
                            "content type must be application/json",
                            request_id,
                        ),
                    )
                )
            content_encoding = request.headers.get("content-encoding", "identity")
            if content_encoding.lower() != "identity":
                return finish(
                    JSONResponse(
                        status_code=415,
                        content=_error(
                            "invalid_request",
                            "encoded request bodies are not supported",
                            request_id,
                        ),
                    )
                )
        content_length = request.headers.get("content-length")
        if request.method == "POST" and content_length is None:
            return finish(
                JSONResponse(
                    status_code=411,
                    content=_error(
                        "length_required",
                        "content length is required",
                        request_id,
                    ),
                )
            )
        if content_length is not None:
            try:
                length = int(content_length)
                too_large = length < 0 or length > MAX_REQUEST_BYTES
            except ValueError:
                too_large = True
            if too_large:
                return finish(
                    JSONResponse(
                        status_code=413,
                        content=_error(
                            "request_too_large",
                            "request exceeds the configured byte limit",
                            request_id,
                        ),
                    )
                )
        response = await call_next(request)
        return finish(response)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        invalid_json = any(
            item.get("type") == "json_invalid" for item in error.errors()
        )
        status = 400 if invalid_json else 422
        return JSONResponse(
            status_code=status,
            content=_error(
                "invalid_request" if invalid_json else "unsupported_question",
                "request validation failed",
                request.state.request_id,
            ),
        )

    @app.get("/healthz", response_model=None)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health", response_model=None)
    @app.get("/readyz", response_model=None)
    def ready():
        if not service.ready:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable"},
            )
        return service.readiness()

    @app.get("/v1/model", response_model=None)
    def model() -> dict[str, Any]:
        return service.model_info()

    @app.get("/models", response_model=None)
    @app.get("/v1/models", response_model=None)
    def models() -> dict[str, Any]:
        return model_list(service.identity)

    @app.post("/api/alpha/decisions", response_model=None)
    @app.post("/v1/systemone", response_model=None)
    def system_one(body: EvaluationRequest, request: Request):
        try:
            return service.decide(body)
        except ContextLimitExceededError:
            return JSONResponse(
                status_code=413,
                content=_error(
                    "context_limit_exceeded",
                    "request exceeds the model context limit",
                    request.state.request_id,
                ),
            )
        except OverloadedError:
            return JSONResponse(
                status_code=429,
                content=_error(
                    "overloaded",
                    "the local model is busy; retry shortly",
                    request.state.request_id,
                ),
                headers={"Retry-After": "1"},
            )
        except Exception:
            return JSONResponse(
                status_code=503,
                content=_error(
                    "model_unavailable",
                    "the model could not complete this request",
                    request.state.request_id,
                ),
            )

    @app.post("/chat/completions", response_model=None)
    @app.post("/v1/chat/completions", response_model=None)
    def chat_completions(body: ChatCompletionRequest, request: Request):
        try:
            decision = decision_from_chat(body)
            result = service.decide(decision)
            response_id = f"chatcmpl-{uuid.uuid4().hex}"
            if body.stream:
                return StreamingResponse(
                    completion_stream(
                        request_id=response_id,
                        identity=service.identity,
                        result=result,
                        include_usage=True,
                    ),
                    media_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no"},
                )
            return completion_response(
                request_id=response_id,
                identity=service.identity,
                result=result,
            )
        except ContextLimitExceededError:
            return JSONResponse(
                status_code=413,
                content=_error(
                    "context_limit_exceeded",
                    "request exceeds the model context limit",
                    request.state.request_id,
                ),
            )
        except (ChatDecisionError, ValidationError):
            return JSONResponse(
                status_code=400,
                content=_error(
                    "invalid_request",
                    "chat content must be one valid Commerce-1 decision JSON object",
                    request.state.request_id,
                ),
            )
        except OverloadedError:
            return JSONResponse(
                status_code=429,
                content=_error(
                    "overloaded",
                    "the local model is busy; retry shortly",
                    request.state.request_id,
                ),
                headers={"Retry-After": "1"},
            )
        except Exception:
            return JSONResponse(
                status_code=503,
                content=_error(
                    "model_unavailable",
                    "the model could not complete this request",
                    request.state.request_id,
                ),
            )

    return app


def _runtime_factory(value: str) -> RuntimeFactory:
    module_name, separator, attribute = value.partition(":")
    if not separator or not module_name or not attribute:
        raise RuntimeUnavailableError(
            "runtime factory must use the form package.module:callable"
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as error:
        raise RuntimeUnavailableError("runtime factory is unavailable") from error
    if not callable(factory):
        raise RuntimeUnavailableError("runtime factory is not callable")
    return factory


def _loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commerce-1-v3-serve",
        description=(
            "Serve a content-identified Commerce-1 runtime. This command does "
            "not claim that the supplied model has been qualified or released."
        ),
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--runtime-factory", default=DEFAULT_RUNTIME_FACTORY)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-non-loopback", action="store_true")
    parser.add_argument(
        "--api-key-file",
        type=Path,
        help="Owner-only file containing the bearer token; required off loopback.",
    )
    return parser


def _read_api_key(path: Path) -> str:
    """Load a bounded bearer token from an owner-only, non-symlink file."""

    if path.is_symlink():
        raise RuntimeUnavailableError("API key file cannot be a symlink")
    try:
        resolved = path.expanduser().resolve(strict=True)
        details = resolved.stat()
    except OSError as error:
        raise RuntimeUnavailableError("API key file is unavailable") from error
    if not stat.S_ISREG(details.st_mode):
        raise RuntimeUnavailableError("API key file must be a regular file")
    if hasattr(os, "getuid") and details.st_uid != os.getuid():
        raise RuntimeUnavailableError("API key file must be owned by this process user")
    if stat.S_IMODE(details.st_mode) not in {0o400, 0o600}:
        raise RuntimeUnavailableError("API key file mode must be 0400 or 0600")
    try:
        token = resolved.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise RuntimeUnavailableError("API key file cannot be read") from error
    invalid = (
        not token
        or len(token) > 4096
        or any(character.isspace() for character in token)
    )
    if invalid:
        raise RuntimeUnavailableError("API key file contains an invalid token")
    return token


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not _loopback(args.host) and not args.allow_non_loopback:
        raise RuntimeUnavailableError(
            "non-loopback serving requires --allow-non-loopback"
        )
    if not _loopback(args.host) and args.api_key_file is None:
        raise RuntimeUnavailableError(
            "non-loopback serving requires an owner-only --api-key-file"
        )
    api_key = _read_api_key(args.api_key_file) if args.api_key_file else None
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    base_dir = args.base_dir.expanduser().resolve(strict=True)
    if not model_dir.is_dir() or not base_dir.is_dir():
        raise RuntimeUnavailableError("model and base paths must be directories")
    factory = _runtime_factory(args.runtime_factory)
    runtime = factory(
        model_dir=model_dir,
        base_dir=base_dir,
        offline=args.offline,
    )
    app = create_app(runtime, api_key=api_key)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
