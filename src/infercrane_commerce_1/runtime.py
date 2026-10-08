"""Small, model-neutral runtime contract used by the Commerce-1 HTTP server."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

MODEL_ID = "infercrane/commerce-1"
API_SCHEMA_VERSION = "infercrane-commerce-1-systemone/v1"
RELEASE_STATUS = "qualification_external"
QUESTION_TYPES = ("choice", "noul", "score")
MAX_INPUT_TOKENS = 65_536
QUESTION_MICROBATCH_SIZE = 1

_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BACKEND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,127}\Z")


class RuntimeContractError(ValueError):
    """The backend identity or output violates the public serving contract."""


class RuntimeUnavailableError(RuntimeError):
    """No qualified runtime is ready to answer requests."""


class ContextLimitExceededError(RuntimeUnavailableError):
    """The exact tokenizer found more input tokens than the release contract."""


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Content identities safe to expose from ``GET /v1/model``."""

    model: str
    revision: str
    runtime_identity_sha256: str
    evidence_sha256: str
    backend: str

    def __post_init__(self) -> None:
        if self.model != MODEL_ID:
            raise RuntimeContractError(f"model must be {MODEL_ID}")
        if _IMMUTABLE_REVISION.fullmatch(self.revision) is None:
            raise RuntimeContractError(
                "revision must be an immutable lowercase Git commit or SHA-256"
            )
        for field, value in (
            ("runtime_identity_sha256", self.runtime_identity_sha256),
            ("evidence_sha256", self.evidence_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise RuntimeContractError(f"{field} must be a lowercase SHA-256")
        if _BACKEND_ID.fullmatch(self.backend) is None:
            raise RuntimeContractError("backend must be a bounded public identifier")

    def public_payload(self) -> dict[str, Any]:
        """Return identity and capability data without paths or environment state."""

        return {
            "schema_version": API_SCHEMA_VERSION,
            "object": "model",
            "model": self.model,
            "revision": self.revision,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "evidence_sha256": self.evidence_sha256,
            "provider": "InferCrane",
            "release_status": RELEASE_STATUS,
            "qualification": {
                "status": "external_attestation_required",
                "reason": (
                    "Runtime qualification is bound to an immutable deployment "
                    "receipt, not inferred from package presence."
                ),
            },
            "supported_question_types": list(QUESTION_TYPES),
            "limits": {
                "maximum_input_tokens": MAX_INPUT_TOKENS,
                "maximum_questions": 64,
                "maximum_choice_options": 255,
                "maximum_score_levels": 10,
                "question_microbatch_size": QUESTION_MICROBATCH_SIZE,
            },
        }


@dataclass(frozen=True, slots=True)
class RuntimePrediction:
    """Unprojected option probabilities returned by the exact model backend."""

    distributions: Mapping[str, Sequence[float]]
    input_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens < 0
        ):
            raise RuntimeContractError("input_tokens must be a non-negative integer")


class DecisionRuntime(Protocol):
    """Backend ABI intentionally independent of Torch and model layout."""

    @property
    def identity(self) -> ModelIdentity: ...

    def predict(
        self,
        *,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
    ) -> RuntimePrediction: ...

    def close(self) -> None: ...


def normalized_distribution(
    values: Sequence[Any], *, expected: int, question_id: str
) -> list[float]:
    """Validate and normalize one backend distribution without changing order."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise RuntimeContractError(
            f"runtime distribution for {question_id!r} must be a sequence"
        )
    if len(values) != expected:
        raise RuntimeContractError(
            f"runtime distribution for {question_id!r} changed option coverage"
        )
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeContractError(
                f"runtime distribution for {question_id!r} must be numeric"
            )
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise RuntimeContractError(
                f"runtime distribution for {question_id!r} is invalid"
            )
        normalized.append(number)
    total = math.fsum(normalized)
    if total <= 0:
        raise RuntimeContractError(
            f"runtime distribution for {question_id!r} has no probability mass"
        )
    result = [value / total for value in normalized]
    if not math.isclose(math.fsum(result), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeContractError(
            f"runtime distribution for {question_id!r} is not normalized"
        )
    return result
