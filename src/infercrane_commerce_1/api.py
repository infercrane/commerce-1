"""Typed request models and answer conversion for Commerce-1 system-one."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

from .runtime import MODEL_ID, RuntimeContractError, normalized_distribution

type Content = str | dict[str, JsonValue] | list[JsonValue]


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instructions: Content | None = None


class ChoiceQuestion(Question):
    type: Literal["choice"]
    criteria: dict[str, Content | None] = Field(min_length=2, max_length=255)

    @field_validator("criteria")
    @classmethod
    def valid_choice_keys(
        cls, value: dict[str, Content | None]
    ) -> dict[str, Content | None]:
        if any(not key or key != key.strip() or len(key) > 256 for key in value):
            raise ValueError("Choice option names must be bounded non-empty strings")
        return value


class NoulQuestion(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Content | None] | None = None

    @field_validator("criteria")
    @classmethod
    def complete_noul_criteria(
        cls, value: dict[Literal["true", "false"], Content | None] | None
    ) -> dict[Literal["true", "false"], Content | None] | None:
        if value is not None and set(value) != {"false", "true"}:
            raise ValueError("Noul criteria must contain exactly false and true")
        return value


class ScoreQuestion(Question):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)


TypedQuestion = Annotated[
    ChoiceQuestion | NoulQuestion | ScoreQuestion,
    Field(discriminator="type"),
]


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    state: Content
    questions: dict[str, TypedQuestion] = Field(min_length=1, max_length=64)

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        if value != MODEL_ID:
            raise ValueError(f"model must be {MODEL_ID}")
        return value

    @field_validator("questions")
    @classmethod
    def valid_question_ids(
        cls, value: dict[str, TypedQuestion]
    ) -> dict[str, TypedQuestion]:
        if any(not key or key != key.strip() or len(key) > 256 for key in value):
            raise ValueError("question IDs must be bounded non-empty strings")
        return value


def option_keys(question: TypedQuestion) -> list[str]:
    if question.type == "choice":
        return list(question.criteria)
    if question.type == "score":
        return [str(index) for index in range(len(question.criteria))]
    return ["false", "true"]


def answer_from_distribution(
    question: TypedQuestion,
    values: list[float],
    *,
    question_id: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build the stable typed answer and separately exposed raw distribution."""

    keys = option_keys(question)
    probabilities = normalized_distribution(
        values, expected=len(keys), question_id=question_id
    )
    distribution = dict(zip(keys, probabilities, strict=True))
    if question.type == "noul":
        return {"type": "noul", "noul": probabilities[1]}, distribution

    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    if question.type == "choice":
        baseline = 1.0 / len(probabilities)
        confidence = (probabilities[best] - baseline) / (1.0 - baseline)
        return (
            {
                "type": "choice",
                "choice": keys[best],
                "probabilities": distribution,
                "confidence": max(0.0, min(1.0, confidence)),
            },
            distribution,
        )

    midpoint = (len(probabilities) - 1) / 2
    baseline_distance = math.fsum(
        abs(index - midpoint) for index in range(len(probabilities))
    ) / len(probabilities)
    selected_distance = math.fsum(
        probability * abs(index - best)
        for index, probability in enumerate(probabilities)
    )
    if baseline_distance <= 0:
        raise RuntimeContractError("Score questions require at least two levels")
    return (
        {
            "type": "score",
            "score": math.fsum(
                index * probability for index, probability in enumerate(probabilities)
            ),
            "legend": {
                str(index): description
                for index, description in enumerate(question.criteria)
            },
            "probabilities": distribution,
            "confidence": max(0.0, 1.0 - selected_distance / baseline_distance),
        },
        distribution,
    )
