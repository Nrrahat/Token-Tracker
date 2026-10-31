"""Validated automatic and human evaluation contracts."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluatorType(StrEnum):
    AUTOMATIC = "automatic"
    HUMAN = "human"


class HumanEvaluationSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    evaluator_name: str = Field(min_length=1, max_length=120)
    response_relevance: int | None = Field(default=None, ge=1, le=5)
    response_completeness: int | None = Field(default=None, ge=1, le=5)
    output_quality: int | None = Field(default=None, ge=1, le=5)
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def require_at_least_one_score(self) -> "HumanEvaluationSubmission":
        if all(
            value is None
            for value in (
                self.response_relevance,
                self.response_completeness,
                self.output_quality,
            )
        ):
            raise ValueError("A human evaluation requires at least one quality score")
        return self


class EvaluationView(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    run_id: str
    evaluator_type: EvaluatorType
    evaluator_name: str
    task_success: bool
    tool_use_success: bool | None
    response_relevance: int | None
    response_completeness: int | None
    output_quality: int | None
    notes: str
    evaluated_at: datetime


class EvaluationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    automatic_evaluations: int
    automatic_success_rate: float | None
    human_evaluations: int
    average_relevance: float | None
    average_completeness: float | None
    average_quality: float | None