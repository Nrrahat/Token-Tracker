"""Validated experiment, dataset, and comparison result contracts."""

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.enums import ExecutionStatus


class ExperimentArm(StrEnum):
    BASELINE = "baseline"
    OPTIMIZED = "optimized"


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=160)
    prompt: str = Field(min_length=1, max_length=25_000)
    expected_outcome: str | None = Field(default=None, max_length=4000)


class ExperimentDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=2000)
    cases: list[ExperimentCase] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "ExperimentDataset":
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Dataset case IDs must be unique")
        return self


class AgentConfigSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    name: str
    config: dict[str, Any]
    skill_versions: dict[str, str]


class ExperimentDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    baseline_snapshot: AgentConfigSnapshot
    optimized_snapshot: AgentConfigSnapshot
    dataset: ExperimentDataset
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error_type: str | None


class ExperimentResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    arm: ExperimentArm
    run_id: str
    status: ExecutionStatus


class ExperimentArmMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    runs: int
    successful_runs: int
    task_success_rate: float | None
    reported_input_tokens: int
    reported_output_tokens: int
    reported_total_tokens: int
    reported_reasoning_tokens: int
    cached_input_tokens: int
    llm_calls: int
    average_llm_calls_per_run: float | None
    tool_calls: int
    average_tool_calls_per_run: float | None
    average_latency_ms: float | None
    estimated_costs: dict[str, Decimal]
    human_evaluations: int
    average_human_relevance: float | None
    average_human_completeness: float | None
    average_human_quality: float | None


class ExperimentComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment: ExperimentDefinition
    baseline: ExperimentArmMetrics
    optimized: ExperimentArmMetrics
    token_reduction_percent: float | None
    success_rate_change: float | None
    human_quality_change: float | None
    cost_reduction_percent_by_currency: dict[str, float]
    results: list[ExperimentResult]