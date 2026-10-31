"""Stable result contracts for analytics pages and exports."""

from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict


class TokenTotals(BaseModel):
    model_config = ConfigDict(frozen=True)

    llm_calls: int
    reported_calls: int
    reported_input_tokens: int
    reported_output_tokens: int
    reported_reasoning_tokens: int
    cached_input_tokens: int
    estimated_component_tokens: int
    attribution_gap_tokens: int
    estimated_costs: dict[str, Decimal]


class TokenGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    calls: int
    input_tokens: int
    output_tokens: int
    estimated_costs: dict[str, Decimal]


class TokenDay(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: date
    input_tokens: int
    output_tokens: int


class ComponentTokenGroup(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: str
    source_id: str | None
    estimated_tokens: int


class TokenAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    totals: TokenTotals
    by_model: list[TokenGroup]
    by_provider: list[TokenGroup]
    by_agent: list[TokenGroup]
    by_day: list[TokenDay]
    by_component: list[ComponentTokenGroup]


class ToolAnalyticsRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    tool_name: str
    calls: int
    failures: int
    duplicate_calls: int
    empty_results: int
    total_duration_ms: int
    total_output_size_bytes: int
    estimated_output_tokens: int
    included_result_tokens: int


class AgentAnalyticsRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    agent_name: str
    executions: int
    successful_executions: int
    success_rate: float
    llm_calls: int
    average_llm_calls_per_execution: float
    reported_input_tokens: int
    reported_output_tokens: int
    average_reported_tokens_per_execution: float
    tool_calls: int
    average_tool_calls_per_execution: float
    estimated_costs: dict[str, Decimal]


class AgentAnalytics(BaseModel):
    model_config = ConfigDict(frozen=True)

    agents: list[AgentAnalyticsRow]