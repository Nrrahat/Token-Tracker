"""Validated data contracts used at application boundaries."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from core.enums import ProviderName


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    provider: ProviderName
    model: str = Field(min_length=1, max_length=200)
    system_instructions: str = ""
    developer_instructions: str = ""
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=1, le=100)
    max_tool_calls: int = Field(default=DEFAULT_MAX_TOOL_CALLS, ge=0, le=500)
    request_timeout_seconds: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        gt=0.0,
        le=600.0,
    )
    skill_names: list[str] = Field(default_factory=list, max_length=50)
    tool_names: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("skill_names", "tool_names")
    @classmethod
    def names_must_be_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("Attached skill and tool names must be unique")
        return values


class TaskSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    prompt: str = Field(min_length=1, max_length=100_000)


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    is_estimated: bool = False


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class ProviderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    tools: list[ToolDefinition] | None = None
    timeout_seconds: float = Field(default=60.0, gt=0.0, le=600.0)


class ProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = Field(ge=0.0)
    finish_reason: str | None = None