"""Validated tool contract and execution result types."""

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.schemas import ToolDefinition as ProviderToolDefinition
from token_engine.schemas import PromptComponent, PromptComponentKind


class ToolStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class BaseTool(ABC):
    """Base class for explicitly registered, schema-validated tools."""

    name: str
    description: str
    input_model: type[BaseModel]
    timeout_seconds: float = 5.0

    @abstractmethod
    def execute(self, arguments: BaseModel) -> Any:
        """Execute validated input and return JSON-serializable output."""

    def provider_definition(self) -> ProviderToolDefinition:
        return ProviderToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: ToolStatus
    output: Any = None
    output_text: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    duration_ms: float = Field(ge=0.0)
    output_size_bytes: int = Field(default=0, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)
    output_metadata: dict[str, Any] = Field(default_factory=dict)

    def as_tool_result_message(self, source_id: str | None = None) -> PromptComponent:
        """Explicitly mark successful output for inclusion in a later LLM prompt."""
        if self.status is not ToolStatus.SUCCEEDED or self.output_text is None:
            raise ValueError("Only successful tool results can be added to a prompt")
        return PromptComponent(
            kind=PromptComponentKind.TOOL_RESULT,
            source_id=source_id or self.tool_name,
            text=self.output_text,
        )


class ToolMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    duplicate_call_count: int = 0
    empty_result_count: int = 0
    total_duration_ms: float = 0.0
    total_output_size_bytes: int = 0
    estimated_output_tokens: int = 0