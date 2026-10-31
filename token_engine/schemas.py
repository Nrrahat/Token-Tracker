"""Contracts for estimates and token attribution."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.enums import ProviderName
from core.schemas import TokenUsage


class PromptComponentKind(StrEnum):
    SYSTEM_INSTRUCTION = "system_instruction"
    DEVELOPER_INSTRUCTION = "developer_instruction"
    USER_PROMPT = "user_prompt"
    SKILL_INSTRUCTION = "skill_instruction"
    TOOL_SCHEMA = "tool_schema"
    TOOL_RESULT = "tool_result"
    CONVERSATION_HISTORY = "conversation_history"


class PromptComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PromptComponentKind
    text: str
    source_id: str | None = None


class ComponentTokenEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: PromptComponentKind
    source_id: str | None = None
    estimated_tokens: int = Field(ge=0)
    is_estimated: bool = True


class PromptTokenBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    encoding_name: str
    components: list[ComponentTokenEstimate]
    total_estimated_tokens: int = Field(ge=0)
    is_estimated: bool = True


class UsageExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    model: str
    usage: TokenUsage
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    reported_total_tokens: int | None = Field(default=None, ge=0)
    has_reported_usage: bool


class TokenAttributionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: PromptTokenBreakdown
    provider_usage: UsageExtraction
    attribution_gap_tokens: int | None = None