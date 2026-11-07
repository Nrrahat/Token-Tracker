"""Shared enumerations for domain and persistence contracts."""

from enum import StrEnum


class ProviderName(StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    OLLAMA = "ollama"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    TASK_SUBMITTED = "task_submitted"
    AGENT_STARTED = "agent_started"
    LLM_REQUEST_STARTED = "llm_request_started"
    LLM_RESPONSE_RECEIVED = "llm_response_received"
    LLM_REQUEST_FAILED = "llm_request_failed"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    ITERATION_COMPLETED = "iteration_completed"
    AGENT_FINISHED = "agent_finished"
    AGENT_FAILED = "agent_failed"
    AGENT_CANCELLED = "agent_cancelled"