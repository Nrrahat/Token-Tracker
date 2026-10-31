"""Agent-management and execution result contracts."""

from pydantic import BaseModel, ConfigDict

from core.enums import ExecutionStatus
from core.schemas import AgentConfig


class AgentSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent_id: str
    config: AgentConfig
    is_active: bool
    skill_versions: dict[str, str]


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str
    run_id: str
    status: ExecutionStatus
    answer: str | None = None
    error_type: str | None = None
    iterations: int = 0
    tool_calls: int = 0