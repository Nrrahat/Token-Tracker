"""CRUD and attachment validation for independent agent configurations."""

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from core.enums import ProviderName
from core.schemas import AgentConfig
from database.models import (
    AgentRecord,
    AgentSkillAttachmentRecord,
    AgentToolAttachmentRecord,
    TaskRecord,
)
from agents.schemas import AgentSnapshot
from skills.skill_registry import SkillRegistry
from tools.tool_registry import ToolRegistry


class AgentNotFoundError(LookupError):
    """Raised when an agent ID does not exist."""


class AgentInUseError(RuntimeError):
    """Raised when deleting an agent that owns task history."""


class AgentManager:
    def __init__(
        self,
        session_factory: sessionmaker,
        skill_registry: SkillRegistry,
        tool_registry: ToolRegistry,
    ) -> None:
        self._session_factory = session_factory
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry

    def create_agent(self, config: AgentConfig) -> AgentSnapshot:
        skill_versions = self._validate_attachments(config)
        with self._session_factory() as session:
            record = _agent_record(config)
            session.add(record)
            session.flush()
            record.skill_attachments = [
                AgentSkillAttachmentRecord(skill_name=name, skill_version=skill_versions[name])
                for name in config.skill_names
            ]
            record.tool_attachments = [
                AgentToolAttachmentRecord(tool_name=name) for name in config.tool_names
            ]
            session.commit()
            session.refresh(record)
            return _snapshot(record, skill_versions)

    def update_agent(self, agent_id: str, config: AgentConfig) -> AgentSnapshot:
        skill_versions = self._validate_attachments(config)
        with self._session_factory() as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                raise AgentNotFoundError(f"Agent does not exist: {agent_id}")
            _update_record(record, config)
            record.skill_attachments.clear()
            record.tool_attachments.clear()
            session.flush()
            record.skill_attachments = [
                AgentSkillAttachmentRecord(
                    agent_id=agent_id,
                    skill_name=name,
                    skill_version=skill_versions[name],
                )
                for name in config.skill_names
            ]
            record.tool_attachments = [
                AgentToolAttachmentRecord(agent_id=agent_id, tool_name=name)
                for name in config.tool_names
            ]
            session.commit()
            session.refresh(record)
            return _snapshot(record, skill_versions)

    def get_agent(self, agent_id: str) -> AgentSnapshot:
        with self._session_factory() as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                raise AgentNotFoundError(f"Agent does not exist: {agent_id}")
            skill_versions = {
                item.skill_name: item.skill_version for item in record.skill_attachments
            }
            return _snapshot(record, skill_versions)

    def list_agents(self) -> list[AgentSnapshot]:
        with self._session_factory() as session:
            records = session.scalars(select(AgentRecord).order_by(AgentRecord.name)).all()
            return [
                _snapshot(
                    record,
                    {item.skill_name: item.skill_version for item in record.skill_attachments},
                )
                for record in records
            ]

    def set_active(self, agent_id: str, is_active: bool) -> AgentSnapshot:
        with self._session_factory() as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                raise AgentNotFoundError(f"Agent does not exist: {agent_id}")
            record.is_active = is_active
            session.commit()
            session.refresh(record)
            skill_versions = {
                item.skill_name: item.skill_version for item in record.skill_attachments
            }
            return _snapshot(record, skill_versions)

    def delete_agent(self, agent_id: str) -> None:
        with self._session_factory() as session:
            record = session.get(AgentRecord, agent_id)
            if record is None:
                raise AgentNotFoundError(f"Agent does not exist: {agent_id}")
            task_count = session.scalar(
                select(func.count()).select_from(TaskRecord).where(TaskRecord.agent_id == agent_id)
            )
            if task_count:
                raise AgentInUseError("Agents with execution history cannot be deleted")
            session.delete(record)
            session.commit()

    def _validate_attachments(self, config: AgentConfig) -> dict[str, str]:
        versions: dict[str, str] = {}
        for skill_name in config.skill_names:
            skill = self._skill_registry.load(skill_name)
            versions[skill_name] = skill.metadata.version
        for tool_name in config.tool_names:
            self._tool_registry.get(tool_name)
        return versions


def _agent_record(config: AgentConfig) -> AgentRecord:
    return AgentRecord(
        name=config.name,
        description=config.description,
        provider=config.provider.value,
        model=config.model,
        system_instructions=config.system_instructions,
        developer_instructions=config.developer_instructions,
        temperature=config.temperature,
        max_iterations=config.max_iterations,
        max_tool_calls=config.max_tool_calls,
        request_timeout_seconds=config.request_timeout_seconds,
    )


def _update_record(record: AgentRecord, config: AgentConfig) -> None:
    record.name = config.name
    record.description = config.description
    record.provider = config.provider.value
    record.model = config.model
    record.system_instructions = config.system_instructions
    record.developer_instructions = config.developer_instructions
    record.temperature = config.temperature
    record.max_iterations = config.max_iterations
    record.max_tool_calls = config.max_tool_calls
    record.request_timeout_seconds = config.request_timeout_seconds


def _snapshot(record: AgentRecord, skill_versions: dict[str, str]) -> AgentSnapshot:
    config = AgentConfig(
        name=record.name,
        description=record.description,
        provider=ProviderName(record.provider),
        model=record.model,
        system_instructions=record.system_instructions,
        developer_instructions=record.developer_instructions,
        temperature=record.temperature,
        max_iterations=record.max_iterations,
        max_tool_calls=record.max_tool_calls,
        request_timeout_seconds=record.request_timeout_seconds,
        skill_names=sorted(skill_versions),
        tool_names=sorted(item.tool_name for item in record.tool_attachments),
    )
    return AgentSnapshot(
        agent_id=record.id,
        config=config,
        is_active=record.is_active,
        skill_versions=skill_versions,
    )