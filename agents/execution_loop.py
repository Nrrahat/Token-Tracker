"""Transparent synchronous agent loop with provider/tool instrumentation."""

import hashlib
import json
from datetime import datetime, timezone
from threading import Event
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.orm import sessionmaker

from agents.agent_manager import AgentManager
from agents.schemas import AgentSnapshot, ExecutionResult
from agents.stopping_conditions import StoppingConditions
from core.enums import EventType, ExecutionStatus, ProviderName
from core.schemas import ChatMessage, ProviderRequest, ProviderResponse, TaskSubmission
from database.models import ExecutionRunRecord, TaskRecord
from instrumentation.event_tracker import EventTracker
from instrumentation.llm_tracker import LLMTracker
from skills.skill_registry import SkillRegistry
from token_engine.schemas import PromptComponent, PromptComponentKind
from tools.base_tool import ToolExecutionResult, ToolStatus
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolNotFoundError, ToolRegistry
from providers.base_provider import LLMProvider


class _StopExecution(Exception):
    def __init__(self, status: ExecutionStatus, error_type: str) -> None:
        super().__init__(error_type)
        self.status = status
        self.error_type = error_type


class AgentExecutor:
    def __init__(
        self,
        session_factory: sessionmaker,
        agent_manager: AgentManager,
        skill_registry: SkillRegistry,
        tool_registry: ToolRegistry,
        tool_manager: ToolManager,
        llm_tracker: LLMTracker,
        event_tracker: EventTracker | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._agent_manager = agent_manager
        self._skill_registry = skill_registry
        self._tool_registry = tool_registry
        self._tool_manager = tool_manager
        self._llm_tracker = llm_tracker
        self._event_tracker = event_tracker or EventTracker(session_factory)

    def execute(
        self,
        agent_id: str,
        prompt: str,
        provider: LLMProvider,
        cancellation_event: Event | None = None,
    ) -> ExecutionResult:
        agent = self._agent_manager.get_agent(agent_id)
        if not agent.is_active:
            raise ValueError("Agent is inactive")
        if provider.provider_name is not agent.config.provider:
            raise ValueError("Provider adapter does not match the agent configuration")
        submission = TaskSubmission(agent_id=UUID(agent_id), prompt=prompt)
        task_id, run_id = self._create_run(agent, submission.prompt)
        iterations = 0
        tool_calls_count = 0
        answer: str | None = None
        error_type: str | None = None
        status = ExecutionStatus.SUCCEEDED

        self._event_tracker.record_event(
            run_id,
            EventType.TASK_SUBMITTED,
            status=ExecutionStatus.RUNNING,
            metadata={"task_id": task_id, "agent_id": agent_id},
        )
        self._event_tracker.record_event(
            run_id,
            EventType.AGENT_STARTED,
            status=ExecutionStatus.RUNNING,
            metadata={"agent_id": agent_id},
        )

        try:
            skills = self._load_skills(agent)
            tools = [self._tool_registry.get(name) for name in agent.config.tool_names]
            messages, fixed_components = self._initial_context(agent, skills, submission.prompt, tools)
            history_components: list[PromptComponent] = []
            tool_definitions = [tool.provider_definition() for tool in tools]

            while not StoppingConditions.iteration_limit_reached(
                iterations, agent.config.max_iterations
            ):
                if StoppingConditions.cancellation_requested(cancellation_event):
                    raise _StopExecution(ExecutionStatus.CANCELLED, "ExecutionCancelled")

                request = ProviderRequest(
                    model=agent.config.model,
                    messages=messages.copy(),
                    temperature=agent.config.temperature,
                    tools=tool_definitions or None,
                    timeout_seconds=agent.config.request_timeout_seconds,
                )
                response = self._llm_tracker.complete(
                    run_id,
                    provider,
                    request,
                    prompt_components=[*fixed_components, *history_components],
                )
                iterations += 1

                if not response.tool_calls:
                    answer = response.content or ""
                    self._record_iteration(run_id, iterations, tool_calls_count)
                    break

                self._append_assistant_tool_request(messages, history_components, response)
                for tool_call in response.tool_calls:
                    if StoppingConditions.cancellation_requested(cancellation_event):
                        raise _StopExecution(ExecutionStatus.CANCELLED, "ExecutionCancelled")
                    tool_calls_count += 1
                    requested_event = self._event_tracker.record_event(
                        run_id,
                        EventType.TOOL_REQUESTED,
                        status=ExecutionStatus.RUNNING,
                        metadata={
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.id,
                            "argument_fingerprint": _argument_fingerprint(tool_call.arguments),
                        },
                    )
                    if tool_calls_count > agent.config.max_tool_calls:
                        self._event_tracker.record_event(
                            run_id,
                            EventType.TOOL_FAILED,
                            status=ExecutionStatus.FAILED,
                            metadata={
                                "tool_name": tool_call.name,
                                "tool_call_id": tool_call.id,
                                "argument_fingerprint": _argument_fingerprint(tool_call.arguments),
                            },
                            parent_event_id=requested_event.id,
                            error_details="Maximum tool-call limit reached",
                        )
                        raise _StopExecution(ExecutionStatus.FAILED, "MaxToolCallsExceeded")
                    result = self._execute_tool(
                        run_id,
                        agent,
                        tool_call.name,
                        tool_call.arguments,
                        tool_call.id,
                        requested_event.id,
                    )
                    tool_message = _tool_message(tool_call.id, tool_call.name, result)
                    messages.append(tool_message)
                    history_components.append(
                        PromptComponent(
                            kind=PromptComponentKind.TOOL_RESULT,
                            source_id=tool_call.name,
                            text=tool_message.content or "",
                        )
                    )
                self._record_iteration(run_id, iterations, tool_calls_count)
            else:
                raise _StopExecution(ExecutionStatus.FAILED, "MaxIterationsExceeded")
        except _StopExecution as stop:
            status = stop.status
            error_type = stop.error_type
        except Exception as error:
            status = ExecutionStatus.FAILED
            error_type = type(error).__name__

        self._finish_run(run_id, status, answer, error_type)
        final_event = (
            EventType.AGENT_FINISHED
            if status is ExecutionStatus.SUCCEEDED
            else EventType.AGENT_CANCELLED
            if status is ExecutionStatus.CANCELLED
            else EventType.AGENT_FAILED
        )
        self._event_tracker.record_event(
            run_id,
            final_event,
            status=status,
            metadata={
                "iterations": iterations,
                "tool_calls": tool_calls_count,
                "error_type": error_type,
            },
            error_details=error_type,
        )
        return ExecutionResult(
            task_id=task_id,
            run_id=run_id,
            status=status,
            answer=answer,
            error_type=error_type,
            iterations=iterations,
            tool_calls=tool_calls_count,
        )

    def _create_run(self, agent: AgentSnapshot, prompt: str) -> tuple[str, str]:
        with self._session_factory() as session:
            task = TaskRecord(agent_id=agent.agent_id, prompt=prompt)
            session.add(task)
            session.flush()
            run = ExecutionRunRecord(
                task_id=task.id,
                agent_id=agent.agent_id,
                status=ExecutionStatus.RUNNING.value,
                started_at=datetime.now(timezone.utc),
            )
            session.add(run)
            session.commit()
            return task.id, run.id

    def _load_skills(self, agent: AgentSnapshot) -> list[Any]:
        skills = self._skill_registry.load_many(agent.config.skill_names)
        for skill in skills:
            if agent.skill_versions.get(skill.metadata.name) != skill.metadata.version:
                raise ValueError(f"Attached skill version changed: {skill.metadata.name}")
        return skills

    def _initial_context(
        self,
        agent: AgentSnapshot,
        skills: list[Any],
        prompt: str,
        tools: list[Any],
    ) -> tuple[list[ChatMessage], list[PromptComponent]]:
        messages: list[ChatMessage] = []
        components: list[PromptComponent] = []
        config = agent.config
        if config.system_instructions:
            messages.append(ChatMessage(role="system", content=config.system_instructions))
            components.append(
                PromptComponent(
                    kind=PromptComponentKind.SYSTEM_INSTRUCTION,
                    text=config.system_instructions,
                )
            )
        for skill in skills:
            messages.append(ChatMessage(role="system", content=skill.instructions))
            components.append(
                PromptComponent(
                    kind=PromptComponentKind.SKILL_INSTRUCTION,
                    source_id=f"{skill.metadata.name}@{skill.metadata.version}",
                    text=skill.instructions,
                )
            )
        if config.developer_instructions:
            messages.append(ChatMessage(role="developer", content=config.developer_instructions))
            components.append(
                PromptComponent(
                    kind=PromptComponentKind.DEVELOPER_INSTRUCTION,
                    text=config.developer_instructions,
                )
            )
        messages.append(ChatMessage(role="user", content=prompt))
        components.append(PromptComponent(kind=PromptComponentKind.USER_PROMPT, text=prompt))

        for tool in tools:
            definition = tool.provider_definition()
            tool_text = json.dumps(
                {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            components.append(
                PromptComponent(
                    kind=PromptComponentKind.TOOL_SCHEMA,
                    source_id=tool.name,
                    text=tool_text,
                )
            )
        return messages, components

    def _execute_tool(
        self,
        run_id: str,
        agent: AgentSnapshot,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        requested_event_id: str,
    ) -> ToolExecutionResult:
        started_event = self._event_tracker.record_event(
            run_id,
            EventType.TOOL_STARTED,
            status=ExecutionStatus.RUNNING,
            metadata={"tool_name": tool_name, "tool_call_id": tool_call_id},
            parent_event_id=requested_event_id,
        )
        if tool_name not in agent.config.tool_names:
            result = ToolExecutionResult(
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error_type="ToolNotAllowed",
                error_message="Tool is not enabled for this agent.",
                duration_ms=0,
            )
        else:
            try:
                result = self._tool_manager.execute(tool_name, arguments)
            except ToolNotFoundError:
                result = ToolExecutionResult(
                    tool_name=tool_name,
                    status=ToolStatus.FAILED,
                    error_type="ToolNotRegistered",
                    error_message="Tool is not available.",
                    duration_ms=0,
                )

        succeeded = result.status is ToolStatus.SUCCEEDED
        self._event_tracker.record_event(
            run_id,
            EventType.TOOL_COMPLETED if succeeded else EventType.TOOL_FAILED,
            status=ExecutionStatus.SUCCEEDED if succeeded else ExecutionStatus.FAILED,
            metadata={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "argument_fingerprint": _argument_fingerprint(arguments),
                "duration_ms": result.duration_ms,
                "output_size_bytes": result.output_size_bytes,
                "estimated_output_tokens": result.estimated_output_tokens,
                "is_empty": result.output_metadata.get("is_empty"),
                "error_type": result.error_type,
            },
            duration_ms=result.duration_ms,
            parent_event_id=started_event.id,
            error_details=result.error_type,
        )
        return result

    @staticmethod
    def _append_assistant_tool_request(
        messages: list[ChatMessage],
        history_components: list[PromptComponent],
        response: ProviderResponse,
    ) -> None:
        messages.append(
            ChatMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)
        )
        if response.content:
            history_components.append(
                PromptComponent(
                    kind=PromptComponentKind.CONVERSATION_HISTORY,
                    text=response.content,
                )
            )
        history_components.append(
            PromptComponent(
                kind=PromptComponentKind.CONVERSATION_HISTORY,
                text=json.dumps(
                    [tool_call.model_dump(mode="json") for tool_call in response.tool_calls],
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    def _record_iteration(self, run_id: str, iteration: int, tool_calls: int) -> None:
        self._event_tracker.record_event(
            run_id,
            EventType.ITERATION_COMPLETED,
            status=ExecutionStatus.RUNNING,
            metadata={"iteration": iteration, "tool_calls": tool_calls},
        )

    def _finish_run(
        self,
        run_id: str,
        status: ExecutionStatus,
        answer: str | None,
        error_type: str | None,
    ) -> None:
        with self._session_factory() as session:
            run = session.get(ExecutionRunRecord, run_id)
            if run is None:
                raise ValueError(f"Execution run does not exist: {run_id}")
            run.status = status.value
            run.finished_at = datetime.now(timezone.utc)
            run.result = answer
            run.error_message = error_type
            session.commit()

def _tool_message(tool_call_id: str, tool_name: str, result: ToolExecutionResult) -> ChatMessage:
    content = result.output_text if result.status is ToolStatus.SUCCEEDED else json.dumps(
        {"error": result.error_message or "Tool execution failed."},
        separators=(",", ":"),
    )
    return ChatMessage(role="tool", name=tool_name, tool_call_id=tool_call_id, content=content)


def _argument_fingerprint(arguments: dict[str, Any]) -> str | None:
    try:
        canonical_arguments = json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canonical_arguments.encode("utf-8")).hexdigest()