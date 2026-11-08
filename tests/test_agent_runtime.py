from pathlib import Path
from threading import Event

import pytest
from sqlalchemy import select

from agents.agent_manager import AgentInUseError, AgentManager
from agents.execution_loop import AgentExecutor
from core.enums import EventType, ExecutionStatus, ProviderName
from core.schemas import AgentConfig, ProviderRequest, ProviderResponse, TokenUsage, ToolCall
from database.connection import create_database_engine, create_schema, create_session_factory
from database.models import ExecutionRunRecord
from instrumentation.event_tracker import EventTracker
from instrumentation.llm_tracker import LLMTracker
from instrumentation.trace_manager import TraceManager
from providers.base_provider import LLMProvider
from skills.skill_registry import SkillRegistry
from tools.available.calculator import CalculatorTool
from tools.custom_python_tool import sync_custom_tools
from tools.custom_tool_store import CustomToolStore
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    provider_name = ProviderName.OPENAI

    def __init__(self, responses: list[ProviderResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[ProviderRequest] = []

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return next(self._responses)


@pytest.fixture
def runtime_environment(tmp_path):
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    tool_registry = ToolRegistry()
    tool_registry.register(CalculatorTool())
    custom_tool_store = CustomToolStore(tmp_path / "custom_tools")
    skill_root = Path(__file__).parents[1] / "skills" / "available"
    skill_registry = SkillRegistry(skill_root)
    agent_manager = AgentManager(session_factory, skill_registry, tool_registry)
    tool_manager = ToolManager(tool_registry)
    event_tracker = EventTracker(session_factory)
    llm_tracker = LLMTracker(session_factory, event_tracker=event_tracker)
    executor = AgentExecutor(
        session_factory,
        agent_manager,
        skill_registry,
        tool_registry,
        tool_manager,
        llm_tracker,
        event_tracker,
    )
    yield session_factory, agent_manager, tool_manager, executor, tool_registry, custom_tool_store
    tool_manager.shutdown()
    engine.dispose()


def make_tool_response() -> ProviderResponse:
    return ProviderResponse(
        provider=ProviderName.OPENAI,
        model="test-model",
        tool_calls=[
            ToolCall(id="call-1", name="calculator", arguments={"expression": "3 * 7"})
        ],
        usage=TokenUsage(input_tokens=50, output_tokens=10),
        raw_usage={"total_tokens": 60},
        latency_ms=3.0,
        finish_reason="tool_calls",
    )


def make_final_response(answer: str = "The result is 21.") -> ProviderResponse:
    return ProviderResponse(
        provider=ProviderName.OPENAI,
        model="test-model",
        content=answer,
        usage=TokenUsage(input_tokens=70, output_tokens=8),
        raw_usage={"total_tokens": 78},
        latency_ms=2.0,
        finish_reason="stop",
    )


def make_agent_config(**overrides) -> AgentConfig:
    values = {
        "name": "Calculator agent",
        "provider": ProviderName.OPENAI,
        "model": "test-model",
        "system_instructions": "Answer accurately.",
        "developer_instructions": "Show concise results.",
        "max_iterations": 3,
        "max_tool_calls": 2,
        "request_timeout_seconds": 45.0,
        "skill_names": ["document_analysis"],
        "tool_names": ["calculator"],
    }
    values.update(overrides)
    return AgentConfig(**values)


def test_agent_manager_persists_config_and_versioned_attachments(runtime_environment) -> None:
    _, agent_manager, _, _, _, _ = runtime_environment

    agent = agent_manager.create_agent(make_agent_config())
    fetched = agent_manager.get_agent(agent.agent_id)

    assert fetched.config.name == "Calculator agent"
    assert fetched.config.skill_names == ["document_analysis"]
    assert fetched.config.tool_names == ["calculator"]
    assert fetched.skill_versions == {"document_analysis": "1.0.0"}
    assert agent_manager.list_agents()[0].agent_id == agent.agent_id

    deactivated = agent_manager.set_active(agent.agent_id, False)
    assert deactivated.is_active is False


def test_agent_manager_replaces_attachments_and_prevents_deleting_task_history(runtime_environment) -> None:
    session_factory, agent_manager, _, executor, _, _ = runtime_environment
    agent = agent_manager.create_agent(make_agent_config())
    updated = agent_manager.update_agent(
        agent.agent_id,
        make_agent_config(
            name="Updated calculator",
            skill_names=[],
            tool_names=[],
            max_tool_calls=0,
        ),
    )

    assert updated.config.name == "Updated calculator"
    assert updated.config.skill_names == []
    assert updated.config.tool_names == []

    provider = ScriptedProvider([make_final_response()])
    result = executor.execute(agent.agent_id, "Say hello", provider)
    assert result.status is ExecutionStatus.SUCCEEDED
    with pytest.raises(AgentInUseError):
        agent_manager.delete_agent(agent.agent_id)

    with session_factory() as session:
        run = session.get(ExecutionRunRecord, result.run_id)
        assert run is not None


def test_complete_task_calls_tool_then_finishes_with_persisted_trace(runtime_environment) -> None:
    session_factory, agent_manager, tool_manager, executor, _, _ = runtime_environment
    agent = agent_manager.create_agent(make_agent_config())
    provider = ScriptedProvider([make_tool_response(), make_final_response()])

    result = executor.execute(agent.agent_id, "What is 3 times 7?", provider)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert result.answer == "The result is 21."
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert len(provider.requests) == 2
    assert provider.requests[0].timeout_seconds == 45.0
    assert [message.role for message in provider.requests[0].messages] == [
        "system",
        "system",
        "developer",
        "user",
    ]
    assert provider.requests[0].tools[0].name == "calculator"
    tool_reply = provider.requests[1].messages[-1]
    assert tool_reply.role == "tool"
    assert tool_reply.tool_call_id == "call-1"
    assert '"result":"21"' in tool_reply.content

    trace = TraceManager(session_factory).get_trace(result.run_id)
    assert trace is not None
    assert [event.event_type for event in trace.events].count(EventType.TOOL_COMPLETED.value) == 1
    completed_tool = next(
        event for event in trace.events if event.event_type == EventType.TOOL_COMPLETED.value
    )
    assert len(completed_tool.metadata["argument_fingerprint"]) == 64
    assert "arguments" not in completed_tool.metadata
    assert [event.event_type for event in trace.events][-1] == EventType.AGENT_FINISHED.value
    assert len(trace.llm_calls) == 2
    assert trace.llm_calls[0].estimated_component_tokens is not None
    assert tool_manager.metrics()["calculator"].call_count == 1
    with session_factory() as session:
        run = session.get(ExecutionRunRecord, result.run_id)
        assert run is not None
        assert run.status == ExecutionStatus.SUCCEEDED.value
        assert run.result == result.answer


def test_execution_stops_at_iteration_limit(runtime_environment) -> None:
    _, agent_manager, _, executor, _, _ = runtime_environment
    agent = agent_manager.create_agent(make_agent_config(max_iterations=1))
    provider = ScriptedProvider([make_tool_response()])

    result = executor.execute(agent.agent_id, "Calculate 3 times 7", provider)

    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "MaxIterationsExceeded"
    assert result.iterations == 1
    assert len(provider.requests) == 1


def test_execution_enforces_tool_call_limit(runtime_environment) -> None:
    _, agent_manager, tool_manager, executor, _, _ = runtime_environment
    agent = agent_manager.create_agent(make_agent_config(max_tool_calls=0))
    provider = ScriptedProvider([make_tool_response()])

    result = executor.execute(agent.agent_id, "Calculate 3 times 7", provider)

    assert result.status is ExecutionStatus.FAILED
    assert result.error_type == "MaxToolCallsExceeded"
    assert result.tool_calls == 1
    assert tool_manager.metrics() == {}


def test_cancellation_before_first_request_is_persisted(runtime_environment) -> None:
    session_factory, agent_manager, _, executor, _, _ = runtime_environment
    agent = agent_manager.create_agent(make_agent_config())
    provider = ScriptedProvider([])
    cancellation = Event()
    cancellation.set()

    result = executor.execute(
        agent.agent_id,
        "This task is cancelled before execution.",
        provider,
        cancellation_event=cancellation,
    )

    assert result.status is ExecutionStatus.CANCELLED
    assert not provider.requests
    trace = TraceManager(session_factory).get_trace(result.run_id)
    assert trace is not None
    assert trace.events[-1].event_type == EventType.AGENT_CANCELLED.value
    with session_factory() as session:
        run = session.scalar(select(ExecutionRunRecord).where(ExecutionRunRecord.id == result.run_id))
        assert run is not None
        assert run.status == ExecutionStatus.CANCELLED.value


def test_agent_manager_rejects_unregistered_tools(runtime_environment) -> None:
    _, agent_manager, _, _, _, _ = runtime_environment

    with pytest.raises(LookupError):
        agent_manager.create_agent(make_agent_config(tool_names=["unregistered"]))


def test_agent_executes_persisted_custom_tool_from_registry(runtime_environment) -> None:
    session_factory, agent_manager, tool_manager, executor, registry, custom_store = runtime_environment
    custom_store.create(
        "answer_tool",
        "Add one to the supplied value.",
        "run",
        "def run(arguments):\n    return {'answer': arguments['value'] + 1}\n",
    )
    assert sync_custom_tools(CustomToolStore(custom_store.root), registry) == ["answer_tool"]
    agent = agent_manager.create_agent(
        make_agent_config(skill_names=[], tool_names=["answer_tool"])
    )
    provider = ScriptedProvider(
        [
            ProviderResponse(
                provider=ProviderName.OPENAI,
                model="test-model",
                tool_calls=[
                    ToolCall(
                        id="custom-call-1",
                        name="answer_tool",
                        arguments={"value": 41},
                    )
                ],
                latency_ms=1.0,
                finish_reason="tool_calls",
            ),
            make_final_response("The answer is 42."),
        ]
    )

    result = executor.execute(agent.agent_id, "Calculate the answer.", provider)

    assert result.status is ExecutionStatus.SUCCEEDED
    assert len(provider.requests[0].tools) == 1
    assert provider.requests[0].tools[0].name == "answer_tool"
    assert provider.requests[0].tools[0].parameters["additionalProperties"] is True
    assert provider.requests[1].messages[-1].tool_call_id == "custom-call-1"
    assert provider.requests[1].messages[-1].content == '{"answer":42}'
    assert tool_manager.metrics()["answer_tool"].call_count == 1
    trace = TraceManager(session_factory).get_trace(result.run_id)
    assert trace is not None
    assert any(event.event_type == EventType.TOOL_COMPLETED.value for event in trace.events)

    custom_store.update(
        "answer_tool",
        "Add two to the supplied value.",
        "run",
        "def run(arguments):\n    return {'answer': arguments['value'] + 2}\n",
    )
    sync_custom_tools(custom_store, registry)
    updated_provider = ScriptedProvider(
        [
            ProviderResponse(
                provider=ProviderName.OPENAI,
                model="test-model",
                tool_calls=[
                    ToolCall(
                        id="custom-call-2",
                        name="answer_tool",
                        arguments={"value": 40},
                    )
                ],
                latency_ms=1.0,
                finish_reason="tool_calls",
            ),
            make_final_response("The answer is 42."),
        ]
    )
    updated_result = executor.execute(agent.agent_id, "Calculate again.", updated_provider)
    assert updated_result.status is ExecutionStatus.SUCCEEDED
    assert updated_provider.requests[1].messages[-1].content == '{"answer":42}'

    custom_store.delete("answer_tool")
    sync_custom_tools(custom_store, registry)
    assert "answer_tool" not in {tool.name for tool in registry.list_tools()}