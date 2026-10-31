from pathlib import Path

import pytest

from agents.agent_manager import AgentManager
from agents.execution_loop import AgentExecutor
from core.enums import ExecutionStatus, ProviderName
from core.schemas import AgentConfig, ProviderRequest, ProviderResponse, TokenUsage
from database.connection import create_database_engine, create_schema, create_session_factory
from evaluation.schemas import HumanEvaluationSubmission
from evaluation.service import EvaluationManager
from experiments.comparison_engine import ComparisonEngine
from experiments.dataset_manager import DatasetManager
from experiments.experiment_manager import ExperimentConfigurationError, ExperimentManager
from experiments.experiment_runner import ExperimentRunner
from experiments.schemas import ExperimentDataset, ExperimentStatus
from instrumentation.event_tracker import EventTracker
from instrumentation.llm_tracker import LLMTracker
from skills.skill_registry import SkillRegistry
from tools.tool_manager import ToolManager
from tools.tool_registry import ToolRegistry


class SequentialProvider:
    provider_name = ProviderName.OPENAI

    def __init__(self) -> None:
        self._token_usages = iter(
            [
                TokenUsage(input_tokens=100, output_tokens=50),
                TokenUsage(input_tokens=70, output_tokens=30),
            ]
        )

    def complete(self, request: ProviderRequest) -> ProviderResponse:
        usage = next(self._token_usages)
        return ProviderResponse(
            provider=ProviderName.OPENAI,
            model=request.model,
            content="Result is 42.",
            usage=usage,
            raw_usage={"total_tokens": usage.input_tokens + usage.output_tokens},
            latency_ms=5.0,
            finish_reason="stop",
        )


@pytest.fixture
def experiment_environment():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry(Path(__file__).parents[1] / "skills" / "available")
    agent_manager = AgentManager(session_factory, skill_registry, tool_registry)
    event_tracker = EventTracker(session_factory)
    tool_manager = ToolManager(tool_registry)
    executor = AgentExecutor(
        session_factory,
        agent_manager,
        skill_registry,
        tool_registry,
        tool_manager,
        LLMTracker(session_factory, event_tracker=event_tracker),
        event_tracker,
    )
    evaluation_manager = EvaluationManager(session_factory)
    experiment_manager = ExperimentManager(session_factory, agent_manager)
    comparison_engine = ComparisonEngine(session_factory, experiment_manager)
    runner = ExperimentRunner(
        experiment_manager,
        agent_manager,
        executor,
        evaluation_manager,
        comparison_engine,
    )
    yield {
        "session_factory": session_factory,
        "agent_manager": agent_manager,
        "experiment_manager": experiment_manager,
        "comparison_engine": comparison_engine,
        "runner": runner,
        "evaluation_manager": evaluation_manager,
        "tool_manager": tool_manager,
        "engine": engine,
    }
    tool_manager.shutdown()
    engine.dispose()


def make_agent(agent_manager: AgentManager, name: str, model: str):
    return agent_manager.create_agent(
        AgentConfig(
            name=name,
            provider=ProviderName.OPENAI,
            model=model,
            max_iterations=2,
            max_tool_calls=0,
            skill_names=[],
            tool_names=[],
        )
    )


def test_experiment_runner_records_paired_runs_and_reductions(experiment_environment) -> None:
    services = experiment_environment
    baseline = make_agent(services["agent_manager"], "Baseline", "baseline-model")
    optimized = make_agent(services["agent_manager"], "Optimized", "optimized-model")
    dataset = ExperimentDataset(
        name="Arithmetic smoke set",
        description="One deterministic task",
        cases=[
            {
                "case_id": "case-001",
                "name": "Answer 42",
                "prompt": "What is the answer to life, the universe, and everything?",
                "expected_outcome": "42",
            }
        ],
    )
    experiment = services["experiment_manager"].create(
        "Prompt compression trial",
        "Compare two saved configurations.",
        baseline.agent_id,
        optimized.agent_id,
        dataset,
    )
    provider = SequentialProvider()

    comparison = services["runner"].run(experiment.experiment_id, lambda _: provider)

    assert comparison.experiment.status is ExperimentStatus.COMPLETED
    assert len(comparison.results) == 2
    assert comparison.baseline.reported_total_tokens == 150
    assert comparison.optimized.reported_total_tokens == 100
    assert comparison.token_reduction_percent == pytest.approx(100 / 3)
    assert comparison.baseline.task_success_rate == 1.0
    assert comparison.optimized.task_success_rate == 1.0
    assert comparison.success_rate_change == 0
    assert comparison.human_quality_change is None
    assert comparison.baseline.human_evaluations == 0
    assert comparison.optimized.human_evaluations == 0

    # Completed experiments are read-only and do not create duplicate runs.
    repeated = services["runner"].run(experiment.experiment_id, lambda _: provider)
    assert len(repeated.results) == 2


def test_human_quality_change_is_reported_alongside_success_and_tokens(experiment_environment) -> None:
    services = experiment_environment
    baseline = make_agent(services["agent_manager"], "Baseline", "baseline-model")
    optimized = make_agent(services["agent_manager"], "Optimized", "optimized-model")
    dataset = DatasetManager().parse_json(
        '{"name":"One case","cases":[{"case_id":"c1","name":"Task","prompt":"Do task"}]}'
    )
    experiment = services["experiment_manager"].create(
        "Quality comparison", "", baseline.agent_id, optimized.agent_id, dataset
    )
    provider = SequentialProvider()
    comparison = services["runner"].run(experiment.experiment_id, lambda _: provider)
    baseline_result = next(result for result in comparison.results if result.arm.value == "baseline")
    optimized_result = next(result for result in comparison.results if result.arm.value == "optimized")
    services["evaluation_manager"].submit_human_evaluation(
        HumanEvaluationSubmission(
            run_id=baseline_result.run_id,
            evaluator_name="reviewer",
            response_relevance=4,
            response_completeness=4,
            output_quality=4,
        )
    )
    services["evaluation_manager"].submit_human_evaluation(
        HumanEvaluationSubmission(
            run_id=optimized_result.run_id,
            evaluator_name="reviewer",
            response_relevance=3,
            response_completeness=3,
            output_quality=3,
        )
    )

    compared = services["comparison_engine"].compare(experiment.experiment_id)

    assert compared.token_reduction_percent == pytest.approx(100 / 3)
    assert compared.success_rate_change == 0
    assert compared.human_quality_change == -1
    assert compared.baseline.average_human_quality == 4
    assert compared.optimized.average_human_quality == 3


def test_experiment_refuses_agent_config_drift(experiment_environment) -> None:
    services = experiment_environment
    baseline = make_agent(services["agent_manager"], "Baseline", "baseline-model")
    optimized = make_agent(services["agent_manager"], "Optimized", "optimized-model")
    experiment = services["experiment_manager"].create(
        "Snapshot check",
        "",
        baseline.agent_id,
        optimized.agent_id,
        ExperimentDataset(
            name="One case",
            cases=[{"case_id": "c1", "name": "Task", "prompt": "Do task"}],
        ),
    )
    services["agent_manager"].update_agent(
        baseline.agent_id,
        AgentConfig(
            name="Changed baseline",
            provider=ProviderName.OPENAI,
            model="baseline-model",
            max_iterations=2,
            max_tool_calls=0,
        ),
    )

    with pytest.raises(ExperimentConfigurationError, match="configuration changed"):
        services["runner"].run(experiment.experiment_id, lambda _: SequentialProvider())

    saved = services["experiment_manager"].get(experiment.experiment_id)
    assert saved.status is ExperimentStatus.FAILED


def test_dataset_manager_rejects_duplicate_case_ids() -> None:
    raw = (
        '{"name":"Bad dataset","cases":['
        '{"case_id":"same","name":"One","prompt":"First"},'
        '{"case_id":"same","name":"Two","prompt":"Second"}]}'
    )

    with pytest.raises(ValueError, match="invalid"):
        DatasetManager().parse_json(raw)