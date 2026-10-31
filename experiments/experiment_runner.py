"""Run each dataset case against baseline and optimized configurations."""

import json
from collections.abc import Callable

from agents.agent_manager import AgentManager
from agents.execution_loop import AgentExecutor
from core.enums import ProviderName
from evaluation.service import EvaluationManager
from experiments.comparison_engine import ComparisonEngine
from experiments.experiment_manager import ExperimentConfigurationError, ExperimentManager
from experiments.schemas import ExperimentArm, ExperimentComparison, ExperimentStatus
from providers.base_provider import LLMProvider


class ExperimentRunner:
    def __init__(
        self,
        experiment_manager: ExperimentManager,
        agent_manager: AgentManager,
        agent_executor: AgentExecutor,
        evaluation_manager: EvaluationManager,
        comparison_engine: ComparisonEngine,
    ) -> None:
        self._experiment_manager = experiment_manager
        self._agent_manager = agent_manager
        self._agent_executor = agent_executor
        self._evaluation_manager = evaluation_manager
        self._comparison_engine = comparison_engine

    def run(
        self,
        experiment_id: str,
        provider_factory: Callable[[ProviderName], LLMProvider],
    ) -> ExperimentComparison:
        experiment = self._experiment_manager.get(experiment_id)
        if experiment.status is ExperimentStatus.COMPLETED:
            return self._comparison_engine.compare(experiment_id)
        self._experiment_manager.set_status(experiment_id, ExperimentStatus.RUNNING)
        try:
            for case in experiment.dataset.cases:
                for arm, snapshot in (
                    (ExperimentArm.BASELINE, experiment.baseline_snapshot),
                    (ExperimentArm.OPTIMIZED, experiment.optimized_snapshot),
                ):
                    if self._experiment_manager.result_exists(experiment_id, case.case_id, arm):
                        continue
                    agent = self._agent_manager.get_agent(snapshot.agent_id)
                    if not agent.is_active or not _snapshot_matches(agent, snapshot.model_dump(mode="json")):
                        raise ExperimentConfigurationError(
                            f"Agent configuration changed since experiment creation: {snapshot.name}"
                        )
                    provider = provider_factory(agent.config.provider)
                    result = self._agent_executor.execute(
                        agent.agent_id,
                        case.prompt,
                        provider,
                    )
                    self._evaluation_manager.evaluate_execution(result.run_id)
                    self._experiment_manager.record_result(
                        experiment_id,
                        case,
                        arm,
                        result.run_id,
                        result.status.value,
                    )
            self._experiment_manager.set_status(experiment_id, ExperimentStatus.COMPLETED)
        except Exception as error:
            self._experiment_manager.set_status(
                experiment_id,
                ExperimentStatus.FAILED,
                error_type=type(error).__name__,
            )
            raise
        return self._comparison_engine.compare(experiment_id)


def _snapshot_matches(agent, snapshot: dict[str, object]) -> bool:
    current = {
        "agent_id": agent.agent_id,
        "name": agent.config.name,
        "config": agent.config.model_dump(mode="json"),
        "skill_versions": agent.skill_versions,
    }
    return json.dumps(current, sort_keys=True) == json.dumps(snapshot, sort_keys=True)