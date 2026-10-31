"""Persist immutable experiment definitions and paired run references."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from agents.agent_manager import AgentManager
from database.models import ExperimentRecord, ExperimentResultRecord
from experiments.dataset_manager import DatasetManager
from experiments.schemas import (
    AgentConfigSnapshot,
    ExperimentArm,
    ExperimentCase,
    ExperimentDataset,
    ExperimentDefinition,
    ExperimentResult,
    ExperimentStatus,
)


class ExperimentNotFoundError(LookupError):
    """Raised when an experiment ID does not exist."""


class ExperimentConfigurationError(ValueError):
    """Raised when experiment arms cannot provide a valid reproducible comparison."""


class ExperimentManager:
    def __init__(
        self,
        session_factory: sessionmaker,
        agent_manager: AgentManager,
        dataset_manager: DatasetManager | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._agent_manager = agent_manager
        self._dataset_manager = dataset_manager or DatasetManager()

    def create(
        self,
        name: str,
        description: str,
        baseline_agent_id: str,
        optimized_agent_id: str,
        dataset: ExperimentDataset,
    ) -> ExperimentDefinition:
        if not name.strip() or len(name) > 160:
            raise ExperimentConfigurationError("Experiment name must contain 1 to 160 characters")
        if len(description) > 2000:
            raise ExperimentConfigurationError("Experiment description cannot exceed 2000 characters")
        if baseline_agent_id == optimized_agent_id:
            raise ExperimentConfigurationError("Baseline and optimized arms must use different agents")
        baseline = self._agent_manager.get_agent(baseline_agent_id)
        optimized = self._agent_manager.get_agent(optimized_agent_id)
        if not baseline.is_active or not optimized.is_active:
            raise ExperimentConfigurationError("Both experiment agents must be active")

        baseline_snapshot = _agent_snapshot(baseline)
        optimized_snapshot = _agent_snapshot(optimized)
        with self._session_factory() as session:
            record = ExperimentRecord(
                id=str(uuid4()),
                name=name,
                description=description,
                status=ExperimentStatus.DRAFT.value,
                baseline_agent_id=baseline_agent_id,
                optimized_agent_id=optimized_agent_id,
                baseline_snapshot=baseline_snapshot.model_dump(mode="json"),
                optimized_snapshot=optimized_snapshot.model_dump(mode="json"),
                dataset_snapshot=self._dataset_manager.serialize(dataset),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return _definition(record)

    def get(self, experiment_id: str) -> ExperimentDefinition:
        with self._session_factory() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise ExperimentNotFoundError(f"Experiment does not exist: {experiment_id}")
            return _definition(record)

    def list_experiments(self) -> list[ExperimentDefinition]:
        with self._session_factory() as session:
            records = session.scalars(
                select(ExperimentRecord).order_by(ExperimentRecord.created_at.desc())
            ).all()
            return [_definition(record) for record in records]

    def results(self, experiment_id: str) -> list[ExperimentResult]:
        with self._session_factory() as session:
            if session.get(ExperimentRecord, experiment_id) is None:
                raise ExperimentNotFoundError(f"Experiment does not exist: {experiment_id}")
            records = session.scalars(
                select(ExperimentResultRecord)
                .where(ExperimentResultRecord.experiment_id == experiment_id)
                .order_by(ExperimentResultRecord.case_id, ExperimentResultRecord.arm)
            ).all()
            return [
                ExperimentResult(
                    case_id=record.case_id,
                    arm=ExperimentArm(record.arm),
                    run_id=record.run_id,
                    status=record.status,
                )
                for record in records
            ]

    def result_exists(self, experiment_id: str, case_id: str, arm: ExperimentArm) -> bool:
        with self._session_factory() as session:
            return session.scalar(
                select(ExperimentResultRecord.id).where(
                    ExperimentResultRecord.experiment_id == experiment_id,
                    ExperimentResultRecord.case_id == case_id,
                    ExperimentResultRecord.arm == arm.value,
                )
            ) is not None

    def record_result(
        self,
        experiment_id: str,
        case: ExperimentCase,
        arm: ExperimentArm,
        run_id: str,
        status: str,
    ) -> ExperimentResult:
        with self._session_factory() as session:
            if session.get(ExperimentRecord, experiment_id) is None:
                raise ExperimentNotFoundError(f"Experiment does not exist: {experiment_id}")
            existing = session.scalar(
                select(ExperimentResultRecord).where(
                    ExperimentResultRecord.experiment_id == experiment_id,
                    ExperimentResultRecord.case_id == case.case_id,
                    ExperimentResultRecord.arm == arm.value,
                )
            )
            if existing is not None:
                return ExperimentResult(
                    case_id=existing.case_id,
                    arm=ExperimentArm(existing.arm),
                    run_id=existing.run_id,
                    status=existing.status,
                )
            record = ExperimentResultRecord(
                id=str(uuid4()),
                experiment_id=experiment_id,
                case_id=case.case_id,
                arm=arm.value,
                run_id=run_id,
                status=status,
                recorded_at=datetime.now(timezone.utc),
            )
            session.add(record)
            session.commit()
            return ExperimentResult(
                case_id=case.case_id,
                arm=arm,
                run_id=run_id,
                status=status,
            )

    def set_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        error_type: str | None = None,
    ) -> ExperimentDefinition:
        with self._session_factory() as session:
            record = session.get(ExperimentRecord, experiment_id)
            if record is None:
                raise ExperimentNotFoundError(f"Experiment does not exist: {experiment_id}")
            record.status = status.value
            record.error_type = error_type
            if status is ExperimentStatus.RUNNING and record.started_at is None:
                record.started_at = datetime.now(timezone.utc)
            if status in {ExperimentStatus.COMPLETED, ExperimentStatus.FAILED}:
                record.finished_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(record)
            return _definition(record)


def _agent_snapshot(agent) -> AgentConfigSnapshot:
    return AgentConfigSnapshot(
        agent_id=agent.agent_id,
        name=agent.config.name,
        config=agent.config.model_dump(mode="json"),
        skill_versions=agent.skill_versions,
    )


def _definition(record: ExperimentRecord) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id=record.id,
        name=record.name,
        description=record.description,
        status=ExperimentStatus(record.status),
        baseline_snapshot=AgentConfigSnapshot.model_validate(record.baseline_snapshot),
        optimized_snapshot=AgentConfigSnapshot.model_validate(record.optimized_snapshot),
        dataset=ExperimentDataset.model_validate(record.dataset_snapshot),
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        error_type=record.error_type,
    )