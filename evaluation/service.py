"""Execution-derived and human evaluation persistence."""

from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from core.enums import EventType, ExecutionStatus
from database.models import EvaluationRecord, ExecutionEventRecord, ExecutionRunRecord
from evaluation.schemas import (
    EvaluationSummary,
    EvaluationView,
    EvaluatorType,
    HumanEvaluationSubmission,
)


class EvaluationRunNotFoundError(LookupError):
    """Raised when evaluation targets an unknown execution run."""


class EvaluationRunIncompleteError(RuntimeError):
    """Raised when evaluation is requested before an execution has finished."""


class EvaluationManager:
    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def evaluate_execution(self, run_id: str) -> EvaluationView:
        with self._session_factory() as session:
            run = session.get(ExecutionRunRecord, run_id)
            if run is None:
                raise EvaluationRunNotFoundError(f"Execution run does not exist: {run_id}")
            if run.finished_at is None:
                raise EvaluationRunIncompleteError("An execution must finish before it can be evaluated")
            existing = session.scalar(
                select(EvaluationRecord).where(
                    EvaluationRecord.run_id == run_id,
                    EvaluationRecord.evaluator_type == EvaluatorType.AUTOMATIC.value,
                    EvaluationRecord.evaluator_name == "execution_runtime",
                )
            )
            tool_events = session.scalars(
                select(ExecutionEventRecord).where(
                    ExecutionEventRecord.run_id == run_id,
                    ExecutionEventRecord.event_type.in_(
                        [EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value]
                    ),
                )
            ).all()
            tool_use_success = None
            if tool_events:
                tool_use_success = all(
                    event.event_type == EventType.TOOL_COMPLETED.value for event in tool_events
                )
            record = existing or EvaluationRecord(
                id=str(uuid4()),
                run_id=run_id,
                evaluator_type=EvaluatorType.AUTOMATIC.value,
                evaluator_name="execution_runtime",
            )
            record.task_success = run.status == ExecutionStatus.SUCCEEDED.value
            record.tool_use_success = tool_use_success
            record.notes = "Derived from persisted execution status and tool events; no response-quality judgment was made."
            record.evaluated_at = datetime.now(timezone.utc)
            session.add(record)
            session.commit()
            session.refresh(record)
            return _view(record)

    def submit_human_evaluation(self, submission: HumanEvaluationSubmission) -> EvaluationView:
        with self._session_factory() as session:
            run = session.get(ExecutionRunRecord, submission.run_id)
            if run is None:
                raise EvaluationRunNotFoundError(
                    f"Execution run does not exist: {submission.run_id}"
                )
            if run.finished_at is None:
                raise EvaluationRunIncompleteError("An execution must finish before it can be evaluated")
            tool_events = session.scalars(
                select(ExecutionEventRecord).where(
                    ExecutionEventRecord.run_id == submission.run_id,
                    ExecutionEventRecord.event_type.in_(
                        [EventType.TOOL_COMPLETED.value, EventType.TOOL_FAILED.value]
                    ),
                )
            ).all()
            tool_use_success = None
            if tool_events:
                tool_use_success = all(
                    event.event_type == EventType.TOOL_COMPLETED.value for event in tool_events
                )
            record = EvaluationRecord(
                id=str(uuid4()),
                run_id=submission.run_id,
                evaluator_type=EvaluatorType.HUMAN.value,
                evaluator_name=submission.evaluator_name,
                task_success=run.status == ExecutionStatus.SUCCEEDED.value,
                tool_use_success=tool_use_success,
                response_relevance=submission.response_relevance,
                response_completeness=submission.response_completeness,
                output_quality=submission.output_quality,
                notes=submission.notes,
                evaluated_at=datetime.now(timezone.utc),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return _view(record)

    def list_evaluations(self, run_id: str | None = None) -> list[EvaluationView]:
        query = select(EvaluationRecord).order_by(EvaluationRecord.evaluated_at.desc())
        if run_id is not None:
            query = query.where(EvaluationRecord.run_id == run_id)
        with self._session_factory() as session:
            records = session.scalars(query).all()
            return [_view(record) for record in records]

    def summary(self) -> EvaluationSummary:
        evaluations = self.list_evaluations()
        automatic = [item for item in evaluations if item.evaluator_type is EvaluatorType.AUTOMATIC]
        human = [item for item in evaluations if item.evaluator_type is EvaluatorType.HUMAN]
        return EvaluationSummary(
            automatic_evaluations=len(automatic),
            automatic_success_rate=(
                sum(item.task_success for item in automatic) / len(automatic) if automatic else None
            ),
            human_evaluations=len(human),
            average_relevance=_average(item.response_relevance for item in human),
            average_completeness=_average(item.response_completeness for item in human),
            average_quality=_average(item.output_quality for item in human),
        )


def _view(record: EvaluationRecord) -> EvaluationView:
    return EvaluationView(
        id=record.id,
        run_id=record.run_id,
        evaluator_type=EvaluatorType(record.evaluator_type),
        evaluator_name=record.evaluator_name,
        task_success=record.task_success,
        tool_use_success=record.tool_use_success,
        response_relevance=record.response_relevance,
        response_completeness=record.response_completeness,
        output_quality=record.output_quality,
        notes=record.notes,
        evaluated_at=record.evaluated_at,
    )


def _average(values) -> float | None:
    scores = [value for value in values if value is not None]
    return sum(scores) / len(scores) if scores else None