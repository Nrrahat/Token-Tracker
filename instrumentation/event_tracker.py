"""Persist ordered execution events."""

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from core.enums import EventType, ExecutionStatus
from database.models import ExecutionEventRecord, ExecutionRunRecord


class EventTracker:
    """Append ordered events to an existing execution run."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def record_event(
        self,
        run_id: str,
        event_type: EventType,
        *,
        status: ExecutionStatus | str | None = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: float | None = None,
        parent_event_id: str | None = None,
        error_details: str | None = None,
    ) -> ExecutionEventRecord:
        with self._session_factory() as session:
            if session.get(ExecutionRunRecord, run_id) is None:
                raise ValueError(f"Execution run does not exist: {run_id}")
            if parent_event_id is not None:
                parent = session.get(ExecutionEventRecord, parent_event_id)
                if parent is None or parent.run_id != run_id:
                    raise ValueError("Parent event must belong to the same execution run")

            current_sequence = session.scalar(
                select(func.max(ExecutionEventRecord.sequence)).where(
                    ExecutionEventRecord.run_id == run_id
                )
            )
            status_value = status.value if isinstance(status, ExecutionStatus) else status
            record = ExecutionEventRecord(
                run_id=run_id,
                parent_event_id=parent_event_id,
                sequence=(current_sequence or 0) + 1,
                event_type=event_type.value,
                status=status_value,
                duration_ms=round(duration_ms) if duration_ms is not None else None,
                event_data=metadata or {},
                error_details=error_details,
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record