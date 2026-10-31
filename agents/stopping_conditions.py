"""Agent runtime stop checks."""

from threading import Event


class StoppingConditions:
    @staticmethod
    def cancellation_requested(cancellation_event: Event | None) -> bool:
        return cancellation_event is not None and cancellation_event.is_set()

    @staticmethod
    def iteration_limit_reached(iterations_completed: int, maximum: int) -> bool:
        return iterations_completed >= maximum

    @staticmethod
    def tool_call_limit_reached(tool_calls_completed: int, maximum: int) -> bool:
        return tool_calls_completed >= maximum