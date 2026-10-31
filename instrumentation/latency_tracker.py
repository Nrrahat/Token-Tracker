"""Monotonic elapsed-time measurement for provider requests."""

from time import perf_counter


class LatencyTracker:
    def start(self) -> float:
        return perf_counter()

    @staticmethod
    def elapsed_ms(started_at: float) -> float:
        return max(0.0, (perf_counter() - started_at) * 1000)