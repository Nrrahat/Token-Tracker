"""Execute allowlisted tools with validation, deadlines, and usage metrics."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from typing import Any

from pydantic import ValidationError

from tools.base_tool import ToolExecutionResult, ToolMetrics, ToolStatus
from tools.tool_registry import ToolRegistry


@dataclass
class _MutableMetrics:
    call_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    duplicate_call_count: int = 0
    empty_result_count: int = 0
    total_duration_ms: float = 0.0
    total_output_size_bytes: int = 0
    estimated_output_tokens: int = 0


class ToolManager:
    """Manage execution only for tools explicitly installed in the registry."""

    def __init__(self, registry: ToolRegistry, tokenizer: Any | None = None, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self._registry = registry
        self._tokenizer = tokenizer
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aop-tool")
        self._lock = Lock()
        self._metrics: dict[str, _MutableMetrics] = {}
        self._seen_calls: set[tuple[str, str]] = set()

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        tool = self._registry.get(name)
        canonical_arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        signature = (name, hashlib.sha256(canonical_arguments.encode("utf-8")).hexdigest())
        with self._lock:
            metrics = self._metrics.setdefault(name, _MutableMetrics())
            metrics.call_count += 1
            if signature in self._seen_calls:
                metrics.duplicate_call_count += 1
            self._seen_calls.add(signature)

        try:
            validated_arguments = tool.input_model.model_validate(arguments)
        except ValidationError:
            return self._finish(
                ToolExecutionResult(
                    tool_name=name,
                    status=ToolStatus.FAILED,
                    error_type="ValidationError",
                    error_message="Tool arguments did not match the input schema.",
                    duration_ms=0,
                )
            )

        started = perf_counter()
        future = self._executor.submit(tool.execute, validated_arguments)
        try:
            output = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError:
            future.cancel()
            duration_ms = (perf_counter() - started) * 1000
            return self._finish(
                ToolExecutionResult(
                    tool_name=name,
                    status=ToolStatus.TIMED_OUT,
                    error_type="TimeoutError",
                    error_message="Tool execution exceeded its time limit.",
                    duration_ms=duration_ms,
                )
            )
        except Exception as error:
            duration_ms = (perf_counter() - started) * 1000
            return self._finish(
                ToolExecutionResult(
                    tool_name=name,
                    status=ToolStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message="Tool execution failed.",
                    duration_ms=duration_ms,
                )
            )

        duration_ms = (perf_counter() - started) * 1000
        try:
            output_text = json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return self._finish(
                ToolExecutionResult(
                    tool_name=name,
                    status=ToolStatus.FAILED,
                    error_type="SerializationError",
                    error_message="Tool output was not JSON serializable.",
                    duration_ms=duration_ms,
                )
            )

        output_size_bytes = len(output_text.encode("utf-8"))
        estimated_tokens = None
        if self._tokenizer is not None:
            try:
                estimated_tokens = self._tokenizer.count_text(output_text)
            except Exception:
                estimated_tokens = None
        is_empty = output is None or output == "" or output == [] or output == {}
        return self._finish(
            ToolExecutionResult(
                tool_name=name,
                status=ToolStatus.SUCCEEDED,
                output=output,
                output_text=output_text,
                duration_ms=duration_ms,
                output_size_bytes=output_size_bytes,
                estimated_output_tokens=estimated_tokens,
                output_metadata={"output_type": type(output).__name__, "is_empty": is_empty},
            )
        )

    def metrics(self) -> dict[str, ToolMetrics]:
        with self._lock:
            return {
                name: ToolMetrics(**vars(values))
                for name, values in self._metrics.items()
            }

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _finish(self, result: ToolExecutionResult) -> ToolExecutionResult:
        with self._lock:
            metrics = self._metrics[result.tool_name]
            metrics.total_duration_ms += result.duration_ms
            metrics.total_output_size_bytes += result.output_size_bytes
            if result.status is not ToolStatus.SUCCEEDED:
                metrics.failure_count += 1
            if result.status is ToolStatus.TIMED_OUT:
                metrics.timeout_count += 1
            if result.output_metadata.get("is_empty") is True:
                metrics.empty_result_count += 1
            if result.estimated_output_tokens is not None:
                metrics.estimated_output_tokens += result.estimated_output_tokens
        return result