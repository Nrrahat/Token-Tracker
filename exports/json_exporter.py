"""JSON exports for traces and experiment comparisons."""

import json

from experiments.schemas import ExperimentComparison
from instrumentation.trace_manager import ExecutionTrace


def execution_trace_json(trace: ExecutionTrace) -> str:
    return json.dumps(trace.model_dump(mode="json"), indent=2, ensure_ascii=False)


def experiment_comparison_json(comparison: ExperimentComparison) -> str:
    return json.dumps(comparison.model_dump(mode="json"), indent=2, ensure_ascii=False)