"""Human-readable summary reports with explicit estimate labels."""

from analytics.schemas import TokenAnalytics
from evaluation.schemas import EvaluationSummary
from experiments.schemas import ExperimentComparison


def summary_report_markdown(
    token_analytics: TokenAnalytics,
    evaluation_summary: EvaluationSummary,
    comparison: ExperimentComparison | None = None,
) -> str:
    totals = token_analytics.totals
    lines = [
        "# Agent Observatory Summary",
        "",
        "## Token usage",
        "",
        f"- LLM calls: {totals.llm_calls}",
        f"- Calls with provider-reported usage: {totals.reported_calls}",
        f"- Reported input tokens: {totals.reported_input_tokens}",
        f"- Reported output tokens: {totals.reported_output_tokens}",
        f"- Estimated prompt-component tokens: {totals.estimated_component_tokens}",
        f"- Attribution gap (reported input minus component estimate): {totals.attribution_gap_tokens}",
        "",
        "Reported counts and local estimates are different measurements. Attribution gaps do not establish hidden prompt content.",
        "",
        "Estimated costs:",
    ]
    if totals.estimated_costs:
        lines.extend(f"- {currency}: {amount} (estimate)" for currency, amount in sorted(totals.estimated_costs.items()))
    else:
        lines.append("- Not available; configure verified model prices and record complete usage.")
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- Automatic evaluations: {evaluation_summary.automatic_evaluations}",
            f"- Automatic task success rate: {_percent(evaluation_summary.automatic_success_rate)}",
            f"- Human evaluations: {evaluation_summary.human_evaluations}",
            f"- Mean human quality: {_score(evaluation_summary.average_quality)}",
        ]
    )
    if comparison is not None:
        lines.extend(
            [
                "",
                "## Experiment comparison",
                "",
                f"- Experiment: {comparison.experiment.name}",
                f"- Reported-token reduction: {_percent(comparison.token_reduction_percent)}",
                f"- Task-success-rate change: {_signed_percent(comparison.success_rate_change)}",
                f"- Human-quality change: {_signed_score(comparison.human_quality_change)}",
                "",
                "Token reduction must be interpreted together with success and human quality; it is not sufficient by itself to establish an optimization benefit.",
            ]
        )
    return "\n".join(lines) + "\n"


def experiment_report_markdown(comparison: ExperimentComparison) -> str:
    lines = [
        f"# Experiment Report: {comparison.experiment.name}",
        "",
        comparison.experiment.description or "No experiment description supplied.",
        "",
        "## Paired results",
        "",
        f"- Dataset: {comparison.experiment.dataset.name}",
        f"- Cases: {len(comparison.experiment.dataset.cases)}",
        f"- Baseline agent: {comparison.experiment.baseline_snapshot.name}",
        f"- Optimized agent: {comparison.experiment.optimized_snapshot.name}",
        f"- Reported-token reduction: {_percent(comparison.token_reduction_percent)}",
        f"- Task-success-rate change: {_signed_percent(comparison.success_rate_change)}",
        f"- Human-quality change: {_signed_score(comparison.human_quality_change)}",
        "",
        "| Metric | Baseline | Optimized |",
        "| --- | ---: | ---: |",
    ]
    for label, baseline, optimized in (
        ("Runs", comparison.baseline.runs, comparison.optimized.runs),
        ("Task success rate", _percent(comparison.baseline.task_success_rate), _percent(comparison.optimized.task_success_rate)),
        ("Reported input tokens", comparison.baseline.reported_input_tokens, comparison.optimized.reported_input_tokens),
        ("Reported output tokens", comparison.baseline.reported_output_tokens, comparison.optimized.reported_output_tokens),
        ("Reasoning tokens", comparison.baseline.reported_reasoning_tokens, comparison.optimized.reported_reasoning_tokens),
        ("Cached input tokens", comparison.baseline.cached_input_tokens, comparison.optimized.cached_input_tokens),
        ("LLM calls", comparison.baseline.llm_calls, comparison.optimized.llm_calls),
        ("Tool calls", comparison.baseline.tool_calls, comparison.optimized.tool_calls),
        ("Average latency ms", _number(comparison.baseline.average_latency_ms), _number(comparison.optimized.average_latency_ms)),
        ("Human quality / 5", _number(comparison.baseline.average_human_quality), _number(comparison.optimized.average_human_quality)),
    ):
        lines.append(f"| {label} | {baseline} | {optimized} |")
    lines.extend(
        [
            "",
            "Estimated costs by currency:",
            "",
            f"- Baseline: {_currency_costs(comparison.baseline.estimated_costs)}",
            f"- Optimized: {_currency_costs(comparison.optimized.estimated_costs)}",
            "",
            "Token reduction alone does not establish an optimization benefit. Interpret it together with task success and human response-quality ratings. Token counts labeled reported come from provider usage; cost is estimated from configured pricing.",
            "",
            "## Trials",
            "",
            "| Case | Arm | Run ID | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {result.case_id} | {result.arm.value} | {result.run_id} | {result.status.value} |"
        for result in comparison.results
    )
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "Not rated"


def _signed_percent(value: float | None) -> str:
    return f"{value:+.1%}" if value is not None else "Not available"


def _score(value: float | None) -> str:
    return f"{value:.2f} / 5" if value is not None else "Not rated"


def _signed_score(value: float | None) -> str:
    return f"{value:+.2f} / 5" if value is not None else "Not rated"


def _number(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "Not available"


def _currency_costs(costs) -> str:
    if not costs:
        return "Not available"
    return " · ".join(f"{currency} {amount} (estimate)" for currency, amount in sorted(costs.items()))