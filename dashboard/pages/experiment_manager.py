"""Create, run, and compare paired baseline/optimized experiments."""

import json

import pandas as pd
import streamlit as st

from core.enums import ProviderName
from dashboard.components import empty_state, page_header
from dashboard.provider_configuration import create_session_provider, provider_ready
from dashboard.runtime import AppServices
from experiments.dataset_manager import DatasetManager
from experiments.schemas import ExperimentArm, ExperimentDataset
from exports.csv_exporter import experiment_comparison_csv
from exports.json_exporter import experiment_comparison_json
from exports.report_generator import experiment_report_markdown


SAMPLE_DATASET = json.dumps(
    {
        "name": "Small repeatable arithmetic set",
        "description": "A small example dataset. Replace these cases with a representative task set before drawing research conclusions.",
        "cases": [
            {
                "case_id": "case-001",
                "name": "Simple calculation",
                "prompt": "What is 17 multiplied by 24? Use the calculator tool if available.",
                "expected_outcome": "408",
            },
            {
                "case_id": "case-002",
                "name": "Explain a result",
                "prompt": "Explain briefly why 12 squared is 144.",
                "expected_outcome": "12 multiplied by 12 equals 144.",
            },
        ],
    },
    indent=2,
)


def render(services: AppServices) -> None:
    page_header("Experiment manager", "Run the same task set against two frozen agent configurations and compare outcomes.")
    agents = services.agent_manager.list_agents()
    active_agents = [agent for agent in agents if agent.is_active]
    if len(active_agents) < 2:
        empty_state("Create and activate at least two agents to define baseline and optimized arms.")
        _render_existing_experiments(services)
        return

    with st.expander("Create experiment", expanded=not services.experiment_manager.list_experiments()):
        with st.form("create_experiment"):
            name = st.text_input("Experiment name", max_chars=160)
            description = st.text_area("Research note", max_chars=2000, height=70)
            baseline_id = st.selectbox(
                "Baseline agent",
                [agent.agent_id for agent in active_agents],
                format_func=lambda value: _agent_label(active_agents, value),
            )
            optimized_options = [agent.agent_id for agent in active_agents if agent.agent_id != baseline_id]
            optimized_id = st.selectbox(
                "Optimized agent",
                optimized_options,
                format_func=lambda value: _agent_label(active_agents, value),
            )
            dataset_json = st.text_area("Repeatable dataset (JSON)", value=SAMPLE_DATASET, height=280)
            create_submitted = st.form_submit_button("Save experiment", type="primary")
        if create_submitted:
            try:
                dataset = DatasetManager().parse_json(dataset_json)
                experiment = services.experiment_manager.create(
                    name.strip(),
                    description.strip(),
                    baseline_id,
                    optimized_id,
                    dataset,
                )
                st.success(f"Saved experiment {experiment.name}")
                st.rerun()
            except (ValueError, LookupError) as error:
                st.error(str(error))

    experiments = services.experiment_manager.list_experiments()
    if experiments:
        _render_existing_experiments(services)


def _render_existing_experiments(services: AppServices) -> None:
    experiments = services.experiment_manager.list_experiments()
    if not experiments:
        return
    st.subheader("Experiments")
    selected_id = st.selectbox(
        "Experiment",
        [experiment.experiment_id for experiment in experiments],
        format_func=lambda value: next(
            f"{item.name} · {item.status.value} · {item.experiment_id[:8]}"
            for item in experiments
            if item.experiment_id == value
        ),
    )
    definition = services.experiment_manager.get(selected_id)
    left, right = st.columns(2)
    left.markdown(f"**Baseline:** {definition.baseline_snapshot.name}")
    right.markdown(f"**Optimized:** {definition.optimized_snapshot.name}")
    st.caption(
        f"{definition.dataset.name} · {len(definition.dataset.cases)} cases · "
        "configuration and skill-version snapshots are fixed for this experiment"
    )

    providers = {
        ProviderName(snapshot.config["provider"])
        for snapshot in (definition.baseline_snapshot, definition.optimized_snapshot)
    }
    providers_available = all(provider_ready(provider) for provider in providers)
    can_run = definition.status.value != "completed" and providers_available
    if st.button("Run or resume paired trials", type="primary", disabled=not can_run):
        def provider_factory(provider_name: ProviderName):
            return create_session_provider(provider_name)

        try:
            with st.spinner("Running baseline and optimized task pairs…"):
                services.experiment_runner.run(selected_id, provider_factory)
            st.success("All dataset cases have paired results.")
            st.rerun()
        except Exception as error:
            st.error(f"Experiment stopped: {type(error).__name__}")
            st.caption("Completed pairs are retained; rerun to resume missing cases.")
    if not providers_available and definition.status.value != "completed":
        st.info("Configure credentials/endpoints for both providers in Provider settings before running.")

    comparison = services.comparison_engine.compare(selected_id)
    _render_comparison(comparison)


def _render_comparison(comparison) -> None:
    st.subheader("Paired comparison")
    reduction = comparison.token_reduction_percent
    success_change = comparison.success_rate_change
    quality_change = comparison.human_quality_change
    metrics = st.columns(3)
    metrics[0].metric(
        "Reported token reduction",
        f"{reduction:.1f}%" if reduction is not None else "—",
        help="Calculated from reported input + output counts. A negative value means token use increased.",
    )
    metrics[1].metric(
        "Task success change",
        f"{success_change:+.0%}" if success_change is not None else "—",
        help="Optimized success rate minus baseline success rate.",
    )
    metrics[2].metric(
        "Human quality change",
        f"{quality_change:+.2f} / 5" if quality_change is not None else "Not rated",
        help="Optimized mean human overall-quality score minus baseline. Automatic evaluation does not supply quality scores.",
    )

    rows = []
    for label, arm in (("Baseline", comparison.baseline), ("Optimized", comparison.optimized)):
        rows.append(
            {
                "Arm": label,
                "Runs": arm.runs,
                "Success rate": f"{arm.task_success_rate:.0%}" if arm.task_success_rate is not None else "—",
                "Reported input": arm.reported_input_tokens,
                "Reported output": arm.reported_output_tokens,
                "Reasoning": arm.reported_reasoning_tokens,
                "Cached input": arm.cached_input_tokens,
                "LLM calls / run": round(arm.average_llm_calls_per_run, 2) if arm.average_llm_calls_per_run is not None else None,
                "Tool calls / run": round(arm.average_tool_calls_per_run, 2) if arm.average_tool_calls_per_run is not None else None,
                "Latency ms": round(arm.average_latency_ms, 1) if arm.average_latency_ms is not None else None,
                "Human quality / 5": round(arm.average_human_quality, 2) if arm.average_human_quality is not None else None,
                "Est. costs by currency": json.dumps(
                    {currency: str(amount) for currency, amount in arm.estimated_costs.items()},
                    sort_keys=True,
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if comparison.cost_reduction_percent_by_currency:
        st.caption(
            "Estimated cost reduction: "
            + " · ".join(
                f"{currency} {percent:.1f}%"
                for currency, percent in sorted(comparison.cost_reduction_percent_by_currency.items())
            )
        )

    result_rows = [
        {"Case": result.case_id, "Arm": result.arm.value, "Run": result.run_id, "Status": result.status.value}
        for result in comparison.results
    ]
    if result_rows:
        with st.expander("Trial records"):
            st.dataframe(pd.DataFrame(result_rows), use_container_width=True, hide_index=True)
    downloads = st.columns(3)
    downloads[0].download_button(
        "CSV",
        data=experiment_comparison_csv(comparison),
        file_name=f"experiment-{comparison.experiment.experiment_id}.csv",
        mime="text/csv",
        icon="⬇",
    )
    downloads[1].download_button(
        "JSON",
        data=experiment_comparison_json(comparison),
        file_name=f"experiment-{comparison.experiment.experiment_id}.json",
        mime="application/json",
        icon="⬇",
    )
    downloads[2].download_button(
        "Summary report",
        data=experiment_report_markdown(comparison),
        file_name=f"experiment-{comparison.experiment.experiment_id}-summary.md",
        mime="text/markdown",
        icon="⬇",
    )


def _agent_label(agents, agent_id: str) -> str:
    agent = next(agent for agent in agents if agent.agent_id == agent_id)
    return f"{agent.config.name} · {agent.config.provider.value}/{agent.config.model}"