from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

from dashboard.pages import task_execution
from database.models import ExecutionEventRecord, ExecutionRunRecord


def test_open_execution_trace_callback_updates_navigation(monkeypatch) -> None:
    state = {"workspace_page": "Task execution"}
    monkeypatch.setattr(task_execution.st, "session_state", state)

    task_execution._open_execution_trace()

    assert state["workspace_page"] == "Execution traces"


def test_latest_result_stays_visible_until_cleared(monkeypatch) -> None:
    state = {"last_run_id": "run-123"}
    run = SimpleNamespace(
        status="succeeded",
        result="Saved response",
        error_message=None,
    )

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def get(self, model, run_id):
            assert model is ExecutionRunRecord
            assert run_id == "run-123"
            return run

    services = SimpleNamespace(session_factory=Session)
    monkeypatch.setattr(task_execution.st, "session_state", state)
    monkeypatch.setattr(task_execution.st, "divider", Mock())
    monkeypatch.setattr(task_execution.st, "subheader", Mock())
    markdown = Mock()
    monkeypatch.setattr(task_execution.st, "markdown", markdown)
    monkeypatch.setattr(task_execution.st, "success", Mock())
    monkeypatch.setattr(task_execution.st, "error", Mock())
    monkeypatch.setattr(task_execution.st, "info", Mock())
    monkeypatch.setattr(task_execution.st, "rerun", Mock())
    clear_on_click = {"enabled": False}
    monkeypatch.setattr(
        task_execution.st,
        "button",
        lambda _label, **kwargs: (
            kwargs["key"] == "clear_latest_task_result"
            and clear_on_click["enabled"]
            and state.pop("last_run_id", None)
        ),
    )

    task_execution._render_last_result(services)
    task_execution._render_last_result(services)

    assert markdown.call_count == 2
    assert state["last_run_id"] == "run-123"

    clear_on_click["enabled"] = True
    task_execution._render_last_result(services)

    assert "last_run_id" not in state


def test_failed_run_shows_failure_reason_and_provider_context(monkeypatch) -> None:
    state = {"last_run_id": "run-456"}
    run = SimpleNamespace(
        status="failed",
        result=None,
        error_message="ProviderRequestError",
    )
    failure_event = SimpleNamespace(
        event_type="llm_request_failed",
        error_details="ProviderRequestError",
        event_data={"provider": "groq", "model": "llama-test"},
    )

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def get(self, model, run_id):
            assert model is ExecutionRunRecord
            assert run_id == "run-456"
            return run

        def scalars(self, _query):
            return SimpleNamespace(all=lambda: [failure_event])

    services = SimpleNamespace(session_factory=Session)
    monkeypatch.setattr(task_execution.st, "session_state", state)
    monkeypatch.setattr(task_execution.st, "divider", Mock())
    monkeypatch.setattr(task_execution.st, "subheader", Mock())
    monkeypatch.setattr(task_execution.st, "error", Mock())
    monkeypatch.setattr(task_execution.st, "info", Mock())
    monkeypatch.setattr(task_execution.st, "success", Mock())
    monkeypatch.setattr(task_execution.st, "expander", lambda *_args, **_kwargs: nullcontext())
    markdown = Mock()
    write = Mock()
    caption = Mock()
    monkeypatch.setattr(task_execution.st, "markdown", markdown)
    monkeypatch.setattr(task_execution.st, "write", write)
    monkeypatch.setattr(task_execution.st, "caption", caption)
    monkeypatch.setattr(task_execution.st, "button", Mock(return_value=False))

    task_execution._render_last_result(services)

    assert any("llm request failed" in str(call).lower() for call in markdown.call_args_list)
    assert any("Reason: ProviderRequestError" in str(call) for call in write.call_args_list)
    assert any("Provider: groq" in str(call) and "Model: llama-test" in str(call) for call in caption.call_args_list)
