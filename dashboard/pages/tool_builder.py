"""Custom tool authoring, testing, and management UI."""

import json
from typing import Any

import streamlit as st
from streamlit_ace import st_ace

from agents.agent_manager import AgentManager
from dashboard.components import empty_state, page_header
from tools.custom_python_tool import CustomPythonTool, sync_custom_tools
from tools.custom_tool_store import CustomToolStorageError, CustomToolStore
from tools.custom_tool_testing import CustomToolTestResult, test_custom_tool
from tools.tool_registry import ToolNotFoundError, ToolRegistry


_STARTER_CODE = '''def run(arguments: dict[str, object]) -> dict[str, object]:
    """Return a JSON-serializable result."""
    return {"result": "Replace this with your tool logic."}
'''


def render(
    custom_tool_store: CustomToolStore,
    tool_registry: ToolRegistry,
    agent_manager: AgentManager,
) -> None:
    page_header(
        "Tool Builder",
        "Create, test, edit, and remove custom tools stored on this machine.",
    )
    mode = st.radio("Tool action", ["Create", "Manage"], horizontal=True)
    st.caption(
        "Tests run in a separate process with restricted Python syntax, bounded inputs, "
        "and a two-second timeout. Imports, attribute access, and file/network APIs are disabled. "
        "This test runner is not an OS-level sandbox for hostile code."
    )
    if mode == "Create":
        _render_create(custom_tool_store, tool_registry)
    else:
        _render_manage(custom_tool_store, tool_registry, agent_manager)


def _render_create(custom_tool_store: CustomToolStore, tool_registry: ToolRegistry) -> None:
    name = st.text_input("Tool name", placeholder="my_custom_tool", key="new_custom_tool_name")
    description = st.text_input(
        "Description",
        placeholder="What this tool does",
        key="new_custom_tool_description",
    )
    function_name = st.text_input(
        "Function name",
        value="run",
        key="new_custom_tool_function",
    )
    source = _code_editor(_STARTER_CODE, "new_custom_tool_source")
    arguments_json = st.text_area(
        "Test arguments (JSON object)",
        value="{}",
        key="new_custom_tool_arguments",
        height=100,
    )
    _show_test_button(source, function_name, arguments_json, "new_custom_tool")
    if st.button("Save tool", type="primary", key="save_new_custom_tool"):
        try:
            registered_tool = tool_registry.get(name)
        except ToolNotFoundError:
            registered_tool = None
        if registered_tool is not None and not isinstance(registered_tool, CustomPythonTool):
            st.error(f"'{name}' is already used by a built-in tool.")
            return
        try:
            metadata = custom_tool_store.create(name, description, function_name, source)
        except (ValueError, CustomToolStorageError) as error:
            st.error(str(error))
        else:
            try:
                sync_custom_tools(custom_tool_store, tool_registry)
            except (ValueError, CustomToolStorageError) as error:
                st.error(f"Saved the tool, but could not load it into the runtime: {error}")
            else:
                st.success(f"Saved custom tool '{metadata.name}'.")


def _render_manage(
    custom_tool_store: CustomToolStore,
    tool_registry: ToolRegistry,
    agent_manager: AgentManager,
) -> None:
    try:
        tools = custom_tool_store.list_tools()
    except CustomToolStorageError as error:
        st.error(str(error))
        return
    if not tools:
        empty_state("No custom tools have been saved yet.")
        return

    selected_name = st.selectbox(
        "Saved tool",
        options=[tool.name for tool in tools],
        format_func=lambda name: next(
            f"{tool.name} — {tool.description}" if tool.description else tool.name
            for tool in tools
            if tool.name == name
        ),
        key="custom_tool_selected",
    )
    try:
        saved = custom_tool_store.get(selected_name)
    except (LookupError, CustomToolStorageError) as error:
        st.error(str(error))
        return

    st.text_input(
        "Tool name",
        value=saved.metadata.name,
        disabled=True,
        help="Tool names are fixed to preserve references from agent configurations.",
    )
    description = st.text_input(
        "Description",
        value=saved.metadata.description,
        key=f"custom_tool_description_{selected_name}",
    )
    function_name = st.text_input(
        "Function name",
        value=saved.metadata.function_name,
        key=f"custom_tool_function_{selected_name}",
    )
    source = _code_editor(saved.source, f"custom_tool_source_{selected_name}")
    arguments_json = st.text_area(
        "Test arguments (JSON object)",
        value="{}",
        key=f"custom_tool_arguments_{selected_name}",
        height=100,
    )

    columns = st.columns(3)
    test_clicked = columns[0].button("Test code", key=f"test_custom_tool_{selected_name}")
    save_clicked = columns[1].button(
        "Save changes",
        type="primary",
        key=f"update_custom_tool_{selected_name}",
    )
    confirm_delete = st.checkbox(
        f"Confirm deleting '{selected_name}'",
        key=f"confirm_delete_custom_tool_{selected_name}",
    )
    attached_agents = [
        agent.config.name
        for agent in agent_manager.list_agents()
        if selected_name in agent.config.tool_names
    ]
    delete_clicked = columns[2].button(
        "Delete tool",
        disabled=not confirm_delete or bool(attached_agents),
        key=f"delete_custom_tool_{selected_name}",
    )
    if attached_agents:
        st.caption(
            "Remove this tool from these agents before deleting: "
            + ", ".join(attached_agents)
        )

    if test_clicked:
        _run_test(source, function_name, arguments_json)
    if save_clicked:
        try:
            custom_tool_store.update(
                selected_name,
                description,
                function_name,
                source,
            )
        except (LookupError, ValueError, CustomToolStorageError) as error:
            st.error(str(error))
        else:
            try:
                sync_custom_tools(custom_tool_store, tool_registry)
            except (ValueError, CustomToolStorageError) as error:
                st.error(f"Saved the changes, but could not reload the tool: {error}")
            else:
                st.success(f"Updated custom tool '{selected_name}'.")
    if delete_clicked:
        try:
            custom_tool_store.delete(selected_name)
        except (LookupError, CustomToolStorageError) as error:
            st.error(str(error))
        else:
            try:
                sync_custom_tools(custom_tool_store, tool_registry)
            except (ValueError, CustomToolStorageError) as error:
                st.error(f"Deleted the tool, but could not refresh the runtime: {error}")
                return
            st.toast(f"Deleted custom tool '{selected_name}'.")
            st.rerun()


def _code_editor(source: str, key: str) -> str:
    code = st_ace(
        value=source,
        language="python",
        theme="chrome",
        key=key,
        height=360,
        auto_update=True,
    )
    return code or ""


def _show_test_button(
    source: str,
    function_name: str,
    arguments_json: str,
    key_prefix: str,
) -> None:
    if st.button("Test code", key=f"test_{key_prefix}"):
        _run_test(source, function_name, arguments_json)


def _run_test(source: str, function_name: str, arguments_json: str) -> None:
    if len(arguments_json) > 32_000:
        st.error("Test arguments exceed the 32,000-character limit.")
        return
    try:
        arguments: Any = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        st.error(f"Invalid test JSON on line {error.lineno}: {error.msg}")
        return
    except RecursionError:
        st.error("Test arguments contain too many nested JSON values.")
        return
    if not isinstance(arguments, dict):
        st.error("Test arguments must be a JSON object.")
        return

    result: CustomToolTestResult = test_custom_tool(source, function_name, arguments)
    if result.stdout:
        st.caption("Printed output")
        st.code(result.stdout, language="text")
    if not result.success:
        st.error(result.error or "Custom tool test failed.")
        return
    st.success("Custom tool test succeeded.")
    st.write("Returned value")
    st.json(result.output)
