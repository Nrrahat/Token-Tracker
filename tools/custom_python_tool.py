"""Adapter for persisted custom tools in the existing tool execution system."""

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from core.schemas import ToolDefinition
from tools.base_tool import BaseTool
from tools.custom_tool_store import CustomToolStore, StoredCustomTool
from tools.custom_tool_testing import test_custom_tool
from tools.tool_registry import ToolNotFoundError, ToolRegistry


_CUSTOM_INPUT_CONFIG = ConfigDict(extra="allow")


class CustomPythonTool(BaseTool):
    """Expose a persisted Python tool through the validated tool interface."""

    timeout_seconds = 3.0

    def __init__(self, stored_tool: StoredCustomTool) -> None:
        metadata = stored_tool.metadata
        self.name = metadata.name
        self.description = metadata.description
        model_name = f"CustomToolInput_{re.sub(r'[^a-zA-Z0-9_]', '_', metadata.name)}"
        self.input_model: type[BaseModel] = create_model(
            model_name,
            __config__=_CUSTOM_INPUT_CONFIG,
        )
        self._function_name = metadata.function_name
        self._source = stored_tool.source

    def execute(self, arguments: BaseModel) -> Any:
        argument_values = arguments.model_dump()
        result = test_custom_tool(
            self._source,
            self._function_name,
            argument_values,
        )
        if not result.success:
            raise RuntimeError(result.error or "Custom tool execution failed.")
        if result.stdout and isinstance(result.output, dict):
            output = dict(result.output)
            output.setdefault("stdout", result.stdout)
            return output
        elif result.stdout:
            return {"result": result.output, "stdout": result.stdout}
        return result.output

    def provider_definition(self) -> ToolDefinition:
        definition = super().provider_definition()
        definition.parameters["additionalProperties"] = True
        return definition


def register_custom_tools(store: CustomToolStore, registry: ToolRegistry) -> list[str]:
    """Load persisted tools into the runtime allowlist and return their names."""
    loaded: list[str] = []
    for metadata in store.list_tools():
        registry.register(CustomPythonTool(store.get(metadata.name)))
        loaded.append(metadata.name)
    return loaded


def sync_custom_tools(store: CustomToolStore, registry: ToolRegistry) -> list[str]:
    """Synchronize saved custom tools into the live allowlist."""
    metadata = store.list_tools()
    custom_tools = [CustomPythonTool(store.get(item.name)) for item in metadata]
    saved_names = {tool.name for tool in custom_tools}

    for existing in registry.list_tools():
        if isinstance(existing, CustomPythonTool) and existing.name not in saved_names:
            registry.unregister(existing.name)

    for tool in custom_tools:
        try:
            current = registry.get(tool.name)
        except ToolNotFoundError:
            registry.register(tool)
        else:
            if not isinstance(current, CustomPythonTool):
                raise ValueError(f"Custom tool name conflicts with a registered built-in: {tool.name}")
            registry.replace(tool)
    return [tool.name for tool in custom_tools]
