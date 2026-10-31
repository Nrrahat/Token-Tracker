"""Explicit allowlist registry for tool implementations."""

import math
import re

from tools.base_tool import BaseTool


class ToolNotFoundError(LookupError):
    """Raised when a caller requests a tool that is not allowlisted."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", tool.name):
            raise ValueError("Tool names must contain only letters, digits, '_' or '-' and be at most 64 chars")
        if not math.isfinite(tool.timeout_seconds) or tool.timeout_seconds <= 0:
            raise ValueError("Tool timeout must be a positive finite number")
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotFoundError(f"Tool is not registered: {name}") from error

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())