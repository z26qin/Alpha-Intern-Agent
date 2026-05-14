"""Tool surface for AlphaInternAgent.

Each capability in `alpha_intern/` (data, features, models, backtest,
memory, skills) is exposed here as a typed `Tool`. Tools can be
dispatched by name via a `ToolRegistry`, and every dispatch is
optionally recorded to a `RunLog`. This is the surface a future LLM
agent will call.
"""

from __future__ import annotations

from alpha_intern.tools import (
    backtest_tools,
    data_tools,
    feature_tools,
    memory_tools,
    model_tools,
    skill_tools,
)
from alpha_intern.tools.registry import (
    Tool,
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
    get_default_registry,
    reset_default_registry,
)
from alpha_intern.tools.workspace import Workspace

__all__ = [
    "Tool",
    "ToolContext",
    "ToolError",
    "ToolInput",
    "ToolOutput",
    "ToolRegistry",
    "Workspace",
    "get_default_registry",
    "reset_default_registry",
]


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """Populate a registry with every built-in tool. Idempotent per registry."""
    data_tools.register(registry)
    feature_tools.register(registry)
    model_tools.register(registry)
    backtest_tools.register(registry)
    memory_tools.register(registry)
    skill_tools.register(registry)
