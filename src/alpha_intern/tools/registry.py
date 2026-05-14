"""Tool registry, context, and dispatcher.

A `Tool` is a pydantic-typed callable. Each tool declares an input
model, an output model, a name, and a description. Tools are dispatched
by name with kwargs; inputs are validated; the result is validated; and
the call is optionally logged to a `RunLog`.

This is the surface a future LLM agent will call. It is also directly
usable from Python.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Type

from pydantic import BaseModel, ConfigDict, ValidationError


class ToolInput(BaseModel):
    """Base class for tool input schemas."""

    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    """Base class for tool output schemas."""

    model_config = ConfigDict(extra="allow")


@dataclass
class ToolContext:
    """In-process objects tools may read or mutate.

    Kept off the JSON-input surface on purpose: tools take artifact
    *names*, then resolve them via this context.
    """

    workspace: Optional[Any] = None  # Workspace
    memory: Optional[Any] = None  # ResearchMemoryStore
    skills: Optional[Any] = None  # SkillRegistry
    run_log: Optional[Any] = None  # RunLog
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """A registered tool spec."""

    name: str
    description: str
    input_model: Type[ToolInput]
    output_model: Type[ToolOutput]
    func: Callable[[ToolInput, ToolContext], ToolOutput]
    tags: tuple[str, ...] = ()

    def json_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "input_schema": self.input_model.model_json_schema(),
            "output_schema": self.output_model.model_json_schema(),
        }


class ToolError(RuntimeError):
    """Raised when tool dispatch fails (invalid inputs, missing tool, etc.)."""


class ToolRegistry:
    """Named collection of `Tool`s with dispatch + JSON-schema export."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        input_model: Type[ToolInput],
        output_model: Type[ToolOutput],
        tags: tuple[str, ...] = (),
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a function as a Tool."""

        def deco(func: Callable[[ToolInput, ToolContext], ToolOutput]) -> Callable[..., Any]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    input_model=input_model,
                    output_model=output_model,
                    func=func,
                    tags=tags,
                )
            )
            return func

        return deco

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"No such tool: {name!r}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return [self._tools[k] for k in sorted(self._tools)]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def json_schemas(self) -> list[dict[str, Any]]:
        return [t.json_schema() for t in self.list_tools()]

    def dispatch(
        self,
        name: str,
        inputs: Optional[dict[str, Any]] = None,
        ctx: Optional[ToolContext] = None,
    ) -> ToolOutput:
        if ctx is None:
            ctx = ToolContext()
        tool = self.get(name)
        raw = inputs or {}

        try:
            validated_in = tool.input_model.model_validate(raw)
        except ValidationError as exc:
            self._maybe_log_error(ctx, name, raw, exc)
            raise ToolError(f"Invalid inputs for tool {name!r}: {exc}") from exc

        started = time.time()
        try:
            result = tool.func(validated_in, ctx)
        except Exception as exc:
            self._maybe_log_error(ctx, name, raw, exc, started=started)
            raise

        if not isinstance(result, tool.output_model):
            try:
                result = tool.output_model.model_validate(
                    result if isinstance(result, dict) else result.model_dump()
                )
            except ValidationError as exc:
                self._maybe_log_error(ctx, name, raw, exc, started=started)
                raise ToolError(
                    f"Tool {name!r} returned invalid output: {exc}"
                ) from exc

        self._maybe_log_success(ctx, name, raw, result, started=started)
        return result

    @staticmethod
    def _maybe_log_success(
        ctx: ToolContext,
        name: str,
        raw_inputs: dict[str, Any],
        output: ToolOutput,
        started: Optional[float] = None,
    ) -> None:
        if ctx.run_log is None:
            return
        ctx.run_log.log_tool_call(
            tool_name=name,
            input=raw_inputs,
            output=output.model_dump(),
            duration_s=None if started is None else round(time.time() - started, 6),
        )

    @staticmethod
    def _maybe_log_error(
        ctx: ToolContext,
        name: str,
        raw_inputs: dict[str, Any],
        exc: Exception,
        started: Optional[float] = None,
    ) -> None:
        if ctx.run_log is None:
            return
        ctx.run_log.log_tool_error(
            tool_name=name,
            input=raw_inputs,
            error=f"{type(exc).__name__}: {exc}",
            duration_s=None if started is None else round(time.time() - started, 6),
        )


_GLOBAL_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """Return the process-global registry, populating it on first call."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry()
        # Import inside the function to avoid circular imports during init.
        from alpha_intern.tools import _register_builtin_tools

        _register_builtin_tools(_GLOBAL_REGISTRY)
    return _GLOBAL_REGISTRY


def reset_default_registry() -> None:
    """Test helper — drop the global registry so it's rebuilt on next call."""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None
