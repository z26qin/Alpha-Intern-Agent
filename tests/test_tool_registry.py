"""Tests for the ToolRegistry core (validation, dispatch, schemas)."""

from __future__ import annotations

import pytest
from pydantic import Field

from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class EchoIn(ToolInput):
    message: str = Field(..., description="text to echo back")


class EchoOut(ToolOutput):
    echoed: str


def _make_registry_with_echo() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        name="echo",
        description="Echo a string.",
        input_model=EchoIn,
        output_model=EchoOut,
    )
    def _echo(inp: EchoIn, _ctx: ToolContext) -> EchoOut:
        return EchoOut(echoed=inp.message)

    return reg


def test_dispatch_validates_and_runs() -> None:
    reg = _make_registry_with_echo()
    out = reg.dispatch("echo", inputs={"message": "hi"})
    assert isinstance(out, EchoOut)
    assert out.echoed == "hi"


def test_dispatch_unknown_tool_raises() -> None:
    reg = ToolRegistry()
    with pytest.raises(ToolError):
        reg.dispatch("nope", inputs={})


def test_dispatch_invalid_inputs_raises() -> None:
    reg = _make_registry_with_echo()
    with pytest.raises(ToolError):
        reg.dispatch("echo", inputs={"wrong": 1})


def test_register_rejects_duplicates() -> None:
    reg = _make_registry_with_echo()

    with pytest.raises(ValueError):

        @reg.tool(
            name="echo",
            description="dup",
            input_model=EchoIn,
            output_model=EchoOut,
        )
        def _dup(inp: EchoIn, _ctx: ToolContext) -> EchoOut:  # pragma: no cover
            return EchoOut(echoed=inp.message)


def test_json_schemas_includes_input_and_output() -> None:
    reg = _make_registry_with_echo()
    schemas = reg.json_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "echo"
    assert "input_schema" in schema and "output_schema" in schema
    assert "message" in schema["input_schema"]["properties"]


def test_default_registry_has_builtin_tools() -> None:
    from alpha_intern.tools import get_default_registry, reset_default_registry

    reset_default_registry()
    reg = get_default_registry()
    names = set(reg.names())
    expected = {
        "normalize_prices",
        "build_features",
        "train_signal",
        "predict_signal",
        "run_rank_backtest",
        "compute_metrics",
        "add_memory",
        "search_memory",
        "list_recent_memories",
        "list_skills",
        "get_skill",
    }
    assert expected.issubset(names)
    reset_default_registry()
