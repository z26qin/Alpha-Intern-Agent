"""Tests for the RunLog + its integration with the tool dispatcher."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import Field

from alpha_intern.agent.run_log import RunLog, iter_run_ids
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class EchoIn(ToolInput):
    message: str = Field(...)


class EchoOut(ToolOutput):
    echoed: str


def _registry_with_echo_and_boom() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("echo", "echo", EchoIn, EchoOut)
    def _echo(inp: EchoIn, _ctx: ToolContext) -> EchoOut:
        return EchoOut(echoed=inp.message)

    @reg.tool("boom", "always errors", EchoIn, EchoOut)
    def _boom(_inp: EchoIn, _ctx: ToolContext) -> EchoOut:
        raise RuntimeError("kaboom")

    return reg


def test_run_log_records_tool_call(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "run.jsonl")
    reg = _registry_with_echo_and_boom()
    ctx = ToolContext(run_log=log)

    reg.dispatch("echo", inputs={"message": "hello"}, ctx=ctx)
    log.close()

    entries = log.current_run_entries()
    step_types = [e.step_type for e in entries]
    assert step_types[0] == "run_start"
    assert "tool_call" in step_types
    assert step_types[-1] == "run_end"

    tool_calls = [e for e in entries if e.step_type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "echo"
    assert tool_calls[0].input == {"message": "hello"}
    assert tool_calls[0].output == {"echoed": "hello"}


def test_run_log_records_tool_error(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "run.jsonl")
    reg = _registry_with_echo_and_boom()
    ctx = ToolContext(run_log=log)

    with pytest.raises(RuntimeError):
        reg.dispatch("boom", inputs={"message": "x"}, ctx=ctx)

    log.close()

    errs = [e for e in log.current_run_entries() if e.step_type == "tool_error"]
    assert len(errs) == 1
    assert errs[0].tool_name == "boom"
    assert "kaboom" in (errs[0].error or "")


def test_run_log_records_validation_error(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "run.jsonl")
    reg = _registry_with_echo_and_boom()
    ctx = ToolContext(run_log=log)

    with pytest.raises(ToolError):
        reg.dispatch("echo", inputs={"wrong": 1}, ctx=ctx)

    log.close()

    errs = [e for e in log.current_run_entries() if e.step_type == "tool_error"]
    assert len(errs) == 1
    assert errs[0].tool_name == "echo"


def test_distinct_runs_isolated(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    reg = _registry_with_echo_and_boom()

    log_a = RunLog(path)
    reg.dispatch("echo", inputs={"message": "a"}, ctx=ToolContext(run_log=log_a))
    log_a.close()

    log_b = RunLog(path)
    reg.dispatch("echo", inputs={"message": "b"}, ctx=ToolContext(run_log=log_b))
    log_b.close()

    assert log_a.run_id != log_b.run_id

    ids = list(iter_run_ids(path))
    assert log_a.run_id in ids
    assert log_b.run_id in ids

    a_entries = log_a.current_run_entries()
    b_entries = log_b.current_run_entries()
    a_tool = [e for e in a_entries if e.step_type == "tool_call"][0]
    b_tool = [e for e in b_entries if e.step_type == "tool_call"][0]
    assert a_tool.input == {"message": "a"}
    assert b_tool.input == {"message": "b"}


def test_run_log_context_manager(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    with RunLog(path) as log:
        log.log_note("hello world", source="test")

    entries = list(RunLog(path, write_start_event=False).iter_entries())
    # 1 run_start + 1 note + 1 run_end from the with-block, then run-start from the reader
    # We just assert that note + run_end exist for the original run.
    notes = [e for e in entries if e.step_type == "note"]
    assert any(n.metadata.get("content") == "hello world" for n in notes)
