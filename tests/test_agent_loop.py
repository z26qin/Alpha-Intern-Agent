"""Tests for the LLM agent loop using a ScriptedProvider (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_intern.agent.loop import run_agent
from alpha_intern.agent.provider import LLMResponse, ScriptedProvider, ToolUse
from alpha_intern.agent.run_log import RunLog
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools import (
    Workspace,
    get_default_registry,
    reset_default_registry,
)
from alpha_intern.tools.registry import ToolContext


def _ctx(tmp_path: Path, with_log: bool = True) -> tuple[ToolContext, RunLog | None]:
    workspace = Workspace()
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    skills = SkillRegistry(tmp_path / "skills.json")
    log = RunLog(tmp_path / "run.jsonl") if with_log else None
    return (
        ToolContext(workspace=workspace, memory=memory, skills=skills, run_log=log),
        log,
    )


def test_loop_terminates_on_end_turn(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    provider = ScriptedProvider(
        [LLMResponse(text="Nothing to do — done.", tool_uses=[], stop_reason="end_turn")]
    )

    result = run_agent(
        "what is 2+2?",
        provider=provider,
        registry=reg,
        ctx=ctx,
        max_steps=4,
    )

    assert result.stopped_reason == "end_turn"
    assert result.steps_used == 1
    assert "done" in result.final_text.lower()
    assert result.tool_calls == []
    if log is not None:
        log.close()
    reset_default_registry()


def test_loop_executes_tool_and_writes_memory(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    provider = ScriptedProvider(
        [
            LLMResponse(
                text="I'll record a memory note.",
                tool_uses=[
                    ToolUse(
                        id="t1",
                        name="add_memory",
                        input={
                            "title": "momentum quick look",
                            "content": "Tentative finding: short-horizon momentum is weak this year.",
                            "tags": ["momentum", "auto"],
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Saved the note. Done.", tool_uses=[], stop_reason="end_turn"),
        ]
    )

    result = run_agent(
        "Take a quick look at momentum.",
        provider=provider,
        registry=reg,
        ctx=ctx,
        max_steps=4,
    )

    assert result.stopped_reason == "end_turn"
    assert result.steps_used == 2
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc["tool"] == "add_memory"
    assert tc["ok"] is True
    assert "id" in tc["output"]

    assert ctx.memory is not None
    notes = ctx.memory.list_recent(limit=5)
    assert any(n.title == "momentum quick look" for n in notes)

    if log is not None:
        log.close()
        entries = log.current_run_entries()
        tool_call_logs = [e for e in entries if e.step_type == "tool_call"]
        assert len(tool_call_logs) == 1
        plan_logs = [e for e in entries if e.step_type == "plan"]
        assert len(plan_logs) == 1
        summary = [e for e in entries if e.step_type == "run_end_summary"]
        assert summary and summary[0].metadata["stopped_reason"] == "end_turn"
    reset_default_registry()


def test_loop_passes_tool_schemas_to_provider(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path, with_log=False)

    provider = ScriptedProvider([LLMResponse(text="ok", tool_uses=[])])
    run_agent("g", provider=provider, registry=reg, ctx=ctx, max_steps=1)

    assert provider.calls, "provider was not called"
    call = provider.calls[0]
    tool_names = {t["name"] for t in call["tools"]}
    assert "add_memory" in tool_names
    assert "build_features" in tool_names
    assert call["system"]  # system prompt is non-empty
    reset_default_registry()


def test_loop_stops_at_max_steps(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    looping_tool_use = LLMResponse(
        tool_uses=[
            ToolUse(id="x", name="list_skills", input={}),
        ],
        stop_reason="tool_use",
    )
    provider = ScriptedProvider([looping_tool_use, looping_tool_use, looping_tool_use])

    result = run_agent(
        "loop forever",
        provider=provider,
        registry=reg,
        ctx=ctx,
        max_steps=2,
    )
    assert result.stopped_reason == "step_budget"
    assert result.steps_used == 2
    assert len(result.tool_calls) == 2

    if log is not None:
        log.close()
    reset_default_registry()


def test_loop_records_tool_error_without_crashing(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    provider = ScriptedProvider(
        [
            LLMResponse(
                tool_uses=[
                    ToolUse(
                        id="bad",
                        name="get_skill",
                        input={"name": "does_not_exist"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="acknowledged error, stopping", stop_reason="end_turn"),
        ]
    )

    result = run_agent(
        "exercise an error",
        provider=provider,
        registry=reg,
        ctx=ctx,
        max_steps=4,
    )
    assert result.stopped_reason == "end_turn"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["ok"] is False
    assert "does_not_exist" in result.tool_calls[0]["error"]

    if log is not None:
        log.close()
        errs = [e for e in log.current_run_entries() if e.step_type == "tool_error"]
        assert errs and errs[0].tool_name == "get_skill"
    reset_default_registry()


def test_loop_records_provider_error(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    class Boom:
        def generate(self, system, messages, tools, max_tokens=4096):
            raise RuntimeError("provider down")

    result = run_agent(
        "trigger provider error",
        provider=Boom(),
        registry=reg,
        ctx=ctx,
        max_steps=3,
    )
    assert result.stopped_reason == "error"
    assert "provider down" in (result.error or "")

    if log is not None:
        log.close()
    reset_default_registry()


def test_loop_rejects_zero_max_steps(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, _ = _ctx(tmp_path, with_log=False)
    provider = ScriptedProvider([LLMResponse(text="x")])
    with pytest.raises(ValueError):
        run_agent("g", provider=provider, registry=reg, ctx=ctx, max_steps=0)
    reset_default_registry()
