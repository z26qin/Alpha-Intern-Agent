"""Tests for the reflection module + its integration with run_agent."""

from __future__ import annotations

import json
from pathlib import Path

from alpha_intern.agent.loop import run_agent
from alpha_intern.agent.provider import LLMResponse, ScriptedProvider, ToolUse
from alpha_intern.agent.reflection import (
    ReflectionPayload,
    reflect_on_run,
    summarize_trace,
)
from alpha_intern.agent.run_log import RunLog
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools import (
    Workspace,
    get_default_registry,
    reset_default_registry,
)
from alpha_intern.tools.registry import ToolContext


def _ctx(tmp_path: Path) -> tuple[ToolContext, RunLog]:
    workspace = Workspace()
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    skills = SkillRegistry(tmp_path / "skills.json")
    log = RunLog(tmp_path / "run.jsonl")
    return (
        ToolContext(workspace=workspace, memory=memory, skills=skills, run_log=log),
        log,
    )


def _good_reflection_json() -> str:
    return json.dumps(
        {
            "summary": "Saved a momentum memo after a quick smoke test.",
            "what_worked": ["wrote a memory note via add_memory"],
            "what_failed": [],
            "recommendations": ["next time, also call run_skill"],
            "skill_suggestion": None,
        }
    )


def test_reflect_on_run_parses_json_and_writes_memory(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    log = RunLog(tmp_path / "run.jsonl")
    log.log_event("plan", goal="momentum")
    log.log_tool_call(
        tool_name="add_memory",
        input={"title": "x"},
        output={"id": "m1", "timestamp": "t"},
    )
    log.close()

    provider = ScriptedProvider([LLMResponse(text=_good_reflection_json())])

    result = reflect_on_run(
        provider=provider,
        run_log=log,
        goal="study momentum",
        memory=memory,
    )

    assert result.parse_error is None
    assert result.payload.summary.startswith("Saved a momentum memo")
    assert result.memory_id is not None

    saved = memory.list_recent(limit=5)
    lessons = [m for m in saved if m.memory_type == "lesson"]
    assert lessons, "no lesson memory written"
    assert "momentum" in lessons[0].content.lower()
    assert "reflection" in lessons[0].tags


def test_reflect_on_run_handles_markdown_fenced_json(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    fenced = "```json\n" + _good_reflection_json() + "\n```"
    provider = ScriptedProvider([LLMResponse(text=fenced)])

    result = reflect_on_run(
        provider=provider,
        run_log=log,
        memory=memory,
    )
    assert result.parse_error is None
    assert result.payload.recommendations == ["next time, also call run_skill"]


def test_reflect_on_run_extracts_json_from_prose(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    text = (
        "Sure, here is my reflection. "
        + _good_reflection_json()
        + " — let me know if you want more."
    )
    provider = ScriptedProvider([LLMResponse(text=text)])

    result = reflect_on_run(
        provider=provider,
        run_log=log,
        memory=memory,
    )
    assert result.parse_error is None
    assert result.payload.what_worked == ["wrote a memory note via add_memory"]


def test_reflect_on_run_falls_back_when_unparseable(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    provider = ScriptedProvider([LLMResponse(text="totally not JSON at all")])

    result = reflect_on_run(
        provider=provider,
        run_log=log,
        memory=memory,
    )
    assert result.parse_error is not None
    assert "totally not JSON" in result.payload.summary
    # Lesson still written so the failure isn't silent.
    assert result.memory_id is not None


def test_reflect_on_run_can_skip_memory(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    log = RunLog(tmp_path / "run.jsonl")
    log.close()
    provider = ScriptedProvider([LLMResponse(text=_good_reflection_json())])

    result = reflect_on_run(
        provider=provider,
        run_log=log,
        memory=memory,
        write_memory=False,
    )
    assert result.memory_id is None
    assert memory.list_recent(limit=5) == []


def test_summarize_trace_truncates_and_labels(tmp_path: Path) -> None:
    log = RunLog(tmp_path / "run.jsonl")
    for _ in range(40):
        log.log_tool_call(
            tool_name="add_memory",
            input={"title": "t"},
            output={"id": "x"},
        )
    log.log_tool_error(tool_name="get_skill", input={}, error="missing")
    log.close()

    text = summarize_trace(log.current_run_entries(), max_tool_calls=10)
    assert "tool_call" in text
    assert "tool_error" in text
    assert "more tool calls" in text


def test_run_agent_with_reflect_at_end(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx, log = _ctx(tmp_path)

    main_provider = ScriptedProvider(
        [
            LLMResponse(
                text="I'll note this.",
                tool_uses=[
                    ToolUse(
                        id="t1",
                        name="add_memory",
                        input={"title": "auto", "content": "auto note"},
                    )
                ],
                stop_reason="tool_use",
            ),
            LLMResponse(text="Done.", stop_reason="end_turn"),
        ]
    )

    reflection_provider = ScriptedProvider([LLMResponse(text=_good_reflection_json())])

    result = run_agent(
        goal="trial run",
        provider=main_provider,
        registry=reg,
        ctx=ctx,
        max_steps=4,
        reflect_at_end=True,
        reflection_provider=reflection_provider,
    )

    assert result.reflection is not None
    assert result.reflection.parse_error is None
    assert result.reflection.memory_id is not None

    log.close()
    notes = [
        e
        for e in log.current_run_entries()
        if e.step_type == "note" and (e.metadata or {}).get("source") == "reflection"
    ]
    assert notes, "expected a 'reflection' note entry in the run log"

    assert ctx.memory is not None
    lessons = [m for m in ctx.memory.list_recent(limit=5) if m.memory_type == "lesson"]
    assert lessons, "expected a lesson memory written by reflection"

    reset_default_registry()
