"""Tests for context assembly."""

from __future__ import annotations

from pathlib import Path

from alpha_intern.agent.context import assemble_context
from alpha_intern.agent.prompts import SYSTEM_PROMPT
from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools.workspace import Workspace


def test_minimal_context_only_has_goal() -> None:
    out = assemble_context(goal="study mid-cap momentum")
    assert out.system == SYSTEM_PROMPT
    assert "study mid-cap momentum" in out.initial_user_message
    assert "Available skill recipes" not in out.initial_user_message
    assert "Relevant prior memory notes" not in out.initial_user_message
    assert "Workspace artifacts" not in out.initial_user_message
    assert out.skill_names == []
    assert out.memory_ids == []
    assert out.artifact_names == []


def test_context_includes_skills_when_registry_present(tmp_path: Path) -> None:
    skills = SkillRegistry(tmp_path / "skills.json")
    out = assemble_context(goal="momentum study", skills=skills)
    assert "Available skill recipes" in out.initial_user_message
    assert "momentum_signal_research" in out.initial_user_message
    assert "momentum_signal_research" in out.skill_names


def test_context_includes_relevant_memories(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    memory.add_memory(title="momentum notes", content="12-1 momentum on tech")
    memory.add_memory(title="value notes", content="cheap defensives")

    out = assemble_context(
        goal="anything", memory=memory, memory_query="momentum"
    )
    assert "Relevant prior memory notes" in out.initial_user_message
    assert "momentum notes" in out.initial_user_message
    assert "value notes" not in out.initial_user_message
    assert len(out.memory_ids) == 1


def test_context_falls_back_to_recent_memories_when_no_match(tmp_path: Path) -> None:
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    memory.add_memory(title="totally unrelated", content="x")
    out = assemble_context(goal="momentum", memory=memory)
    assert len(out.memory_ids) == 1


def test_context_includes_workspace_artifacts() -> None:
    ws = Workspace()
    ws.put("prices", object())
    ws.put("features", object())
    out = assemble_context(goal="g", workspace=ws)
    assert "Workspace artifacts" in out.initial_user_message
    assert "prices" in out.initial_user_message
    assert set(out.artifact_names) == {"prices", "features"}


def test_context_respects_char_budget() -> None:
    long_goal = "x" * 50_000
    out = assemble_context(goal=long_goal, char_budget=500)
    assert len(out.initial_user_message) <= 520
    assert "truncated" in out.initial_user_message
