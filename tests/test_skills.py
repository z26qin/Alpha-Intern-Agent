"""Tests for the deterministic skill registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_intern.memory.skills import ResearchSkill, SkillRegistry


def test_defaults_are_seeded(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills.json")
    names = {s.name for s in registry.list_skills()}
    assert {
        "momentum_signal_research",
        "earnings_reaction_backtest",
        "factor_decay_analysis",
        "transaction_cost_sensitivity",
    }.issubset(names)


def test_add_and_get_skill(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills.json", seed_defaults=False)
    skill = ResearchSkill(
        name="rsi_meanrev",
        description="Mean-reversion using RSI.",
        inputs=["price_panel"],
        steps=["Compute RSI.", "Long oversold, short overbought."],
        outputs=["backtest_returns"],
        tags=["mean_reversion"],
    )
    registry.add_skill(skill)
    fetched = registry.get_skill("rsi_meanrev")
    assert fetched.description == "Mean-reversion using RSI."


def test_add_skill_rejects_duplicates(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills.json", seed_defaults=False)
    skill = ResearchSkill(name="dup", description="x")
    registry.add_skill(skill)
    with pytest.raises(ValueError):
        registry.add_skill(skill)


def test_update_skill_changes_fields_and_timestamp(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills.json")
    before = registry.get_skill("momentum_signal_research")

    updated = registry.update_skill(
        "momentum_signal_research",
        description="Updated description.",
    )
    assert updated.description == "Updated description."
    assert updated.updated_at >= before.updated_at


def test_update_skill_rejects_unknown_field(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills.json")
    with pytest.raises(ValueError):
        registry.update_skill("momentum_signal_research", nonsense="x")


def test_registry_persists_to_disk(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    r1 = SkillRegistry(path, seed_defaults=False)
    r1.add_skill(ResearchSkill(name="persisted", description="d"))

    r2 = SkillRegistry(path, seed_defaults=False)
    assert r2.get_skill("persisted").description == "d"
