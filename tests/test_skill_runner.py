"""Tests for the SkillRunner + the `run_skill` tool."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_intern.memory.skill_runner import SkillRunner
from alpha_intern.memory.skills import ExecutableStep, ResearchSkill, SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools import (
    Workspace,
    get_default_registry,
    reset_default_registry,
)
from alpha_intern.tools.registry import ToolContext


def _ctx(tmp_path: Path) -> ToolContext:
    workspace = Workspace()
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    skills = SkillRegistry(tmp_path / "skills.json")
    return ToolContext(workspace=workspace, memory=memory, skills=skills)


def _synthetic_panel(n_days: int = 160, seed: int = 13) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    frames = []
    for ticker, drift in [("AAA", 0.0007), ("BBB", -0.0001), ("CCC", 0.0002)]:
        rets = rng.normal(loc=drift, scale=0.01, size=n_days)
        price = 100.0 * np.cumprod(1.0 + rets)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": rng.integers(100_000, 500_000, size=n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_seeded_momentum_skill_runs_end_to_end(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)
    ctx.workspace.put("prices_raw", _synthetic_panel())

    runner = SkillRunner(registry=reg, ctx=ctx)
    result = runner.run("momentum_signal_research")

    assert result.ok, f"skill failed: {result.error}\n{[s.error for s in result.step_results if s.error]}"
    assert result.completed_steps == result.total_steps
    assert result.total_steps == 6
    tool_sequence = [s.tool for s in result.step_results]
    assert tool_sequence == [
        "normalize_prices",
        "build_features",
        "train_signal",
        "predict_signal",
        "run_rank_backtest",
        "compute_metrics",
    ]
    metrics_step = result.step_results[-1]
    assert metrics_step.output is not None
    assert "sharpe_ratio" in metrics_step.output

    # Artifacts produced along the way are in the workspace.
    assert ctx.workspace.has("prices")
    assert ctx.workspace.has("features")
    assert ctx.workspace.has("signal")
    assert ctx.workspace.has("bt_returns")

    reset_default_registry()


def test_skill_runner_resolves_param_and_step_refs(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    # Tiny skill that uses both $param and $step.<alias>.<field>.
    skill = ResearchSkill(
        name="memo_chain",
        description="Add a memory, then search for it by title prefix.",
        default_params={"prefix": "hello"},
        executable_steps=[
            ExecutableStep(
                tool="add_memory",
                inputs={
                    "title": "$param.prefix",
                    "content": "auto-written",
                    "tags": ["chain", "$param.prefix"],
                },
                save_outputs_as="added",
            ),
            ExecutableStep(
                tool="search_memory",
                inputs={"query": "$param.prefix", "limit": 5},
                save_outputs_as="searched",
            ),
        ],
    )

    runner = SkillRunner(registry=reg, ctx=ctx)
    result = runner.run(skill)
    assert result.ok
    search_output = result.step_results[1].output
    assert search_output is not None
    items = search_output["items"]
    assert any(item["title"] == "hello" for item in items)

    reset_default_registry()


def test_skill_runner_aborts_on_missing_param(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    skill = ResearchSkill(
        name="needs_param",
        description="x",
        executable_steps=[
            ExecutableStep(
                tool="add_memory",
                inputs={
                    "title": "$param.title",
                    "content": "x",
                },
            )
        ],
    )
    result = SkillRunner(reg, ctx).run(skill, params={})
    assert not result.ok
    assert "title" in (result.error or "")
    assert result.completed_steps == 0

    reset_default_registry()


def test_skill_runner_propagates_tool_error(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    skill = ResearchSkill(
        name="bad_skill",
        description="x",
        executable_steps=[
            ExecutableStep(
                tool="get_skill",
                inputs={"name": "does_not_exist"},
            )
        ],
    )
    result = SkillRunner(reg, ctx).run(skill)
    assert not result.ok
    assert result.completed_steps == 0
    assert "does_not_exist" in (result.error or "")

    reset_default_registry()


def test_skill_runner_rejects_empty_executable_steps(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    skill = ResearchSkill(name="empty", description="x")
    result = SkillRunner(reg, ctx).run(skill)
    assert not result.ok
    assert result.total_steps == 0

    reset_default_registry()


def test_run_skill_tool(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)
    ctx.workspace.put("prices_raw", _synthetic_panel())

    out = reg.dispatch(
        "run_skill",
        inputs={"skill_name": "momentum_signal_research", "params": {}},
        ctx=ctx,
    )
    assert out.ok is True
    assert out.completed_steps == 6
    assert out.total_steps == 6
    tools_run = [s["tool"] for s in out.step_results]
    assert "compute_metrics" in tools_run

    reset_default_registry()


def test_run_skill_tool_resolves_param_overrides(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)
    ctx.workspace.put("my_prices", _synthetic_panel())

    out = reg.dispatch(
        "run_skill",
        inputs={
            "skill_name": "momentum_signal_research",
            "params": {"raw_prices": "my_prices", "cost_bps": 0.0},
        },
        ctx=ctx,
    )
    assert out.ok is True
    # cost_bps override flowed through to backtest step
    backtest_step = next(
        s for s in out.step_results if s["tool"] == "run_rank_backtest"
    )
    assert backtest_step["inputs"]["cost_bps"] == 0.0

    reset_default_registry()


def test_skills_with_executable_steps_roundtrip_on_disk(tmp_path: Path) -> None:
    path = tmp_path / "skills.json"
    r1 = SkillRegistry(path)
    momentum = r1.get_skill("momentum_signal_research")
    assert len(momentum.executable_steps) == 6
    assert momentum.default_params["raw_prices"] == "prices_raw"

    r2 = SkillRegistry(path)
    reloaded = r2.get_skill("momentum_signal_research")
    assert len(reloaded.executable_steps) == 6
    assert reloaded.executable_steps[0].tool == "normalize_prices"
