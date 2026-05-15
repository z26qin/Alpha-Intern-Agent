"""Tests for the data-loading tools."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools import (
    Workspace,
    get_default_registry,
    reset_default_registry,
)
from alpha_intern.tools.registry import ToolContext, ToolError


def _ctx(tmp_path: Path) -> ToolContext:
    workspace = Workspace()
    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    skills = SkillRegistry(tmp_path / "skills.json")
    return ToolContext(workspace=workspace, memory=memory, skills=skills)


def test_load_synthetic_prices_populates_workspace(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    out = reg.dispatch(
        "load_synthetic_prices",
        inputs={"output_artifact": "prices_raw"},
        ctx=ctx,
    )
    assert out.output_artifact == "prices_raw"
    assert out.n_tickers == 3  # default tickers
    assert out.n_rows == 3 * 240
    assert ctx.workspace.has("prices_raw")

    df = ctx.workspace.get("prices_raw")
    expected_cols = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    assert expected_cols.issubset(df.columns)

    reset_default_registry()


def test_load_synthetic_prices_is_deterministic(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx_a = _ctx(tmp_path / "a")
    ctx_b = _ctx(tmp_path / "b")

    reg.dispatch(
        "load_synthetic_prices",
        inputs={"output_artifact": "p", "seed": 42, "n_days": 50},
        ctx=ctx_a,
    )
    reg.dispatch(
        "load_synthetic_prices",
        inputs={"output_artifact": "p", "seed": 42, "n_days": 50},
        ctx=ctx_b,
    )

    a = ctx_a.workspace.get("p")
    b = ctx_b.workspace.get("p")
    import pandas as pd

    pd.testing.assert_frame_equal(a, b)
    reset_default_registry()


def test_load_synthetic_prices_unlocks_momentum_skill(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)

    # Bootstrap data via the load tool, then run the seeded momentum skill.
    reg.dispatch(
        "load_synthetic_prices",
        inputs={"output_artifact": "prices_raw", "n_days": 180},
        ctx=ctx,
    )
    out = reg.dispatch(
        "run_skill",
        inputs={"skill_name": "momentum_signal_research", "params": {}},
        ctx=ctx,
    )
    assert out.ok, f"skill failed: {out.error}"
    assert out.completed_steps == 6
    reset_default_registry()


def test_load_synthetic_prices_rejects_empty_tickers(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)
    with pytest.raises(Exception):
        reg.dispatch(
            "load_synthetic_prices",
            inputs={"output_artifact": "p", "tickers": []},
            ctx=ctx,
        )
    reset_default_registry()


@pytest.mark.skipif(
    importlib.util.find_spec("yfinance") is None,
    reason="yfinance not installed",
)
def test_load_prices_yfinance_tool_is_registered() -> None:
    reset_default_registry()
    reg = get_default_registry()
    assert "load_prices_yfinance" in reg.names()
    spec = reg.get("load_prices_yfinance").json_schema()
    assert "tickers" in spec["input_schema"]["properties"]
    reset_default_registry()


def test_load_prices_yfinance_surfaces_missing_dep_as_tool_error(tmp_path: Path) -> None:
    """If yfinance isn't installed, the tool returns a clean ToolError, not a crash."""
    if importlib.util.find_spec("yfinance") is not None:
        pytest.skip("yfinance is installed; cannot test the missing-dep path")

    reset_default_registry()
    reg = get_default_registry()
    ctx = _ctx(tmp_path)
    with pytest.raises(ToolError):
        reg.dispatch(
            "load_prices_yfinance",
            inputs={"tickers": ["AAPL"], "start": "2024-01-01"},
            ctx=ctx,
        )
    reset_default_registry()
