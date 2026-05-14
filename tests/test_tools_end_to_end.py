"""End-to-end pipeline test through the tool surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha_intern.memory.skills import SkillRegistry
from alpha_intern.memory.store import ResearchMemoryStore
from alpha_intern.tools import (
    ToolContext,
    Workspace,
    get_default_registry,
    reset_default_registry,
)


def _synthetic_panel(n_days: int = 120, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")

    frames = []
    for ticker, drift in [("AAA", 0.0006), ("BBB", -0.0001), ("CCC", 0.0002)]:
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


def test_full_pipeline_through_tools(tmp_path: Path) -> None:
    reset_default_registry()
    reg = get_default_registry()

    ws = Workspace()
    ws.put("prices_raw", _synthetic_panel())

    memory = ResearchMemoryStore(tmp_path / "mem.jsonl")
    skills = SkillRegistry(tmp_path / "skills.json")
    ctx = ToolContext(workspace=ws, memory=memory, skills=skills)

    reg.dispatch(
        "normalize_prices",
        inputs={"input_artifact": "prices_raw", "output_artifact": "prices"},
        ctx=ctx,
    )

    reg.dispatch(
        "build_features",
        inputs={"input_artifact": "prices", "output_artifact": "features"},
        ctx=ctx,
    )

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "moving_average_20d",
        "volume_zscore_20d",
    ]

    reg.dispatch(
        "train_signal",
        inputs={
            "input_artifact": "features",
            "feature_columns": feature_cols,
            "target_column": "target_return_5d_forward",
            "model_kind": "ridge",
            "model_artifact": "model",
        },
        ctx=ctx,
    )

    reg.dispatch(
        "predict_signal",
        inputs={
            "input_artifact": "features",
            "model_artifact": "model",
            "output_artifact": "signal",
        },
        ctx=ctx,
    )

    feats = ws.get("features")
    preds = ws.get("signal")
    merged = feats[["date", "ticker", "target_return_5d_forward"]].merge(
        preds[["date", "ticker", "signal"]], on=["date", "ticker"]
    )
    ws.put("scored", merged)

    bt_out = reg.dispatch(
        "run_rank_backtest",
        inputs={
            "input_artifact": "scored",
            "signal_column": "signal",
            "forward_return_column": "target_return_5d_forward",
            "top_quantile": 0.34,
            "bottom_quantile": 0.34,
            "cost_bps": 1.0,
            "output_artifact": "bt_returns",
        },
        ctx=ctx,
    )
    assert bt_out.n_periods > 0

    metrics = reg.dispatch(
        "compute_metrics",
        inputs={"input_artifact": "bt_returns", "return_column": "net_return"},
        ctx=ctx,
    )
    for field_name in (
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
    ):
        assert hasattr(metrics, field_name)

    add_out = reg.dispatch(
        "add_memory",
        inputs={
            "title": "first run",
            "content": "ran the pipeline through tools",
            "tags": ["smoke", "pipeline"],
        },
        ctx=ctx,
    )
    assert add_out.id

    search_out = reg.dispatch(
        "search_memory",
        inputs={"query": "pipeline"},
        ctx=ctx,
    )
    assert len(search_out.items) == 1

    skills_out = reg.dispatch("list_skills", inputs={}, ctx=ctx)
    assert any(s["name"] == "momentum_signal_research" for s in skills_out.skills)

    reset_default_registry()
