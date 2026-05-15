"""Tests for the walk-forward fold generator + run_walk_forward."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_intern.backtest.walk_forward import (
    generate_walk_forward_folds,
    run_walk_forward,
)
from alpha_intern.data.loader import normalize_price_dataframe
from alpha_intern.features.technical import build_basic_features


def _synthetic_panel(n_days: int = 240, seed: int = 7) -> pd.DataFrame:
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


def test_fold_generator_produces_non_overlapping_train_test_windows() -> None:
    dates = pd.date_range("2022-01-03", periods=300, freq="B")
    folds = generate_walk_forward_folds(
        list(dates),
        mode="expanding",
        min_train_size=120,
        refit_every_days=20,
        test_window_days=20,
        target_horizon_days=5,
    )
    assert len(folds) > 0
    for f in folds:
        # train_end is target-horizon-shifted, so it's strictly < as_of_date
        assert f.train_end_date < f.as_of_date
        # test starts strictly after the as_of_date
        assert f.test_start_date > f.as_of_date
        # train range is non-empty
        assert f.train_start_date <= f.train_end_date


def test_fold_generator_expanding_vs_rolling() -> None:
    dates = pd.date_range("2022-01-03", periods=300, freq="B")
    exp = generate_walk_forward_folds(
        list(dates),
        mode="expanding",
        min_train_size=120,
        refit_every_days=20,
        test_window_days=20,
    )
    rol = generate_walk_forward_folds(
        list(dates),
        mode="rolling",
        train_lookback_days=80,
        min_train_size=120,
        refit_every_days=20,
        test_window_days=20,
    )
    # Expanding: train_start is always the first date.
    for f in exp:
        assert f.train_start_date == dates[0]
    # Rolling: train_start moves forward over time.
    starts = [f.train_start_date for f in rol]
    assert starts == sorted(starts)
    assert starts[-1] > starts[0]


def test_fold_generator_rejects_bad_inputs() -> None:
    dates = pd.date_range("2022-01-03", periods=50, freq="B")
    with pytest.raises(ValueError):
        generate_walk_forward_folds(list(dates), min_train_size=3, target_horizon_days=5)
    with pytest.raises(ValueError):
        generate_walk_forward_folds(list(dates), mode="unsupported")  # type: ignore[arg-type]


def test_run_walk_forward_no_lookahead() -> None:
    panel = normalize_price_dataframe(_synthetic_panel(n_days=240))
    feats = build_basic_features(panel)

    feature_cols = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "moving_average_20d",
        "volume_zscore_20d",
    ]

    preds = run_walk_forward(
        feats,
        feature_cols=feature_cols,
        target_col="target_return_5d_forward",
        mode="expanding",
        min_train_size=120,
        refit_every_days=20,
        test_window_days=20,
        target_horizon_days=5,
    )

    assert len(preds) > 0
    assert {"date", "ticker", "signal", "fold_id", "as_of_date"}.issubset(preds.columns)
    # passthrough default attaches the target column
    assert "target_return_5d_forward" in preds.columns

    # Every prediction's date must be STRICTLY AFTER its fold's as_of_date.
    bad = preds[preds["date"] <= preds["as_of_date"]]
    assert len(bad) == 0, "Walk-forward leaked: prediction date <= as_of_date"


def test_run_walk_forward_handles_passthrough_columns() -> None:
    panel = normalize_price_dataframe(_synthetic_panel(n_days=200))
    feats = build_basic_features(panel)

    feature_cols = ["return_5d", "return_20d", "volatility_20d"]
    preds = run_walk_forward(
        feats,
        feature_cols=feature_cols,
        target_col="target_return_5d_forward",
        passthrough_columns=[],
        min_train_size=100,
        refit_every_days=40,
        test_window_days=40,
    )
    assert "target_return_5d_forward" not in preds.columns


def test_run_walk_forward_returns_empty_when_no_folds() -> None:
    panel = normalize_price_dataframe(_synthetic_panel(n_days=20))
    feats = build_basic_features(panel)
    preds = run_walk_forward(
        feats,
        feature_cols=["return_5d"],
        target_col="target_return_5d_forward",
        min_train_size=200,
        refit_every_days=20,
        test_window_days=20,
    )
    assert len(preds) == 0
