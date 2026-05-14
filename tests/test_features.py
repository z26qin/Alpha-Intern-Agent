"""Feature engineering tests on synthetic data — no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_intern.data.loader import normalize_price_dataframe
from alpha_intern.features.technical import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_basic_features,
)


def _synthetic_panel(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")

    frames: list[pd.DataFrame] = []
    for ticker, drift in [("AAA", 0.0005), ("BBB", -0.0002)]:
        rets = rng.normal(loc=drift, scale=0.01, size=n_days)
        price = 100.0 * np.cumprod(1.0 + rets)
        frame = pd.DataFrame(
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
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def test_normalize_preserves_required_columns() -> None:
    df = _synthetic_panel(n_days=10)
    out = normalize_price_dataframe(df)
    for col in [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]:
        assert col in out.columns
    assert (out.groupby("ticker")["date"].is_monotonic_increasing).all()


def test_build_basic_features_emits_expected_columns() -> None:
    df = normalize_price_dataframe(_synthetic_panel())
    feats = build_basic_features(df)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"missing feature: {col}"
    assert TARGET_COLUMN in feats.columns


def test_features_are_backward_looking() -> None:
    """The first 19 rows per ticker must have NaN for 20d-window features."""
    df = normalize_price_dataframe(_synthetic_panel())
    feats = build_basic_features(df)

    for _, g in feats.groupby("ticker"):
        head = g.head(19)
        assert head["return_20d"].isna().all()
        assert head["volatility_20d"].isna().all()
        assert head["moving_average_20d"].isna().all()


def test_target_uses_future_data_only_for_target() -> None:
    """target_return_5d_forward must be NaN at the last 5 rows per ticker."""
    df = normalize_price_dataframe(_synthetic_panel())
    feats = build_basic_features(df)

    for _, g in feats.groupby("ticker"):
        tail = g.tail(5)
        assert tail[TARGET_COLUMN].isna().all()
