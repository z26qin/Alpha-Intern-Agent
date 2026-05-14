"""Backward-looking technical features.

All features are computed per ticker, after sorting by date. The only
column that is allowed to use future data is `target_return_5d_forward`,
which is clearly named as a target and must never be used as an input
feature in a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

FEATURE_COLUMNS: tuple[str, ...] = (
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "moving_average_20d",
    "moving_average_60d",
    "volume_zscore_20d",
)

TARGET_COLUMN: str = "target_return_5d_forward"


def _per_ticker_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    price = g["adj_close"]
    volume = g["volume"]

    g["return_1d"] = price.pct_change(1)
    g["return_5d"] = price.pct_change(5)
    g["return_20d"] = price.pct_change(20)
    g["volatility_20d"] = g["return_1d"].rolling(20, min_periods=20).std()
    g["moving_average_20d"] = price.rolling(20, min_periods=20).mean()
    g["moving_average_60d"] = price.rolling(60, min_periods=60).mean()

    vol_mean = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    g["volume_zscore_20d"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    g[TARGET_COLUMN] = price.shift(-5) / price - 1.0
    return g


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking features (+ one clearly-named forward target).

    Input must contain the columns produced by
    `alpha_intern.data.loader.normalize_price_dataframe`.
    """
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"build_basic_features missing input columns: {missing}")

    work = df.copy()
    work = work.sort_values(["ticker", "date"]).reset_index(drop=True)

    pieces = [
        _per_ticker_features(g) for _, g in work.groupby("ticker", sort=False)
    ]
    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
