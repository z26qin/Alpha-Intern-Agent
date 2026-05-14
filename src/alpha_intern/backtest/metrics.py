"""Performance metrics for backtest return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(returns: pd.Series | np.ndarray | list[float]) -> pd.Series:
    s = pd.Series(returns, dtype="float64").dropna()
    return s


def annualized_return(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    mean = r.mean()
    return float((1.0 + mean) ** periods_per_year - 1.0)


def annualized_volatility(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series | np.ndarray | list[float],
    periods_per_year: int = 252,
) -> float:
    r = _as_series(returns)
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float((r.mean() / sd) * np.sqrt(periods_per_year))


def max_drawdown(
    returns: pd.Series | np.ndarray | list[float],
) -> float:
    """Return max drawdown as a non-positive float (e.g. -0.23)."""
    r = _as_series(returns)
    if len(r) == 0:
        return float("nan")
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    return float(drawdown.min())
