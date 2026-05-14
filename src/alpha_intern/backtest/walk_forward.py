"""Simple long/short rank backtest.

This is intentionally minimal: on each date, rank stocks by the signal,
go long the top quantile and short the bottom quantile, equal-weight,
and compute the next-period return implied by the precomputed forward
return column. A flat per-period cost (in basis points) may be applied.

This module does not refit a model. A proper walk-forward refit loop
will land in a later PR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_simple_rank_backtest(
    df: pd.DataFrame,
    signal_col: str,
    forward_return_col: str,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Equal-weight long/short rank backtest.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``date``, ``ticker``, ``signal_col`` and
        ``forward_return_col``.
    signal_col : str
        Column with the model signal used to rank stocks on each date.
    forward_return_col : str
        Column with the realized forward return for each row.
    top_quantile, bottom_quantile : float
        Fraction of the cross-section to go long / short. Both in (0, 1).
    cost_bps : float
        Flat per-period transaction-cost haircut, in basis points,
        applied to the gross return.

    Returns
    -------
    DataFrame with columns: date, gross_return, cost, net_return.
    """
    if not 0 < top_quantile < 1 or not 0 < bottom_quantile < 1:
        raise ValueError("top_quantile and bottom_quantile must be in (0, 1)")
    if top_quantile + bottom_quantile > 1:
        raise ValueError("top_quantile + bottom_quantile must be <= 1")

    required = {"date", "ticker", signal_col, forward_return_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"run_simple_rank_backtest missing columns: {sorted(missing)}")

    work = df[["date", "ticker", signal_col, forward_return_col]].dropna(
        subset=[signal_col, forward_return_col]
    )

    rows: list[dict[str, object]] = []
    for date, g in work.groupby("date", sort=True):
        n = len(g)
        if n < 2:
            continue

        g = g.sort_values(signal_col)
        n_short = max(1, int(np.floor(n * bottom_quantile)))
        n_long = max(1, int(np.floor(n * top_quantile)))
        if n_short + n_long > n:
            continue

        shorts = g.head(n_short)
        longs = g.tail(n_long)

        long_ret = longs[forward_return_col].mean()
        short_ret = shorts[forward_return_col].mean()
        gross = float(long_ret - short_ret)

        cost = float(cost_bps) / 10_000.0
        net = gross - cost

        rows.append(
            {
                "date": date,
                "gross_return": gross,
                "cost": cost,
                "net_return": net,
            }
        )

    return pd.DataFrame(rows, columns=["date", "gross_return", "cost", "net_return"])
