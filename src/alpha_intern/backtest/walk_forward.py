"""Walk-forward backtesting utilities.

Two pieces:

- `run_simple_rank_backtest` — single-pass long/short rank backtest over
  a precomputed signal & forward-return panel. (Was the original v0 API.)
- `run_walk_forward` — refits an `AlphaSignalModel` on rolling/expanding
  training windows and produces out-of-sample signals for each test
  window. Output is a long DataFrame ready to feed into
  `run_simple_rank_backtest`.

The walk-forward fold generator filters training rows that would have
used future information beyond the fold's as-of date: rows are kept
only if `date + target_horizon_days <= as_of_date`. This is the only
safe way to use a forward-looking target without leaking the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from alpha_intern.models.signal_model import AlphaSignalModel, ModelKind


@dataclass(frozen=True)
class WalkForwardFold:
    """One refit window in a walk-forward run."""

    fold_id: int
    as_of_date: pd.Timestamp
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp


def generate_walk_forward_folds(
    dates: list[pd.Timestamp] | np.ndarray,
    *,
    mode: Literal["expanding", "rolling"] = "expanding",
    train_lookback_days: int = 252,
    refit_every_days: int = 21,
    test_window_days: int = 21,
    min_train_size: int = 252,
    target_horizon_days: int = 5,
) -> list[WalkForwardFold]:
    """Generate fold definitions over a sorted unique-date axis.

    "Days" here means *positions* in the unique-date array, not calendar
    days. For business-day panels the two coincide.
    """
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"Unknown mode {mode!r}; use 'expanding' or 'rolling'")
    if min_train_size <= target_horizon_days:
        raise ValueError("min_train_size must exceed target_horizon_days")
    if refit_every_days < 1 or test_window_days < 1:
        raise ValueError("refit_every_days and test_window_days must be >= 1")

    unique_dates = list(pd.to_datetime(pd.Index(dates)))
    n = len(unique_dates)
    if n == 0:
        return []

    folds: list[WalkForwardFold] = []
    cursor = min_train_size - 1  # zero-indexed position of train_end (as-of)

    while cursor + 1 < n:
        as_of_idx = cursor
        # Filter out the last `target_horizon_days` rows of the training
        # window so the target was actually observable by as_of_date.
        effective_train_end_idx = as_of_idx - target_horizon_days
        if mode == "expanding":
            train_start_idx = 0
        else:
            train_start_idx = max(
                0, as_of_idx - train_lookback_days + 1
            )
        if effective_train_end_idx < train_start_idx:
            cursor += refit_every_days
            continue

        test_start_idx = as_of_idx + 1
        test_end_idx = min(test_start_idx + test_window_days - 1, n - 1)
        if test_start_idx > n - 1:
            break

        folds.append(
            WalkForwardFold(
                fold_id=len(folds),
                as_of_date=unique_dates[as_of_idx],
                train_start_date=unique_dates[train_start_idx],
                train_end_date=unique_dates[effective_train_end_idx],
                test_start_date=unique_dates[test_start_idx],
                test_end_date=unique_dates[test_end_idx],
            )
        )
        cursor += refit_every_days

    return folds


def run_walk_forward(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    mode: Literal["expanding", "rolling"] = "expanding",
    train_lookback_days: int = 252,
    refit_every_days: int = 21,
    test_window_days: int = 21,
    min_train_size: int = 252,
    target_horizon_days: int = 5,
    model_kind: ModelKind = "ridge",
    model_factory: Optional[Callable[[], AlphaSignalModel]] = None,
    passthrough_columns: Optional[list[str]] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Refit a signal model on a moving window and emit OOS signals.

    Parameters
    ----------
    df : DataFrame
        Long-form feature panel; must contain ``date_col``, ``ticker_col``,
        every name in ``feature_cols``, and ``target_col``.
    passthrough_columns : list[str], optional
        Columns to copy through to the output frame (e.g. the realized
        forward return so the result can be fed straight into
        ``run_simple_rank_backtest``). Defaults to ``[target_col]``.

    Returns
    -------
    DataFrame with columns: date, ticker, signal, fold_id, as_of_date,
    plus any passthrough columns.
    """
    required = {date_col, ticker_col, target_col, *feature_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"run_walk_forward missing columns: {sorted(missing)}")

    if passthrough_columns is None:
        passthrough_columns = [target_col]
    passthrough_columns = [c for c in passthrough_columns if c in df.columns]

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work = work.sort_values([date_col, ticker_col]).reset_index(drop=True)

    unique_dates = list(work[date_col].drop_duplicates().sort_values())
    folds = generate_walk_forward_folds(
        unique_dates,
        mode=mode,
        train_lookback_days=train_lookback_days,
        refit_every_days=refit_every_days,
        test_window_days=test_window_days,
        min_train_size=min_train_size,
        target_horizon_days=target_horizon_days,
    )

    if not folds:
        empty_cols = [date_col, ticker_col, "signal", "fold_id", "as_of_date"]
        empty_cols += [c for c in passthrough_columns if c not in empty_cols]
        return pd.DataFrame(columns=empty_cols)

    pieces: list[pd.DataFrame] = []
    factory = model_factory or (lambda: AlphaSignalModel(kind=model_kind))

    for fold in folds:
        train_mask = (work[date_col] >= fold.train_start_date) & (
            work[date_col] <= fold.train_end_date
        )
        train_df = work.loc[train_mask].dropna(subset=[target_col])
        if len(train_df) == 0:
            continue

        model = factory()
        model.fit(train_df, feature_cols=feature_cols, target_col=target_col)

        test_mask = (work[date_col] >= fold.test_start_date) & (
            work[date_col] <= fold.test_end_date
        )
        test_df = work.loc[test_mask]
        if len(test_df) == 0:
            continue

        preds = model.predict(test_df)
        preds = preds.reset_index(drop=True)
        preds["fold_id"] = fold.fold_id
        preds["as_of_date"] = fold.as_of_date

        if passthrough_columns:
            extras = (
                test_df[[date_col, ticker_col, *passthrough_columns]]
                .reset_index(drop=True)
            )
            preds = preds.merge(extras, on=[date_col, ticker_col], how="left")

        pieces.append(preds)

    if not pieces:
        empty_cols = [date_col, ticker_col, "signal", "fold_id", "as_of_date"]
        empty_cols += [c for c in passthrough_columns if c not in empty_cols]
        return pd.DataFrame(columns=empty_cols)

    out = pd.concat(pieces, ignore_index=True)
    out = out.sort_values([date_col, ticker_col]).reset_index(drop=True)
    return out


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
