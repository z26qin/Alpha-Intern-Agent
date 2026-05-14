"""Minimal market-data loader.

This module exposes a single normalization function plus an optional
thin wrapper around `yfinance`. Tests must never hit the network — they
should construct synthetic DataFrames and feed them through
`normalize_price_dataframe`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)

_COLUMN_ALIASES: dict[str, str] = {
    "Date": "date",
    "Datetime": "date",
    "timestamp": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "AdjClose": "adj_close",
    "adjclose": "adj_close",
    "Volume": "volume",
    "Ticker": "ticker",
    "symbol": "ticker",
    "Symbol": "ticker",
}


def normalize_price_dataframe(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
) -> pd.DataFrame:
    """Normalize a price DataFrame into the canonical schema.

    Output columns: date, ticker, open, high, low, close, adj_close, volume.
    Rows are sorted by (ticker, date) and duplicates are dropped.
    """
    if df is None or len(df) == 0:
        raise ValueError("normalize_price_dataframe received an empty DataFrame")

    out = df.copy()

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index()
        if "index" in out.columns and "date" not in out.columns:
            out = out.rename(columns={"index": "date"})

    out = out.rename(columns={c: _COLUMN_ALIASES.get(c, c) for c in out.columns})
    out.columns = [str(c).lower().strip().replace(" ", "_") for c in out.columns]

    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]

    if "ticker" not in out.columns:
        if ticker is None:
            raise ValueError(
                "Input has no 'ticker' column; pass `ticker=` to normalize_price_dataframe"
            )
        out["ticker"] = ticker

    missing = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")

    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out["ticker"] = out["ticker"].astype(str)
    for col in ("open", "high", "low", "close", "adj_close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[list(REQUIRED_COLUMNS)]
    out = out.drop_duplicates(subset=["ticker", "date"], keep="last")
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def download_prices_yfinance(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:  # pragma: no cover - network call, not exercised by tests
    """Thin wrapper around `yfinance.download` returning normalized data.

    Not used in tests. Imported lazily so `yfinance` is optional.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is not installed. Install with `pip install alpha-intern-agent[data]`."
        ) from exc

    tickers = list(tickers)
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    frames: list[pd.DataFrame] = []
    if len(tickers) == 1:
        frames.append(normalize_price_dataframe(raw, ticker=tickers[0]))
    else:
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                sub = raw[t].copy()
                frames.append(normalize_price_dataframe(sub, ticker=t))

    if not frames:
        raise RuntimeError("yfinance returned no data for the requested tickers")

    return pd.concat(frames, ignore_index=True)
