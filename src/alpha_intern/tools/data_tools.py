"""Tool wrappers around alpha_intern.data."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import Field

from alpha_intern.data.loader import normalize_price_dataframe
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class NormalizePricesIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of the raw price DataFrame.")
    output_artifact: str = Field(..., description="Workspace name to write the normalized DataFrame to.")
    ticker: Optional[str] = Field(
        default=None,
        description="Optional ticker to assign if the input lacks a 'ticker' column.",
    )


class NormalizePricesOut(ToolOutput):
    output_artifact: str
    n_rows: int
    n_tickers: int


class LoadSyntheticPricesIn(ToolInput):
    output_artifact: str = Field(
        default="prices_raw",
        description="Workspace name to store the generated DataFrame under.",
    )
    tickers: list[str] = Field(
        default_factory=lambda: ["AAA", "BBB", "CCC"],
        description="Ticker symbols to generate.",
    )
    n_days: int = Field(default=240, gt=0, le=5_000)
    start: str = Field(default="2022-01-03", description="Start date (YYYY-MM-DD).")
    seed: int = Field(default=7, description="RNG seed for determinism.")
    daily_drift: float = Field(
        default=0.0003, description="Per-ticker daily mean return (small)."
    )
    daily_vol: float = Field(default=0.01, gt=0.0, description="Per-day return std.")


class LoadSyntheticPricesOut(ToolOutput):
    output_artifact: str
    n_rows: int
    n_tickers: int
    start_date: str
    end_date: str


class LoadPricesYfinanceIn(ToolInput):
    tickers: list[str] = Field(..., description="Ticker symbols to download.")
    start: str = Field(..., description="Start date (YYYY-MM-DD).")
    end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD), exclusive.")
    output_artifact: str = Field(
        default="prices_raw",
        description="Workspace name to store the downloaded DataFrame under.",
    )


class LoadPricesYfinanceOut(ToolOutput):
    output_artifact: str
    n_rows: int
    n_tickers: int


def _build_synthetic_panel(
    *,
    tickers: list[str],
    n_days: int,
    start: str,
    seed: int,
    daily_drift: float,
    daily_vol: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    frames: list[pd.DataFrame] = []
    for i, ticker in enumerate(tickers):
        # Slight per-ticker drift variation so cross-section isn't degenerate.
        drift = daily_drift * (1.0 + 0.5 * (i - (len(tickers) - 1) / 2))
        rets = rng.normal(loc=drift, scale=daily_vol, size=n_days)
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


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="load_synthetic_prices",
        description=(
            "Generate a deterministic synthetic OHLCV panel and store it in the "
            "workspace. Use this to bootstrap a research session when no real "
            "price data is preloaded. Output schema is already canonical "
            "(date, ticker, OHLC, adj_close, volume), so normalize_prices is a "
            "no-op afterwards but safe to run."
        ),
        input_model=LoadSyntheticPricesIn,
        output_model=LoadSyntheticPricesOut,
        tags=("data",),
    )
    def _load_synth(inp: LoadSyntheticPricesIn, ctx: ToolContext) -> LoadSyntheticPricesOut:
        if ctx.workspace is None:
            raise ToolError("load_synthetic_prices requires a workspace in ToolContext")
        if not inp.tickers:
            raise ToolError("load_synthetic_prices requires at least one ticker")
        panel = _build_synthetic_panel(
            tickers=list(inp.tickers),
            n_days=inp.n_days,
            start=inp.start,
            seed=inp.seed,
            daily_drift=inp.daily_drift,
            daily_vol=inp.daily_vol,
        )
        ctx.workspace.put(inp.output_artifact, panel)
        return LoadSyntheticPricesOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(panel)),
            n_tickers=int(panel["ticker"].nunique()),
            start_date=str(panel["date"].min().date()),
            end_date=str(panel["date"].max().date()),
        )

    @registry.tool(
        name="normalize_prices",
        description=(
            "Normalize a raw OHLCV DataFrame in the workspace into the canonical "
            "schema (date, ticker, OHLC, adj_close, volume)."
        ),
        input_model=NormalizePricesIn,
        output_model=NormalizePricesOut,
        tags=("data",),
    )
    def _normalize(inp: NormalizePricesIn, ctx: ToolContext) -> NormalizePricesOut:
        if ctx.workspace is None:
            raise ToolError("normalize_prices requires a workspace in ToolContext")
        raw = ctx.workspace.get(inp.input_artifact)
        normalized = normalize_price_dataframe(raw, ticker=inp.ticker)
        ctx.workspace.put(inp.output_artifact, normalized)
        return NormalizePricesOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(normalized)),
            n_tickers=int(normalized["ticker"].nunique()),
        )

    @registry.tool(
        name="load_prices_yfinance",
        description=(
            "Download historical OHLCV data from Yahoo Finance via the yfinance "
            "library and store a normalized panel in the workspace. Requires "
            "network access and the optional [data] install extra. For "
            "deterministic offline runs prefer load_synthetic_prices."
        ),
        input_model=LoadPricesYfinanceIn,
        output_model=LoadPricesYfinanceOut,
        tags=("data", "network"),
    )
    def _load_yfinance(inp: LoadPricesYfinanceIn, ctx: ToolContext) -> LoadPricesYfinanceOut:
        if ctx.workspace is None:
            raise ToolError("load_prices_yfinance requires a workspace in ToolContext")
        if not inp.tickers:
            raise ToolError("load_prices_yfinance requires at least one ticker")
        # Lazy import so missing yfinance only breaks this tool, not the package.
        from alpha_intern.data.loader import download_prices_yfinance

        try:
            panel = download_prices_yfinance(
                tickers=inp.tickers, start=inp.start, end=inp.end
            )
        except ImportError as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            raise ToolError(
                f"yfinance download failed: {type(exc).__name__}: {exc}"
            ) from exc

        ctx.workspace.put(inp.output_artifact, panel)
        return LoadPricesYfinanceOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(panel)),
            n_tickers=int(panel["ticker"].nunique()),
        )
