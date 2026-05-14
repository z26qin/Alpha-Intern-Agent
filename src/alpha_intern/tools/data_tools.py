"""Tool wrappers around alpha_intern.data."""

from __future__ import annotations

from typing import Optional

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


def register(registry: ToolRegistry) -> None:
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
    def _run(inp: NormalizePricesIn, ctx: ToolContext) -> NormalizePricesOut:
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
