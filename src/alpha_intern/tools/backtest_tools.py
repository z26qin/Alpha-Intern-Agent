"""Tool wrappers around alpha_intern.backtest."""

from __future__ import annotations

from pydantic import Field

from alpha_intern.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)
from alpha_intern.backtest.walk_forward import run_simple_rank_backtest
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class RunRankBacktestIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of a DataFrame with signal + forward return columns.")
    signal_column: str = Field(..., description="Signal column to rank stocks by on each date.")
    forward_return_column: str = Field(..., description="Forward-return column used to score the rank portfolio.")
    top_quantile: float = Field(default=0.2, gt=0.0, lt=1.0)
    bottom_quantile: float = Field(default=0.2, gt=0.0, lt=1.0)
    cost_bps: float = Field(default=0.0, ge=0.0)
    output_artifact: str = Field(..., description="Workspace name to store the per-date returns DataFrame.")


class RunRankBacktestOut(ToolOutput):
    output_artifact: str
    n_periods: int


class ComputeMetricsIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of a return series DataFrame.")
    return_column: str = Field(default="net_return", description="Column with per-period returns.")
    periods_per_year: int = Field(default=252, gt=0)


class ComputeMetricsOut(ToolOutput):
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="run_rank_backtest",
        description="Equal-weight long/short rank backtest over a precomputed signal & forward-return panel.",
        input_model=RunRankBacktestIn,
        output_model=RunRankBacktestOut,
        tags=("backtest",),
    )
    def _backtest(inp: RunRankBacktestIn, ctx: ToolContext) -> RunRankBacktestOut:
        if ctx.workspace is None:
            raise ToolError("run_rank_backtest requires a workspace in ToolContext")
        df = ctx.workspace.get(inp.input_artifact)
        returns = run_simple_rank_backtest(
            df=df,
            signal_col=inp.signal_column,
            forward_return_col=inp.forward_return_column,
            top_quantile=inp.top_quantile,
            bottom_quantile=inp.bottom_quantile,
            cost_bps=inp.cost_bps,
        )
        ctx.workspace.put(inp.output_artifact, returns)
        return RunRankBacktestOut(
            output_artifact=inp.output_artifact,
            n_periods=int(len(returns)),
        )

    @registry.tool(
        name="compute_metrics",
        description="Compute annualized return / volatility / Sharpe / max drawdown over a return series.",
        input_model=ComputeMetricsIn,
        output_model=ComputeMetricsOut,
        tags=("backtest",),
    )
    def _metrics(inp: ComputeMetricsIn, ctx: ToolContext) -> ComputeMetricsOut:
        if ctx.workspace is None:
            raise ToolError("compute_metrics requires a workspace in ToolContext")
        df = ctx.workspace.get(inp.input_artifact)
        if inp.return_column not in df.columns:
            raise ToolError(
                f"return_column {inp.return_column!r} missing from artifact {inp.input_artifact!r}"
            )
        s = df[inp.return_column]
        return ComputeMetricsOut(
            annualized_return=annualized_return(s, periods_per_year=inp.periods_per_year),
            annualized_volatility=annualized_volatility(s, periods_per_year=inp.periods_per_year),
            sharpe_ratio=sharpe_ratio(s, periods_per_year=inp.periods_per_year),
            max_drawdown=max_drawdown(s),
        )
