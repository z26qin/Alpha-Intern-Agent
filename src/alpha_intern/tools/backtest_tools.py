"""Tool wrappers around alpha_intern.backtest."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from alpha_intern.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
)
from alpha_intern.backtest.walk_forward import (
    run_simple_rank_backtest,
    run_walk_forward,
)
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class RunRankBacktestIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of a DataFrame with signal + forward return columns.")
    signal_column: str = Field(
        default="signal",
        description=(
            "Signal column to rank stocks by on each date. Defaults to 'signal', "
            "which is what both walk_forward_signal and predict_signal emit."
        ),
    )
    forward_return_column: str = Field(
        default="target_return_5d_forward",
        description=(
            "Forward-return column used to score the rank portfolio. Defaults to "
            "'target_return_5d_forward' (build_features' standard target). Pass "
            "this column via walk_forward_signal's passthrough_columns."
        ),
    )
    top_quantile: float = Field(default=0.2, gt=0.0, lt=1.0)
    bottom_quantile: float = Field(default=0.2, gt=0.0, lt=1.0)
    cost_bps: float = Field(default=0.0, ge=0.0)
    output_artifact: str = Field(..., description="Workspace name to store the per-date returns DataFrame.")


class RunRankBacktestOut(ToolOutput):
    output_artifact: str
    n_periods: int


class WalkForwardSignalIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of a feature DataFrame.")
    feature_columns: list[str] = Field(..., description="Columns to use as features.")
    target_column: str = Field(..., description="Forward-looking target column.")
    output_artifact: str = Field(..., description="Workspace name to store the OOS signal DataFrame.")
    mode: Literal["expanding", "rolling"] = Field(default="expanding")
    train_lookback_days: int = Field(default=252, gt=0)
    refit_every_days: int = Field(default=21, gt=0)
    test_window_days: int = Field(default=21, gt=0)
    min_train_size: int = Field(default=252, gt=0)
    target_horizon_days: int = Field(
        default=5,
        gt=0,
        description=(
            "Number of periods of forward information baked into target_column. "
            "Training rows whose targets would not have been observable by the "
            "as-of date are dropped, preventing look-ahead leakage."
        ),
    )
    model_kind: Literal["ridge", "random_forest"] = Field(default="ridge")
    passthrough_columns: Optional[list[str]] = Field(
        default=None,
        description=(
            "Columns to carry over from the input frame to the signal frame. "
            "Defaults to [target_column] so the output is rank-backtest-ready."
        ),
    )


class WalkForwardSignalOut(ToolOutput):
    output_artifact: str
    n_rows: int
    n_folds: int


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
        name="walk_forward_signal",
        description=(
            "Walk-forward: refit an AlphaSignalModel on rolling/expanding "
            "training windows and emit out-of-sample signals. Output is "
            "rank-backtest-ready when passthrough_columns includes the "
            "realized forward return."
        ),
        input_model=WalkForwardSignalIn,
        output_model=WalkForwardSignalOut,
        tags=("backtest", "model"),
    )
    def _walk_forward(inp: WalkForwardSignalIn, ctx: ToolContext) -> WalkForwardSignalOut:
        if ctx.workspace is None:
            raise ToolError("walk_forward_signal requires a workspace in ToolContext")
        df = ctx.workspace.get(inp.input_artifact)
        preds = run_walk_forward(
            df,
            feature_cols=inp.feature_columns,
            target_col=inp.target_column,
            mode=inp.mode,
            train_lookback_days=inp.train_lookback_days,
            refit_every_days=inp.refit_every_days,
            test_window_days=inp.test_window_days,
            min_train_size=inp.min_train_size,
            target_horizon_days=inp.target_horizon_days,
            model_kind=inp.model_kind,
            passthrough_columns=inp.passthrough_columns,
        )
        ctx.workspace.put(inp.output_artifact, preds)
        n_folds = int(preds["fold_id"].nunique()) if "fold_id" in preds.columns and len(preds) else 0
        return WalkForwardSignalOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(preds)),
            n_folds=n_folds,
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
