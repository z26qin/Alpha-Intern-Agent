"""Tool wrappers around alpha_intern.models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from alpha_intern.models.signal_model import AlphaSignalModel
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class TrainSignalIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of the feature DataFrame.")
    feature_columns: list[str] = Field(..., description="Column names to use as features.")
    target_column: str = Field(..., description="Column name to use as the regression target.")
    model_kind: Literal["ridge", "random_forest"] = Field(default="ridge")
    model_artifact: str = Field(..., description="Workspace name to store the fitted model under.")


class TrainSignalOut(ToolOutput):
    model_artifact: str
    model_kind: str
    n_train_rows: int


class PredictSignalIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of the feature DataFrame to score.")
    model_artifact: str = Field(..., description="Workspace name of a fitted AlphaSignalModel.")
    output_artifact: str = Field(..., description="Workspace name to store the signal DataFrame under.")


class PredictSignalOut(ToolOutput):
    output_artifact: str
    n_rows: int


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="train_signal",
        description="Fit an AlphaSignalModel on a feature DataFrame and store it in the workspace.",
        input_model=TrainSignalIn,
        output_model=TrainSignalOut,
        tags=("model",),
    )
    def _train(inp: TrainSignalIn, ctx: ToolContext) -> TrainSignalOut:
        if ctx.workspace is None:
            raise ToolError("train_signal requires a workspace in ToolContext")
        df = ctx.workspace.get(inp.input_artifact)
        model = AlphaSignalModel(kind=inp.model_kind)
        model.fit(df, feature_cols=inp.feature_columns, target_col=inp.target_column)
        ctx.workspace.put(inp.model_artifact, model)
        n_train = int(df.dropna(subset=[inp.target_column]).shape[0])
        return TrainSignalOut(
            model_artifact=inp.model_artifact,
            model_kind=inp.model_kind,
            n_train_rows=n_train,
        )

    @registry.tool(
        name="predict_signal",
        description="Run a fitted AlphaSignalModel over a feature DataFrame and store the signal frame.",
        input_model=PredictSignalIn,
        output_model=PredictSignalOut,
        tags=("model",),
    )
    def _predict(inp: PredictSignalIn, ctx: ToolContext) -> PredictSignalOut:
        if ctx.workspace is None:
            raise ToolError("predict_signal requires a workspace in ToolContext")
        df = ctx.workspace.get(inp.input_artifact)
        model = ctx.workspace.get(inp.model_artifact)
        if not isinstance(model, AlphaSignalModel):
            raise ToolError(
                f"Artifact {inp.model_artifact!r} is not an AlphaSignalModel"
            )
        preds = model.predict(df)
        ctx.workspace.put(inp.output_artifact, preds)
        return PredictSignalOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(preds)),
        )
