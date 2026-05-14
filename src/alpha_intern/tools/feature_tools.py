"""Tool wrappers around alpha_intern.features."""

from __future__ import annotations

from pydantic import Field

from alpha_intern.features.technical import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_basic_features,
)
from alpha_intern.tools.registry import (
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    ToolRegistry,
)


class BuildFeaturesIn(ToolInput):
    input_artifact: str = Field(..., description="Workspace name of a normalized price DataFrame.")
    output_artifact: str = Field(..., description="Workspace name to write the feature DataFrame to.")


class BuildFeaturesOut(ToolOutput):
    output_artifact: str
    n_rows: int
    feature_columns: list[str]
    target_column: str


def register(registry: ToolRegistry) -> None:
    @registry.tool(
        name="build_features",
        description=(
            "Compute backward-looking technical features (and one named forward "
            "target) from a normalized price DataFrame."
        ),
        input_model=BuildFeaturesIn,
        output_model=BuildFeaturesOut,
        tags=("features",),
    )
    def _run(inp: BuildFeaturesIn, ctx: ToolContext) -> BuildFeaturesOut:
        if ctx.workspace is None:
            raise ToolError("build_features requires a workspace in ToolContext")
        prices = ctx.workspace.get(inp.input_artifact)
        feats = build_basic_features(prices)
        ctx.workspace.put(inp.output_artifact, feats)
        return BuildFeaturesOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(feats)),
            feature_columns=list(FEATURE_COLUMNS),
            target_column=TARGET_COLUMN,
        )
