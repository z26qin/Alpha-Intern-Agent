"""Tool wrappers around alpha_intern.features."""

from __future__ import annotations

from pydantic import Field

from alpha_intern.features.technical import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_basic_features,
)
from alpha_intern.features.cross_sectional import (
    CS_FEATURE_COLUMNS,
    CrossSectionalSpec,
    build_cross_sectional_features,
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


class BuildCrossSectionalIn(ToolInput):
    input_artifact: str = Field(
        ...,
        description=(
            "Workspace name of a DataFrame that already contains technical "
            "features (output of build_features).  Must include at least a "
            "'date' and 'ticker' column."
        ),
    )
    output_artifact: str = Field(
        ..., description="Workspace name to write the enriched DataFrame to."
    )
    extra_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Names of additional numeric columns already present in the "
            "input DataFrame to cross-sectionalise with both rank and "
            "z-score.  Use this to incorporate columns added by external "
            "data sources (fundamentals, alt-data, API feeds) without "
            "modifying the global spec registry."
        ),
    )


class BuildCrossSectionalOut(ToolOutput):
    output_artifact: str
    n_rows: int
    cs_feature_columns: list[str]


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

    @registry.tool(
        name="build_cross_sectional_features",
        description=(
            "Enrich a feature DataFrame with cross-sectional statistics "
            "(percentile ranks and z-scores) computed across all tickers on "
            "each date.  Accepts an optional list of extra column names from "
            "external data sources (fundamentals, alt-data, API feeds) to "
            "cross-sectionalise alongside the default technical features."
        ),
        input_model=BuildCrossSectionalIn,
        output_model=BuildCrossSectionalOut,
        tags=("features",),
    )
    def _run_cs(inp: BuildCrossSectionalIn, ctx: ToolContext) -> BuildCrossSectionalOut:
        if ctx.workspace is None:
            raise ToolError("build_cross_sectional_features requires a workspace in ToolContext")
        feats = ctx.workspace.get(inp.input_artifact)
        extra_specs = [
            CrossSectionalSpec(col, compute_rank=True, compute_zscore=True)
            for col in inp.extra_columns
        ]
        enriched = build_cross_sectional_features(feats, extra_specs=extra_specs)
        ctx.workspace.put(inp.output_artifact, enriched)
        all_cs_cols = list(CS_FEATURE_COLUMNS) + [
            c for spec in extra_specs for c in spec.output_columns
        ]
        return BuildCrossSectionalOut(
            output_artifact=inp.output_artifact,
            n_rows=int(len(enriched)),
            cs_feature_columns=all_cs_cols,
        )
