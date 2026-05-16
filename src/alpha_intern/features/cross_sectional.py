"""Cross-sectional feature engineering.

Cross-sectional features rank or normalise a metric *across all tickers on
the same date*, complementing the per-ticker time-series features in
technical.py.

Extension point
---------------
Other data sources (fundamentals, alternative data, external APIs) can
contribute columns to the cross-sectional pipeline by calling
``register_spec()`` before ``build_cross_sectional_features()`` is invoked:

    from alpha_intern.features.cross_sectional import register_spec, CrossSectionalSpec
    register_spec(CrossSectionalSpec("pe_ratio", compute_rank=True, compute_zscore=True))

The new column will be picked up automatically on the next call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class CrossSectionalSpec:
    """Declares how one source column should be cross-sectionalised.

    Attributes
    ----------
    source_column:
        Name of the column already present in the input DataFrame.
    compute_rank:
        If True, add ``cs_{source_column}_rank`` — percentile rank in [0, 1]
        across tickers at each date (NaN propagated).
    compute_zscore:
        If True, add ``cs_{source_column}_zscore`` — z-score across tickers at
        each date, clipped to ``[-zscore_clip, zscore_clip]``.
    zscore_clip:
        Symmetric clip applied after z-scoring (default 3.0 σ).
    """

    source_column: str
    compute_rank: bool = True
    compute_zscore: bool = False
    zscore_clip: float = 3.0

    @property
    def output_columns(self) -> list[str]:
        cols: list[str] = []
        if self.compute_rank:
            cols.append(f"cs_{self.source_column}_rank")
        if self.compute_zscore:
            cols.append(f"cs_{self.source_column}_zscore")
        return cols


# ---------------------------------------------------------------------------
# Global spec registry
# ---------------------------------------------------------------------------

_REGISTRY: list[CrossSectionalSpec] = []


def register_spec(spec: CrossSectionalSpec) -> None:
    """Register a cross-sectional spec so it is included in every future call
    to ``build_cross_sectional_features``."""
    _REGISTRY.append(spec)


def get_registered_specs() -> list[CrossSectionalSpec]:
    """Return a copy of the currently registered specs."""
    return list(_REGISTRY)


def clear_registered_specs() -> None:
    """Remove all registered specs (useful in tests for isolation)."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Default specs (from technical.py feature columns)
# ---------------------------------------------------------------------------

_DEFAULT_SPECS: list[CrossSectionalSpec] = [
    CrossSectionalSpec("return_1d",        compute_rank=True, compute_zscore=True),
    CrossSectionalSpec("return_5d",        compute_rank=True, compute_zscore=True),
    CrossSectionalSpec("return_20d",       compute_rank=True, compute_zscore=True),
    CrossSectionalSpec("volatility_20d",   compute_rank=True, compute_zscore=False),
    CrossSectionalSpec("volume_zscore_20d", compute_rank=True, compute_zscore=False),
]

CS_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    col for spec in _DEFAULT_SPECS for col in spec.output_columns
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _apply_spec(group: pd.DataFrame, spec: CrossSectionalSpec) -> pd.DataFrame:
    """Apply one spec to a single date-group (modifies in place)."""
    col = spec.source_column
    if col not in group.columns:
        return group

    series = group[col]

    if spec.compute_rank:
        out_col = f"cs_{col}_rank"
        group[out_col] = series.rank(pct=True, na_option="keep")

    if spec.compute_zscore:
        out_col = f"cs_{col}_zscore"
        mu = series.mean(skipna=True)
        sigma = series.std(skipna=True, ddof=0)
        if sigma == 0 or np.isnan(sigma):
            group[out_col] = np.nan
        else:
            zscores = (series - mu) / sigma
            group[out_col] = zscores.clip(lower=-spec.zscore_clip, upper=spec.zscore_clip)

    return group


def build_cross_sectional_features(
    df: pd.DataFrame,
    extra_specs: Sequence[CrossSectionalSpec] | None = None,
) -> pd.DataFrame:
    """Add cross-sectional rank/zscore columns to a feature DataFrame.

    Parameters
    ----------
    df:
        Output of ``build_basic_features`` (or any DataFrame that already
        contains the source columns referenced by the active specs).
    extra_specs:
        Additional specs to apply on top of the registered ones.  Use this to
        pass columns originating from external data sources (fundamentals,
        alternative data, API feeds) without permanently mutating the registry.

    Returns
    -------
    DataFrame with all original columns preserved plus the new ``cs_*`` columns.
    Cross-sectional statistics are computed *within each date*, so at least two
    tickers are needed for meaningful ranks/z-scores.
    """
    specs: list[CrossSectionalSpec] = _DEFAULT_SPECS + list(_REGISTRY) + list(extra_specs or [])

    # Deduplicate by source_column + flags (last registration wins for a given
    # source_column so callers can override defaults).
    seen: dict[str, CrossSectionalSpec] = {}
    for spec in specs:
        seen[spec.source_column] = spec
    active_specs = list(seen.values())

    work = df.copy()

    # Group by date and apply each spec across the ticker universe.
    result_groups: list[pd.DataFrame] = []
    for _, date_group in work.groupby("date", sort=False):
        g = date_group.copy()
        for spec in active_specs:
            g = _apply_spec(g, spec)
        result_groups.append(g)

    if not result_groups:
        return work

    out = pd.concat(result_groups, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out
