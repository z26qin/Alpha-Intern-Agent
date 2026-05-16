"""Tests for cross-sectional feature engineering — no network, deterministic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_intern.features.cross_sectional import (
    CS_FEATURE_COLUMNS,
    CrossSectionalSpec,
    build_cross_sectional_features,
    clear_registered_specs,
    get_registered_specs,
    register_spec,
)
from alpha_intern.features.technical import build_basic_features
from alpha_intern.data.loader import normalize_price_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_panel(n_days: int = 120, n_tickers: int = 3, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        rets = rng.normal(loc=0.0002, scale=0.012, size=n_days)
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
    return normalize_price_dataframe(pd.concat(frames, ignore_index=True))


def _featurized_panel(**kw) -> pd.DataFrame:
    return build_basic_features(_synthetic_panel(**kw))


# Isolate registry between tests
@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registered_specs()
    yield
    clear_registered_specs()


# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------

def test_default_cs_columns_present() -> None:
    out = build_cross_sectional_features(_featurized_panel())
    for col in CS_FEATURE_COLUMNS:
        assert col in out.columns, f"missing default cs column: {col}"


def test_original_columns_preserved() -> None:
    feats = _featurized_panel()
    original_cols = set(feats.columns)
    out = build_cross_sectional_features(feats)
    assert original_cols.issubset(set(out.columns)), "original columns must be preserved"


def test_row_count_unchanged() -> None:
    feats = _featurized_panel()
    out = build_cross_sectional_features(feats)
    assert len(out) == len(feats)


# ---------------------------------------------------------------------------
# Rank semantics
# ---------------------------------------------------------------------------

def test_rank_values_in_unit_interval() -> None:
    out = build_cross_sectional_features(_featurized_panel())
    rank_cols = [c for c in CS_FEATURE_COLUMNS if c.endswith("_rank")]
    for col in rank_cols:
        valid = out[col].dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all(), \
            f"{col} has values outside [0, 1]"


def test_rank_nan_propagated_from_source() -> None:
    """Rows where the source feature is NaN should have NaN rank too."""
    out = build_cross_sectional_features(_featurized_panel())
    # return_20d is NaN for first 19 rows per ticker
    mask_nan_source = out["return_20d"].isna()
    assert out.loc[mask_nan_source, "cs_return_20d_rank"].isna().all()


def test_single_ticker_rank_is_one() -> None:
    """With only one ticker, pct_rank returns 1.0 (not NaN)."""
    feats = build_basic_features(_synthetic_panel(n_tickers=1))
    out = build_cross_sectional_features(feats)
    valid = out["cs_return_1d_rank"].dropna()
    assert (valid == 1.0).all()


# ---------------------------------------------------------------------------
# Z-score semantics
# ---------------------------------------------------------------------------

def test_zscore_clipped() -> None:
    out = build_cross_sectional_features(_featurized_panel())
    zscore_cols = [c for c in CS_FEATURE_COLUMNS if c.endswith("_zscore")]
    for col in zscore_cols:
        valid = out[col].dropna()
        assert (valid >= -3.0).all() and (valid <= 3.0).all(), \
            f"{col} has values outside [-3, 3]"


def test_custom_zscore_clip() -> None:
    spec = CrossSectionalSpec("return_1d", compute_rank=False, compute_zscore=True, zscore_clip=1.5)
    out = build_cross_sectional_features(_featurized_panel(), extra_specs=[spec])
    valid = out["cs_return_1d_zscore"].dropna()
    assert (valid >= -1.5).all() and (valid <= 1.5).all()


def test_constant_column_zscore_is_nan() -> None:
    """When all tickers have the same value, std=0 → zscore should be NaN."""
    feats = _featurized_panel()
    feats["flat"] = 1.0
    spec = CrossSectionalSpec("flat", compute_rank=False, compute_zscore=True)
    out = build_cross_sectional_features(feats, extra_specs=[spec])
    assert out["cs_flat_zscore"].isna().all()


# ---------------------------------------------------------------------------
# Extension: extra_specs
# ---------------------------------------------------------------------------

def test_extra_specs_adds_columns() -> None:
    feats = _featurized_panel()
    feats["my_signal"] = np.random.default_rng(1).normal(size=len(feats))
    spec = CrossSectionalSpec("my_signal", compute_rank=True, compute_zscore=True)
    out = build_cross_sectional_features(feats, extra_specs=[spec])
    assert "cs_my_signal_rank" in out.columns
    assert "cs_my_signal_zscore" in out.columns


def test_extra_spec_column_missing_from_df_is_skipped() -> None:
    """If a spec references a column not in the DataFrame, it is silently skipped."""
    spec = CrossSectionalSpec("nonexistent_col", compute_rank=True)
    out = build_cross_sectional_features(_featurized_panel(), extra_specs=[spec])
    assert "cs_nonexistent_col_rank" not in out.columns


# ---------------------------------------------------------------------------
# Extension: global registry
# ---------------------------------------------------------------------------

def test_register_spec_included_in_output() -> None:
    feats = _featurized_panel()
    feats["ext_signal"] = np.random.default_rng(2).normal(size=len(feats))
    register_spec(CrossSectionalSpec("ext_signal", compute_rank=True))
    out = build_cross_sectional_features(feats)
    assert "cs_ext_signal_rank" in out.columns


def test_get_registered_specs_reflects_registrations() -> None:
    assert get_registered_specs() == []
    spec = CrossSectionalSpec("pe_ratio", compute_rank=True)
    register_spec(spec)
    assert len(get_registered_specs()) == 1
    assert get_registered_specs()[0].source_column == "pe_ratio"


def test_clear_registered_specs() -> None:
    register_spec(CrossSectionalSpec("ev_ebitda", compute_rank=True))
    clear_registered_specs()
    assert get_registered_specs() == []


def test_extra_spec_overrides_default_for_same_source_column() -> None:
    """Passing an extra_spec for a default column should override the default."""
    # Default for return_1d has compute_zscore=True; override to False
    spec = CrossSectionalSpec("return_1d", compute_rank=True, compute_zscore=False)
    out = build_cross_sectional_features(_featurized_panel(), extra_specs=[spec])
    assert "cs_return_1d_rank" in out.columns
    assert "cs_return_1d_zscore" not in out.columns


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_missing_date_column_raises() -> None:
    feats = _featurized_panel().drop(columns=["date"])
    with pytest.raises(ValueError, match="'date'"):
        build_cross_sectional_features(feats)


def test_missing_ticker_column_raises() -> None:
    feats = _featurized_panel().drop(columns=["ticker"])
    with pytest.raises(ValueError, match="'ticker'"):
        build_cross_sectional_features(feats)


# ---------------------------------------------------------------------------
# Sorting / determinism
# ---------------------------------------------------------------------------

def test_output_sorted_by_ticker_date() -> None:
    out = build_cross_sectional_features(_featurized_panel())
    for _, g in out.groupby("ticker"):
        assert g["date"].is_monotonic_increasing
