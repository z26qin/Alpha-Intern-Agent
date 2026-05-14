"""A small sklearn-based signal model.

Deliberately simple: impute → standardize → linear (Ridge) or
RandomForest. No hyperparameter search yet. This module is meant to be
a stable interface that more sophisticated models can implement later.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ModelKind = Literal["ridge", "random_forest"]


class AlphaSignalModel:
    """Lightweight cross-sectional signal model."""

    def __init__(
        self,
        kind: ModelKind = "ridge",
        random_state: int = 7,
    ) -> None:
        if kind == "ridge":
            estimator = Ridge(alpha=1.0, random_state=random_state)
        elif kind == "random_forest":
            estimator = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
                n_jobs=1,
            )
        else:
            raise ValueError(f"Unknown model kind: {kind!r}")

        self.kind: ModelKind = kind
        self.pipeline: Pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("estimator", estimator),
            ]
        )
        self.feature_cols_: Optional[list[str]] = None
        self.target_col_: Optional[str] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: Iterable[str],
        target_col: str,
    ) -> "AlphaSignalModel":
        feature_cols = list(feature_cols)
        if not feature_cols:
            raise ValueError("feature_cols must be a non-empty iterable")
        if target_col in feature_cols:
            raise ValueError(
                f"target_col {target_col!r} must not appear in feature_cols"
            )

        needed = feature_cols + [target_col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for fit: {missing}")

        train = df.dropna(subset=[target_col]).copy()
        X = train[feature_cols]
        y = train[target_col].astype(float)

        if len(train) == 0:
            raise ValueError("No training rows after dropping rows with missing target")

        self.pipeline.fit(X, y)
        self.feature_cols_ = feature_cols
        self.target_col_ = target_col
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_ or self.feature_cols_ is None:
            raise RuntimeError("AlphaSignalModel.predict called before fit()")

        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for predict: {missing}")

        preds = self.pipeline.predict(df[self.feature_cols_])
        out = pd.DataFrame(index=df.index)
        if "date" in df.columns:
            out["date"] = df["date"].values
        if "ticker" in df.columns:
            out["ticker"] = df["ticker"].values
        out["signal"] = preds
        return out
