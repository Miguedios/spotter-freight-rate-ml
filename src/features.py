from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_START = pd.Timestamp("2025-01-01")

FULL_NUMERIC_FEATURES = [
    "distance",
    "weight",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_year",
    "trend_day",
]
FULL_CATEGORICAL_FEATURES = ["equipment"]

LANE_NUMERIC_FEATURES = [
    "distance",
    "weight",
    "month",
    "day_of_week",
    "day_of_year",
    "trend_day",
]
LANE_CATEGORICAL_FEATURES = ["pickup", "delivery", "equipment"]


def clean_weight(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.loc[result["weight"] <= 0, "weight"] = pd.NA
    result["weight"] = pd.to_numeric(result["weight"], errors="coerce")
    return result


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    parsed = pd.to_datetime(result["date"])
    result["month"] = parsed.dt.month
    result["day_of_week"] = parsed.dt.dayofweek
    result["day_of_year"] = parsed.dt.dayofyear
    result["trend_day"] = (parsed - TRAIN_START).dt.days
    return result


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    return add_date_features(clean_weight(frame))


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
