from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from features import (
    FULL_CATEGORICAL_FEATURES,
    FULL_NUMERIC_FEATURES,
    LANE_CATEGORICAL_FEATURES,
    LANE_NUMERIC_FEATURES,
    build_preprocessor,
    prepare,
)

DATA_PATH = Path("data/train_test.csv")
MODELS_DIR = Path("models")
SPLIT_DATE = pd.Timestamp("2025-10-01")
TARGET = "posted_rate"

def build_candidate_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(frame["date"])
    train = frame.loc[dates < SPLIT_DATE]
    holdout = frame.loc[dates >= SPLIT_DATE]
    return train, holdout


def evaluate(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    predictions = model.predict(features)
    return {
        "mae": mean_absolute_error(target, predictions),
        "rmse": root_mean_squared_error(target, predictions),
        "r2": r2_score(target, predictions),
    }


def fit_best_model(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[Pipeline, dict]:
    train_df, holdout_df = chronological_split(frame)
    feature_columns = numeric_features + categorical_features

    results = {}
    fitted = {}
    for name, estimator in build_candidate_models().items():
        pipeline = Pipeline(
            [
                ("preprocess", build_preprocessor(numeric_features, categorical_features)),
                ("model", estimator),
            ]
        )
        pipeline.fit(train_df[feature_columns], train_df[TARGET])
        results[name] = evaluate(pipeline, holdout_df[feature_columns], holdout_df[TARGET])
        fitted[name] = pipeline

    best_name = min(results, key=lambda name: results[name]["mae"])
    best_pipeline = fitted[best_name].fit(frame[feature_columns], frame[TARGET])
    return best_pipeline, {"scores": results, "selected": best_name}


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    frame = prepare(raw)

    full_model, full_report = fit_best_model(frame, FULL_NUMERIC_FEATURES, FULL_CATEGORICAL_FEATURES)
    lane_model, lane_report = fit_best_model(frame, LANE_NUMERIC_FEATURES, LANE_CATEGORICAL_FEATURES)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(full_model, MODELS_DIR / "full_model.joblib")
    joblib.dump(lane_model, MODELS_DIR / "lane_model.joblib")

    report = {"full_model": full_report, "lane_model": lane_report}
    with open(MODELS_DIR / "training_report.json", "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
