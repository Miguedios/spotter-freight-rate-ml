from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import (
    FULL_CATEGORICAL_FEATURES,
    FULL_NUMERIC_FEATURES,
    LANE_CATEGORICAL_FEATURES,
    LANE_NUMERIC_FEATURES,
    prepare,
)

MODELS_DIR = Path("models")
DATA_DIR = Path("data")


def predict_validation() -> None:
    model = joblib.load(MODELS_DIR / "full_model.joblib")
    frame = prepare(pd.read_csv(DATA_DIR / "validation.csv"))
    feature_columns = FULL_NUMERIC_FEATURES + FULL_CATEGORICAL_FEATURES
    frame["predicted_rate"] = model.predict(frame[feature_columns])
    frame[["load_id", "predicted_rate"]].to_csv("validation_predictions.csv", index=False)


def predict_december() -> None:
    model = joblib.load(MODELS_DIR / "lane_model.joblib")
    original = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")
    frame = prepare(original)
    feature_columns = LANE_NUMERIC_FEATURES + LANE_CATEGORICAL_FEATURES
    original["predicted_rate"] = model.predict(frame[feature_columns])
    original.to_csv("december_predictions.csv", index=False)


def main() -> None:
    predict_validation()
    predict_december()


if __name__ == "__main__":
    main()
