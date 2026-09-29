# Freight Rate Prediction Challenge

See `Freight_Rate_ML_Assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `data/train_test.csv`.
2. Predict every load in `data/validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `data/validation_predictions_template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `data/december_chart_inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link

## Solution layout

- `src/features.py` — data cleaning, date feature engineering, and preprocessing pipeline builders.
- `src/train.py` — chronological train/holdout split, trains and compares Linear Regression, Random Forest, and Gradient Boosting, saves the best model per use case.
- `src/predict.py` — loads the trained models and writes `validation_predictions.csv` and `december_predictions.csv`.

Two models are trained:

- `full_model`: uses every feature available in `data/validation.csv` (distance, weight, coordinates, equipment, date, market index, quote signal) to score the 12,000 validation loads.
- `lane_model`: uses only the features available in `data/december_chart_inputs.csv` (distance, weight, equipment, pickup/delivery city, date) — `market_index` and `quote_signal` are not provided for December, so a second model is trained without them instead of guessing their values.

## Results (chronological holdout — October 2025)

| Model | Full model MAE | Full model R² | Lane model MAE | Lane model R² |
|---|---|---|---|---|
| Linear Regression | **$141.74** | **0.818** | $172.61 | 0.816 |
| Random Forest | $287.50 | 0.740 | $218.74 | 0.757 |
| Gradient Boosting | $159.37 | 0.814 | **$154.00** | **0.816** |

Bold = selected model (lowest MAE on the October holdout) for each use case.

## Key design decisions

- **Chronological split, not random/k-fold.** Training on January–September and holding out October mirrors the real task — forecasting a future period from a past one — and avoids leaking market conditions from the test period into training.
- **Coordinates instead of city names for the full model.** `validation.csv` contains 8 pickup/delivery cities (Chicago, Norfolk, San Diego, Charlotte, Knoxville, Allentown, Laredo, Jackson) that never appear in `train_test.csv`. One-hot encoding city names would give the model no signal for these rows; continuous latitude/longitude lets it generalize geographically instead.
- **Two models, not one.** `december_chart_inputs.csv` doesn't include `market_index`, `quote_signal`, or coordinates. Rather than invent values for missing columns, `lane_model` is trained on the same 48,000 rows using only the columns that file actually provides.
- **Raw data is intentionally not committed** (see `.gitignore`) — it was shared for this assessment, not for public redistribution.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/train.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```