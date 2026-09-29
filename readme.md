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

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python src/train.py
python src/predict.py
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```