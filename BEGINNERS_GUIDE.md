# Datathon Workflow — Beginner Edition

This is a hand-holding guide for running the project end-to-end. Follow it in order.

## 1) One-time setup
1. Install Python 3.12+.
2. Install dependencies (uses uv):
   ```bash
   uv sync
   ```
3. Activate the environment when you run commands:
   ```bash
   uv run python --version
   ```
   (Use `uv run <command>` for everything below.)

## 2) Know your folders (where things go)
- `config/paths.yaml` — canonical locations for data, figures, artifacts.
- `config/experiments/` — YAML/JSON configs that describe each model run.
- `data/` — raw → interim → processed data. You will read from `data/processed/` for modeling.
- `figs/` — saved plots: `eda/`, `modeling/`, `final/` for slides.
- `scripts/` — all reusable code (loading, cleaning, processing, plotting, modeling, evaluation).
- `notebooks/` — human-friendly steps that call code from `scripts/`.
- `slides/` — outline + template, pull final images from `figs/final/`.
- `artifacts/` — saved models (`.joblib`) and metrics (`.json`).

## 3) Put data in the right place
1. Drop your raw CSVs into `data/raw/` (e.g., `train.csv`, `test.csv`).
2. If you have intermediate/cleaned versions, put them in `data/interim/` or `data/processed/`.
3. For this scaffold, modeling expects a processed CSV in `data/processed/your_data.csv`.

## 4) Quick inspection (optional but recommended)
Run the first notebook to understand the data shape/types:
```bash
uv run jupyter lab
```
Open `notebooks/00_data_overview.ipynb` and explore head/dtypes. Note target column name.

## 5) Basic cleaning + features (scripts)
If you want scripted cleaning/processing:
- Edit `scripts/cleaning/clean_data.py` to adjust missing-value logic.
- Edit `scripts/processing/feature_engineering.py` to add domain features.
- Save your cleaned file to `data/processed/your_data.csv`.

## 6) Create an experiment config
Make a YAML file in `config/experiments/`, e.g., `exp001_baseline.yaml`:
```yaml
task: classification           # or regression
data_path: data/processed/your_data.csv
target: your_target_column
model_name: exp001_baseline_lr
model_type: logistic_regression # options: logistic_regression, random_forest_classifier, random_forest_regressor, gbdt_classifier, gbdt_regressor, xgb_classifier, xgb_regressor
model_params: {}
test_size: 0.2
random_state: 42
```
(For regression, pick a regressor model type and set `task: regression`.)

## 7) Run a baseline experiment (no notebook needed)
```bash
uv run python -m scripts.evaluation.experiment_runner config/experiments/exp001_baseline.yaml
```
This will:
- Load the data/target
- Split train/val
- Train the chosen model
- Save metrics to `artifacts/metrics/<exp-name>.json`
- Save the trained model to `artifacts/models/<model_name>.joblib`

## 8) Generate EDA figures
If you already have processed data in `data/processed/train.csv` (or change the name in the script), run:
```bash
uv run python -m scripts.visualization.generate_eda_figures
```
Outputs land in `figs/eda/` with sensible filenames.

## 9) Generate modeling figures
After training, if you have predictions and probabilities, call the plotting utilities from your pipeline, or adapt `scripts/visualization/generate_model_figures.py` by passing `y_true`, `y_pred`, `y_proba`, labels, and optional feature importances. Save outputs to `figs/modeling/`.

## 10) Prepare slides
- Pick 5–10 best figures and copy them (or save directly) into `figs/final/` with descriptive names like `final_best_model_roc.png`.
- Open `slides/outline.md` and map each slide to a figure + one-sentence takeaway.
- Update `slides/deck_template.pptx` with the Tailwind sky colors and Inter font (see `styles/`).

## 11) Keep notes
- Log experiments and observations in `reports/logbook.md` (what you tried, what worked).
- Record decisions/assumptions in `reports/decisions.md` (metrics chosen, handling of leakage, etc.).

## 12) Common tweaks
- Need more model power? Try `model_type: random_forest_classifier` or `xgb_classifier` with tuned `model_params`.
- Imbalanced classes? Adjust class weights in `model_params` for logistic regression or use `scale_pos_weight` for XGBoost.
- Too slow? Reduce `n_estimators` or depth for forests/boosting.

## 13) Resetting for new data
- Swap in new raw data under `data/raw/`.
- Re-run cleaning/processing to `data/processed/new_data.csv`.
- Create a new experiment config (e.g., `exp002_*`) and rerun steps 7–9.

That’s it—stay in `uv run` for commands, keep configs in `config/experiments/`, and save outputs to `figs/` + `artifacts/` as you go.
