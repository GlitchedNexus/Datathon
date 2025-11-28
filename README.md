# Loyalty Program & Rebranding Analysis

### Quickstart
- Ensure Python 3.12+ is available.
- Install deps: `uv sync`
- Activate shell: `uv run python main.py` (placeholder) or `uv run jupyter lab` for notebooks.
- Optional conda env: `conda env create -f config/env.yml` then `conda activate datathon`.

### Repository Layout
- Data: `data/raw`, `data/interim`, `data/processed`, `data/external`
- Figures: `figs/eda`, `figs/modeling`, `figs/final` (default format: `.png`)
- Styles: `styles/` (`colors.yaml`, `typography.md`, `matplotlib.json`, `plotly_theme.json`)
- Scripts: `scripts/` for loading, cleaning, processing, plotting, models, evaluation, visualization orchestrators, utils
- Config: `config/paths.yaml`, `config/env.yml`, `config/experiments/`
- Notebooks: `notebooks/00_data_overview.ipynb` … `04_report_figures.ipynb`
- Slides: `slides/outline.md`, `slides/deck_template.pptx` (placeholder), `slides/assets/`
- Reports: `reports/logbook.md`, `reports/decisions.md`
- Artifacts: `artifacts/models`, `artifacts/metrics`, `artifacts/logs`

```
project-root/
├─ config/           # env + paths + experiment configs
├─ data/             # raw → interim → processed → external
├─ figs/             # saved figures (eda/modeling/final)
├─ styles/           # color/typography + plotting themes
├─ scripts/          # loading, cleaning, processing, plots, models, evaluation, utils
├─ notebooks/        # 00..04 notebooks (story driven)
├─ slides/           # outline, template, assets
├─ reports/          # logbook + decision record
├─ artifacts/        # saved models/metrics/logs
└─ main.py           # placeholder entrypoint
```

### Folder Reference
- `config/paths.yaml`: canonical locations for data/figs/artifacts; keep scripts aligned.
- `config/env.yml`: conda spec (Python 3.12, pandas, numpy, sklearn, matplotlib, seaborn, plotly, xgboost, pyyaml).
- `config/experiments/`: drop YAML/JSON configs for runs (name, task, data_path, target, model_type, params).
- `data/`: store source data; interim/processed mirror stages; use `scripts/loading/load_data.py` helpers.
- `figs/`: `eda/`, `modeling/`, `final/` for slide-ready assets; plotting helpers save here automatically.
- `styles/`: Tailwind sky palette + typography + matplotlib/plotly themes.
- `scripts/`: reusable code—cleaning, processing, model builders, evaluation, plotting, visualization orchestration, utils.
- `notebooks/`: guided workflows; keep logic inside `scripts/`, call functions here.
- `slides/`: outline + template placeholder; point to `figs/final/` assets.
- `reports/`: logbook and decisions to narrate choices.
- `artifacts/`: persisted models (`.joblib`) and metrics (`.json`).

### Figure & Experiment Naming
- EDA: `eda_distribution_<column>.png`, `eda_corr_heatmap.png`, `eda_scatter_<x>_vs_<y>.png`
- Modeling: `model_<modelname>_roc.png`, `model_<modelname>_pr.png`, `model_<modelname>_cm.png`, `model_<modelname>_fi.png`
- Finals for slides: `final_corr_heatmap.png`, `final_class_balance.png`, `final_best_model_roc.png`, `final_feature_vs_target.png`
- Experiments/metrics: `exp001_baseline_lr.json`, increment IDs per run

### Notebook Objectives
- `00_data_overview`: load raw data, inspect schema, list assumptions/questions
- `01_eda`: run cleaning/processing, generate standard EDA figures into `figs/eda`
- `02_baseline_models`: train quick baselines with evaluation plots/metrics
- `03_advanced_models`: improved features/models, compare to baselines
- `04_report_figures`: curate best visuals, export into `figs/final`

### Styling Decisions (Tailwind Sky)
- Palette in `styles/colors.yaml`; typography in `styles/typography.md`
- Matplotlib/Plotly themes pre-set for light background, sky colorway, Inter font

### Next Steps
- Add data and create experiment configs under `config/experiments/` (see schema in `scripts/models/train_model.py`).
- Run `uv sync`, then `uv run python -m scripts.visualization.generate_eda_figures` after placing processed data.
- Wire notebooks to call the now-implemented scripts; export final slide figures to `figs/final/`.
- Update `slides/deck_template.pptx` with the sky palette + Inter typography before presenting.
