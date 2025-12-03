Neighborhood Safety Index (NSI) Forecasting
===========================================

This repo contains a light-weight pipeline for building a **Neighborhood Safety Index** from the Toronto Major Crime Indicators open data set and forecasting short-term safety trends with an LSTM model. The project was created for CSCC11 and covers cleaning the raw incident CSV, aggregating crimes into a monthly NSI per neighbourhood, engineering temporal features, and learning multi-step dynamics with PyTorch.

## Data

- `data/Major_Crime_Indicators_Open_Data.csv` – raw open data published by Toronto Police Services.
- Each record is transformed into monthly aggregates that summarize crime severity, counts, and geography (x/y centroid) before being scaled into an NSI value in `[0, 1]`.

## Repository layout

```
├── README.md
├── data/
│   └── Major_Crime_Indicators_Open_Data.csv
├── notebooks/                
└── src/
    ├── preprocess.py         # fills OCC_* columns derived from OCC_DATE
    ├── build_feature_matrix.py
    ├── data_prep.py          # one-stop helper to produce monthly modeling set
    ├── config/feature_def.py # curated feature lists & target column
    └── models/
```

## Usage

- **Environment setup**: create a virtual environment in the repo root and install dependencies
  ```
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install pandas numpy scikit-learn torch matplotlib jupyter
  ```
- **Data**: the raw CSV `data/Major_Crime_Indicators_Open_Data.csv` is already included; download a fresh copy from the open data portal if you want the latest snapshot.
- **Run the core notebook**: open `notebooks/linear_regression_pipeline.ipynb` (and any other notebook in the folder) and simply **Run All** inside Jupyter / JupyterLab. The notebook orchestrates:
  - calling `src/preprocess.py`, `src/build_feature_matrix.py`, `src/data_prep.py` for cleaning, monthly aggregation, NSI computation, and feature engineering
  - building the feature matrix with the FEATURE_* bundles defined in `src/config/feature_def.py`
  - training/evaluating the models (linear regression, LSTM, etc.) and emitting metrics/plots
- **Explore & iterate**: use the notebooks for ad-hoc EDA, hyper-parameter tuning, or model comparisons; once the workflow solidifies, port the logic into `src/` modules for repeatable runs.
