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

## Getting started

1. **Create an environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install pandas numpy scikit-learn torch matplotlib
   ```
2. **(Optional) Update the raw data** – drop a fresh CSV from the open portal into `data/` and adjust the path you pass to `prepare_monthly_dataset`.

## Usage

- **环境准备**：在仓库根目录创建虚拟环境并安装依赖
  ```
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install pandas numpy scikit-learn torch matplotlib jupyter
  ```
- **数据就绪**：`data/Major_Crime_Indicators_Open_Data.csv` 已提供；若需要最新数据，可从开放门户下载后覆盖或修改路径。
- **运行核心 Notebook**：打开 `notebooks/linear_regression_pipeline.ipynb`（以及目录下其他 Notebook），在 Jupyter / JupyterLab 中直接 **Run All**。Notebook 会按顺序完成：
  - 调用 `src/preprocess.py`, `src/build_feature_matrix.py`, `src/data_prep.py` 进行清洗、月度聚合、NSI 计算、特征工程
  - 基于 `src/config/feature_def.py` 的 FEATURE_* 组合构建特征矩阵
  - 训练、评估模型（线性回归、LSTM 等）并输出指标/可视化
- **命令行脚本（可选）**：若希望在 Python 脚本中复现 LSTM 流程，可参考：
  ```python
  from src.data_prep import prepare_monthly_dataset
  from src.config.feature_def import FEATURE_BASE, TARGET_COL
  from src.models.LSTM_model import train_lstm_pipeline, create_sequences_per_neighborhood

  monthly_df = prepare_monthly_dataset("data/Major_Crime_Indicators_Open_Data.csv")
  model, y_true, y_pred, metrics, _ = train_lstm_pipeline(
      df=monthly_df,
      feature_cols=FEATURE_BASE,
      target_col=TARGET_COL,
      seq_length=12,
      create_sequences_fn=create_sequences_per_neighborhood,
  )
  print(metrics)
  ```
- **探索与迭代**：在 Notebook 中做 ad-hoc EDA、调参或模型比较，效果确定后可把逻辑抽到 `src/` 模块里以便脚本化复现。

## Feature sets

`src/config/feature_def.py` defines curated bundles that you can swap into the pipeline:

| Config | Description |
| --- | --- |
| `FEATURE_BASE` | Minimal inputs: neighbourhood id, report year/month, previous NSI, and centroid coordinates. |
| `FEATURE_NSI_3M` | Adds a 3-month rolling NSI average for short-term trend awareness. |
| `FEATURE_NSI_YEAR` | Adds previous-year NSI for the same month to capture seasonality. |
| `FEATURE_CRIME_6M` | Adds crime counts and 6-month rolling averages for richer temporal dynamics. |

Choose the list that matches the modeling question; all include `TARGET_COL = "NSI"`.

## Notebooks & experiments

The `notebooks/` directory holds the end-to-end pipelines used in the course project (for example, `linear_regression_pipeline.ipynb`). After installing dependencies, open each notebook and simply **Run All** to reproduce the entire workflow—from data prep through model training/evaluation—without touching the CLI scripts. Use notebooks for exploratory data analysis, quick hypothesis testing, or plotting alternative models; when something becomes production-ready, migrate the logic into `src/` for reusability.

## Next steps

- Tune hyperparameters (layers, hidden size, learning rate) or swap in other architectures (Temporal CNN, Transformers).
- Integrate cross-validation and backtesting utilities under `src/models`.
- Expose an inference script or simple API/UI that consumes the trained LSTM checkpoint for real-time NSI projections.
