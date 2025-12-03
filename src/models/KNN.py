import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from src.models.utils import (
    split_data,
    evaluate_model,
    plot_predictions,
    plot_residuals
)


############################################
# KNN Model Core
############################################

def build_model(n_neighbors=5, weights='distance', metric='euclidean'):
    """Initialize a KNN regressor with selected hyperparameters."""
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )


def train_model(model, X_train, y_train):
    """Fit KNN model."""
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Predict NSI values."""
    return model.predict(X_test)


def train_knn_model(df, feature_cols, target_col,
                    scale=True, n_neighbors=5,
                    weights='distance', metric='euclidean'):
    """
    Full KNN pipeline:
    split → build → train → predict → evaluate
    """
    df = df.dropna(subset=['Prev_Month_NSI']).copy()

    X_train, X_test, y_train, y_test, scaler = split_data(
        df, feature_cols, target_col, scale
    )

    model = build_model(n_neighbors, weights, metric)
    model = train_model(model, X_train, y_train)
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)

    return model, y_test, y_pred, metrics


############################################
# Hyperparameter Exploration
############################################

def knn_hparam_search(df, feature_cols, target_col,
                      k_values, weight_options, metric_options,
                      scale=True):
    """
    Grid-search over:
      • k ∈ k_values
      • weights ∈ {'uniform', 'distance'}
      • metric ∈ {'euclidean', 'manhattan'}

    Returns:
        results: list of dict
        best:    selected config by highest R2 & lowest RMSE
    """

    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    results = []

    for k in k_values:
        for w in weight_options:
            for m in metric_options:

                model = build_model(k, w, m)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = evaluate_model(y_test, y_pred)

                results.append({
                    "k": k,
                    "weights": w,
                    "metric": m,
                    "R2": metrics["R2"],
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"]
                })

    # Best = highest R2 and lowest RMSE
    best = max(results, key=lambda r: (r["R2"], -r["RMSE"]))
    return results, best


############################################
# Visualization Utilities
############################################

def plot_knn_metric_vs_k(results, metric="R2"):
    """
    Plot performance metric vs k for each weighting scheme.
    metric ∈ {"R2", "RMSE", "MAE"}
    """
    ks = sorted(set(r["k"] for r in results))
    for w in set(r["weights"] for r in results):

        vals = [
            next(r[metric] for r in results if r["k"] == k and r["weights"] == w)
            for k in ks
        ]

        plt.plot(ks, vals, marker='o', label=f"weights={w}")

    plt.title(f"KNN {metric} vs K")
    plt.xlabel("k (number of neighbors)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_knn_results(y_test, y_pred, n_samples=200):
    """Plot predictions + residuals."""
    plot_predictions(y_test, y_pred, n_samples)
    plot_residuals(y_test, y_pred)