import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

from src.models.utils import (
    split_data,
    evaluate_model,
    plot_predictions,
    plot_residuals
)


##############################
# KNN core model functions
##############################

def build_model(n_neighbors=5, weights='distance'):
    """Initialize a KNN regressor."""
    return KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights
    )


def train_model(model, X_train, y_train):
    """Fit model."""
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Predict NSI."""
    return model.predict(X_test)


def train_knn_model(df, feature_cols, target_col, scale=True, n_neighbors=5):
    """
    Full single-run KNN pipeline:
    split → build → train → predict → evaluate
    """
    df = df.dropna(subset=['Prev_Month_NSI']).copy()

    X_train, X_test, y_train, y_test, scaler = split_data(
        df, feature_cols, target_col, scale
    )

    model = build_model(n_neighbors=n_neighbors)
    model = train_model(model, X_train, y_train)
    y_pred = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred)

    return model, y_test, y_pred, metrics


##############################
# K exploration functions
##############################

def explore_k_values(df, feature_cols, target_col, k_values, scale=True):
    """
    Iterate multiple K values and record performance metrics.
    """
    results = []

    for k in k_values:
        _, y_test, y_pred, metrics = train_knn_model(
            df,
            feature_cols,
            target_col,
            scale=scale,
            n_neighbors=k
        )
        results.append({
            "k": k,
            "R2": metrics["R2"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"]
        })

    return results


def plot_k_performance(results, metric="R2"):
    """
    Plot performance metrics vs K.
    metric ∈ {"R2", "MAE", "RMSE"}
    """
    ks = [res["k"] for res in results]
    vals = [res[metric] for res in results]

    plt.figure(figsize=(10, 5))
    plt.plot(ks, vals, marker='o')
    plt.title(f"KNN Performance vs K ({metric})")
    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()


##############################
# Combined visualization
##############################

def visualize_knn_results(y_test, y_pred, n_samples=200):
    """Plot predictions + residuals."""
    plot_predictions(y_test, y_pred, n_samples)
    plot_residuals(y_test, y_pred)