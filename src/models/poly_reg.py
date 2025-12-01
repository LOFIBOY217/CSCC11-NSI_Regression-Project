import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import itertools
import matplotlib.pyplot as plt

from src.models.utils import (
    split_data,
    evaluate_model,
    plot_predictions,
    plot_residuals
)


############################################
# Polynomial Regression - Core Components
############################################

def build_model(degree=2, interaction_only=False):
    """
    Initialize polynomial transformer + linear regression model.
    degree: int, complexity of polynomial
    interaction_only: if True only generate cross terms, no powers
    """
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=interaction_only
    )
    model = LinearRegression()
    return poly, model


def train_model(poly, model, X_train, y_train):
    """Fit polynomial transformer + regression model."""
    X_poly = poly.fit_transform(X_train)
    model.fit(X_poly, y_train)
    return poly, model


def predict(poly, model, X_test):
    """Predict NSI values from test set."""
    X_poly = poly.transform(X_test)
    return model.predict(X_poly)


def train_poly_model(df, feature_cols, target_col, scale=True, degree=2, interaction_only=False):
    """
    Full polynomial regression pipeline:
    split → build → train → predict → evaluate
    """
    df = df.dropna(subset=['Prev_Month_NSI']).copy()

    X_train, X_test, y_train, y_test, scaler = split_data(
        df, feature_cols, target_col, scale
    )

    poly, model = build_model(degree, interaction_only)
    poly, model = train_model(poly, model, X_train, y_train)
    y_pred = predict(poly, model, X_test)

    metrics = evaluate_model(y_test, y_pred)

    return model, poly, y_test, y_pred, metrics


############################################
# Model Diagnostics
############################################

def visualize_poly_results(y_test, y_pred, n_samples=200):
    """Plot predictions and residuals for polynomial regression."""
    plot_predictions(y_test, y_pred, n_samples)
    plot_residuals(y_test, y_pred)


############################################
# Hyperparameter Grid Search
############################################

def explore_poly_params(df, feature_cols, target_col, degrees, interaction_flags, scale=True):
    """
    Grid search over:
        degree × interaction_only

    Returns:
        list of dict metrics for each combination.
    """

    results = []

    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    for degree, inter in itertools.product(degrees, interaction_flags):
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False,
            interaction_only=inter
        )

        X_poly_train = poly.fit_transform(X_train)
        X_poly_test  = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        y_pred = model.predict(X_poly_test)
        metrics = evaluate_model(y_test, y_pred)

        results.append({
            "degree": degree,
            "interaction_only": inter,
            "R2":  metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"]
        })

    return results


def plot_poly_grid(results, metric="R2"):
    """
    Visualize hyperparameter search matrix for given metric.
    metric ∈ {'R2','RMSE','MAE'}
    """

    degs = sorted(set(r["degree"] for r in results))
    inters = [False, True]

    matrix = [[
        next(r[metric] for r in results if r["degree"] == d and r["interaction_only"] == i)
        for i in inters
    ] for d in degs]

    plt.figure(figsize=(8, len(degs) * 0.6 + 3))
    plt.imshow(matrix, cmap="viridis", aspect="auto")
    plt.colorbar(label=metric)

    plt.xticks([0, 1], ["interaction=False", "interaction=True"])
    plt.yticks(range(len(degs)), [f"degree={d}" for d in degs])

    plt.title(f"Polynomial Regression Grid Search ({metric})")
    plt.show()