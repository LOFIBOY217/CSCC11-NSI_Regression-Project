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
def poly_hparam_search(df, feature_cols, target_col, degrees=range(1, 6), scale=True):
    """
    Search polynomial configurations over:
        degree ∈ [1..7], interaction_only ∈ {False, True}

    Returns:
        results: list of dict
        best_config: dict
    """
    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    results = []

    for degree, inter in itertools.product(degrees, [False, True]):
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False,
            interaction_only=inter
        )
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        metrics = evaluate_model(y_test, y_pred)

        results.append({
            "degree": degree,
            "interaction": inter,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"]
        })

    # Best = maximize R2 and minimize RMSE
    best = max(results, key=lambda r: (r["R2"], -r["RMSE"]))
    return results, best


############################################
# RMSE Plot (Fixed Version)
############################################
def plot_poly_rmse_lines(results):
    """Plot RMSE vs degree for both interaction settings."""

    # separate two result groups correctly
    no_inter = sorted([r for r in results if not r["interaction"]], key=lambda r: r["degree"])
    inter    = sorted([r for r in results if r["interaction"]], key=lambda r: r["degree"])

    degrees = [r["degree"] for r in no_inter]
    rmse_no_inter = [r["RMSE"] for r in no_inter]
    rmse_inter    = [r["RMSE"] for r in inter]

    # Plot 1: No interaction
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, rmse_no_inter, marker='o', linewidth=2)
    plt.title("Polynomial Regression RMSE vs Degree (interaction=False)")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()

    # Plot 2: Interaction
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, rmse_inter, marker='o', linewidth=2, color='darkred')
    plt.title("Polynomial Regression RMSE vs Degree (interaction=True)")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.show()


############################################
# R² Plot (Fixed Version)
############################################
def plot_poly_r2_lines(results):
    """Plot R² vs degree for both interaction settings."""

    no_inter = sorted([r for r in results if not r["interaction"]], key=lambda r: r["degree"])
    inter    = sorted([r for r in results if r["interaction"]], key=lambda r: r["degree"])

    degrees = [r["degree"] for r in no_inter]
    r2_no_inter = [r["R2"] for r in no_inter]
    r2_inter    = [r["R2"] for r in inter]

    # Plot 1: No interaction
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, r2_no_inter, marker='o', linewidth=2)
    plt.title("Polynomial Regression R² vs Degree (interaction=False)")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.show()

    # Plot 2: Interaction
    plt.figure(figsize=(8, 4))
    plt.plot(degrees, r2_inter, marker='o', linewidth=2, color='darkred')
    plt.title("Polynomial Regression R² vs Degree (interaction=True)")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("R² Score")
    plt.grid(True)
    plt.show()