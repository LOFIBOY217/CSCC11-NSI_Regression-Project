import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import itertools
import matplotlib.pyplot as plt

from src.models.utils import (
    split_data,
    evaluate_model,
    plot_predictions,
    plot_residuals
)

############################################
# Regularized Polynomial Regression - Core
############################################

def build_model(degree=2, interaction_only=False, reg_type="ridge", alpha=1.0):
    """
    Initialize polynomial transformer + regularized linear model.

    Parameters:
        degree: polynomial degree
        interaction_only: generate only interaction terms
        reg_type: "ridge", "lasso", or "elasticnet"
        alpha: regularization strength (λ)

    Returns:
        (poly_transformer, model)
    """

    poly = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=interaction_only
    )

    if reg_type == "ridge":
        model = Ridge(alpha=alpha)
    elif reg_type == "lasso":
        model = Lasso(alpha=alpha, max_iter=2000)
    elif reg_type == "elasticnet":
        model = ElasticNet(alpha=alpha, max_iter=2000, l1_ratio=0.5)
    else:
        raise ValueError("reg_type must be one of: ridge, lasso, elasticnet")

    return poly, model


def train_model(poly, model, X_train, y_train):
    """Fit polynomial transformer + regularized regression model."""
    X_poly = poly.fit_transform(X_train)
    model.fit(X_poly, y_train)
    return poly, model


def predict(poly, model, X_test):
    """Generate predictions from test set."""
    return model.predict(poly.transform(X_test))


def train_poly_model(df, feature_cols, target_col,
                     scale=True, degree=2, interaction_only=False,
                     reg_type="ridge", alpha=1.0):
    """
    Full training pipeline with Regularization.

    reg_type ∈ {"ridge", "lasso", "elasticnet"}
    """

    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    poly, model = build_model(degree, interaction_only, reg_type, alpha)
    poly, model = train_model(poly, model, X_train, y_train)

    y_pred = predict(poly, model, X_test)
    metrics = evaluate_model(y_test, y_pred)

    return model, poly, y_test, y_pred, metrics


############################################
# Hyperparameter Search
############################################

def poly_reg_hparam_search(df, feature_cols, target_col,
                           degrees=range(1, 6),
                           alphas=[0.01, 0.1, 1.0, 5.0],
                           reg_types=["ridge", "lasso", "elasticnet"],
                           scale=True):
    """
    Grid search over:
        degree × alpha × reg_type

    Returns:
        results: list of dict
        best_config: dict
    """

    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    results = []

    for degree, reg, alpha in itertools.product(degrees, reg_types, alphas):
        poly, model = build_model(degree, False, reg, alpha)
        X_tr = poly.fit_transform(X_train)
        X_te = poly.transform(X_test)

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        metrics = evaluate_model(y_test, y_pred)
        results.append({
            "degree": degree,
            "alpha": alpha,
            "reg_type": reg,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"]
        })

    best = max(results, key=lambda r: (r["R2"], -r["RMSE"]))
    return results, best


############################################
# Visualization Utilities
############################################

def plot_alpha_effect(results, reg_type="ridge"):
    """Plot RMSE vs alpha for a given regularization type."""

    filtered = [r for r in results if r["reg_type"] == reg_type]
    filtered = sorted(filtered, key=lambda r: (r["degree"], r["alpha"]))

    degrees = sorted(set(r["degree"] for r in filtered))

    for d in degrees:
        subset = [r for r in filtered if r["degree"] == d]
        xs = [r["alpha"] for r in subset]
        ys = [r["RMSE"] for r in subset]

        plt.plot(xs, ys, marker='o', label=f"degree={d}")

    plt.title(f"{reg_type.title()} Regression: RMSE vs alpha")
    plt.xlabel("alpha (λ)")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.show()