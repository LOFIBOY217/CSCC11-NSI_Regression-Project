import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from src.models.utils import (
    split_data,
    evaluate_model,
    plot_predictions,
    plot_residuals
)


############################################
# Neural Network (Fully Connected) Core
############################################

def build_model(hidden_layers=(64, 32), activation='relu'):
    """
    Initialize a fully-connected feedforward neural network (MLP).
    hidden_layers: tuple specifying neurons per hidden layer
    activation: activation function ('relu', 'logistic', 'sigmoid-like')
                • 'relu'     → piecewise linear, modern default
                • 'logistic' → sigmoid, bounded, smooth
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation=activation,  # 'relu' or 'logistic'
        solver='adam',
        max_iter=500,
        random_state=42
    )


def train_model(model, X_train, y_train):
    """Fit neural network model."""
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Predict NSI values."""
    return model.predict(X_test)


def train_nn_model(df, feature_cols, target_col,
                   scale=True,
                   hidden_layers=(64, 32),
                   activation='relu'):
    """
    Full NN pipeline:
    split → build → train → predict → evaluate
    """
    df = df.dropna(subset=['Prev_Month_NSI']).copy()

    X_train, X_test, y_train, y_test, scaler = split_data(
        df, feature_cols, target_col, scale
    )

    model = build_model(hidden_layers, activation)
    model = train_model(model, X_train, y_train)
    y_pred = predict(model, X_test)

    metrics = evaluate_model(y_test, y_pred)
    return model, y_test, y_pred, metrics


############################################
# Hyperparameter Search
############################################

def nn_hparam_search(df, feature_cols, target_col,
                     layer_configs, activations, scale=True):
    """
    Grid-search over NN architectures:
      • hidden_layer_sizes ∈ layer_configs
      • activation ∈ {'relu', 'logistic'}   <-- sigmoid used here

    Returns:
      results: list of dict
      best:    config with highest R2 & lowest RMSE
    """

    df = df.dropna(subset=['Prev_Month_NSI']).copy()
    X_train, X_test, y_train, y_test, scaler = split_data(df, feature_cols, target_col, scale)

    results = []

    for layers in layer_configs:
        for act in activations:  # ['relu', 'logistic']
            model = build_model(layers, act)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_model(y_test, y_pred)

            results.append({
                "layers": layers,
                "activation": act,
                "R2": metrics["R2"],
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"]
            })

    best = max(results, key=lambda r: (r["R2"], -r["RMSE"]))
    return results, best


############################################
# Visualization
############################################

def plot_nn_r2(results):
    """
    Plot R2 scores across architectures.
    Each hidden-layer config gets one point.
    """
    labels = [str(r["layers"]) + "_" + r["activation"] for r in results]
    r2_vals = [r["R2"] for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(range(len(results)), r2_vals, marker='o')
    plt.xticks(range(len(results)), labels, rotation=45)
    plt.title("NN R2 Across Architectures")
    plt.xlabel("NN Configuration")
    plt.ylabel("R2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_nn_results(y_test, y_pred, n_samples=200):
    """Plot predictions + residuals."""
    plot_predictions(y_test, y_pred, n_samples)
    plot_residuals(y_test, y_pred)