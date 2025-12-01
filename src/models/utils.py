import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

def split_data(df, feature_cols, target_col, scale=False):
    """
    Time-based split for NSI prediction.
    Train: REPORT_YEAR <= 2023
    Test : REPORT_YEAR >= 2024
    """
    train = df[df['REPORT_YEAR'] <= 2023].copy()
    test  = df[df['REPORT_YEAR'] >= 2024].copy()

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test  = test[feature_cols]
    y_test  = test[target_col]

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def evaluate_model(y_test, y_pred):
    """Return regression metrics dictionary."""
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    }


def plot_residuals(y_test, y_pred):
    """Residual diagnostic plot."""
    residuals = y_test - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted NSI")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def plot_predictions(y_test, y_pred, n_samples=200):
    """Plot actual vs predicted NSI values."""
    y_test = np.array(y_test).reshape(-1)[:n_samples]
    y_pred = np.array(y_pred).reshape(-1)[:n_samples]

    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="Actual NSI", linewidth=2)
    plt.plot(y_pred, label="Predicted NSI", linestyle='--')
    plt.legend()
    plt.title(f"NSI Prediction (First {n_samples} samples)")
    plt.xlabel("Time index")
    plt.ylabel("NSI")
    plt.show()