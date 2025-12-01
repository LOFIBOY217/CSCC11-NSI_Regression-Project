import numpy as np

from sklearn.linear_model import LinearRegression

# import shared utilities
from src.models.utils import (
    split_data,
    evaluate_model
)


def build_model():
    """Initialize linear regression model."""
    return LinearRegression()


def train_model(model, X_train, y_train):
    """Fit model on training data."""
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """Predict test set."""
    return model.predict(X_test)


def train_linear_model(df, feature_cols, target_col, scale=False):
    """
    Full linear model pipeline:
    split → build → train → predict → evaluate
    """

    # ensure lag is valid
    df = df.dropna(subset=['Prev_Month_NSI']).copy()

    X_train, X_test, y_train, y_test, scaler = split_data(
        df, feature_cols, target_col, scale
    )

    model = build_model()
    model = train_model(model, X_train, y_train)
    y_pred = predict(model, X_test)

    metrics = evaluate_model(y_test, y_pred)

    return model, y_test, y_pred, metrics