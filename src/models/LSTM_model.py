import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.models.utils import (
    evaluate_model,
    plot_predictions,
    plot_residuals
)

############################################
# LSTM Core Model
############################################

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last timestep
        return out


############################################
# Training
############################################

def train_model(model, train_loader, epochs=50, lr=0.001, device='cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{epoch+1}/{epochs}] Loss = {total_loss / len(train_loader):.4f}")

    return model


############################################
# Prediction (NO inverse transform here)
############################################

@torch.no_grad()
def predict(model, test_loader, device='cpu'):
    model.eval()
    preds, actuals = [], []

    for X, y in test_loader:
        X = X.to(device)
        out = model(X).cpu().numpy()
        preds.append(out)
        actuals.append(y.numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    return preds, actuals


############################################
# Full Pipeline
############################################

def train_lstm_pipeline(
    df,
    feature_cols,
    target_col,
    seq_length=12,
    hidden_size=256,
    num_layers=2,
    batch_size=32,
    epochs=50,
    device='cpu',
    create_sequences_fn=None,
):
    """
    Full LSTM pipeline:
        1) convert df → sequences
        2) split 80/20
        3) train model
        4) predict
        5) inverse-scale both preds and actuals TOGETHER
        6) evaluate
    """
    if create_sequences_fn is None:
        raise ValueError("Missing create_sequences_fn function")

    # create_sequences_fn must return (X, y, target_scaler)
    X, y, target_scaler = create_sequences_fn(df, feature_cols, target_col, seq_length)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    test_loader = DataLoader(TensorDataset(X_test, y_test),
                             batch_size=batch_size,
                             shuffle=False)

    model = LSTMModel(input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    model = train_model(model, train_loader, epochs=epochs, device=device)

    preds_scaled, actuals_scaled = predict(model, test_loader, device)

    # 🚨 Correct scaling: both inverse-transformed TOGETHER
    preds = target_scaler.inverse_transform(preds_scaled)
    actuals = target_scaler.inverse_transform(actuals_scaled)

    metrics = evaluate_model(actuals, preds)
    return model, actuals, preds, metrics, target_scaler


############################################
# Sequence Builder
############################################

def create_sequences_per_neighborhood(df, feature_columns, target_column, seq_length):
    X_list, y_list = [], []

    all_features = df[feature_columns].values
    all_targets = df[target_column].values

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    feature_scaler.fit(all_features)
    target_scaler.fit(all_targets.reshape(-1, 1))

    for neighborhood in df['NEIGHBOURHOOD_158'].unique():
        nd = df[df['NEIGHBOURHOOD_158'] == neighborhood].copy()
        nd = nd.sort_values(['REPORT_YEAR', 'REPORT_MONTH'])

        features = feature_scaler.transform(nd[feature_columns].values)
        target = target_scaler.transform(nd[target_column].values.reshape(-1, 1))

        for i in range(len(features) - seq_length):
            X_list.append(features[i:i + seq_length])
            y_list.append(target[i + seq_length])

    return np.array(X_list), np.array(y_list), target_scaler


############################################
# Visualization Helpers
############################################

def visualize_lstm_results(y_test, y_pred, n_samples=200):
    plot_predictions(y_test, y_pred, n_samples)
    plot_residuals(y_test, y_pred)