import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
all_results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    features = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
        'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
    ]
    target = 'solar_generation_MW'
    seq_len = 24

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    scaler_x = StandardScaler().fit(df[features].iloc[:train_end])
    scaler_y = StandardScaler().fit(df[[target]].iloc[:train_end])

    X_all = scaler_x.transform(df[features])
    y_all = scaler_y.transform(df[[target]]).ravel()

    def to_sequences(X, y, start, end):
        xs, ys = [], []
        for i in range(start, end - seq_len):
            xs.append(X[i:i+seq_len])
            ys.append(y[i+seq_len])
        return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)

    X_train, y_train = to_sequences(X_all, y_all, 0, train_end)
    X_test, y_test = to_sequences(X_all, y_all, val_end - seq_len, len(df))

    class LSTM(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = LSTM(len(features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    batch_size = 256
    n = X_train.shape[0]

    for epoch in range(20):
        perm = torch.randperm(n)
        epoch_loss = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        print(f"Epoch {epoch+1}, loss: {epoch_loss/n:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
        y_train_pred_scaled = model(X_train).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).ravel()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
    y_train_true = scaler_y.inverse_transform(y_train.numpy().reshape(-1, 1)).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_r2 = r2_score(y_train_true, y_train_pred)

    print(f"{country} TRAIN: MAE={train_mae:.2f}, R2={train_r2*100:.1f}%")
    print(f"{country} TEST:  MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2*100:.1f}%")
    all_results[country] = {"MAE": mae, "RMSE": rmse, "R2": r2 * 100}

results_df = pd.DataFrame(all_results).T
print(results_df)
results_df.to_csv("../../results/lstm_pv_model.csv")