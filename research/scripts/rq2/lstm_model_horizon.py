import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
]
seq_len = 24
EPOCHS = 15
BATCH_SIZE = 256


class LSTM(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def to_sequences(X, y, start, end, horizon):
    xs, ys = [], []
    for i in range(start, end - seq_len - horizon + 1):
        xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len + horizon - 1])
    return torch.tensor(np.array(xs), dtype=torch.float32), torch.tensor(np.array(ys), dtype=torch.float32)


for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    target = "solar_generation_MW"

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    scaler_x = StandardScaler().fit(df[features].iloc[:train_end])
    scaler_y = StandardScaler().fit(df[[target]].iloc[:train_end])

    X_all = scaler_x.transform(df[features])
    y_all = scaler_y.transform(df[[target]]).ravel()

    results = []
    for h in range(1, 25):
        X_train, y_train = to_sequences(X_all, y_all, 0, train_end, h)
        X_test, y_test = to_sequences(X_all, y_all, val_end - seq_len, len(df), h)

        model = LSTM(len(features))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        n = X_train.shape[0]
        for epoch in range(EPOCHS):
            perm = torch.randperm(n)
            for i in range(0, n, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                xb, yb = X_train[idx], y_train[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test).numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_true = scaler_y.inverse_transform(y_test.numpy().reshape(-1, 1)).ravel()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        results.append({"horizon": h, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"{country} h={h}: R2={r2*100:.1f}%")

    pd.DataFrame(results).to_csv(f"../../results/rq2-results/lstm_model/{country}_results.csv", index=False)
    print(f"{country}: ГОТОВО - h=1 R2={results[0]['R2']*100:.1f}%, h=24 R2={results[23]['R2']*100:.1f}%\n")
