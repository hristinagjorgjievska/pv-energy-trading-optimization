import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

pv_features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
]

MODEL_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1, reg_lambda=1, random_state=42
)

all_results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    X_train = df[pv_features].iloc[:train_end]
    X_test = df[pv_features].iloc[val_end:]

    results = []
    predictions = pd.DataFrame(index=df.index[val_end:])
    for h in range(1, 25):
        y_train = df[f"target_pv_h{h}"].iloc[:train_end]
        y_test = df[f"target_pv_h{h}"].iloc[val_end:]

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[f"predicted_P_h{h}"] = y_pred

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        results.append({"horizon": h, "MAE": mae, "RMSE": rmse, "R2": r2})

    all_results[country] = pd.DataFrame(results)
    all_results[country].to_csv(f"../../results/rq2-results/xgboost_model/{country}_results.csv")
    predictions.to_csv(f"../../results/rq2-results/xgboost_model/{country}_predictions.csv", index=False)
    print(f"{country}: h=1 R2={results[0]['R2']*100:.1f}%, h=24 R2={results[23]['R2']*100:.1f}%")
