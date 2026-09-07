import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
all_results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    target = "solar_generation_MW"
    features = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
        'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
    ]

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    X_train = df[features].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_test = df[features].iloc[val_end:]
    y_test = df[target].iloc[val_end:]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=1,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{country}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2*100:.1f}%")
    all_results[country] = {"MAE": mae, "RMSE": rmse, "R2": r2 * 100}

    y_train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"{country} TRAIN: MAE={train_mae:.2f}, R2={train_r2*100:.1f}%")

results_df = pd.DataFrame(all_results).T
print(results_df)
results_df.to_csv("../../results/rq2-results/xgboost_pv_model.csv")