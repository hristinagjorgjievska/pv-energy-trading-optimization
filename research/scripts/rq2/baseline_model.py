import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

all_results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    target = "solar_generation_MW"
    pred_col = "solar_lag_24h"

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    test = df.iloc[val_end:]

    y_true = test[target]
    y_pred = test[pred_col]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"{country}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2 * 100:.1f}%")
    all_results[country] = {"MAE": mae, "RMSE": rmse, "R2": r2 * 100}

results_df = pd.DataFrame(all_results).T
print(results_df)
results_df.to_csv("../../results/rq2-results/baseline_pv_model.csv")