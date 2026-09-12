import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
all_results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    val_end = int(len(df) * 0.85)
    test = df.iloc[val_end:]

    results = []
    for h in range(1, 25):
        y_true = test[f"target_pv_h{h}"]
        y_pred = test["solar_generation_MW"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)

        results.append({"horizon": h, "MAE": mae, "RMSE": rmse, "R2": r2})

    all_results[country] = pd.DataFrame(results)
    all_results[country].to_csv(f"../../results/rq2-results/baseline_model/{country}_results.csv")
    print(f"{country}: h=1 R2={results[0]['R2']*100:.1f}%, h=24 R2={results[23]['R2']*100:.1f}%")
