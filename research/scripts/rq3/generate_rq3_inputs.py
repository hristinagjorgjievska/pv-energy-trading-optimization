import pandas as pd

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
HORIZONS = range(1, 25)

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    val_end = int(len(df) * 0.85)
    test = df.iloc[val_end:].copy()

    price_pred = pd.read_csv(f"../../results/rq1-results/xgboost_model/{country}_predictions.csv")
    pv_pred = pd.read_csv(f"../../results/rq2-results/xgboost_model/{country}_predictions.csv")

    assert len(price_pred) == len(test), f"{country}: rq1 predictions ({len(price_pred)}) не одговараат на test ({len(test)}) - прво пушти rq1/xgboost_model.py"
    assert len(pv_pred) == len(test), f"{country}: rq2 predictions ({len(pv_pred)}) не одговараат на test ({len(test)}) - прво пушти rq2/xgboost_model_horizon.py"

    test = test.reset_index(drop=True)
    for h in HORIZONS:
        test[f"predicted_price_h{h}"] = price_pred[f"predicted_price_h{h}"]
        test[f"predicted_P_h{h}"] = pv_pred[f"predicted_P_h{h}"]

    test = test.rename(columns={'Price (EUR/MWhe)': 'actual_price', 'solar_generation_MW': 'actual_P'})

    out_cols = (
        ['actual_price', 'actual_P', 'hour', 'day_of_week', 'month', 'is_weekend', 'cloud_cover']
        + [f'predicted_price_h{h}' for h in HORIZONS]
        + [f'predicted_P_h{h}' for h in HORIZONS]
    )
    out = test[out_cols]
    out.to_csv(f"inputs/rq3_inputs_{country}.csv", index=False)
    print(f"{country}: {len(out)} редови зачувани (од RQ1/RQ2 предвидувања, без ново тренирање)\n")
