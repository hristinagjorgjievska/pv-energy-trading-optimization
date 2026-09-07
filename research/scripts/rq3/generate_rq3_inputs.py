import pandas as pd
from xgboost import XGBRegressor

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

price_features = [
    'price_lag_1h', 'price_lag_24h', 'price_lag_168h',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h'
]
pv_features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
]

for country in COUNTRIES:
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")

    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    price_model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                                reg_alpha=1, reg_lambda=1, random_state=42)
    price_model.fit(df[price_features].iloc[:train_end], df['Price (EUR/MWhe)'].iloc[:train_end])

    pv_model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                             min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=1, reg_lambda=1, random_state=42)
    pv_model.fit(df[pv_features].iloc[:train_end], df['solar_generation_MW'].iloc[:train_end])

    test = df.iloc[val_end:].copy()
    test['predicted_price'] = price_model.predict(test[price_features])
    test['predicted_P'] = pv_model.predict(test[pv_features])
    test = test.rename(columns={'Price (EUR/MWhe)': 'actual_price', 'solar_generation_MW': 'actual_P'})

    out = test[['actual_price', 'predicted_price', 'actual_P', 'predicted_P']]
    out.to_csv(f"../..scripts/rq3/rq3_inputs_{country}.csv", index=False)
    print(f"{country}: {len(out)} редови зачувани")