import pandas as pd

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

for country in COUNTRIES:
    df = pd.read_csv(f"../data/processed/{country}_merged.csv")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["price_lag_1h"] = df["Price (EUR/MWhe)"].shift(1)
    df["price_lag_24h"] = df["Price (EUR/MWhe)"].shift(24)
    df["price_lag_168h"] = df["Price (EUR/MWhe)"].shift(168)

    df["solar_lag_1h"] = df["solar_generation_MW"].shift(1)
    df["solar_lag_24h"] = df["solar_generation_MW"].shift(24)
    df["solar_lag_168h"] = df["solar_generation_MW"].shift(168)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    for h in range(1, 25):
        df[f"target_h{h}"] = df["Price (EUR/MWhe)"].shift(-h)

    df = df.dropna().reset_index(drop=True)

    print(df.shape)

    df.to_csv(f"../data/processed/fe/{country}_features.csv", index=False)

    print(df.columns)
    print(df.head(5))

