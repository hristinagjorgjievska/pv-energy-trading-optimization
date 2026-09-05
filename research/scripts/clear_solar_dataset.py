
COUNTRIES = ["processed", "spain", "greece", "netherlands", "poland", "sweden", "germany"]

# for country in COUNTRIES:
#     path = f"data/raw/{country}/solar-raw.csv"
#     df = pd.read_csv(path, index_col=0)
#     df.index = pd.to_datetime(df.index, utc=True)
#
#     nan_before = df.iloc[:, 0].isna().sum()
#     df = df.interpolate(method="time")
#     nan_after = df.iloc[:, 0].isna().sum()
#
#     df.to_csv(path)
#     print(f"{country}, NaN пред = {nan_before}, после = {nan_after}")
#
#
# price_de = pd.read_csv("../data/raw/DE/Germany.csv")
# solar_de = pd.read_csv("../data/raw/DE/solar-raw.csv")
# meteo_de = pd.read_csv("../data/raw/DE/meteo-raw-germany.csv")
#
# print("=== ЦЕНИ ===")
# print(price_de.head(2))
# print("\n=== SOLAR ===")
# print(solar_de.head(2))
# print("\n=== ВРЕМЕ ===")
# print(meteo_de.head(2))

import pandas as pd
#
# for country in ["germany", "spain", "greece", "netherlands", "poland", "sweden"]:
#     df = pd.read_csv(f"../data/processed/{country.lower()}_merged.csv")
#     print(f"{country}: {df.shape}, NaN: {df.isna().sum().sum()}, колони: {list(df.columns)}")

for country in ["germany", "spain", "greece", "netherlands", "poland", "sweden"]:

    df = pd.read_csv(f"../data/processed/{country.lower()}_merged.csv")
    print(f"{country}: {df.solar_generation_MW.mean()} is mean")
    print(f"{country}: {df.solar_generation_MW.max()} is max")


    nl_year = df[(df["datetime"] >= "2023-01-01") & (df["datetime"] < "2024-01-01")]
    total_twh = nl_year["solar_generation_MW"].sum() / 1_000_000
    print("total_twh", total_twh)
