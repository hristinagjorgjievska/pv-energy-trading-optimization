import pandas as pd

COUNTRIES = ["processed", "ES", "GR", "NL", "PL", "SE"]

for country in COUNTRIES:
    path = f"data/raw/{country}/solar-raw.csv"
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)

    nan_before = df.iloc[:, 0].isna().sum()
    df = df.interpolate(method="time")
    nan_after = df.iloc[:, 0].isna().sum()

    df.to_csv(path)
    print(f"{country}, NaN пред = {nan_before}, после = {nan_after}")


price_de = pd.read_csv("../data/raw/entso-e-prices/Germany.csv")
solar_de = pd.read_csv("../data/raw/DE/solar-raw.csv")
meteo_de = pd.read_csv("../data/raw/meteo-raw-germany.csv")

print("=== ЦЕНИ ===")
print(price_de.head(2))
print("\n=== SOLAR ===")
print(solar_de.head(2))
print("\n=== ВРЕМЕ ===")
print(meteo_de.head(2))