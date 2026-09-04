import os
import time
import pandas as pd
from entso_data_loader import fetch_solar_generation, COUNTRY_ZONES

START = "2022-01-01"
END = "2026-01-01"

results = {}

for country in COUNTRY_ZONES:
    try:
        solar = fetch_solar_generation(country, START, END)

        os.makedirs(f"data/raw/{country}", exist_ok=True)
        solar.to_csv(f"data/raw/{country}/solar-raw.csv", header=["solar_generation_MW"])

        results[country] = {
            "rows": len(solar),
            "NaN": int(solar.isna().sum()),
            "start": solar.index.min(),
            "end": solar.index.max(),
        }
        print(f" {country} saved: {len(solar)} rows")

    except Exception as e:
        print("error")

    time.sleep(3)

summary = pd.DataFrame(results).T
print(summary)
summary.to_csv("data/raw/solar_fetch_summary.csv")