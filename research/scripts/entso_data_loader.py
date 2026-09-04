import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

load_dotenv()
client = EntsoePandasClient(api_key=os.environ["ENTSOE_TOKEN"])

COUNTRY_ZONES = {
    "processed": "DE_LU",
    "ES": "ES",
    "GR": "GR",
    "NL": "NL",
    "PL": "PL",
    "SE": "SE_3",
}

# def fetch_solar_generation(country: str, start: str, end: str) -> pd.Series:
#     zone = COUNTRY_ZONES[country]
#     ts_start = pd.Timestamp(start, tz="UTC")
#     ts_end = pd.Timestamp(end, tz="UTC")
#
#     generation = client.query_generation(
#         zone, start=ts_start, end=ts_end, psr_type="B16"
#     )
#
#     if isinstance(generation.columns, pd.MultiIndex):
#         solar = generation[("Solar", "Actual Aggregated")]
#     else:
#         solar = generation.iloc[:, 0]
#
#     solar_hourly = solar.resample("1h").mean()
#     return solar_hourly

import time

def fetch_solar_generation(country: str, start: str, end: str, max_retries=3) -> pd.Series:
    zone = COUNTRY_ZONES[country]
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    chunks = []
    current = start_ts
    while current < end_ts:
        chunk_end = min(current + pd.DateOffset(months=1), end_ts)

        for attempt in range(max_retries):
            try:
                gen = client.query_generation(zone, start=current, end=chunk_end, psr_type="B16")
                if isinstance(gen.columns, pd.MultiIndex):
                    solar = gen[("Solar", "Actual Aggregated")]
                else:
                    solar = gen.iloc[:, 0]
                chunks.append(solar)
                print(f"  {current.date()} до {chunk_end.date()}: OK ({len(solar)} редови)")
                break
            except Exception as e:
                print(f"  {current.date()} до {chunk_end.date()}: обид {attempt+1} неуспешен ({type(e).__name__})")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print(f"  ⚠️ конечно неуспешно за {current.date()} — {chunk_end.date()}")

        time.sleep(1)
        current = chunk_end

    full = pd.concat(chunks)
    return full.resample("1h").mean()

def fetch_day_ahead_prices(country: str, start: str, end: str) -> pd.Series:
    zone = COUNTRY_ZONES[country]
    ts_start = pd.Timestamp(start, tz="UTC")
    ts_end = pd.Timestamp(end, tz="UTC")

    prices = client.query_day_ahead_prices(
        zone, start=ts_start, end=ts_end
    )

    return prices
