import pandas as pd

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

for country in COUNTRIES:
    df = pd.read_csv(f"rq3_inputs_{country}.csv")

    avg_production = df['actual_P'].mean()
    CAPACITY = avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * 0.4
    EFFICIENCY = 0.95

    low_price = df['predicted_price'].quantile(0.25)
    high_price = df['predicted_price'].quantile(0.75)

    soc = MIN_SOC
    revenue = 0

    for i, row in df.iterrows():
        production = row['actual_P']
        real_price = row['actual_price']

        charge = 0
        discharge = 0
        sell = production

        if row['predicted_price'] > high_price:
            discharge = min(MAX_RATE, soc - MIN_SOC)
        elif row['predicted_price'] < low_price:
            charge = min(production, MAX_RATE, MAX_SOC - soc)
            sell = production - charge

        revenue += real_price * (sell + discharge * EFFICIENCY) / 1000
        soc = soc + charge * EFFICIENCY - discharge

    print(f"{country}: capacity={CAPACITY:.0f}, baseline revenue = {revenue:.2f} EUR")