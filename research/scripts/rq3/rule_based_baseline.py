import pandas as pd

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
EFFICIENCY = 0.95

results = {}

for country in COUNTRIES:
    df = pd.read_csv(f"inputs/rq3_inputs_{country}.csv")

    fe = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    train_end = int(len(fe) * 0.7)
    train = fe.iloc[:train_end]

    train_avg_production = train['solar_generation_MW'].mean()
    CAPACITY = train_avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * 0.4

    low_price = train['Price (EUR/MWhe)'].quantile(0.25)
    high_price = train['Price (EUR/MWhe)'].quantile(0.75)

    soc = MIN_SOC
    revenue = 0

    for i, row in df.iterrows():
        production = row['actual_P']
        real_price = row['actual_price']

        charge = 0
        discharge = 0
        sell = production

        if real_price > high_price:
            discharge = min(MAX_RATE, soc - MIN_SOC)
        elif real_price < low_price:
            charge = min(production, MAX_RATE, MAX_SOC - soc)
            sell = production - charge

        revenue += real_price * (sell + discharge * EFFICIENCY) / 1000
        soc = soc + charge * EFFICIENCY - discharge

    print(f"{country}: capacity={CAPACITY:.0f}, baseline revenue = {revenue:.2f} EUR")
    results[country] = {"capacity": CAPACITY, "revenue": revenue}

pd.DataFrame(results).T.to_csv("../../results/rq3-results/rule_based_results.csv")
