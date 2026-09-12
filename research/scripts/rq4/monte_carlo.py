import pandas as pd
import numpy as np
import pulp
import time

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
N_ITERATIONS = 10
UNCERTAINTY = 0.20
EFFICIENCY = 0.95
HORIZON = 24


def solve_window(prices, productions, soc, max_soc, min_soc, max_rate):
    H = len(prices)
    prob = pulp.LpProblem("mpc", pulp.LpMaximize)
    sell = [pulp.LpVariable(f"sell_{t}", lowBound=0) for t in range(H)]
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_rate) for t in range(H)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_rate) for t in range(H)]
    battery = [pulp.LpVariable(f"soc_{t}", lowBound=min_soc, upBound=max_soc) for t in range(H)]
    prob += pulp.lpSum(prices[t] * (sell[t] + discharge[t] * EFFICIENCY) for t in range(H))
    for t in range(H):
        prob += sell[t] + charge[t] == productions[t]
        prev_soc = soc if t == 0 else battery[t - 1]
        prob += battery[t] == prev_soc + charge[t] * EFFICIENCY - discharge[t]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sell[0].varValue, charge[0].varValue, discharge[0].varValue


def run_mpc(actual_prices, actual_P, price_h, pv_h, max_soc, min_soc, max_rate):
    n = len(actual_prices)
    soc = min_soc
    revenue = 0
    for t in range(n):
        remaining = min(HORIZON, n - t)
        future_prices = price_h[t, :remaining - 1]
        future_prod = pv_h[t, :remaining - 1]
        window_prices = [actual_prices[t]] + list(future_prices)
        window_prod = [actual_P[t]] + list(future_prod)

        sell, charge, discharge = solve_window(window_prices, window_prod, soc, max_soc, min_soc, max_rate)

        real_price = actual_prices[t]
        real_production = actual_P[t]
        actual_sell = min(sell, real_production)
        actual_charge = min(charge, real_production - actual_sell, max_soc - soc)
        actual_discharge = min(discharge, soc - min_soc)
        surplus = real_production - actual_sell - actual_charge
        actual_sell += surplus
        revenue += real_price * (actual_sell + actual_discharge * EFFICIENCY) / 1000
        soc = soc + actual_charge * EFFICIENCY - actual_discharge
    return revenue


def perturb(values, std, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, std, values.shape)
    return values * noise


for country in COUNTRIES:
    df = pd.read_csv(f"../rq3/inputs/rq3_inputs_{country}.csv")

    fe = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    train_end = int(len(fe) * 0.7)
    train_avg_production = fe['solar_generation_MW'].iloc[:train_end].mean()
    CAPACITY = train_avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * 0.4

    actual_prices = df['actual_price'].values
    actual_P = df['actual_P'].values
    price_h = df[[f'predicted_price_h{h}' for h in range(1, HORIZON)]].values
    pv_h = df[[f'predicted_P_h{h}' for h in range(1, HORIZON)]].values

    print(f"--- {country} ---")

    price_revenues = []
    for i in range(N_ITERATIONS):
        start = time.time()
        perturbed_price_h = perturb(price_h, UNCERTAINTY, seed=i)
        rev = run_mpc(actual_prices, actual_P, perturbed_price_h, pv_h, MAX_SOC, MIN_SOC, MAX_RATE)
        price_revenues.append(rev)
        print(f"price iter {i+1}/{N_ITERATIONS}: {rev:.2f} EUR ({time.time()-start:.1f}s)")

    production_revenues = []
    for i in range(N_ITERATIONS):
        start = time.time()
        perturbed_pv_h = perturb(pv_h, UNCERTAINTY, seed=i)
        rev = run_mpc(actual_prices, actual_P, price_h, perturbed_pv_h, MAX_SOC, MIN_SOC, MAX_RATE)
        production_revenues.append(rev)
        print(f"production iter {i+1}/{N_ITERATIONS}: {rev:.2f} EUR ({time.time()-start:.1f}s)")

    price_revenues = np.array(price_revenues)
    production_revenues = np.array(production_revenues)

    print(f"{country} PRICE uncertainty: mean={price_revenues.mean():.2f}, std={price_revenues.std():.2f}")
    print(f"{country} PRODUCTION uncertainty: mean={production_revenues.mean():.2f}, std={production_revenues.std():.2f}")

    pd.DataFrame({
        'price_uncertainty_revenue': price_revenues,
        'production_uncertainty_revenue': production_revenues
    }).to_csv(f"../../results/rq4-results/monte_carlo_{country}.csv", index=False)
