import pandas as pd
import pulp

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
EFFICIENCY = 0.95
HORIZON = 24
SCALES = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]

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
        prev_soc = soc if t == 0 else battery[t-1]
        prob += battery[t] == prev_soc + charge[t] * EFFICIENCY - discharge[t]
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sell[0].varValue, charge[0].varValue, discharge[0].varValue

def run_mpc(prices, productions, actual_prices, actual_P, max_soc, min_soc, max_rate):
    soc = min_soc
    revenue = 0
    n = len(prices)
    for t in range(n):
        end = min(t + HORIZON, n)
        sell, charge, discharge = solve_window(
            prices[t:end], productions[t:end], soc, max_soc, min_soc, max_rate
        )
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

for country in COUNTRIES:
    df = pd.read_csv(f"../../scripts/rq3/inputs/rq3_inputs_{country}.csv")

    avg_production = df['actual_P'].mean()
    CAPACITY = avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * 0.4

    predicted_prices = df['predicted_price'].values
    predicted_P = df['predicted_P'].values
    actual_prices = df['actual_price'].values
    actual_P = df['actual_P'].values

    print(f"--- {country} ---")
    for scale in SCALES:
        scaled_prices = predicted_prices * (1 + scale)
        revenue = run_mpc(scaled_prices, predicted_P, actual_prices, actual_P, MAX_SOC, MIN_SOC, MAX_RATE)
        print(f"price scale {scale*100:+.0f}%: revenue = {revenue:.2f} EUR")