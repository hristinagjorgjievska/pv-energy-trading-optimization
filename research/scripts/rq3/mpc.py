import pandas as pd
import pulp

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

MAX_RATE_FACTOR = 0.4
EFFICIENCY = 0.95
HORIZON = 24

def solve_window(prices, productions, soc, max_soc, max_rate):
    H = len(prices)
    prob = pulp.LpProblem("mpc", pulp.LpMaximize)

    sell = [pulp.LpVariable(f"sell_{t}", lowBound=0) for t in range(H)]
    charge = [pulp.LpVariable(f"charge_{t}", lowBound=0, upBound=max_rate) for t in range(H)]
    discharge = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_rate) for t in range(H)]
    battery = [pulp.LpVariable(f"soc_{t}", lowBound=max_soc*0.1, upBound=max_soc) for t in range(H)]

    prob += pulp.lpSum(prices[t] * (sell[t] + discharge[t] * EFFICIENCY) for t in range(H))

    for t in range(H):
        prob += sell[t] + charge[t] == productions[t]
        prev_soc = soc if t == 0 else battery[t-1]
        prob += battery[t] == prev_soc + charge[t] * EFFICIENCY - discharge[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    return sell[0].varValue, charge[0].varValue, discharge[0].varValue

for country in COUNTRIES:
    df = pd.read_csv(f"rq3_inputs_{country}.csv")

    avg_production = df['actual_P'].mean()
    CAPACITY = avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * MAX_RATE_FACTOR

    soc = MIN_SOC
    revenue = 0
    n = len(df)

    for t in range(n):
        window = df.iloc[t:t+HORIZON]
        sell, charge, discharge = solve_window(
            window['predicted_price'].values, window['predicted_P'].values, soc, MAX_SOC, MAX_RATE
        )

        real_price = df['actual_price'].iloc[t]
        real_production = df['actual_P'].iloc[t]

        actual_sell = min(sell, real_production)
        actual_charge = min(charge, real_production - actual_sell, MAX_SOC - soc)
        actual_discharge = min(discharge, soc - MIN_SOC)

        surplus = real_production - actual_sell - actual_charge
        actual_sell += surplus

        revenue += real_price * (actual_sell + actual_discharge * EFFICIENCY) / 1000
        soc = soc + actual_charge * EFFICIENCY - actual_discharge

    print(f"{country}: capacity={CAPACITY:.0f}, MPC revenue = {revenue:.2f} EUR")