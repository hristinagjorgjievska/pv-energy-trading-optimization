import pandas as pd
import numpy as np
import pulp

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
EFFICIENCY = 0.95
HORIZON = 24
UNCERTAINTY = 0.20
N_SEEDS = 5


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


def subset_hours(actual_prices, actual_P, price_h, pv_h, day_mask):
    n_days = len(actual_prices) // 24
    hour_mask = np.repeat(day_mask[:n_days], 24)
    return (actual_prices[hour_mask], actual_P[hour_mask],
            price_h[hour_mask], pv_h[hour_mask])


def eval_regime(actual_prices, actual_P, price_h, pv_h, mask, max_soc, min_soc, max_rate):
    a_p, a_P, p_h, v_h = subset_hours(actual_prices, actual_P, price_h, pv_h, mask)
    n_days = len(a_p) // 24
    if n_days == 0:
        return np.nan, np.nan, 0

    base_revenue = run_mpc(a_p, a_P, p_h, v_h, max_soc, min_soc, max_rate)

    losses = []
    for seed in range(N_SEEDS):
        perturbed_p_h = perturb(p_h, UNCERTAINTY, seed=seed)
        pert_revenue = run_mpc(a_p, a_P, perturbed_p_h, v_h, max_soc, min_soc, max_rate)
        loss_pct = (base_revenue - pert_revenue) / base_revenue * 100 if base_revenue != 0 else np.nan
        losses.append(loss_pct)

    return float(np.mean(losses)), float(np.std(losses)), n_days


all_rows = []

for country in COUNTRIES:
    df = pd.read_csv(f"../rq3/inputs/rq3_inputs_{country}.csv")

    fe = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    train_end = int(len(fe) * 0.7)
    train_avg_production = fe['solar_generation_MW'].iloc[:train_end].mean()
    CAPACITY = train_avg_production * 2
    MIN_SOC = CAPACITY * 0.1
    MAX_SOC = CAPACITY * 0.9
    MAX_RATE = CAPACITY * 0.4

    n_days = len(df) // 24
    df_days = df.iloc[:n_days * 24].copy()

    actual_prices = df_days['actual_price'].values
    actual_P = df_days['actual_P'].values
    price_h = df_days[[f'predicted_price_h{h}' for h in range(1, HORIZON)]].values
    pv_h = df_days[[f'predicted_P_h{h}' for h in range(1, HORIZON)]].values

    day_idx = np.arange(len(df_days)) // 24

    day_price_std = df_days.groupby(day_idx)['actual_price'].std().values
    day_cloud = df_days.groupby(day_idx)['cloud_cover'].mean().values
    day_production = df_days.groupby(day_idx)['actual_P'].mean().values
    day_is_weekend = df_days.groupby(day_idx)['is_weekend'].max().values.astype(bool)
    day_month = df_days.groupby(day_idx)['month'].first().values
    price_spike_threshold = df_days['actual_price'].quantile(0.90)
    day_has_spike = (df_days.groupby(day_idx)['actual_price'].max() > price_spike_threshold).values

    vol_hi = np.quantile(day_price_std, 0.75)
    vol_lo = np.quantile(day_price_std, 0.25)
    cloud_hi = np.quantile(day_cloud, 0.75)
    cloud_lo = np.quantile(day_cloud, 0.25)
    prod_hi = np.quantile(day_production, 0.75)
    prod_lo = np.quantile(day_production, 0.25)

    row = {"country": country}

    regimes = {
        "high_volatility": day_price_std >= vol_hi, "low_volatility": day_price_std <= vol_lo,
        "cloudy": day_cloud >= cloud_hi, "sunny": day_cloud <= cloud_lo,
        "high_pv": day_production >= prod_hi, "low_pv": day_production <= prod_lo,
        "weekend": day_is_weekend, "weekday": ~day_is_weekend,
        "price_spike": day_has_spike, "no_spike": ~day_has_spike,
    }
    winter_mask = np.isin(day_month, [12, 1, 2])
    summer_mask = np.isin(day_month, [6, 7, 8])
    if winter_mask.sum() > 0 and summer_mask.sum() > 0:
        regimes["winter"] = winter_mask
        regimes["summer"] = summer_mask

    for name, mask in regimes.items():
        loss_mean, loss_std, n = eval_regime(actual_prices, actual_P, price_h, pv_h, mask, MAX_SOC, MIN_SOC, MAX_RATE)
        row[f"{name}_days"] = n
        row[f"{name}_loss_%"] = loss_mean
        row[f"{name}_loss_std"] = loss_std
        print(f"{country} - {name}: {n} days, loss={loss_mean:.2f}% (std={loss_std:.2f})")

    all_rows.append(row)

result_df = pd.DataFrame(all_rows)
result_df.to_csv("../../results/rq4-results/regime_analysis_results.csv", index=False)
print(result_df)
