import pandas as pd
import pulp
import matplotlib.pyplot as plt

COUNTRY = "spain"
EFFICIENCY = 0.95
HORIZON = 24
WEEK_HOURS = 24 * 7


def solve_window_full(prices, productions, soc, max_soc, min_soc, max_rate):
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
    return sell, charge, discharge, battery


def solve_window(*args):
    sell, charge, discharge, _ = solve_window_full(*args)
    return sell[0].varValue, charge[0].varValue, discharge[0].varValue


df = pd.read_csv(f"../rq3/inputs/rq3_inputs_{COUNTRY}.csv")
fe = pd.read_csv(f"../../data/processed/fe/{COUNTRY}_features.csv")
train_end = int(len(fe) * 0.7)
val_end = int(len(fe) * 0.85)

CAPACITY = fe['solar_generation_MW'].iloc[:train_end].mean() * 2
MIN_SOC, MAX_SOC, MAX_RATE = CAPACITY * 0.1, CAPACITY * 0.9, CAPACITY * 0.4
low_price = fe['Price (EUR/MWhe)'].iloc[:train_end].quantile(0.25)
high_price = fe['Price (EUR/MWhe)'].iloc[:train_end].quantile(0.75)

price_cols = [f'predicted_price_h{h}' for h in range(1, HORIZON)]
pv_cols = [f'predicted_P_h{h}' for h in range(1, HORIZON)]
n_total = len(df)


def run_rule_based():
    soc, revenue, log = MIN_SOC, 0.0, []
    for t in range(n_total):
        row = df.iloc[t]
        production, price = row['actual_P'], row['actual_price']
        charge = discharge = 0
        sell = production
        if price > high_price:
            discharge = min(MAX_RATE, soc - MIN_SOC)
        elif price < low_price:
            charge = min(production, MAX_RATE, MAX_SOC - soc)
            sell = production - charge
        revenue += price * (sell + discharge * EFFICIENCY) / 1000
        soc = soc + charge * EFFICIENCY - discharge
        log.append({"price": price, "sell": sell, "charge": charge, "discharge": discharge,
                     "soc": soc, "cum_revenue": revenue})
    return pd.DataFrame(log)


def run_mpc():
    soc, revenue, log = MIN_SOC, 0.0, []
    for t in range(n_total):
        row = df.iloc[t]
        remaining = min(HORIZON, n_total - t)
        window_prices = [row['actual_price']] + list(row[price_cols].values[:remaining - 1])
        window_prod = [row['actual_P']] + list(row[pv_cols].values[:remaining - 1])

        sell, charge, discharge = solve_window(window_prices, window_prod, soc, MAX_SOC, MIN_SOC, MAX_RATE)
        price, production = row['actual_price'], row['actual_P']

        actual_sell = min(sell, production)
        actual_charge = min(charge, production - actual_sell, MAX_SOC - soc)
        actual_discharge = min(discharge, soc - MIN_SOC)
        actual_sell += production - actual_sell - actual_charge

        revenue += price * (actual_sell + actual_discharge * EFFICIENCY) / 1000
        soc = soc + actual_charge * EFFICIENCY - actual_discharge
        log.append({"price": price, "sell": actual_sell, "charge": actual_charge, "discharge": actual_discharge,
                     "soc": soc, "cum_revenue": revenue})
    return pd.DataFrame(log)


rb_full = run_rule_based()
mpc_full = run_mpc()
print(f"rule-based total revenue: {rb_full['cum_revenue'].iloc[-1]:.0f} EUR")
print(f"MPC total revenue: {mpc_full['cum_revenue'].iloc[-1]:.0f} EUR")

dt_full = pd.to_datetime(fe["datetime"].iloc[val_end:].reset_index(drop=True))
dt = dt_full.iloc[:WEEK_HOURS]

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
for ax, data, title in [(axes[0], rb_full.iloc[:WEEK_HOURS], "Rule-based baseline"),
                         (axes[1], mpc_full.iloc[:WEEK_HOURS], "MPC")]:
    ax.plot(dt.values, data["price"].values, color="black", linewidth=1.3, label="Цена (EUR/MWh)")
    ax.set_ylabel("Цена (EUR/MWh)")
    ax.set_title(f"{title} - операции со батеријата, Spain, прва недела од test")
    ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.fill_between(dt.values, data["soc"].values / CAPACITY * 100, color="tab:blue", alpha=0.25, step="mid",
                      label="SOC (% од капацитет)")
    ax2.set_ylabel("SOC (% од капацитет)", color="tab:blue")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    charge_mask = data["charge"].values > 0
    discharge_mask = data["discharge"].values > 0
    ax.scatter(dt.values[charge_mask], data["price"].values[charge_mask], color="tab:green", s=25, zorder=5, label="Полнење")
    ax.scatter(dt.values[discharge_mask], data["price"].values[discharge_mask], color="tab:red", s=25, zorder=5, label="Празнење")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

axes[1].set_xlabel("Датум/час")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../../results/rq3-results/plots/battery_operation_spain.png", dpi=150)
plt.close()
print("Зачувано: battery_operation_spain.png")

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(dt_full.values, rb_full["cum_revenue"].values, label="Rule-based", color="gray")
ax.plot(dt_full.values, mpc_full["cum_revenue"].values, label="MPC", color="tab:blue")
ax.set_ylabel("Кумулативен приход (EUR)")
ax.set_xlabel("Датум")
ax.set_title("RQ3: Кумулативен приход низ test период - Spain (Rule-based vs MPC)")
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../../results/rq3-results/plots/cumulative_revenue_spain.png", dpi=150)
plt.close()
print("Зачувано: cumulative_revenue_spain.png")

sample_t = WEEK_HOURS // 2
row = df.iloc[sample_t]
remaining = min(HORIZON, n_total - sample_t)
window_prices = [row['actual_price']] + list(row[price_cols].values[:remaining - 1])
window_prod = [row['actual_P']] + list(row[pv_cols].values[:remaining - 1])
soc_at_t = mpc_full["soc"].iloc[sample_t - 1] if sample_t > 0 else MIN_SOC

_, _, _, battery = solve_window_full(window_prices, window_prod, soc_at_t, MAX_SOC, MIN_SOC, MAX_RATE)
soc_traj = [b.varValue / CAPACITY * 100 for b in battery]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(soc_traj) + 1), soc_traj, marker="o", color="tab:blue")
ax.axhline(MIN_SOC / CAPACITY * 100, color="tab:red", linestyle="--", label="Min SOC")
ax.axhline(MAX_SOC / CAPACITY * 100, color="tab:green", linestyle="--", label="Max SOC")
ax.set_xlabel("Час внатре во 24-часовниот планирачки прозорец")
ax.set_ylabel("SOC (% од капацитет)")
ax.set_title("RQ3: 'Horizon-end effect' - SOC внатре во еден MPC прозорец (Spain)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../../results/rq3-results/plots/soc_horizon_end_effect.png", dpi=150)
plt.close()
print("Зачувано: soc_horizon_end_effect.png")
