import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

mpc_base_df = pd.read_csv("../../results/rq3-results/mpc_results.csv", index_col=0)
mpc_base = {c: mpc_base_df.loc[c, "revenue"] for c in COUNTRIES}

mc_price_mean = {}
mc_prod_mean = {}
for c in COUNTRIES:
    mc = pd.read_csv(f"../../results/rq4-results/monte_carlo_{c}.csv")
    mc_price_mean[c] = mc["price_uncertainty_revenue"].mean()
    mc_prod_mean[c] = mc["production_uncertainty_revenue"].mean()

price_loss = {c: (mpc_base[c] - mc_price_mean[c]) / mpc_base[c] * 100 for c in COUNTRIES}
prod_loss = {c: (mpc_base[c] - mc_prod_mean[c]) / mpc_base[c] * 100 for c in COUNTRIES}

x_pos = np.arange(len(COUNTRIES))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
ax.bar(x_pos - width / 2, [price_loss[c] for c in COUNTRIES], width, label="Несигурност во цена (±20%)", color="tab:red")
ax.bar(x_pos + width / 2, [prod_loss[c] for c in COUNTRIES], width, label="Несигурност во PV производство (±20%)", color="tab:green")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_ylabel("Загуба на приход (%)")
ax.set_title("RQ4: Monte Carlo - осетливост на MPC приход на несигурност")
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in COUNTRIES])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../../results/rq4-results/plots/monte_carlo_sensitivity.png", dpi=150)
plt.close()
print("Зачувано: monte_carlo_sensitivity.png")

SCALES = [-30, -20, -10, 10, 20, 30]
fixed_pert = {
    "spain": [1386492.14, 1397865.04, 1414936.14, 1401665.54, 1366800.44, 1343970.31],
    "greece": [402800.35, 410671.77, 416268.85, 412591.87, 405316.30, 397780.28],
    "netherlands": [17005.85, 17364.12, 17618.34, 17496.20, 17171.78, 16882.50],
    "poland": [829230.79, 838355.15, 847657.73, 846471.71, 837868.41, 828018.40],
    "sweden": [23528.33, 23596.06, 23603.47, 23586.53, 23477.48, 23358.85],
    "germany": [2167688.87, 2208387.95, 2248028.25, 2232526.28, 2193935.19, 2155512.44],
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, country in enumerate(COUNTRIES):
    vals = fixed_pert[country]
    normalized = [v / max(vals) * 100 for v in vals]
    ax = axes[i]
    ax.plot(SCALES, normalized, marker="o", color="tab:purple", linewidth=2)
    ax.set_title(country.capitalize())
    ax.set_xlabel("Систематско скалирање на идна цена (%)")
    if i % 3 == 0:
        ax.set_ylabel("Приход (% од максимум)")
    ax.grid(alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")

plt.suptitle("RQ4: Асиметрична чувствителност на MPC (преценување vs потценување на идни цени)", fontsize=14)
plt.tight_layout()
plt.savefig("../../results/rq4-results/plots/fixed_perturbations_asymmetry.png", dpi=150)
plt.close()
print("Зачувано: fixed_perturbations_asymmetry.png")

regime_data = {
    "high_volatility": {"spain": 2.77, "greece": 1.41, "netherlands": 1.88, "poland": 0.82, "sweden": 0.19, "germany": 1.97},
    "low_volatility":  {"spain": 1.16, "greece": 0.90, "netherlands": 0.92, "poland": 1.10, "sweden": 0.22, "germany": 0.87},
    "cloudy":          {"spain": 1.68, "greece": 1.44, "netherlands": 1.47, "poland": 0.94, "sweden": 0.11, "germany": 1.08},
    "sunny":           {"spain": 1.98, "greece": 1.10, "netherlands": 1.43, "poland": 1.14, "sweden": 0.23, "germany": 1.49},
    "high_pv":         {"spain": 1.98, "greece": 1.20, "netherlands": 1.36, "poland": 1.02, "sweden": 0.30, "germany": 1.45},
    "low_pv":          {"spain": 2.05, "greece": 1.34, "netherlands": 1.62, "poland": 1.47, "sweden": 0.37, "germany": 1.36},
    "weekend":         {"spain": 2.19, "greece": 1.03, "netherlands": 1.48, "poland": 1.15, "sweden": 0.39, "germany": 1.37},
    "weekday":         {"spain": 1.55, "greece": 1.02, "netherlands": 1.32, "poland": 1.14, "sweden": 0.27, "germany": 1.27},
    "price_spike":     {"spain": 1.73, "greece": 1.33, "netherlands": 1.46, "poland": 1.30, "sweden": 0.30, "germany": 1.63},
    "no_spike":        {"spain": 2.33, "greece": 1.35, "netherlands": 1.61, "poland": 1.13, "sweden": 0.22, "germany": 1.05},
    "winter":          {"spain": 0.88, "greece": 0.84, "netherlands": 3.85, "poland": 1.71, "sweden": 2.17, "germany": 1.38},
    "summer":          {"spain": 1.93, "greece": 1.10, "netherlands": 1.40, "poland": 0.86, "sweden": 0.26, "germany": 1.47},
}

df_regime = pd.DataFrame(regime_data).T[COUNTRIES]

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(df_regime.values, cmap="YlOrRd", aspect="auto")

ax.set_xticks(np.arange(len(COUNTRIES)))
ax.set_xticklabels([c.capitalize() for c in COUNTRIES])
ax.set_yticks(np.arange(len(df_regime.index)))
ax.set_yticklabels(df_regime.index)

for i in range(df_regime.shape[0]):
    for j in range(df_regime.shape[1]):
        ax.text(j, i, f"{df_regime.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.colorbar(im, label="Загуба на приход (%)")
ax.set_title("RQ4: Regime анализа - загуба на приход (%) по режим и земја")
plt.tight_layout()
plt.savefig("../../results/rq4-results/plots/regime_heatmap.png", dpi=150)
plt.close()
print("Зачувано: regime_heatmap.png")
