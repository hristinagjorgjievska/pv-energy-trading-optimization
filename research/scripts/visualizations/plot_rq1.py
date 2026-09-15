import pandas as pd
import matplotlib.pyplot as plt

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
COLORS = plt.cm.tab10.colors

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    baseline = pd.read_csv(f"../../results/rq1-results/baseline_model/{country}_results.csv")
    xgb = pd.read_csv(f"../../results/rq1-results/xgboost_model/{country}_results.csv")

    ax = axes[i]
    ax.plot(baseline["horizon"], baseline["R2"] * 100, marker="o", label="Persistence Baseline", color="gray")
    ax.plot(xgb["horizon"], xgb["R2"] * 100, marker="o", label="XGBoost", color="tab:blue")
    ax.set_title(country.capitalize())
    ax.set_xlabel("Хоризонт (часови)")
    if i % 3 == 0:
        ax.set_ylabel("R² (%)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.3)

axes[0].legend(loc="upper right")
plt.suptitle("RQ1: R² по хоризонт на предвидување (Baseline vs XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig("../../results/rq1-results/plots/r2_by_horizon.png", dpi=150)
plt.close()
print("Зачувано: r2_by_horizon.png")

lstm = pd.DataFrame({
    "R2": {
        "spain": 90.113688, "greece": 77.714097, "netherlands": 79.456216,
        "poland": 76.591551, "sweden": 84.802508, "germany": 84.053552,
    }
})

baseline_h1 = []
xgb_h1 = []
for country in COUNTRIES:
    b = pd.read_csv(f"../../results/rq1-results/baseline_model/{country}_results.csv")
    x = pd.read_csv(f"../../results/rq1-results/xgboost_model/{country}_results.csv")
    baseline_h1.append(b[b["horizon"] == 1]["R2"].values[0] * 100)
    xgb_h1.append(x[x["horizon"] == 1]["R2"].values[0] * 100)

lstm_r2 = [lstm.loc[c, "R2"] for c in COUNTRIES]

tft_r2 = {
    "spain": 66.11, "greece": 63.73, "netherlands": 48.61,
    "poland": 42.90, "sweden": 39.78, "germany": 49.09,
}
tft_vals = [tft_r2[c] for c in COUNTRIES]

import numpy as np
x_pos = np.arange(len(COUNTRIES))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x_pos - 1.5 * width, baseline_h1, width, label="Baseline (h=1)", color="gray")
ax.bar(x_pos - 0.5 * width, xgb_h1, width, label="XGBoost (h=1)", color="tab:blue")
ax.bar(x_pos + 0.5 * width, lstm_r2, width, label="LSTM", color="tab:orange")
ax.bar(x_pos + 1.5 * width, tft_vals, width, label="TFT (сите хоризонти)", color="tab:green")

ax.set_ylabel("R² (%)")
ax.set_title("RQ1: Споредба на модели по земја")
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in COUNTRIES])
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../../results/rq1-results/plots/model_comparison.png", dpi=150)
plt.close()
print("Зачувано: model_comparison.png")
