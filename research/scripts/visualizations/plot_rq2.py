import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    baseline = pd.read_csv(f"../../results/rq2-results/baseline_model/{country}_results.csv")
    xgb = pd.read_csv(f"../../results/rq2-results/xgboost_model/{country}_results.csv")

    ax = axes[i]
    ax.plot(baseline["horizon"], baseline["R2"] * 100, marker="o", label="Persistence Baseline", color="gray")
    ax.plot(xgb["horizon"], xgb["R2"] * 100, marker="o", label="XGBoost", color="tab:blue")
    ax.set_title(country.capitalize())
    ax.set_xlabel("Хоризонт (часови)")
    if i % 3 == 0:
        ax.set_ylabel("R² (%)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.3)

axes[0].legend(loc="lower left")
plt.suptitle("RQ2: R² по хоризонт на предвидување на PV производство (Baseline vs XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/r2_by_horizon.png", dpi=150)
plt.close()
print("Зачувано: r2_by_horizon.png")

tft = pd.read_csv("../../results/rq2-results/tft_model/tft_pv_model.csv", index_col=0)


def h1_r2(model_dir, country):
    df = pd.read_csv(f"../../results/rq2-results/{model_dir}/{country}_results.csv")
    return df.loc[df["horizon"] == 1, "R2"].iloc[0] * 100


baseline_r2 = [h1_r2("baseline_model", c) for c in COUNTRIES]
xgb_r2 = [h1_r2("xgboost_model", c) for c in COUNTRIES]
lstm_r2 = [h1_r2("lstm_model", c) for c in COUNTRIES]
tft_r2 = [tft.loc[c, "R2"] for c in COUNTRIES]

x_pos = np.arange(len(COUNTRIES))
width = 0.2

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x_pos - 1.5 * width, baseline_r2, width, label="Persistence Baseline", color="gray")
ax.bar(x_pos - 0.5 * width, xgb_r2, width, label="XGBoost", color="tab:blue")
ax.bar(x_pos + 0.5 * width, lstm_r2, width, label="LSTM", color="tab:orange")
ax.bar(x_pos + 1.5 * width, tft_r2, width, label="TFT", color="tab:green")

ax.set_ylabel("R² (%)")
ax.set_title("RQ2: Споредба на модели за PV производство по земја (h=1)")
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in COUNTRIES])
ax.legend()
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/model_comparison.png", dpi=150)
plt.close()
print("Зачувано: model_comparison.png")
