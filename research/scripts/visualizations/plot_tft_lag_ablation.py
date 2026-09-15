import matplotlib.pyplot as plt

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]
with_lag = {"spain": 66.1, "greece": 63.7, "netherlands": 48.6, "poland": 42.9, "sweden": 39.8, "germany": 49.1}
without_lag = {"spain": 61.2, "greece": 59.0, "netherlands": 37.4, "poland": 36.0, "sweden": 41.6, "germany": 43.6}

fig, ax = plt.subplots(figsize=(8, 7))

for c in COUNTRIES:
    color = "tab:green" if without_lag[c] > with_lag[c] else "tab:red"
    ax.plot([0, 1], [with_lag[c], without_lag[c]], marker="o", color=color, linewidth=2)
    ax.annotate(f"{c.capitalize()} ({with_lag[c]:.1f}%)", (0, with_lag[c]), xytext=(-10, 0),
                textcoords="offset points", ha="right", fontsize=10)
    ax.annotate(f"{without_lag[c]:.1f}%", (1, without_lag[c]), xytext=(10, 0),
                textcoords="offset points", ha="left", fontsize=10)

ax.set_xlim(-0.6, 1.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["TFT со lag features", "TFT без lag features"], fontsize=12)
ax.set_ylabel("R² (%)")
ax.set_title("RQ1 bonus: дали TFT сам може да ги открие временските шаблони?\n(зелено=подобрено, црвено=влошено)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../../results/rq1-results/plots/tft_lag_ablation.png", dpi=150)
plt.close()
print("Зачувано: tft_lag_ablation.png")
