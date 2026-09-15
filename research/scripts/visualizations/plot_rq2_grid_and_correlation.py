import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

pv_features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
]

results = {}

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)

    X_train = df[pv_features].iloc[:train_end]
    y_train = df["target_pv_h1"].iloc[:train_end]
    X_test = df[pv_features].iloc[val_end:]
    y_test = df["target_pv_h1"].iloc[val_end:]

    model = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                          min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                          reg_alpha=1, reg_lambda=1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = 1 - np.sum((y_test.values - y_pred) ** 2) / np.sum((y_test.values - y_test.mean()) ** 2)
    avg_production = df["solar_generation_MW"].iloc[:train_end].mean()
    avg_radiation = df["shortwave_radiation"].iloc[:train_end].mean()
    results[country] = {"R2": r2 * 100, "avg_production": avg_production, "avg_radiation": avg_radiation}

    dt = pd.to_datetime(df["datetime"].iloc[val_end:].values)
    week_actual = y_test.values[:24 * 7]
    week_pred = y_pred[:24 * 7]
    week_dt = dt[:24 * 7]

    ax = axes[i]
    ax.plot(week_dt, week_actual, label="Вистинско", color="black", linewidth=1.3)
    ax.plot(week_dt, week_pred, label="Предвидено", color="tab:green", linestyle="--", linewidth=1.3)
    ax.set_title(f"{country.capitalize()} (R²={r2*100:.1f}%)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=9)

plt.suptitle("RQ2: Вистинско vs предвидено PV производство (XGBoost, h=1) - прва недела од test, сите земји", fontsize=15)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/actual_vs_predicted_all_countries_pv.png", dpi=150)
plt.close()
print("Зачувано: actual_vs_predicted_all_countries_pv.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
for country in COUNTRIES:
    r = results[country]
    ax.scatter(r["avg_production"], r["R2"], s=120, color="tab:green")
    ax.annotate(country.capitalize(), (r["avg_production"], r["R2"]), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Просечно PV производство во train период (MW)")
ax.set_ylabel("R² на XGBoost (h=1, %)")
ax.set_xscale("log")
ax.set_title("R² наспроти големина на PV пазар (log скала)")
ax.grid(alpha=0.3)

ax = axes[1]
for country in COUNTRIES:
    r = results[country]
    ax.scatter(r["avg_radiation"], r["R2"], s=120, color="tab:orange")
    ax.annotate(country.capitalize(), (r["avg_radiation"], r["R2"]), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("Просечно сончево зрачење во train период (W/m²)")
ax.set_ylabel("R² на XGBoost (h=1, %)")
ax.set_title("R² наспроти просечно ниво на зрачење")
ax.grid(alpha=0.3)

plt.suptitle("RQ2: Дали постои врска помеѓу карактеристики на пазарот и предвидливост на производство?", fontsize=14)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/r2_vs_market_characteristics_pv.png", dpi=150)
plt.close()
print("Зачувано: r2_vs_market_characteristics_pv.png")

print()
print(pd.DataFrame(results).T)
