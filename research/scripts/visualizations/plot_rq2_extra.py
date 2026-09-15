import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

pv_features = [
    'hour', 'day_of_week', 'month', 'is_weekend',
    'temperature_2m', 'wind_speed_10m', 'cloud_cover', 'shortwave_radiation',
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h'
]

df = pd.read_csv("../../data/processed/fe/spain_features.csv")
train_end = int(len(df) * 0.7)
val_end = int(len(df) * 0.85)

X_train = df[pv_features].iloc[:train_end]
y_train = df["target_pv_h1"].iloc[:train_end]
X_test = df[pv_features].iloc[val_end:]
y_test = df["target_pv_h1"].iloc[val_end:]

model = XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1, reg_lambda=1, random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

imp = pd.DataFrame({"feature": pv_features, "importance": model.feature_importances_})
imp = imp.sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(imp["feature"], imp["importance"] * 100, color="tab:green")
ax.set_xlabel("Importance (%)")
ax.set_title("RQ2: Feature importance - XGBoost PV производство (Spain, h=1)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/feature_importance_spain_pv.png", dpi=150)
plt.close()
print("Зачувано: feature_importance_spain_pv.png")

avp = pd.DataFrame({
    "datetime": pd.to_datetime(df["datetime"].iloc[val_end:].values),
    "actual": y_test.values,
    "predicted": y_pred,
})
week = avp.iloc[:24 * 7]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(week["datetime"], week["actual"], label="Вистинско производство", color="black", linewidth=1.5)
ax.plot(week["datetime"], week["predicted"], label="Предвидено производство (XGBoost, h=1)", color="tab:green", linestyle="--")
ax.set_ylabel("PV производство (MW)")
ax.set_title("RQ2: Вистинско vs предвидено PV производство - Spain, прва недела од test период")
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("../../results/rq2-results/plots/actual_vs_predicted_spain_pv.png", dpi=150)
plt.close()
print("Зачувано: actual_vs_predicted_spain_pv.png")
