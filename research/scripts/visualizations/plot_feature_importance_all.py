import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
axes = axes.flatten()

for i, country in enumerate(COUNTRIES):
    df = pd.read_csv(f"../../data/processed/fe/{country}_features.csv")
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)
    features_cols = [c for c in df.columns if c != "datetime" and not c.startswith("target_")]

    X_train = df[features_cols].iloc[:train_end]
    y_train = df["target_h1"].iloc[:train_end]

    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1, reg_lambda=1, random_state=42,
    )
    model.fit(X_train, y_train)

    imp = pd.DataFrame({"feature": features_cols, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=True)

    ax = axes[i]
    ax.barh(imp["feature"], imp["importance"] * 100, color="tab:blue")
    ax.set_xlabel("Importance (%)")
    ax.set_title(country.capitalize())
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    print(f"{country}: готово")

plt.suptitle("RQ1: Feature importance - XGBoost (h=1), сите 6 земји", fontsize=15)
plt.tight_layout()
plt.savefig("../../results/rq1-results/plots/feature_importance_all_countries.png", dpi=150)
plt.close()
print("Зачувано: feature_importance_all_countries.png")
