import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COUNTRIES = ["spain", "greece", "netherlands", "poland", "sweden", "germany"]

rule_based = pd.read_csv("../../results/rq3-results/rule_based_results.csv", index_col=0)
mpc = pd.read_csv("../../results/rq3-results/mpc_results.csv", index_col=0)
pf = pd.read_csv("../../results/rq3-results/perfect_foresight_results.csv", index_col=0)
pf.columns = [c.strip() for c in pf.columns]
pf.index = [i.strip() for i in pf.index]
pf["revenue"] = pf["revenue"].astype(str).str.replace("EUR", "", regex=False).str.strip().astype(float)
pf["capacity"] = pd.to_numeric(pf["capacity"], errors="coerce")

x_pos = np.arange(len(COUNTRIES))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ax = axes[0]
rb_vals = [rule_based.loc[c, "revenue"] for c in COUNTRIES]
mpc_vals = [mpc.loc[c, "revenue"] for c in COUNTRIES]
pf_vals = [pf.loc[c, "revenue"] for c in COUNTRIES]

ax.bar(x_pos - width, rb_vals, width, label="Rule-based", color="gray")
ax.bar(x_pos, mpc_vals, width, label="MPC", color="tab:blue")
ax.bar(x_pos + width, pf_vals, width, label="Perfect Foresight", color="tab:green")
ax.set_ylabel("Приход (EUR)")
ax.set_yscale("log")
ax.set_title("Апсолутен приход по земја (log скала)")
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in COUNTRIES], rotation=20)
ax.legend()
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
rb_pct = [rule_based.loc[c, "revenue"] / pf.loc[c, "revenue"] * 100 for c in COUNTRIES]
mpc_pct = [mpc.loc[c, "revenue"] / pf.loc[c, "revenue"] * 100 for c in COUNTRIES]

ax.bar(x_pos - width / 2, rb_pct, width, label="Rule-based", color="gray")
ax.bar(x_pos + width / 2, mpc_pct, width, label="MPC", color="tab:blue")
ax.axhline(100, color="tab:green", linestyle="--", label="Perfect Foresight (100%)")
ax.set_ylabel("% од Perfect Foresight")
ax.set_title("Релативен перформанс (% од теоретски максимум)")
ax.set_xticks(x_pos)
ax.set_xticklabels([c.capitalize() for c in COUNTRIES], rotation=20)
ax.set_ylim(80, 102)
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("../../results/rq3-results/plots/revenue_comparison.png", dpi=150)
plt.close()
print("Зачувано: revenue_comparison.png")
