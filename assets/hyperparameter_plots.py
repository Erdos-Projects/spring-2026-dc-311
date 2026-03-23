import matplotlib.pyplot as plt
import numpy as np
 
data = [
    {"model": "GLM Negative Binomial", "d": 1, "d_f": 7,  "d_p": 14, "d_s": 18, "k_AR": 0,  "l_f": 1,  "l_p": 1,  "l_s": 8},
    {"model": "GLM Negative Binomial", "d": 5, "d_f": 11, "d_p": 9,  "d_s": 17, "k_AR": 0,  "l_f": 10, "l_p": 6,  "l_s": 9},
    {"model": "GLM Negative Binomial", "d": 7, "d_f": 9,  "d_p": 19, "d_s": 19, "k_AR": 0,  "l_f": 10, "l_p": 8,  "l_s": 10},
    {"model": "GLM Poisson",           "d": 1, "d_f": 7,  "d_p": 19, "d_s": 20, "k_AR": 0,  "l_f": 1,  "l_p": 1,  "l_s": 10},
    {"model": "GLM Poisson",           "d": 5, "d_f": 9,  "d_p": 15, "d_s": 20, "k_AR": 0,  "l_f": 8,  "l_p": 10, "l_s": 10},
    {"model": "GLM Poisson",           "d": 7, "d_f": 18, "d_p": 12, "d_s": 19, "k_AR": 0,  "l_f": 7,  "l_p": 6,  "l_s": 6},
    {"model": "XGBoost",               "d": 1, "d_f": 21, "d_p": 8,  "d_s": 8,  "k_AR": 8,  "l_f": 1,  "l_p": 9,  "l_s": 7},
    {"model": "XGBoost",               "d": 5, "d_f": 8,  "d_p": 7,  "d_s": 9,  "k_AR": 0,  "l_f": 10, "l_p": 9,  "l_s": 5},
    {"model": "XGBoost",               "d": 7, "d_f": 16, "d_p": 12, "d_s": 19, "k_AR": 0,  "l_f": 7,  "l_p": 2,  "l_s": 9},
    {"model": "XGBoost SARIMAX",       "d": 1, "d_f": 21, "d_p": 8,  "d_s": 14, "k_AR": 10, "l_f": 10, "l_p": 9,  "l_s": 10},
    {"model": "XGBoost SARIMAX",       "d": 5, "d_f": 8,  "d_p": 21, "d_s": 7,  "k_AR": 0,  "l_f": 9,  "l_p": 10, "l_s": 1},
    {"model": "XGBoost SARIMAX",       "d": 7, "d_f": 11, "d_p": 9,  "d_s": 20, "k_AR": 0,  "l_f": 5,  "l_p": 9,  "l_s": 10},
]
 
params = ["d_f", "d_p", "d_s", "k_AR", "l_f", "l_p", "l_s"]
d_values = [1, 5, 7]
colors = ["#4e79c4", "#e06b3f", "#59a96a"]
 
# Compute averages and std devs per d value
averages = {}
std_devs = {}
for d in d_values:
    rows = [r for r in data if r["d"] == d]
    averages[d] = {p: sum(r[p] for r in rows) / len(rows) for p in params}
    std_devs[d] = {p: (sum((r[p] - averages[d][p])**2 for r in rows) / len(rows)) ** 0.5 for p in params}
 
# Plot
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()
 
x = np.arange(len(d_values))
bar_width = 0.5
 
for i, param in enumerate(params):
    ax = axes[i]
    vals = [averages[d][param] for d in d_values]
    errs = [std_devs[d][param] for d in d_values]
    bars = ax.bar(x, vals, width=bar_width, color=colors, zorder=2,
                  yerr=errs, capsize=5, error_kw={"elinewidth": 1.2, "ecolor": "#555", "capthick": 1.2})
 
    ax.set_title(param, fontsize=13, fontweight="normal", color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d={d}" for d in d_values], fontsize=11)
    ax.set_ylim(0, max(v + e for v, e in zip(vals, errs)) * 1.3 + 1)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.grid(axis="y", color="lightgrey", linewidth=0.5, zorder=1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
 
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=10, color="#444"
        )
 
# Hide the unused 8th subplot
axes[-1].set_visible(False)
 
# Legend
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=c, label=f"d = {d}")
    for c, d in zip(colors, d_values)
]
fig.legend(handles=legend_handles, loc="lower right", bbox_to_anchor=(0.98, 0.08),
           fontsize=11, frameon=False)
 
fig.suptitle("Average hyperparameter values by d", fontsize=15, fontweight="normal", y=1.01)
plt.tight_layout()
plt.savefig("hyperparameter_averages.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to hyperparameter_averages.png")