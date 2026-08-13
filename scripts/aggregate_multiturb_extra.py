# -*- coding: utf-8 -*-
"""
多风机泛化验证 —— 补充图(逐台一致性 + 与风机特性的关系)。
复用 run_records.jsonl(480 条)+ tools/turbine_profile.csv(134 台画像)。

产出:
  outputs/figures/paper/fig_generalization_heatmap.png
      风机(行) x 步长(列) 的 proposed vs 最强基线 改善率热力图
      —— 证明"每一台"都赢,不是靠均值拉高
  outputs/figures/paper/fig_generalization_vs_characteristics.png
      改善率 vs 风机特性(平均出力 / 波动系数CV) 散点 + 趋势线
      —— 证明优势不依赖风机的出力档位或波动程度
  outputs/tables/generalization_winrate.csv
      逐风机胜率表(proposed 在 10 台中赢了几台,对每个基线每个步长)
"""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master")
RECORDS = REPO / "outputs/multiturb/outputs/runs/run_records.jsonl"
PROFILE = REPO / "tools/turbine_profile.csv"
TABLES_DIR = REPO / "outputs/tables"
FIG_DIR = REPO / "outputs/figures/paper"

MODEL_ORDER = ["proposed", "dlinear", "itransformer", "patchtst"]
MODEL_LABEL = {"proposed": "Proposed", "dlinear": "DLinear",
               "itransformer": "iTransformer", "patchtst": "PatchTST"}
HORIZONS = [1, 6, 12, 24]
RUN_ID_RE = re.compile(r"^t(\d+)_(.+)_h(\d+)_seed(\d+)$")

# ---- load records ----
rows = []
with open(RECORDS, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("status") != "success":
            continue
        m = RUN_ID_RE.match(rec["run_id"])
        if not m:
            continue
        tid, model, h, seed = m.groups()
        rows.append({"turbine": int(tid), "model": model, "horizon": int(h),
                      "seed": int(seed), "mae": rec["metrics"]["mae"]})
df = pd.DataFrame(rows)
per_turb = df.groupby(["horizon", "model", "turbine"], as_index=False)["mae"].mean()

# ---- best-baseline MAE per (horizon, turbine) ----
baselines = [m for m in MODEL_ORDER if m != "proposed"]
piv = per_turb.pivot_table(index=["horizon", "turbine"], columns="model", values="mae")
piv["best_baseline"] = piv[baselines].min(axis=1)
piv["best_baseline_name"] = piv[baselines].idxmin(axis=1)
piv["improvement_pct"] = (piv["best_baseline"] - piv["proposed"]) / piv["best_baseline"] * 100
piv = piv.reset_index()

# ===========================================================================
# Figure A: heatmap turbines x horizons, value = improvement% of proposed
#           over the strongest baseline for that cell
# ===========================================================================
turbines_sorted = sorted(piv["turbine"].unique())
mat = piv.pivot(index="turbine", columns="horizon", values="improvement_pct").loc[turbines_sorted, HORIZONS]

fig, ax = plt.subplots(figsize=(6.4, 5.6))
vmax = np.ceil(np.abs(mat.values).max() / 5) * 5
im = ax.imshow(mat.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(HORIZONS)))
ax.set_xticklabels([f"h={h}" for h in HORIZONS])
ax.set_yticks(range(len(turbines_sorted)))
ax.set_yticklabels([f"Turb{t}" for t in turbines_sorted])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        ax.text(j, i, f"{v:+.0f}%", ha="center", va="center",
                fontsize=8.5, color="black" if abs(v) < vmax * 0.6 else "white")
ax.set_title("Proposed vs. strongest baseline: MAE improvement per turbine\n"
             "(green = proposed better, red = baseline better)", fontsize=10.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Improvement of Proposed over best baseline (%)")
for spine in ax.spines.values():
    spine.set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_generalization_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_generalization_heatmap.png")

# ===========================================================================
# Figure B: improvement% vs turbine characteristics (patv_mean, cv)
# ===========================================================================
prof = pd.read_csv(PROFILE, index_col=0)  # index = TurbID
prof = prof.loc[turbines_sorted, ["patv_mean", "cv"]].reset_index().rename(columns={"index": "turbine", "TurbID": "turbine"})
merged = piv.merge(prof, on="turbine")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(HORIZONS)))
for ax, xcol, xlabel in zip(axes, ["patv_mean", "cv"],
                            ["Turbine mean output (kW)", "Turbine coefficient of variation (volatility)"]):
    for h, c in zip(HORIZONS, colors):
        sub = merged[merged.horizon == h]
        ax.scatter(sub[xcol], sub["improvement_pct"], color=c, s=42, label=f"h={h}",
                   edgecolors="white", linewidths=0.5, zorder=3)
        # trend line (only if enough spread)
        if sub[xcol].std() > 1e-6:
            zc = np.polyfit(sub[xcol], sub["improvement_pct"], 1)
            xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
            ax.plot(xs, np.polyval(zc, xs), color=c, alpha=0.5, linewidth=1.3, zorder=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Improvement of Proposed over\nbest baseline (%)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.35)
axes[0].legend(fontsize=8.5, loc="upper right", ncol=2, framealpha=0.9)
fig.suptitle("Robustness of the improvement across turbine operating regimes", fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_generalization_vs_characteristics.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved fig_generalization_vs_characteristics.png")

# ===========================================================================
# Win-rate table: out of 10 turbines, how many does proposed beat each baseline
# ===========================================================================
winrate_rows = []
for h in HORIZONS:
    sub = piv[piv.horizon == h]
    for b in baselines:
        wins = (sub["proposed"] < piv.set_index(["horizon", "turbine"]).loc[
            list(zip([h]*len(sub), sub["turbine"])), b].values).sum() if False else None
    # simpler: recompute directly from pivoted per-model table
    pm = per_turb[(per_turb.horizon == h)].pivot(index="turbine", columns="model", values="mae")
    for b in baselines:
        wins = int((pm["proposed"] < pm[b]).sum())
        winrate_rows.append({"horizon": h, "baseline": MODEL_LABEL[b],
                              "proposed_wins": wins, "n_turbines": len(pm),
                              "win_rate_pct": round(wins / len(pm) * 100, 0)})
winrate = pd.DataFrame(winrate_rows)
winrate.to_csv(TABLES_DIR / "generalization_winrate.csv", index=False)
print("\n=== win-rate table ===")
print(winrate.to_string(index=False))
print("\nDONE")
