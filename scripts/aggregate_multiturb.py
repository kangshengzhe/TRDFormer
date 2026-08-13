# -*- coding: utf-8 -*-
"""
聚合多风机泛化验证结果 -> 论文用表 + 图。

产出:
  outputs/tables/generalization_table.csv / .tex
      每个 (model, horizon) 在 10 台风机 x 3 种子上的 MAE/RMSE/R2 mean±std
  outputs/figures/paper/fig_generalization_box.png
      跨风机 MAE 分布箱线图(每个 horizon 一个面板, 4 模型并排)
  outputs/tables/generalization_significance.csv
      proposed vs 各基线的配对 t 检验(以风机为配对单位, 每风机取 3 seed 均值)
"""
import re, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master")
RECORDS = REPO / "outputs/multiturb/outputs/runs/run_records.jsonl"
TABLES_DIR = REPO / "outputs/tables"
FIG_DIR = REPO / "outputs/figures/paper"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["proposed", "dlinear", "itransformer", "patchtst"]
MODEL_LABEL = {"proposed": "Proposed", "dlinear": "DLinear",
               "itransformer": "iTransformer", "patchtst": "PatchTST"}
HORIZONS = [1, 6, 12, 24]

RUN_ID_RE = re.compile(r"^t(\d+)_(.+)_h(\d+)_seed(\d+)$")

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
        met = rec["metrics"]
        rows.append({
            "turbine": int(tid), "model": model, "horizon": int(h), "seed": int(seed),
            "mae": met["mae"], "rmse": met["rmse"], "r2": met["r2"],
        })

df = pd.DataFrame(rows)
print("loaded rows:", len(df), "turbines:", sorted(df.turbine.unique()),
      "models:", sorted(df.model.unique()), "horizons:", sorted(df.horizon.unique()))
assert len(df) == 480, f"expected 480 rows, got {len(df)}"

# ---------------------------------------------------------------------------
# 1) generalization_table: per (model, horizon), aggregate across turbines x seeds
# ---------------------------------------------------------------------------
def agg_fmt(s):
    return f"{s.mean():.2f} \u00b1 {s.std():.2f}"

table_rows = []
for h in HORIZONS:
    for model in MODEL_ORDER:
        sub = df[(df.horizon == h) & (df.model == model)]
        table_rows.append({
            "horizon": h,
            "model": MODEL_LABEL[model],
            "n_runs": len(sub),
            "n_turbines": sub.turbine.nunique(),
            "MAE": agg_fmt(sub["mae"]),
            "RMSE": agg_fmt(sub["rmse"]),
            "R2": agg_fmt(sub["r2"]),
            "mae_mean": sub["mae"].mean(),
        })
gen_table = pd.DataFrame(table_rows)
gen_table_out = gen_table.drop(columns=["mae_mean"])
gen_table_out.to_csv(TABLES_DIR / "generalization_table.csv", index=False)
with open(TABLES_DIR / "generalization_table.tex", "w", encoding="utf-8") as f:
    f.write(gen_table_out.to_latex(index=False, escape=True,
        caption="Generalization across 10 representative turbines (mean $\\pm$ std over turbines $\\times$ 3 seeds).",
        label="tab:generalization"))
print("\n=== generalization_table ===")
print(gen_table_out.to_string(index=False))

# ---------------------------------------------------------------------------
# 2) significance: proposed vs each baseline, paired by turbine (avg over seeds), per horizon
# ---------------------------------------------------------------------------
sig_rows = []
for h in HORIZONS:
    prop = (df[(df.horizon == h) & (df.model == "proposed")]
            .groupby("turbine")["mae"].mean())
    for model in MODEL_ORDER:
        if model == "proposed":
            continue
        base = (df[(df.horizon == h) & (df.model == model)]
                 .groupby("turbine")["mae"].mean())
        common = prop.index.intersection(base.index)
        a = prop.loc[common].values
        b = base.loc[common].values
        t, p = stats.ttest_rel(b, a)  # baseline - proposed paired
        improvement_pct = (b.mean() - a.mean()) / b.mean() * 100
        sig_rows.append({
            "horizon": h, "baseline": MODEL_LABEL[model], "n_turbines": len(common),
            "proposed_mae_mean": round(a.mean(), 2), "baseline_mae_mean": round(b.mean(), 2),
            "improvement_pct": round(improvement_pct, 1),
            "p_value": p, "significant_p<0.05": bool(p < 0.05),
        })
sig_table = pd.DataFrame(sig_rows)
sig_table.to_csv(TABLES_DIR / "generalization_significance.csv", index=False)
print("\n=== generalization_significance (proposed vs baselines, paired by turbine) ===")
print(sig_table.to_string(index=False))

# ---------------------------------------------------------------------------
# 3) boxplot: MAE distribution across turbines, per horizon, grouped by model
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
})
COLORS = {"proposed": "#C0392B", "dlinear": "#6C8EBF", "itransformer": "#7B4FA3", "patchtst": "#2E8B57"}

# 每台风机取 3 seed 均值,得到 10 个点 per (model, horizon)
per_turb = df.groupby(["horizon", "model", "turbine"], as_index=False)["mae"].mean()

fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), sharey=False)
for ax, h in zip(axes, HORIZONS):
    data = [per_turb[(per_turb.horizon == h) & (per_turb.model == m)]["mae"].values
            for m in MODEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.3),
                     flierprops=dict(marker="o", markersize=3, alpha=0.6))
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.35 if m != "proposed" else 0.55)
        patch.set_edgecolor(COLORS[m])
        patch.set_linewidth(1.3)
    # 叠加每台风机的散点(抖动)
    rng = np.random.default_rng(0)
    for i, m in enumerate(MODEL_ORDER, start=1):
        vals = per_turb[(per_turb.horizon == h) & (per_turb.model == m)]["mae"].values
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                   s=14, color=COLORS[m], alpha=0.85, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_title(f"h = {h}", fontsize=11, fontweight="bold")
    if ax is axes[0]:
        ax.set_ylabel("MAE across 10 turbines (kW)")

fig.suptitle("Generalization across 10 representative turbines (each point = one turbine, mean over 3 seeds)",
             fontsize=11, y=1.02)
fig.tight_layout()
out_path = FIG_DIR / "fig_generalization_box.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print("\nsaved figure:", out_path)

print("\nDONE")
