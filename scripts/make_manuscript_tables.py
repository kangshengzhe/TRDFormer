# -*- coding: utf-8 -*-
"""
为 manuscript 生成精简版 LaTeX 表(正文用):
  - 主对比表:仅 proposed vs 3 个代表性基线(DLinear/iTransformer/最强的TimesNet or best) 的 MAE (mean±std),4 步长
    完整 11 基线全指标表放附录(用原始 outputs/tables/*.tex)
  - 消融精简表:MAE only, 4 步长
  - 多风机泛化精简表:直接复用 generalization_table.csv,已经是精简格式
  - 追溯矩阵:直接复用 traceability_matrix.csv,转成规整 LaTeX

输出到 manuscript/tables/*.tex
"""
import re
from pathlib import Path
import pandas as pd

REPO = Path(r"C:\Users\kangs\Desktop\windpower_model\iTansformer_LSTM_CA_KAN-master")
TAB_SRC = REPO / "outputs/tables"
TAB_OUT = REPO / "manuscript/tables"
TAB_OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 6, 12, 24]
MAIN_BASELINES = ["dlinear", "itransformer", "patchtst", "timesnet"]
LABEL = {"proposed": "\\textbf{Proposed}", "dlinear": "DLinear", "itransformer": "iTransformer",
         "patchtst": "PatchTST", "timesnet": "TimesNet"}


def esc(s):
    return str(s).replace("_", "\\_").replace("±", "$\\pm$")


def esc_round(s, ndigits=2):
    """Round a 'mean±std' string to `ndigits` decimals before escaping,
    to keep table cells short enough to fit the page width."""
    s = str(s)
    m = re.match(r"^([+-]?\d+(?:\.\d+)?)±(\d+(?:\.\d+)?)$", s)
    if not m:
        return esc(s)
    mean, std = float(m.group(1)), float(m.group(2))
    return f"{mean:.{ndigits}f}$\\pm${std:.{ndigits}f}"


# ---------------------------------------------------------------------------
# 1) Main baseline comparison (condensed): MAE mean±std, proposed + 4 baselines, 4 horizons
# ---------------------------------------------------------------------------
# CSV has a two-row header (row0=horizon, row1=metric) and a spurious 'model'
# header-marker row before the actual data. Parse explicitly rather than
# relying on pandas' default single-row header inference.
def _load_wide_metrics_table(path):
    raw = pd.read_csv(path, header=None)
    horizons_row = raw.iloc[0, 1:].tolist()
    metrics_row = raw.iloc[1, 1:].tolist()
    cols = pd.MultiIndex.from_arrays([
        [int(float(h)) for h in horizons_row], metrics_row
    ], names=["horizon", "metric"])
    data = raw.iloc[3:].reset_index(drop=True)   # skip horizon/metric/model-marker rows
    data.columns = ["model"] + list(cols)
    data = data.set_index("model")
    return data

bdf_wide = _load_wide_metrics_table(TAB_SRC / "baseline_comparison_table.csv")

def get_metric(df, name, h, metric):
    try:
        return df.loc[name][(h, metric)]
    except KeyError:
        return None

rows = []
for name in ["proposed"] + MAIN_BASELINES:
    if name not in bdf_wide.index:
        continue
    row = {"Model": LABEL[name]}
    for h in HORIZONS:
        v = get_metric(bdf_wide, name, h, "mae")
        row[f"h{h}"] = esc_round(v) if v is not None else "NA"
    rows.append(row)
main_tab = pd.DataFrame(rows)

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Test-set MAE (kW, mean $\\pm$ std over 10 seeds) of the proposed model against four representative baselines across forecast horizons. Full results for all 11 baselines and 5 metrics are given in Appendix~\\ref{app:full_baseline}.}")
lines.append("\\label{tab:main_baseline}")
lines.append("\\small")
lines.append("\\begin{tabular}{l" + "c" * len(HORIZONS) + "}")
lines.append("\\toprule")
lines.append("Model & " + " & ".join(f"$h={h}$" for h in HORIZONS) + " \\\\")
lines.append("\\midrule")
for _, r in main_tab.iterrows():
    lines.append(f"{r['Model']} & " + " & ".join(r[f"h{h}"] for h in HORIZONS) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TAB_OUT / "main_baseline.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote main_baseline.tex")

# ---------------------------------------------------------------------------
# 2) Ablation condensed: MAE mean±std, per horizon
# ---------------------------------------------------------------------------
adf_wide = _load_wide_metrics_table(TAB_SRC / "ablation_table.csv")
ABL_LABEL = {
    "proposed": "\\textbf{Proposed (full)}",
    "ablation:vmd_off": "w/o VMD [A]",
    "ablation:itrans_off": "w/o iTransformer branch [B]",
    "ablation:lstm_off": "w/o LSTM branch",
    "ablation:fusion_concat": "Fusion: concat [C]",
    "ablation:fusion_sum": "Fusion: sum [C]",
    "ablation:fusion_cross_attention": "Fusion: cross-attn.\\ [C]",
    "ablation:head_linear": "Head: linear [C]",
    "ablation:head_mlp": "Head: MLP [C]",
}
rows = []
for key, label in ABL_LABEL.items():
    if key not in adf_wide.index:
        continue
    row = {"Variant": label}
    for h in HORIZONS:
        v = get_metric(adf_wide, key, h, "mae")
        row[f"h{h}"] = esc_round(v) if v is not None else "NA"
    rows.append(row)
abl_tab = pd.DataFrame(rows)

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Ablation study: test-set MAE (kW, mean $\\pm$ std over 10 seeds) for the full proposed model and each architectural variant. [A]/[B]/[C] mark which innovation the variant tests; \\emph{cross-attn.} denotes the fixed cross-attention fusion superseded by adaptive gating.}")
lines.append("\\label{tab:ablation}")
lines.append("\\small")
lines.append("\\begin{tabular}{p{0.34\\linewidth}cccc}")
lines.append("\\toprule")
lines.append("Variant & " + " & ".join(f"$h={h}$" for h in HORIZONS) + " \\\\")
lines.append("\\midrule")
for _, r in abl_tab.iterrows():
    lines.append(f"{r['Variant']} & " + " & ".join(r[f"h{h}"] for h in HORIZONS) + " \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TAB_OUT / "ablation.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote ablation.tex")

# ---------------------------------------------------------------------------
# 3) Traceability matrix -> clean LaTeX
# ---------------------------------------------------------------------------
tdf = pd.read_csv(TAB_SRC / "traceability_matrix.csv", index_col=0)
tdf.columns = [str(c) for c in tdf.columns]

def short_claim(s):
    s = s.split(":", 1)
    return s[0], s[1].strip() if len(s) > 1 else s[0]

lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Innovation--experiment traceability matrix: verification status of each claim per forecast horizon.}")
lines.append("\\label{tab:traceability}")
lines.append("\\small")
lines.append("\\begin{tabular}{p{0.10\\linewidth}p{0.52\\linewidth}cccc}")
lines.append("\\toprule")
lines.append("Innov. & Claim & " + " & ".join(f"$h={h}$" for h in HORIZONS) + " \\\\")
lines.append("\\midrule")
for idx, row in tdf.iterrows():
    inno, claim_txt = short_claim(idx)
    inno_short = inno.replace("Innovation ", "")
    vals = " & ".join(row[str(h)] if str(h) in row.index else row[h] for h in HORIZONS)
    lines.append(f"{esc(inno_short)} & {esc(claim_txt)} & {vals} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TAB_OUT / "traceability.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote traceability.tex")

# ---------------------------------------------------------------------------
# 4) Generalization table (already condensed) -> pivot to model-rows x horizon-cols per metric (MAE)
# ---------------------------------------------------------------------------
gdf = pd.read_csv(TAB_SRC / "generalization_table.csv")
piv = gdf.pivot(index="model", columns="horizon", values="MAE")
piv = piv.reindex(["Proposed", "DLinear", "iTransformer", "PatchTST"])
lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Generalization across 10 representative turbines: test-set MAE (kW, mean $\\pm$ std over 10 turbines $\\times$ 3 seeds).}")
lines.append("\\label{tab:generalization}")
lines.append("\\small")
lines.append("\\begin{tabular}{l" + "c" * len(HORIZONS) + "}")
lines.append("\\toprule")
lines.append("Model & " + " & ".join(f"$h={h}$" for h in HORIZONS) + " \\\\")
lines.append("\\midrule")
for name in piv.index:
    label = "\\textbf{Proposed}" if name == "Proposed" else name
    vals = " & ".join(esc_round(piv.loc[name, h]) for h in HORIZONS)
    lines.append(f"{label} & {vals} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TAB_OUT / "generalization.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote generalization.tex")

# ---------------------------------------------------------------------------
# 5) Win-rate table (multi-turbine)
# ---------------------------------------------------------------------------
wdf = pd.read_csv(TAB_SRC / "generalization_winrate.csv")
lines = []
lines.append("\\begin{table}[htbp]")
lines.append("\\centering")
lines.append("\\caption{Per-turbine win rate: number of the 10 representative turbines on which the proposed model achieves lower MAE than each baseline.}")
lines.append("\\label{tab:winrate}")
lines.append("\\begin{tabular}{lcccc}")
lines.append("\\toprule")
lines.append("Baseline & $h=1$ & $h=6$ & $h=12$ & $h=24$ \\\\")
lines.append("\\midrule")
for base in ["DLinear", "iTransformer", "PatchTST"]:
    sub = wdf[wdf.baseline == base].set_index("horizon")
    vals = " & ".join(f"{int(sub.loc[h,'proposed_wins'])}/{int(sub.loc[h,'n_turbines'])}" for h in HORIZONS)
    lines.append(f"{base} & {vals} \\\\")
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")
(TAB_OUT / "winrate.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote winrate.tex")

print("\nDONE")

# ---------------------------------------------------------------------------
# 6) Appendix: full baseline table (all 11 baselines, MAE/RMSE/R2), split by horizon
#    into 4 sub-tables to keep column count manageable.
# ---------------------------------------------------------------------------
ALL_MODEL_LABEL = {
    "proposed": "\\textbf{Proposed}", "lstm": "LSTM", "transformer": "Transformer",
    "informer": "Informer", "fedformer": "FEDformer", "dlinear": "DLinear",
    "patchtst": "PatchTST", "itransformer": "iTransformer", "timesnet": "TimesNet",
    "autoformer": "Autoformer", "nonstationary_transformer": "Nonstationary Transf.",
    "timexer": "TimeXer",
}
lines = []
for h in HORIZONS:
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Full comparison against 11 SOTA baselines at horizon $h={h}$ (mean $\\pm$ std over 10 seeds).}}")
    lines.append(f"\\label{{tab:appendix_baseline_h{h}}}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Model & MAE (kW) & RMSE (kW) & $R^2$ \\\\")
    lines.append("\\midrule")
    for key, label in ALL_MODEL_LABEL.items():
        if key not in bdf_wide.index:
            continue
        mae_v = get_metric(bdf_wide, key, h, "mae")
        rmse_v = get_metric(bdf_wide, key, h, "rmse")
        r2_v = get_metric(bdf_wide, key, h, "r2")
        mae = esc_round(mae_v) if mae_v is not None else "NA"
        rmse = esc_round(rmse_v) if rmse_v is not None else "NA"
        r2 = esc_round(r2_v, ndigits=3) if r2_v is not None else "NA"
        lines.append(f"{label} & {mae} & {rmse} & {r2} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
(TAB_OUT / "appendix_baseline_full.tex").write_text("\n".join(lines), encoding="utf-8")
print("wrote appendix_baseline_full.tex")
print("\nALL DONE")
