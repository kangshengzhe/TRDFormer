# -*- coding: utf-8 -*-
"""
论文级图表批量生成脚本
======================
从 outputs/runs/run_records.jsonl(840条,10种子)与各 *_preds.npz 生成一套
投稿用高清图(300 dpi),覆盖:基线对比、消融、多指标雷达、改进热力图、
预测散点、误差分布、种子稳定性、VMD分解示意、指标-步长趋势等。

用法:
    python -m scripts.make_paper_figures
输出:outputs/figures/paper/*.png
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

warnings.filterwarnings("ignore")

# ── 全局风格:干净、期刊风 ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "outputs" / "runs"
RECORDS = RUNS / "run_records.jsonl"
MANIFESTS = ROOT / "outputs" / "manifests"
OUT = ROOT / "outputs" / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 6, 12, 24]
PROPOSED = "proposed"
BASELINES = ["lstm", "transformer", "informer", "fedformer", "dlinear",
             "patchtst", "itransformer", "timesnet", "autoformer",
             "nonstationary_transformer", "timexer"]
# 展示名(更规范的论文写法)
DISPLAY = {
    "proposed": "Proposed", "lstm": "LSTM", "transformer": "Transformer",
    "informer": "Informer", "fedformer": "FEDformer", "dlinear": "DLinear",
    "patchtst": "PatchTST", "itransformer": "iTransformer", "timesnet": "TimesNet",
    "autoformer": "Autoformer", "nonstationary_transformer": "NSTransformer",
    "timexer": "TimeXer",
}
PROPOSED_COLOR = "#d62728"   # 红色高亮 proposed


# ── 数据加载 ───────────────────────────────────────────────────────────────
def load_records():
    recs = []
    with open(RECORDS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return [r for r in recs if r.get("status") == "success"]


def metric_by_model_horizon(recs, metric):
    """返回 {model: {horizon: (mean, std)}}。"""
    bucket = defaultdict(lambda: defaultdict(list))
    for r in recs:
        m = r.get("metrics", {}).get(metric)
        if m is not None and np.isfinite(m):
            bucket[r["model_name"]][int(r["horizon"])].append(float(m))
    out = {}
    for model, hd in bucket.items():
        out[model] = {h: (float(np.mean(v)), float(np.std(v))) for h, v in hd.items()}
    return out


def load_preds(run_id):
    p = RUNS / f"{run_id}_preds.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["predictions"], d["actuals"]


RECS = load_records()
MAE = metric_by_model_horizon(RECS, "mae")
RMSE = metric_by_model_horizon(RECS, "rmse")
R2 = metric_by_model_horizon(RECS, "r2")
SMAPE = metric_by_model_horizon(RECS, "smape")


# 柔和配色(proposed 之外的对比模型),期刊风:低饱和、易区分
SOFT_PALETTE = {
    "dlinear": "#3B6FB6", "lstm": "#E8A33D", "patchtst": "#2E8B57",
    "itransformer": "#7B4FA3", "timesnet": "#1B9E9E", "transformer": "#C9A227",
    "informer": "#C64B8C", "fedformer": "#8C8C8C", "autoformer": "#9C6644",
    "nonstationary_transformer": "#5B8C5A",
}
MARKERS = {"dlinear": "s", "lstm": "^", "patchtst": "D", "itransformer": "v",
           "timesnet": "P", "transformer": "X", "informer": "*"}


# ── 图1:指标-步长趋势(MAE / RMSE / R²,带 ±std 阴影带) ────────────────────
def fig_metrics_vs_horizon():
    show = ["proposed", "dlinear", "lstm", "patchtst", "itransformer", "timesnet"]
    panels = [("mae", MAE, "MAE (kW)"), ("rmse", RMSE, "RMSE (kW)"), ("r2", R2, "R$^2$")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    x = np.array(HORIZONS)
    for ax, (key, data, ylab) in zip(axes, panels):
        for model in show:
            if model not in data:
                continue
            means = np.array([data[model].get(h, (np.nan, 0))[0] for h in HORIZONS])
            stds = np.array([data[model].get(h, (np.nan, 0))[1] for h in HORIZONS])
            is_prop = model == "proposed"
            c = PROPOSED_COLOR if is_prop else SOFT_PALETTE.get(model, "#888888")
            # ±std 阴影带(呼应残差带的视觉语言)
            ax.fill_between(x, means - stds, means + stds,
                            color=c, alpha=0.18 if is_prop else 0.10, lw=0, zorder=1)
            ax.plot(x, means, marker="o" if is_prop else MARKERS.get(model, "s"),
                    ms=7 if is_prop else 5,
                    lw=2.8 if is_prop else 1.6, color=c,
                    zorder=6 if is_prop else 3,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=DISPLAY[model] + (" (Ours)" if is_prop else ""))
        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel(ylab)
        ax.set_xticks(HORIZONS)
        ax.margins(x=0.04)
    axes[0].legend(ncol=2, fontsize=8.5, loc="upper left")
    fig.suptitle("Metric trends across forecast horizons (shaded band = ±1 std over 10 seeds)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_metrics_vs_horizon.png", bbox_inches="tight")
    plt.close(fig)


# ── 图2:基线对比分组柱状图(每个步长一组,带误差棒;proposed高亮+数值标注) ──
def fig_baseline_bars():
    models = [PROPOSED] + BASELINES
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for ax, h in zip(axes.flat, HORIZONS):
        means = [MAE.get(m, {}).get(h, (np.nan, 0))[0] for m in models]
        stds = [MAE.get(m, {}).get(h, (np.nan, 0))[1] for m in models]
        # 排序:proposed 固定第一,其余按 MAE 升序
        order = [0] + (1 + np.argsort([means[i] for i in range(1, len(models))])).tolist()
        om = [models[i] for i in order]
        omean = [means[i] for i in order]
        ostd = [stds[i] for i in order]
        colors = [PROPOSED_COLOR if m == PROPOSED else "#6C8EBF" for m in om]
        # timexer 是发散的异常大值,截断显示以免压扁其它
        ymax = np.nanpercentile([m for m in omean if np.isfinite(m)], 90) * 1.55
        bars = ax.bar(range(len(om)), omean, yerr=ostd, color=colors,
                      capsize=3, edgecolor="white", linewidth=0.8,
                      error_kw=dict(ecolor="0.35", lw=1), zorder=3)
        # proposed 柱描黑边强调
        bars[0].set_edgecolor("black"); bars[0].set_linewidth(1.2)
        ax.set_xticks(range(len(om)))
        ax.set_xticklabels([DISPLAY[m] + (" (Ours)" if m == PROPOSED else "")
                            for m in om], rotation=45, ha="right", fontsize=8.5)
        # 高亮 proposed 的刻度标签
        ax.get_xticklabels()[0].set_color(PROPOSED_COLOR)
        ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_ylabel("MAE (kW)")
        ax.set_title(f"Horizon = {h} step" + ("s" if h > 1 else ""),
                     fontsize=12, fontweight="bold")
        ax.set_ylim(0, ymax)
        # 数值标注:正常柱标在柱顶上方;被截断的超大值旋转标注
        for b, v in zip(bars, omean):
            if not np.isfinite(v):
                continue
            if v > ymax:
                ax.text(b.get_x() + b.get_width()/2, ymax*0.96, f"{v:.0f}",
                        ha="center", va="top", fontsize=7.5, rotation=90, color="0.2")
            else:
                ax.text(b.get_x() + b.get_width()/2, v + ymax*0.015, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=7.5, color="0.25")
    fig.suptitle("MAE comparison against 11 SOTA baselines "
                 "(sorted ascending; mean ± std over 10 seeds)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_baseline_bars.png", bbox_inches="tight")
    plt.close(fig)


# ── 图3:改进百分比热力图(proposed 相对每个基线的 MAE 降幅%) ─────────────
def fig_improvement_heatmap():
    mat = np.full((len(BASELINES), len(HORIZONS)), np.nan)
    for i, b in enumerate(BASELINES):
        for j, h in enumerate(HORIZONS):
            pm = MAE.get(PROPOSED, {}).get(h, (np.nan, 0))[0]
            bm = MAE.get(b, {}).get(h, (np.nan, 0))[0]
            if np.isfinite(pm) and np.isfinite(bm) and bm != 0:
                mat[i, j] = (bm - pm) / bm * 100.0  # 正=proposed更好
    fig, ax = plt.subplots(figsize=(7, 8))
    vmax = np.nanpercentile(np.abs(mat), 95)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_yticks(range(len(BASELINES)))
    ax.set_yticklabels([DISPLAY[b] for b in BASELINES])
    for i in range(len(BASELINES)):
        for j in range(len(HORIZONS)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title("MAE improvement of Proposed over each baseline (%)\n"
                 "green = Proposed better", fontsize=11)
    fig.colorbar(im, ax=ax, label="MAE reduction (%)", shrink=0.7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_improvement_heatmap.png", bbox_inches="tight")
    plt.close(fig)


# ── 图4:多指标雷达图(proposed vs 强基线,h=12) ─────────────────────────
def fig_radar(h=12):
    show = ["proposed", "dlinear", "itransformer", "patchtst", "timesnet"]
    # 指标:MAE, RMSE, SMAPE 越小越好 → 取倒数归一;R² 越大越好
    metrics = [("MAE", MAE, False), ("RMSE", RMSE, False),
               ("SMAPE", SMAPE, False), ("R$^2$", R2, True)]
    # 归一化到 [0,1](1=最好)
    norm = {m: [] for m in show}
    labels = []
    for name, data, higher_better in metrics:
        labels.append(name)
        vals = {m: data.get(m, {}).get(h, (np.nan, 0))[0] for m in show}
        finite = [v for v in vals.values() if np.isfinite(v)]
        lo, hi = min(finite), max(finite)
        for m in show:
            v = vals[m]
            if not np.isfinite(v) or hi == lo:
                score = 0.5
            elif higher_better:
                score = (v - lo) / (hi - lo)
            else:
                score = (hi - v) / (hi - lo)
            norm[m].append(score)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for m in show:
        vals = norm[m] + norm[m][:1]
        is_prop = m == "proposed"
        ax.plot(angles, vals, marker="o", lw=2.5 if is_prop else 1.5,
                color=PROPOSED_COLOR if is_prop else None, label=DISPLAY[m],
                zorder=5 if is_prop else 2)
        if is_prop:
            ax.fill(angles, vals, color=PROPOSED_COLOR, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title(f"Normalized multi-metric comparison (horizon = {h})\n"
                 "farther from center = better", fontsize=11, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_radar_multimetric.png", bbox_inches="tight")
    plt.close(fig)


# ── 图5:消融研究(按创新点A/B/C分三面板,MAE-步长折线 + ±std阴影带) ────────
def fig_ablation_bars():
    # 每个创新点一个面板;proposed 作为红色高亮参考线,消融变体柔和配色
    groups = {
        "Innovation A — VMD decomposition": [
            ("ablation:vmd_off", "w/o VMD", "#3B6FB6"),
        ],
        "Innovation B — dual-branch encoder": [
            ("ablation:itrans_off", "w/o iTransformer branch", "#C64B8C"),
            ("ablation:lstm_off", "w/o LSTM branch", "#2E8B57"),
        ],
        "Innovation C — gated fusion + KAN head": [
            ("ablation:fusion_concat", "Concat fusion", "#E8A33D"),
            ("ablation:fusion_sum", "Sum fusion", "#7B4FA3"),
            ("ablation:fusion_cross_attention", "CrossAttn fusion", "#1B9E9E"),
            ("ablation:head_linear", "Linear head", "#9C6644"),
            ("ablation:head_mlp", "MLP head", "#8C8C8C"),
        ],
    }
    x = np.array(HORIZONS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    def _plot_line(ax, model, label, color, is_prop=False):
        means = np.array([MAE.get(model, {}).get(h, (np.nan, 0))[0] for h in HORIZONS])
        stds = np.array([MAE.get(model, {}).get(h, (np.nan, 0))[1] for h in HORIZONS])
        ax.fill_between(x, means - stds, means + stds, color=color,
                        alpha=0.18 if is_prop else 0.10, lw=0, zorder=1)
        ax.plot(x, means, marker="o" if is_prop else "s",
                ms=7 if is_prop else 5, lw=2.8 if is_prop else 1.7, color=color,
                zorder=6 if is_prop else 3, markeredgecolor="white",
                markeredgewidth=0.6, label=label)

    for ax, (title, variants) in zip(axes, groups.items()):
        # proposed 参考(红色高亮)
        _plot_line(ax, PROPOSED, "Proposed (full)", PROPOSED_COLOR, is_prop=True)
        for model, label, color in variants:
            _plot_line(ax, model, label, color)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast horizon (steps)")
        ax.set_ylabel("MAE (kW)")
        ax.set_xticks(HORIZONS)
        ax.margins(x=0.04)
        ax.legend(fontsize=8.2, loc="upper left")
    fig.suptitle("Ablation study: removing each component vs. the full Proposed model "
                 "(shaded band = ±1 std over 10 seeds)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_ablation_bars.png", bbox_inches="tight")
    plt.close(fig)


# ── 图6:预测值 vs 真实值 散点(hexbin密度,4个步长) ─────────────────────
def fig_scatter_pred_actual():
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for ax, h in zip(axes.flat, HORIZONS):
        preds, actuals = load_preds(f"proposed_h{h}_seed42")
        if preds is None:
            continue
        p = preds.flatten(); a = actuals.flatten()
        hb = ax.hexbin(a, p, gridsize=55, cmap="mako" if "mako" in plt.colormaps() else "viridis",
                       mincnt=1, bins="log", linewidths=0.0)
        lim = [min(a.min(), p.min()), max(a.max(), p.max())]
        # 理想线用 proposed 红色,统一主色调
        ax.plot(lim, lim, ls="--", lw=1.8, color=PROPOSED_COLOR, label="Ideal (y = x)", zorder=5)
        r2 = R2.get("proposed", {}).get(h, (np.nan, 0))[0]
        mae = MAE.get("proposed", {}).get(h, (np.nan, 0))[0]
        rmse = RMSE.get("proposed", {}).get(h, (np.nan, 0))[0]
        # 指标注释框(右下角)
        ax.text(0.97, 0.05, f"R$^2$ = {r2:.3f}\nMAE = {mae:.1f}\nRMSE = {rmse:.1f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.6", alpha=0.85))
        ax.set_xlabel("Actual power (kW)")
        ax.set_ylabel("Predicted power (kW)")
        ax.set_title(f"Horizon = {h} step" + ("s" if h > 1 else ""),
                     fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2, ls="--")
        cb = fig.colorbar(hb, ax=ax, label="Point density (log$_{10}$ count)", shrink=0.82)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Predicted vs. actual active power (Proposed model, denser = darker)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_scatter_pred_actual.png", bbox_inches="tight")
    plt.close(fig)


# ── 图7:误差分布小提琴图(proposed vs 强基线,h=12) ───────────────────────
def fig_error_violin(h=12):
    show = ["proposed", "dlinear", "lstm", "patchtst", "itransformer", "timesnet"]
    data, labels, keys = [], [], []
    for m in show:
        preds, actuals = load_preds(f"{m}_h{h}_seed42")
        if preds is None:
            continue
        err = np.abs(preds.flatten() - actuals.flatten())
        if err.size > 20000:
            err = np.random.RandomState(0).choice(err, 20000, replace=False)
        data.append(err); labels.append(DISPLAY[m]); keys.append(m)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    parts = ax.violinplot(data, showmedians=False, showextrema=False, widths=0.85)
    for i, pc in enumerate(parts["bodies"]):
        is_prop = keys[i] == "proposed"
        pc.set_facecolor(PROPOSED_COLOR if is_prop else SOFT_PALETTE.get(keys[i], "#6C8EBF"))
        pc.set_alpha(0.55 if not is_prop else 0.7)
        pc.set_edgecolor("black" if is_prop else "0.4")
        pc.set_linewidth(1.1 if is_prop else 0.6)
    # 叠加均值点 + 中位数横线,增强可读性
    for i, d in enumerate(data, start=1):
        med, mean = np.median(d), np.mean(d)
        ax.hlines(med, i-0.35, i+0.35, color="black", lw=1.4, zorder=5)
        ax.scatter(i, mean, marker="D", s=28, color="white",
                   edgecolor="black", lw=1.0, zorder=6)
    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels([l + (" (Ours)" if keys[i] == "proposed" else "")
                        for i, l in enumerate(labels)], rotation=25, ha="right")
    ax.get_xticklabels()[keys.index("proposed")].set_color(PROPOSED_COLOR)
    ax.get_xticklabels()[keys.index("proposed")].set_fontweight("bold")
    ax.set_ylabel("Absolute error (kW)")
    ax.set_title(f"Absolute-error distribution at horizon = {h}  "
                 "(— median,  ◇ mean)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, np.percentile(np.concatenate(data), 97))
    ax.grid(True, axis="y", alpha=0.25, ls="--")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_error_violin.png", bbox_inches="tight")
    plt.close(fig)


# ── 图8:种子稳定性箱线图(proposed vs 强基线,各步长的每种子MAE) ─────────
def fig_seed_stability():
    show = ["proposed", "dlinear", "lstm", "itransformer", "timesnet"]
    perseed = defaultdict(lambda: defaultdict(list))
    for r in RECS:
        if r["model_name"] in show:
            mae = r["metrics"].get("mae")
            if mae is not None and np.isfinite(mae):
                perseed[r["model_name"]][int(r["horizon"])].append(float(mae))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=False)
    for ax, h in zip(axes, HORIZONS):
        box_data = [perseed[m].get(h, []) for m in show]
        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(PROPOSED_COLOR if show[i] == "proposed" else "#4c72b0")
            box.set_alpha(0.6)
        ax.set_xticks(range(1, len(show)+1))
        ax.set_xticklabels([DISPLAY[m] for m in show], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("MAE (kW)")
        ax.set_title(f"Horizon = {h}")
    fig.suptitle("Per-seed MAE spread (10 seeds) — lower & tighter is better",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_seed_stability.png", bbox_inches="tight")
    plt.close(fig)


# ── 图9:VMD 分解示意(Patv + K个IMF,一个样本窗口) ───────────────────────
def fig_vmd_decomposition():
    imf_path = MANIFESTS / "vmd_imfs.npz"
    if not imf_path.exists():
        print("  [9] skip VMD (no vmd_imfs.npz)")
        return
    d = np.load(imf_path)
    key = "all_imfs" if "all_imfs" in d else list(d.keys())[0]
    imfs = d[key]  # (N, K)
    win = slice(0, 288)  # 两天(288个10min点)
    seg = imfs[win]
    K = seg.shape[1]
    fig, axes = plt.subplots(K + 1, 1, figsize=(11, 1.6*(K+1)), sharex=True)
    total = seg.sum(axis=1)
    axes[0].plot(total, color="black", lw=1.2)
    axes[0].set_ylabel("Sum", fontsize=9)
    axes[0].set_title("VMD decomposition of the active-power series (first 2 days)", fontsize=12)
    cmap = cm.get_cmap("viridis", K)
    for k in range(K):
        axes[k+1].plot(seg[:, k], color=cmap(k), lw=1.0)
        axes[k+1].set_ylabel(f"IMF {k+1}", fontsize=9)
    axes[-1].set_xlabel("Time step (10-min interval)")
    fig.tight_layout()
    fig.savefig(OUT / "fig9_vmd_decomposition.png", bbox_inches="tight")
    plt.close(fig)


# ── 图10:多步预测曲线叠加(proposed vs actual vs 最强简单基线) ────────────
def fig_prediction_overlay():
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    n_show = 300
    for ax, h in zip(axes.flat, HORIZONS):
        pp, aa = load_preds(f"proposed_h{h}_seed42")
        dp, _ = load_preds(f"dlinear_h{h}_seed42")
        if pp is None:
            continue
        # 取每个窗口的第一步预测,拼成一条时间线
        a = aa[:n_show, 0]
        p = pp[:n_show, 0]
        ax.plot(a, color="black", lw=1.6, label="Actual", zorder=3)
        ax.plot(p, color=PROPOSED_COLOR, lw=1.3, label="Proposed", alpha=0.9)
        if dp is not None:
            ax.plot(dp[:n_show, 0], color="#4c72b0", lw=1.0, ls="--",
                    label="DLinear", alpha=0.7)
        ax.set_title(f"Horizon = {h} step" + ("s" if h > 1 else ""))
        ax.set_xlabel("Test time step")
        ax.set_ylabel("Power (kW)")
        ax.legend(fontsize=8, ncol=3)
    fig.suptitle("Predicted vs. actual active power over the test set", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_prediction_overlay.png", bbox_inches="tight")
    plt.close(fig)


# ── 图11:期刊同款「多面板 + 残差阴影带 + 双轴」预测对比图 ──────────────────
# 学习 W. Zhang et al., Applied Energy 2026 的 Fig.5 风格:
#   每个模型一个子图;左轴=功率(真实值黑虚线 + 预测彩色实线);
#   右轴=残差,以浅灰色阴影带(fill_between)铺在曲线后方。
def fig_prediction_panels(h=12, n_show=1200):
    panel_models = ["proposed", "lstm", "dlinear", "patchtst",
                    "itransformer", "timesnet", "transformer", "timexer"]
    palette = {
        "proposed": "#8B1A1A", "lstm": "#E8A33D", "dlinear": "#3B6FB6",
        "patchtst": "#2E8B57", "itransformer": "#7B4FA3", "timesnet": "#1B9E9E",
        "transformer": "#C9A227", "timexer": "#C64B8C",
    }
    title_of = dict(DISPLAY); title_of["proposed"] = "Ours"

    # 先确定统一的功率轴范围与残差轴范围(所有面板一致,便于对比)
    series = {}
    for m in panel_models:
        pp, aa = load_preds(f"{m}_h{h}_seed42")
        if pp is None:
            continue
        # 用第 h 步(最后一列)——最难、最能区分模型,且对得起 "horizon=h" 标签
        a = aa[:n_show, -1]
        p = pp[:n_show, -1]
        series[m] = (a, p, p - a)
    all_pow = np.concatenate([np.concatenate([a, p]) for a, p, _ in series.values()])
    pow_lo, pow_hi = np.nanmin(all_pow), np.nanmax(all_pow)
    pow_pad = (pow_hi - pow_lo) * 0.05
    all_res = np.concatenate([r for _, _, r in series.values()])
    res_absmax = np.nanpercentile(np.abs(all_res), 99)

    fig, axes = plt.subplots(4, 2, figsize=(16, 13), sharex=True)
    x = np.arange(n_show)
    for ax, m in zip(axes.flat, panel_models):
        if m not in series:
            ax.set_visible(False); continue
        a, p, res = series[m]
        c = palette[m]

        # 右轴:残差阴影带(铺在最底层)
        ax2 = ax.twinx()
        ax2.fill_between(x, res, 0, color="0.6", alpha=0.35, lw=0, zorder=1,
                         label="Residual")
        # 残差轴范围放大,使阴影带落在面板下半部,不遮挡功率曲线
        ax2.set_ylim(-res_absmax, res_absmax * 3.2)
        ax2.set_ylabel("Residual (kW)", fontsize=9)
        ax2.tick_params(labelsize=8)

        # 左轴:真实值(黑虚线) + 预测(彩色实线),置于阴影带之上
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ln_true, = ax.plot(x, a, color="black", ls="--", lw=0.8, zorder=3,
                           label="True Values")
        ln_pred, = ax.plot(x, p, color=c, lw=1.3 if m == "proposed" else 1.0,
                           zorder=2, label=title_of[m])
        ax.set_ylim(pow_lo - pow_pad, pow_hi + pow_pad * 3)
        ax.set_ylabel("Power (kW)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(title_of[m], fontsize=12, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, ls="--")

        # 顶部合并图例(真实值 / 模型 / 残差)
        res_patch = plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.35)
        ax.legend([ln_true, ln_pred, res_patch],
                  ["True Values", title_of[m], "Residual"],
                  loc="upper left", ncol=3, fontsize=8,
                  handlelength=1.6, columnspacing=1.2)

    for ax in axes[-1]:
        ax.set_xlabel(f"Test time step (horizon = {h})", fontsize=10)
    fig.suptitle("Forecasting performance comparison: prediction, ground truth, and residual",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(OUT / "fig11_prediction_panels.png", bbox_inches="tight")
    plt.close(fig)


# ── 图: h=1 vs h=24 对比（proposed vs DLinear 预测曲线 + 标题内嵌真实MAE）──
def fig_h1_vs_h24_contrast(n_show=1200, seed=42):
    """两行面板：上 h=1，下 h=24；各含 proposed(红) + DLinear(蓝) + 真实值(黑虚线）
    标题里的 MAE 数字直接从 run_records 读取，保证与实验数据一致。"""

    # 从 run_records 读各模型各步长的平均 MAE（10 种子）
    mae_lookup: dict[tuple, float] = {}
    with open(RECORDS) as f:
        recs_by_key: dict[tuple, list] = defaultdict(list)
        for line in f:
            r = json.loads(line)
            recs_by_key[(r["model_name"], r["horizon"])].append(r["metrics"]["mae"])
    for (m, h), vals in recs_by_key.items():
        mae_lookup[(m, h)] = float(np.mean(vals))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, h in zip(axes, [1, 24]):
        pp, aa = load_preds(f"proposed_h{h}_seed{seed}")
        dp, _  = load_preds(f"dlinear_h{h}_seed{seed}")
        if pp is None:
            ax.set_visible(False)
            continue

        # 取每个预测窗口的最后一步（第 h 步），最能体现模型在该 horizon 下的真实能力
        a = aa[:n_show, -1]
        p = pp[:n_show, -1]
        x = np.arange(len(a))

        mae_prop   = mae_lookup.get(("proposed", h), float("nan"))
        mae_dlin   = mae_lookup.get(("dlinear",  h), float("nan"))

        if h == 1:
            subtitle = (
                f"$h = 1$ (10-min ahead): DLinear tracks the signal more closely "
                f"than the proposed model\n"
                f"(full test-set MAE: Proposed={mae_prop:.1f} kW, "
                f"DLinear={mae_dlin:.1f} kW)"
            )
        else:
            subtitle = (
                f"$h = 24$ (4-h ahead): the proposed model tracks the signal "
                f"markedly better than DLinear\n"
                f"(full test-set MAE: Proposed={mae_prop:.1f} kW, "
                f"DLinear={mae_dlin:.1f} kW)"
            )

        ax.plot(x, a, color="black", ls="--", lw=0.9, label="True values", zorder=3)
        ax.plot(x, p, color="#d62728", lw=1.2, label="Proposed", zorder=2)
        if dp is not None:
            ax.plot(x, dp[:n_show, 0], color="#4472C4", lw=1.0,
                    label="DLinear", zorder=2, alpha=0.85)

        ax.set_title(subtitle, fontsize=10, pad=6)
        ax.set_ylabel("Power (kW)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper right", fontsize=9, ncol=3)
        ax.set_xlim(0, n_show)

    axes[-1].set_xlabel("Test time step", fontsize=10)
    fig.suptitle("Contrast between the shortest and longest forecast horizons",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_h1_vs_h24_contrast.png", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  预测曲线图（时序预测论文核心定性证据）P1 / P2 / P3
# ═══════════════════════════════════════════════════════════════════════════

# ── P1: proposed 模型四步长全景（2×2，每格一个 h，proposed vs 真实值 + 残差带）─
def fig_proposed_horizons_panorama(n_show=1200, seed=42):
    """展示 proposed 模型在全部 4 个步长的预测能力。
    每个子图：真实值(黑虚线) + proposed(红实线) + 残差(灰色阴影带，右轴)。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    horizon_label = {1: "10-min ahead", 6: "1-h ahead",
                     12: "2-h ahead", 24: "4-h ahead"}

    for ax, h in zip(axes.flat, HORIZONS):
        pp, aa = load_preds(f"proposed_h{h}_seed{seed}")
        if pp is None:
            ax.set_visible(False); continue
        # 取第 h 步（最后一列）——最能反映该 horizon 的真实预测能力
        a = aa[:n_show, -1]
        p = pp[:n_show, -1]
        res = p - a
        x = np.arange(len(a))

        mae_mean = MAE.get("proposed", {}).get(h, (float("nan"),))[0]
        r2_mean  = R2.get("proposed", {}).get(h, (float("nan"),))[0]

        # 右轴残差阴影带
        ax2 = ax.twinx()
        res_absmax = np.nanpercentile(np.abs(res), 99)
        ax2.fill_between(x, res, 0, color="0.6", alpha=0.32, lw=0, zorder=1)
        ax2.set_ylim(-res_absmax, res_absmax * 3.2)
        ax2.set_ylabel("Residual (kW)", fontsize=9)
        ax2.tick_params(labelsize=8)

        # 左轴功率曲线
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ln_true, = ax.plot(x, a, color="black", ls="--", lw=0.8, zorder=3,
                           label="True values")
        ln_pred, = ax.plot(x, p, color=PROPOSED_COLOR, lw=1.2, zorder=2,
                           label="Proposed")
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Power (kW)", fontsize=9)
        ax.set_xlabel("Test time step", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(
            f"$h = {h}$ ({horizon_label[h]})   "
            f"MAE={mae_mean:.1f} kW,  $R^2$={r2_mean:.3f}",
            fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, alpha=0.25, ls="--")

        res_patch = plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.32)
        ax.legend([ln_true, ln_pred, res_patch],
                  ["True values", "Proposed", "Residual"],
                  loc="upper left", ncol=3, fontsize=8,
                  handlelength=1.6, columnspacing=1.2)

    fig.suptitle("Proposed model forecasts across all horizons "
                 "(prediction, ground truth, and residual)",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "fig_proposed_horizons_panorama.png", bbox_inches="tight")
    plt.close(fig)


# ── P2: 8 模型面板对比（补齐 h=1/6/24，与现有 fig11 的 h=12 成套）───────────
def fig_prediction_panels_multi(n_show=1200):
    """对 h ∈ {1, 6, 24} 各生成一张 8 模型面板图，与 fig11(h=12) 风格一致。"""
    for h in [1, 6, 24]:
        fig_prediction_panels(h=h, n_show=n_show)
        # fig_prediction_panels 固定写到 fig11_prediction_panels.png，
        # 这里改为按 horizon 命名，避免互相覆盖
        src = OUT / "fig11_prediction_panels.png"
        dst = OUT / f"fig_prediction_panels_h{h}.png"
        if src.exists():
            import shutil
            shutil.copy(src, dst)
    # 最后把 h=12 重新生成回 fig11（保持默认产物一致）
    fig_prediction_panels(h=12, n_show=n_show)


# ── 附录: 全部 11 个 baseline + proposed 完整面板（4×3）────────────────────
def fig_prediction_panels_full(h=12, n_show=1200):
    """正文精选 8 个之外，附录补充全部 11 baseline + proposed 共 12 个面板。
    布局 4 行 × 3 列，与正文风格完全一致。"""
    all_models = [
        "proposed",
        "lstm",       "dlinear",     "transformer",
        "informer",   "fedformer",   "autoformer",
        "patchtst",   "itransformer","timesnet",
        "nonstationary_transformer", "timexer",
    ]
    palette = {
        "proposed":                  "#8B1A1A",
        "lstm":                      "#E8A33D",
        "dlinear":                   "#3B6FB6",
        "transformer":               "#C9A227",
        "informer":                  "#C64B8C",
        "fedformer":                 "#8C8C8C",
        "autoformer":                "#9C6644",
        "patchtst":                  "#2E8B57",
        "itransformer":              "#7B4FA3",
        "timesnet":                  "#1B9E9E",
        "nonstationary_transformer": "#5B8C5A",
        "timexer":                   "#E05C5C",
    }
    title_of = dict(DISPLAY); title_of["proposed"] = "Ours"

    series = {}
    for m in all_models:
        pp, aa = load_preds(f"{m}_h{h}_seed42")
        if pp is None:
            continue
        a = aa[:n_show, -1]
        p = pp[:n_show, -1]
        series[m] = (a, p, p - a)

    all_pow = np.concatenate([np.concatenate([a, p]) for a, p, _ in series.values()])
    pow_lo, pow_hi = np.nanmin(all_pow), np.nanmax(all_pow)
    pow_pad = (pow_hi - pow_lo) * 0.05
    all_res = np.concatenate([r for _, _, r in series.values()])
    res_absmax = np.nanpercentile(np.abs(all_res), 99)

    fig, axes = plt.subplots(4, 3, figsize=(21, 17), sharex=True)
    x = np.arange(n_show)
    for ax, m in zip(axes.flat, all_models):
        if m not in series:
            ax.set_visible(False); continue
        a, p, res = series[m]
        c = palette.get(m, "#555555")

        ax2 = ax.twinx()
        ax2.fill_between(x, res, 0, color="0.6", alpha=0.35, lw=0, zorder=1)
        ax2.set_ylim(-res_absmax, res_absmax * 3.2)
        ax2.set_ylabel("Residual (kW)", fontsize=8)
        ax2.tick_params(labelsize=7)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ln_true, = ax.plot(x, a, color="black", ls="--", lw=0.8, zorder=3,
                           label="True Values")
        lw = 1.4 if m == "proposed" else 1.0
        ln_pred, = ax.plot(x, p, color=c, lw=lw, zorder=2,
                           label=title_of.get(m, m))
        ax.set_ylim(max(0, pow_lo - pow_pad), pow_hi + pow_pad * 3)
        ax.set_ylabel("Power (kW)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title_of.get(m, m), fontsize=11,
                     fontweight="bold" if m == "proposed" else "normal", pad=5)
        ax.grid(True, alpha=0.25, ls="--")

        res_patch = plt.Rectangle((0, 0), 1, 1, fc="0.6", alpha=0.35)
        ax.legend([ln_true, ln_pred, res_patch],
                  ["True Values", title_of.get(m, m), "Residual"],
                  loc="upper left", ncol=3, fontsize=7,
                  handlelength=1.4, columnspacing=1.0)

    for ax in axes[-1]:
        ax.set_xlabel(f"Test time step (horizon = {h})", fontsize=9)
    fig.suptitle(
        f"Complete forecasting comparison: proposed model and all 11 baselines "
        f"(horizon $h={h}$, prediction · ground truth · residual)",
        fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fname = OUT / f"fig_prediction_panels_full_h{h}.png"
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# ── overlay: 多模型单坐标系叠加（spaghetti plot，proposed 高亮）───────────
def fig_prediction_overlay_multi(h=12, seed=42, start=550, span=350):
    """所有模型 + 真实值叠在同一坐标系。真实值黑色醒目，proposed 加粗红色，
    其余基线低饱和细线。截取一段（默认含大爬坡的活跃区间）以保证可读性。"""
    overlay_models = [
        "proposed", "dlinear", "lstm", "patchtst", "itransformer",
        "timesnet", "transformer", "timexer",
    ]
    # proposed 红色高亮，其余低饱和
    line_style = {
        "proposed":     dict(color="#d62728", lw=2.4, alpha=0.95, zorder=5),
        "dlinear":      dict(color="#4C72B0", lw=1.0, alpha=0.65, zorder=2),
        "lstm":         dict(color="#DD8452", lw=1.0, alpha=0.65, zorder=2),
        "patchtst":     dict(color="#55A868", lw=1.0, alpha=0.65, zorder=2),
        "itransformer": dict(color="#8172B3", lw=1.0, alpha=0.65, zorder=2),
        "timesnet":     dict(color="#64B5CD", lw=1.0, alpha=0.65, zorder=2),
        "transformer":  dict(color="#CCB974", lw=1.0, alpha=0.65, zorder=2),
        "timexer":      dict(color="#C44E9C", lw=1.0, alpha=0.65, zorder=2),
    }

    fig, ax = plt.subplots(figsize=(15, 5.2))
    a_ref = None
    s, e = start, start + span
    for m in overlay_models:
        pp, aa = load_preds(f"{m}_h{h}_seed{seed}")
        if pp is None:
            continue
        if a_ref is None:
            a_ref = aa[:, -1]
        p = pp[:, -1]
        x = np.arange(s, e)
        ax.plot(x, p[s:e], label=(DISPLAY.get(m, m) if m != "proposed" else "Proposed"),
                **line_style[m])

    # 真实值最后画（最上层，黑色粗虚线）
    if a_ref is not None:
        ax.plot(np.arange(s, e), a_ref[s:e], color="black", lw=2.0, ls="--",
                label="Actual", zorder=6)

    ax.set_ylim(bottom=0)
    ax.set_xlim(s, e)
    ax.set_xlabel("Test time step", fontsize=11)
    ax.set_ylabel("Wind power (kW)", fontsize=11)
    ax.set_title(f"All-model forecast overlay at $h={h}$ "
                 f"({ {1:'10-min',6:'1-h',12:'2-h',24:'4-h'}[h] } ahead)",
                 fontsize=12, pad=8)
    ax.grid(True, alpha=0.25, ls="--")
    ax.legend(loc="upper right", ncol=3, fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_prediction_overlay_multi.png", bbox_inches="tight")
    plt.close(fig)


# ── P3: 爬坡事件放大图（截取功率骤变片段，proposed vs 强基线）──────────────
def fig_ramp_events(h=24, seed=42, win=120, top_k=2):
    """自动检测真实功率中变化最剧烈的 ramp 片段，放大展示各模型跟踪能力。
    对比对象：proposed + DLinear + LSTM（h>1 时最强的两个简单基线）。"""
    pp, aa = load_preds(f"proposed_h{h}_seed{seed}")
    if pp is None:
        return
    a_full = aa[:, -1]
    # 用滑动窗口内的 max-min 幅度找 ramp 最剧烈的片段
    n = len(a_full)
    amp = np.array([a_full[i:i+win].max() - a_full[i:i+win].min()
                    for i in range(0, n - win, win // 2)])
    starts = np.array([i for i in range(0, n - win, win // 2)])
    order = np.argsort(amp)[::-1]

    # 选出互不重叠的 top_k 段
    chosen = []
    for idx in order:
        s = starts[idx]
        if all(abs(s - c) >= win for c in chosen):
            chosen.append(s)
        if len(chosen) == top_k:
            break
    chosen.sort()

    compare = {
        "proposed": PROPOSED_COLOR,
        "dlinear":  "#3B6FB6",
        "lstm":     "#E8A33D",
    }
    preds = {}
    for m in compare:
        pm, am = load_preds(f"{m}_h{h}_seed{seed}")
        if pm is not None:
            preds[m] = pm[:, -1]

    fig, axes = plt.subplots(1, top_k, figsize=(7.5 * top_k, 4.6))
    if top_k == 1:
        axes = [axes]

    for ax, s in zip(axes, chosen):
        e = s + win
        x = np.arange(s, e)
        ax.plot(x, a_full[s:e], color="black", ls="--", lw=1.1,
                label="True values", zorder=3)
        for m, c in compare.items():
            if m in preds:
                lw = 1.6 if m == "proposed" else 1.1
                ax.plot(x, preds[m][s:e], color=c, lw=lw,
                        label=DISPLAY.get(m, m) if m != "proposed" else "Proposed",
                        zorder=2, alpha=0.9 if m == "proposed" else 0.8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Test time step", fontsize=10)
        ax.set_ylabel("Power (kW)", fontsize=10)
        ax.set_title(f"Ramp segment [{s}, {e}]", fontsize=11, pad=5)
        ax.grid(True, alpha=0.25, ls="--")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Tracking of power ramp events at $h = {h}$ "
                 f"({ {1:'10-min',6:'1-h',12:'2-h',24:'4-h'}[h] } ahead)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ramp_events.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("generating figures ...")
    fig_metrics_vs_horizon(); print("  [1] metrics_vs_horizon")
    fig_baseline_bars();      print("  [2] baseline_bars")
    fig_improvement_heatmap();print("  [3] improvement_heatmap")
    fig_radar();              print("  [4] radar_multimetric")
    fig_ablation_bars();      print("  [5] ablation_bars")
    fig_scatter_pred_actual();print("  [6] scatter_pred_actual")
    fig_error_violin();       print("  [7] error_violin")
    fig_seed_stability();     print("  [8] seed_stability")
    fig_vmd_decomposition();  print("  [9] vmd_decomposition")
    fig_prediction_overlay(); print("  [10] prediction_overlay")
    fig_prediction_panels();  print("  [11] prediction_panels (journal-style)")
    fig_h1_vs_h24_contrast(); print("  [12] h1_vs_h24_contrast")
    fig_proposed_horizons_panorama(); print("  [P1] proposed_horizons_panorama")
    fig_prediction_panels_multi();    print("  [P2] prediction_panels_multi (h=1/6/24)")
    fig_prediction_panels_full();     print("  [P2-full] prediction_panels_full (12 models, h=12)")
    fig_ramp_events();                print("  [P3] ramp_events")
    print(f"done. figures written to {OUT}")
