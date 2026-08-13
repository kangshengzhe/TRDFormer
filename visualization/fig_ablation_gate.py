"""
Figure: what each component contributes, and what the fusion gate does.

REPLACES ``fig_ablation_bars`` AND ``fig_branch_contribution``
--------------------------------------------------------------
The first was a lone bar chart occupying a whole figure; the second showed
two of the same variants in more detail. Merged here, with two additions
that were previously impossible.

**Bars carry innovation identity.** Each variant is coloured by the
contribution it removes, using the same palette as the architecture
diagram, and grouped by contribution with a dashed enclosure. Colour then
answers "which idea does this bar test?" without reading the label.

**The gate is shown actually gating.** ``outputs/analysis/gate_weights.npz``
records the endogenous gate weight for every test window, every horizon and
all 10 seeds. Panels (c) and (d) use it to show (i) that the gate assigns
71-89% of its weight to the endogenous branch, independently corroborating
the 14.6x asymmetry that the ablations infer indirectly, and (ii) that the
weight *shifts further* toward the endogenous branch as ramp magnitude
grows at h=6 and h=12 - evidence the gate is adaptive rather than a learned
constant. The correlation vanishes at h=24, which is reported as-is.

Negative results are shown as negative. Replacing the KAN head with an MLP
*improves* MAE by 2.0%, and the LSTM branch and linear head are within
noise; those bars are hatched and labelled ``ns`` rather than omitted.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _data as D          # noqa: E402
import _style as S         # noqa: E402

logger = logging.getLogger(__name__)

HORIZON = 12
#: Fixed physical bins, in kW. Quantile bins collapse here because >55% of
#: windows have ramp magnitude at or near zero, which left the lowest two
#: quintiles empty.
RAMP_EDGES = [0.0, 25.0, 75.0, 150.0, 300.0, np.inf]
RAMP_TICKS = ["0-25", "25-75", "75-150", "150-300", ">300"]


def build(out_dir: str = "manuscript/figures/results",
          out_name: str = "fig_ablation_gate.png") -> int:
    ab = D.ablation_metrics("mae", HORIZON)
    full = np.array(list(ab["full"].values()), dtype=float)
    base = float(full.mean())

    rows = []
    for v, (label, key) in D.ABLATION_V2.items():
        if v not in ab or not ab[v]:
            continue
        vals = np.array(list(ab[v].values()), dtype=float)
        rows.append({
            "variant": v, "label": label, "key": key,
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)),
            "infl": 100.0 * (float(vals.mean()) - base) / base,
            "p": D.ablation_pvalue(v, "mae", HORIZON),
            "vals": vals,
        })
    rows.sort(key=lambda r: -r["infl"])

    with plt.rc_context(S.rc(base=7.2)):
        fig = plt.figure(figsize=(S.FULL_W, 5.45))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1.0, 0.88],
            width_ratios=[0.86, 1.0, 1.06],
            left=0.178, right=0.978, top=0.888, bottom=0.112,
            hspace=0.66, wspace=0.46,
        )

        # ============ (a) ablation, horizontal bars by innovation ==========
        axa = fig.add_subplot(gs[0, :])
        y = np.arange(len(rows))[::-1]
        for yy, r in zip(y, rows):
            sig = np.isfinite(r["p"]) and r["p"] < 0.05
            col = S.INNOV[r["key"]]
            axa.barh(yy, r["infl"], height=0.62,
                     color=col if sig else "white",
                     edgecolor=col, linewidth=0.9,
                     hatch=None if sig else "////",
                     alpha=0.90 if sig else 1.0, zorder=3)
            star = S.sig_stars(r["p"])
            off = 1.1 if r["infl"] >= 0 else -1.1
            axa.text(r["infl"] + off, yy,
                     f"{r['infl']:+.1f}%  {star}",
                     va="center", ha="left" if r["infl"] >= 0 else "right",
                     fontsize=6.4, color="#1A1A1A",
                     fontweight="bold" if sig else "normal", zorder=5)
        axa.axvline(0, color="#333333", lw=0.7, zorder=4)
        axa.set_yticks(y)
        axa.set_yticklabels([r["label"] for r in rows])
        axa.set_xlabel(f"MAE change vs full TRDFormer at $h$={HORIZON} "
                       f"(%),  base = {base:.1f} kW", labelpad=2)
        axa.set_xlim(-8.5, 52)
        axa.grid(True, axis="x", zorder=0)
        axa.spines["top"].set_visible(False)
        axa.spines["right"].set_visible(False)
        axa.set_title("Removing or replacing one component at a time "
                      "(10 seeds, paired $t$-test)", fontsize=7.0, pad=4)
        S.panel_tag(axa, "a", size=7.0, loc="lower right")
        axa.text(0.985, 0.10,
                 "solid: $p<$0.05    hatched: not significant",
                 transform=axa.transAxes, ha="right", va="bottom",
                 fontsize=5.6, color=S.MUTED)
        S.innovation_key(fig, 0.150, 0.963, size=5.9, dx=0.148, swatch=0.0105)

        # ============ (b) paired branch asymmetry ==========
        axb = fig.add_subplot(gs[1, 0])
        axb.set_facecolor("white")
        pair = [("v2_no_itrans", "C_endo", "Endogenous\n(iTransformer)"),
                ("v2_no_lstm", "C_exo", "Exogenous\n(LSTM)")]
        spans = {}
        for i, (v, key, lab) in enumerate(pair):
            if v not in ab:
                continue
            common = sorted(set(ab["full"]) & set(ab[v]))
            d = np.array([ab[v][s] - ab["full"][s] for s in common])
            spans[i] = d
            xs = np.full(d.size, i, dtype=float) + \
                np.linspace(-0.11, 0.11, d.size)
            axb.scatter(xs, d, s=7, color=S.INNOV[key], alpha=0.75,
                        edgecolors="none", zorder=4)
            bp = axb.boxplot([d], positions=[i], widths=0.44,
                             showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", lw=0.8),
                             whiskerprops=dict(lw=0.5),
                             capprops=dict(lw=0.5), boxprops=dict(lw=0.5),
                             zorder=3)
            bp["boxes"][0].set_facecolor(S.INNOV[key])
            bp["boxes"][0].set_alpha(0.28)
            bp["boxes"][0].set_edgecolor(S.INNOV[key])
        # The two groups sit at opposite ends of the axis, leaving the middle
        # band empty; labels go there rather than above each box, where the
        # upper one collided with the panel title.
        if 0 in spans:
            axb.text(0, spans[0].min() - 1.4, f"+{spans[0].mean():.1f} kW",
                     ha="center", va="top", fontsize=6.4, fontweight="bold",
                     color=S.INNOV["C_endo"], zorder=6)
        if 1 in spans:
            axb.text(1, spans[1].max() + 1.4, f"+{spans[1].mean():.1f} kW",
                     ha="center", va="bottom", fontsize=6.4,
                     fontweight="bold", color=S.INNOV["C_exo"], zorder=6)
        axb.axhline(0, color="#333333", lw=0.6, zorder=2)
        axb.set_xticks(range(len(pair)))
        axb.set_xticklabels([p[2] for p in pair])
        axb.set_ylabel("$\\Delta$MAE when removed (kW)", labelpad=2)
        axb.set_xlim(-0.55, len(pair) - 0.45)
        axb.grid(True, axis="y", zorder=0)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.set_title("Branch asymmetry, seed-paired", fontsize=7.0, pad=4)
        S.panel_tag(axb, "b", size=7.0)

        ratio = None
        try:
            ci = np.mean([ab["v2_no_itrans"][s] - ab["full"][s]
                          for s in sorted(set(ab["full"]) & set(ab["v2_no_itrans"]))])
            ce = np.mean([ab["v2_no_lstm"][s] - ab["full"][s]
                          for s in sorted(set(ab["full"]) & set(ab["v2_no_lstm"]))])
            ratio = ci / ce if ce else None
        except KeyError:
            pass
        if ratio:
            axb.text(0.5, 0.47, f"{ratio:.1f}$\\times$ asymmetry",
                     transform=axb.transAxes, ha="center", va="center",
                     fontsize=6.3, color="#1A1A1A", zorder=6,
                     bbox=dict(boxstyle="round,pad=0.26", facecolor="#F5F5F5",
                               edgecolor="#999999", linewidth=0.5))

        # ============ (c) gate weight distribution per horizon ==========
        axc = fig.add_subplot(gs[1, 1])
        axc.set_facecolor("white")
        parts_data, means = [], []
        for h in D.HORIZONS:
            g = D.load_gate(h)
            parts_data.append(g["alpha"].ravel())
            means.append(g["alpha"].mean())
        vp = axc.violinplot(parts_data, positions=range(len(D.HORIZONS)),
                            widths=0.72, showextrema=False, showmedians=True)
        for body, h in zip(vp["bodies"], D.HORIZONS):
            body.set_facecolor(S.HORIZON_COLOR[h])
            body.set_alpha(0.55)
            body.set_edgecolor(S.HORIZON_COLOR[h])
            body.set_linewidth(0.6)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(0.8)
        axc.axhline(0.5, color="#666666", lw=0.6, ls=":", zorder=2)
        axc.text(0.985, 0.505, "equal weight", transform=axc.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=5.5, color="#666666")
        for i, (h, m) in enumerate(zip(D.HORIZONS, means)):
            axc.text(i, 1.035, f"{m:.2f}", ha="center", va="bottom",
                     fontsize=6.2, color=S.HORIZON_COLOR[h],
                     fontweight="bold")
        axc.set_xticks(range(len(D.HORIZONS)))
        axc.set_xticklabels([f"$h$={h}" for h in D.HORIZONS])
        axc.set_ylim(0.0, 1.13)
        axc.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axc.set_ylabel(r"gate weight $\alpha_{\mathrm{en}}$", labelpad=2)
        axc.grid(True, axis="y", zorder=0)
        axc.spines["top"].set_visible(False)
        axc.spines["right"].set_visible(False)
        axc.set_title("Reliance on the endogenous branch",
                      fontsize=7.0, pad=8)
        S.panel_tag(axc, "c", size=7.0, loc="lower left")

        # ============ (d) is the gate adaptive? alpha vs ramp size =========
        axd = fig.add_subplot(gs[1, 2])
        for h in (6, 12, 24):
            g = D.load_gate(h)
            ramp, a = g["ramp_mag"], g["alpha_mean"]
            xs, ys, es = [], [], []
            for i, (lo, hi) in enumerate(zip(RAMP_EDGES[:-1], RAMP_EDGES[1:])):
                m = (ramp >= lo) & (ramp < hi)
                if m.sum() < 30:
                    continue
                xs.append(i)
                ys.append(a[m].mean())
                es.append(a[m].std() / np.sqrt(m.sum()))
            r = float(np.corrcoef(ramp, a)[0, 1])
            axd.errorbar(xs, ys, yerr=es, marker="o", ms=2.8,
                         lw=1.25 if h == 12 else 0.9,
                         color=S.HORIZON_COLOR[h], capsize=1.4,
                         elinewidth=0.6,
                         label=f"$h$={h}  ($r$={r:+.2f})", zorder=3)
        axd.set_xticks(range(len(RAMP_TICKS)))
        axd.set_xticklabels(RAMP_TICKS, rotation=30, ha="right")
        axd.set_xlabel("ramp magnitude (kW)", labelpad=1.5)
        axd.set_ylabel(r"mean $\alpha_{\mathrm{en}}$", labelpad=2)
        # headroom so the legend does not sit on the h=12 curve
        y0, y1 = axd.get_ylim()
        axd.set_ylim(y0, y1 + 0.35 * (y1 - y0))
        axd.grid(True, zorder=0)
        axd.spines["top"].set_visible(False)
        axd.spines["right"].set_visible(False)
        # upper-left is the only free corner: the h=24 curve dips into the
        # lower right, where the legend previously sat on top of it
        axd.legend(loc="upper left", fontsize=5.5, labelspacing=0.18,
                   borderpad=0.26, handlelength=1.3, framealpha=0.92)
        axd.set_title("The gate shifts during ramps", fontsize=7.0, pad=4)
        S.panel_tag(axd, "d", size=7.0, loc="lower left")

        # ============ grouping boxes ============
        pa = axa.get_position()
        # top edge must clear the panel title but stay under the innovation
        # key, which sits in the strip above the box
        S.group_box(fig, 0.020, pa.y0 - 0.058, 0.992, pa.y1 + 0.036,
                    label="1  Component contributions", color="#4A4A4A",
                    lw=0.8, size=6.4, label_side="bottom left")
        pb, pc, pd = (axb.get_position(), axc.get_position(),
                      axd.get_position())
        S.group_box(fig, 0.020, pb.y0 - 0.086, 0.992,
                    max(pb.y1, pc.y1, pd.y1) + 0.050,
                    label="2  Mechanism: an asymmetric, adaptive gate",
                    color="#4A4A4A", lw=0.8, size=6.4,
                    label_side="bottom left")

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
