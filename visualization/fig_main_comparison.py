"""
Figure: accuracy against 11 SOTA baselines, all four horizons.

REPLACES TWO LEGACY FIGURES
---------------------------
``fig_overall_comparison`` (grouped bars + radar + rank heatmap) and
``fig_performance_overview`` (radar + horizontal bars + one forecast curve +
parity plot) overlapped heavily: both carried a radar, both carried the same
MAE numbers, and the forecast curve duplicated the prediction-matrix figure.

Two things were also wrong rather than merely redundant:

* **The radars were broken.** Both normalised each axis min-max across the
  models on display. With the proposed model best on most axes it received
  1.0 everywhere and every baseline collapsed toward the centre, producing a
  dart shape that says nothing about magnitudes. Min-max over a handful of
  models is not a meaningful normalisation, so no radar survives here.
* **Nothing showed *where within the horizon* the advantage comes from.**
  Aggregate MAE hides it. The lead-time profile in panel (b) is the single
  most informative addition: it shows DLinear is marginally better at the
  first step and then degrades far faster, and it explains the h=24 result
  instead of leaving it as an anomaly.

Panels
------
(a) MAE for 12 models x 4 horizons, cells annotated, coloured by within-
    horizon rank. Replaces both the grouped bars and the separate rank
    heatmap.
(b) MAE by lead time within the horizon, h=12 and h=24.
(c) MAE change relative to DLinear, the strongest baseline, with paired
    t-test significance.
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

REF = "dlinear"            # strongest baseline; the comparison that matters


def build(out_dir: str = "manuscript/figures/results",
          out_name: str = "fig_main_comparison.png") -> int:
    agg = D.agg_metrics(("mae", "rmse", "r2"))
    pv = D.paired_pvalues("mae")
    horizons = list(D.HORIZONS)
    models = [m for m in D.MODEL_ORDER if m in agg]

    mae = np.array([[agg[m][h]["mae"][0] for h in horizons] for m in models])
    sd = np.array([[agg[m][h]["mae"][1] for h in horizons] for m in models])
    rank = np.argsort(np.argsort(mae, axis=0), axis=0) + 1

    with plt.rc_context(S.rc(base=7.2)):
        fig = plt.figure(figsize=(S.FULL_W, 4.55))
        gs = fig.add_gridspec(
            2, 2, width_ratios=[1.06, 1.0], height_ratios=[1.0, 0.80],
            left=0.148, right=0.978, top=0.905, bottom=0.098,
            hspace=0.52, wspace=0.30,
        )

        # ================== (a) MAE / rank heatmap ==================
        axh = fig.add_subplot(gs[:, 0])
        im = axh.imshow(rank, cmap="RdYlGn_r", vmin=1, vmax=len(models),
                        aspect="auto")
        axh.set_xticks(range(len(horizons)))
        axh.set_xticklabels([f"$h$={h}" for h in horizons])
        axh.set_yticks(range(len(models)))
        axh.set_yticklabels([D.PRETTY[m] for m in models])
        axh.tick_params(length=0)
        for lab, m in zip(axh.get_yticklabels(), models):
            if m == D.PROPOSED:
                lab.set_color(S.HERO)
                lab.set_fontweight("bold")
        for i in range(len(models)):
            for j in range(len(horizons)):
                r = rank[i, j]
                axh.text(j, i, f"{mae[i, j]:.1f}", ha="center", va="center",
                         fontsize=6.1,
                         color="white" if (r <= 2 or r >= len(models) - 1)
                         else "#1A1A1A",
                         fontweight="bold" if models[i] == D.PROPOSED
                         else "normal")
        # frame the proposed model's row
        ip = models.index(D.PROPOSED)
        axh.add_patch(plt.Rectangle(
            (-0.5, ip - 0.5), len(horizons), 1.0, fill=False,
            edgecolor=S.HERO, linewidth=1.5, zorder=20, clip_on=False))
        axh.set_title("MAE (kW), cell colour = rank within horizon",
                      fontsize=7.0, pad=4)
        # tag outside the plotting area: every cell here carries a number,
        # so an in-panel badge would sit on top of data
        S.panel_tag(axh, "a", size=7.0, loc="outside left", boxed=False)
        for s in axh.spines.values():
            s.set_visible(False)

        cb = fig.colorbar(im, ax=axh, orientation="horizontal",
                          fraction=0.034, pad=0.055, aspect=34)
        # Labels sit *inside* the bar. Below it they were clipped by the
        # group-box caption; above it they would collide with the h= ticks.
        cb.set_ticks([])
        cb.outline.set_linewidth(0.4)
        cb.ax.text(0.015, 0.5, "best", transform=cb.ax.transAxes,
                   ha="left", va="center", fontsize=5.9, color="white",
                   fontweight="bold")
        cb.ax.text(0.985, 0.5, "worst", transform=cb.ax.transAxes,
                   ha="right", va="center", fontsize=5.9, color="white",
                   fontweight="bold")

        # ================== (b) lead-time profile ==================
        axp = fig.add_subplot(gs[0, 1])
        # three models, not four: eight lines in a 2.3in panel is clutter,
        # and iTransformer is uniformly worse so it adds no comparison here
        show = [D.PROPOSED, REF, "lstm"]
        ends = {}
        # The h=12 solid and h=24 dashed curves of a given model are NOT
        # equally distinguishable, and that misleads the eye (2026-08 fix).
        # Measured overlap over the first 12 steps: DLinear 0.14 kW mean
        # difference (max 0.37), LSTM 4.80, TRDFormer 18.86. So DLinear's
        # solid h=12 line is *completely* hidden under its own h=24 dashed
        # line, which reads as "the blue curve alone runs to 4 h while the
        # red one stops at 2 h" -- i.e. as a truncation bug rather than as
        # two overlapping series. Two cheap disambiguators, no data change:
        #   * dashed h=24 gets thinner and more transparent than solid h=12,
        #     so where they coincide the solid line still shows through;
        #   * every solid h=12 curve gets a filled marker at its final step,
        #     which is the point the text quotes and the only place the two
        #     horizons must not be confused.
        for h, ls in ((12, "-"), (24, (0, (2.6, 1.4)))):
            dashed = h == 24
            for m in show:
                mu, _ = D.per_step_mae(m, h)
                if mu is None:
                    continue
                t = np.arange(1, len(mu) + 1) * 10 / 60.0
                base_lw = 1.35 if m == D.PROPOSED else 0.85
                axp.plot(t, mu, ls=ls, color=D.MODEL_COLOR[m],
                         lw=base_lw * (0.72 if dashed else 1.0),
                         alpha=(0.55 if dashed else 1.0) if m == D.PROPOSED
                         else (0.45 if dashed else 0.9),
                         zorder=(3 if dashed else 5) if m == D.PROPOSED
                         else (2 if dashed else 4))
                if not dashed:
                    axp.plot(t[-1], mu[-1], marker="o",
                             ms=2.9 if m == D.PROPOSED else 2.2,
                             color=D.MODEL_COLOR[m], mec="white",
                             mew=0.45,
                             zorder=6 if m == D.PROPOSED else 5)
                if m == D.PROPOSED:
                    ends[h] = (t[-1], mu[-1])
        axp.set_xlabel("Lead time within horizon (h)", labelpad=1.5)
        axp.set_ylabel("MAE (kW)", labelpad=2)
        axp.grid(True, zorder=0)
        axp.spines["top"].set_visible(False)
        axp.spines["right"].set_visible(False)
        axp.set_title("Error growth inside the horizon", fontsize=7.0, pad=4)
        S.panel_tag(axp, "b", size=7.0, loc="outside left", boxed=False)

        # Horizon is labelled on the curves themselves rather than in a
        # second legend: two stacked legends in this panel covered both the
        # panel tag and the curves they were meant to explain.
        for h, (tx, ty) in ends.items():
            axp.annotate(f"$h$={h}", xy=(tx, ty), xytext=(-1.5, -7.5),
                         textcoords="offset points", ha="right", va="top",
                         fontsize=6.0, color=S.HERO, fontweight="bold")
        h_model = [plt.Line2D([], [], color=D.MODEL_COLOR[m],
                              lw=1.3 if m == D.PROPOSED else 0.9,
                              label=D.PRETTY[m]) for m in show]
        # Spell out the solid/dashed contract explicitly. "dashed: h=24"
        # alone left the solid case implicit, which is exactly the pairing a
        # reader gets wrong where the two coincide.
        h_model.append(plt.Line2D([], [], color="#777777", lw=0.9, ls="-",
                                  marker="o", ms=2.2, mec="white", mew=0.45,
                                  label="solid: $h$=12"))
        h_model.append(plt.Line2D([], [], color="#777777", lw=0.65,
                                  alpha=0.5,
                                  ls=(0, (2.6, 1.4)), label="dashed: $h$=24"))
        axp.legend(handles=h_model, loc="lower right", fontsize=5.8,
                   labelspacing=0.20, borderpad=0.28, handlelength=1.5,
                   framealpha=0.92)

        # A plain text box, not an annotate() callout: the claim is about
        # the shape of the whole curve rather than one point, so there is no
        # sensible anchor, and any arc from a free corner to the first step
        # had to cross every curve in the panel to get there.
        #
        # The horizon qualifier is not decoration (2026-08 fix). Unqualified,
        # the sentence read as a claim about the whole panel, but the panel
        # draws both horizons and "leads ... marginally" only holds at h=12
        # (31.00 vs 33.48 kW, a 2.47 kW gap). At h=24 DLinear leads the first
        # step by 13.24 kW (30.97 vs 44.21) and wins on the horizon average
        # too (100.21 vs 104.67), so an unqualified sentence overstated the
        # h=12 reading and understated the h=24 one. The body text always
        # carried "at h=12"; the box did not.
        axp.text(0.035, 0.945,
                 "$h$=12: DLinear leads at step 1,\nthen degrades faster",
                 transform=axp.transAxes, ha="left", va="top", fontsize=5.7,
                 color="#1F4E79", zorder=1500,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#EEF5FC",
                           edgecolor="#1F4E79", linewidth=0.6, alpha=0.95))

        # ================== (c) advantage over the best baseline ==========
        axb = fig.add_subplot(gs[1, 1])
        rel = [100.0 * (agg[D.PROPOSED][h]["mae"][0] / agg[REF][h]["mae"][0] - 1)
               for h in horizons]
        xs = np.arange(len(horizons))
        cols = [S.INNOV["C_exo"] if v < 0 else "#B0413E" for v in rel]
        axb.bar(xs, rel, width=0.58, color=cols, edgecolor="none", alpha=0.88,
                zorder=3)
        axb.axhline(0, color="#333333", lw=0.6, zorder=4)
        span = max(abs(min(rel)), abs(max(rel)))
        for x, v, h in zip(xs, rel, horizons):
            star = S.sig_stars(pv.get(REF, {}).get(h, float("nan")))
            # stack value then significance outward from the bar's free end
            sign = -1.0 if v < 0 else 1.0
            va = "top" if v < 0 else "bottom"
            axb.text(x, v + sign * 0.05 * span, f"{v:+.1f}%", ha="center",
                     va=va, fontsize=6.3, fontweight="bold",
                     color="#1A1A1A", zorder=5)
            axb.text(x, v + sign * 0.30 * span, star, ha="center", va=va,
                     fontsize=6.6, color="#555555", zorder=5)
        axb.set_xticks(xs)
        axb.set_xticklabels([f"$h$={h}" for h in horizons])
        axb.set_ylabel(f"MAE vs {D.PRETTY[REF]} (%)", labelpad=2)
        lim = max(abs(min(rel)), abs(max(rel)))
        axb.set_ylim(-lim * 1.55, lim * 1.35)
        axb.grid(True, axis="y", zorder=0)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.set_title(f"Advantage over {D.PRETTY[REF]} "
                      "(paired $t$-test, 10 seeds)", fontsize=7.0, pad=4)
        S.panel_tag(axb, "c", size=7.0)
        # top strip is free (all improvements are negative bars); the
        # bottom-right placement previously landed on the h=12 label
        axb.text(0.30, 0.93, "*** $p<$0.001    ns: not significant",
                 transform=axb.transAxes, ha="left", va="top",
                 fontsize=5.6, color=S.MUTED)

        # ================== grouping boxes ==================
        pa = axh.get_position()
        S.group_box(fig, 0.020, pa.y0 - 0.088, pa.x1 + 0.014, pa.y1 + 0.055,
                    label="1  Accuracy landscape", color="#4A4A4A",
                    lw=0.8, size=6.4, label_side="bottom left")
        pb, pc = axp.get_position(), axb.get_position()
        S.group_box(fig, pb.x0 - 0.052, pc.y0 - 0.062, 0.992, pb.y1 + 0.052,
                    label="2  Where the advantage comes from",
                    color="#4A4A4A", lw=0.8, size=6.4,
                    label_side="bottom right")

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
