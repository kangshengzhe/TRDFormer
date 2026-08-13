"""
Figure: does the result hold on other turbines, and at other times?

REPLACES ``fig_multiturb_generalization`` AND ``fig_temporal_robustness``
------------------------------------------------------------------------
Both answered the same question - "is this a single-turbine, single-split
artefact?" - so they are combined into one figure with a spatial band
(panels a-c) and a temporal band (panels d-e), each enclosed and labelled.

The panel order is dictated by the grouping boxes rather than by
convenience: a first attempt interleaved the temporal panel between two
spatial ones, which made a rectangular enclosure impossible and left the
spatial box cutting straight through the heatmap. Spatial panels therefore
occupy the first two rows and the temporal band occupies the third.

Added over the originals: panel (c) regresses the per-turbine advantage
against how variable each turbine's output is. Winning 39 of 40 turbine x
horizon cells is only persuasive if the wins are not concentrated in
turbines resembling the development turbine; a near-flat relationship is
the evidence for that, and it was not previously shown.

The one spatial loss (turbine 70 at h=24) is ringed rather than smoothed
over, as is the one temporal loss (fold W2, DLinear ahead by 0.6%).
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

REF = "dlinear"
FOCUS_H = 12

#: Expanding-window protocol, as fractions of the full chronological series.
#: Training range grows, validation and test slide forward; the scaler and
#: the DWT are refit inside each fold's own training range.
WINDOWS = {
    "W1": {"train": (0.00, 0.55), "valid": (0.55, 0.60), "test": (0.60, 0.70)},
    "W2": {"train": (0.00, 0.65), "valid": (0.65, 0.70), "test": (0.70, 0.80)},
    "W3": {"train": (0.00, 0.75), "valid": (0.75, 0.80), "test": (0.80, 0.90)},
}
SPAN_COLOR = {"train": "#4C72B0", "valid": "#E8A33D", "test": S.HERO}


def build(out_dir: str = "manuscript/figures/results",
          out_name: str = "fig_generalization.png") -> int:
    mt = D.multiturb("mae")
    if D.PROPOSED not in mt or REF not in mt:
        logger.error("multi-turbine records incomplete")
        return 1
    turbs = sorted(mt[D.PROPOSED])
    horizons = list(D.HORIZONS)

    delta = np.array([[100.0 * (np.mean(mt[D.PROPOSED][t][h])
                               / np.mean(mt[REF][t][h]) - 1.0)
                       for t in turbs] for h in horizons])
    prof = D.turbine_profile()
    ro = D.rolling("mae")

    with plt.rc_context(S.rc(base=7.2)):
        fig = plt.figure(figsize=(S.FULL_W, 6.30))
        gs = fig.add_gridspec(
            3, 2, height_ratios=[0.80, 1.0, 0.68], width_ratios=[1.06, 1.0],
            left=0.100, right=0.972, top=0.918, bottom=0.088,
            hspace=0.66, wspace=0.36,
        )

        # ================= (a) per-turbine advantage heatmap =============
        axa = fig.add_subplot(gs[0, :])
        lim = float(np.nanmax(np.abs(delta)))
        axa.imshow(delta, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
        axa.set_xticks(range(len(turbs)))
        axa.set_xticklabels([f"T{t}" for t in turbs])
        axa.set_yticks(range(len(horizons)))
        axa.set_yticklabels([f"$h$={h}" for h in horizons])
        axa.tick_params(length=0)
        for i in range(len(horizons)):
            for j in range(len(turbs)):
                v = delta[i, j]
                axa.text(j, i, f"{v:+.0f}", ha="center", va="center",
                         fontsize=6.0,
                         color="white" if abs(v) > 0.62 * lim else "#1A1A1A")
                if v > 0:                       # the single loss: ring it
                    axa.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="#7A0000", linewidth=1.2, zorder=10))
            wins = int(np.sum(delta[i] < 0))
            axa.text(len(turbs) - 0.30, i, f"{wins}/{len(turbs)}",
                     ha="left", va="center", fontsize=6.3, fontweight="bold",
                     color=S.INNOV["C_exo"] if wins == len(turbs)
                     else "#8A6D00")
        # room inside the axes for the per-row win counts
        axa.set_xlim(-0.5, len(turbs) + 0.45)
        axa.set_title("MAE change vs DLinear per turbine (%),  "
                      "negative = TRDFormer wins", fontsize=7.0, pad=4)
        for s in axa.spines.values():
            s.set_visible(False)
        S.panel_tag(axa, "a", size=7.0, loc="outside left", boxed=False)
        axa.text(0.995, -0.235,
                 f"{int(np.sum(delta < 0))}/{delta.size} "
                 f"turbine\u00d7horizon cells won",
                 transform=axa.transAxes, ha="right", va="top",
                 fontsize=6.3, color="#1A1A1A", fontweight="bold")

        # ================= (b) dumbbell per turbine ======================
        axb = fig.add_subplot(gs[1, 0])
        a = np.array([np.mean(mt[D.PROPOSED][t][FOCUS_H]) for t in turbs])
        b = np.array([np.mean(mt[REF][t][FOCUS_H]) for t in turbs])
        order = np.argsort(b)
        y = np.arange(len(turbs))
        for k, idx in enumerate(order):
            axb.plot([a[idx], b[idx]], [k, k], color="#BBBBBB", lw=0.9,
                     zorder=2, solid_capstyle="round")
        axb.scatter(b[order], y, s=13, color=D.MODEL_COLOR[REF],
                    edgecolors="none", zorder=3, label=D.PRETTY[REF])
        axb.scatter(a[order], y, s=15, color=S.HERO, edgecolors="none",
                    zorder=4, label="TRDFormer")
        axb.set_yticks(y)
        axb.set_yticklabels([f"T{turbs[i]}" for i in order])
        axb.tick_params(axis="y", length=0)
        axb.set_xlabel(f"MAE at $h$={FOCUS_H} (kW)", labelpad=1.5)
        axb.grid(True, axis="x", zorder=0)
        for sp in ("top", "right", "left"):
            axb.spines[sp].set_visible(False)
        axb.legend(loc="lower right", fontsize=5.8, labelspacing=0.2,
                   borderpad=0.28, handletextpad=0.3, framealpha=0.92,
                   scatterpoints=1)
        axb.set_title("Every turbine, same direction", fontsize=7.0, pad=4)
        S.panel_tag(axb, "b", size=7.0, loc="outside left", boxed=False)

        # ================= (c) is the gain turbine-specific? =============
        axc = fig.add_subplot(gs[1, 1])
        cv = np.array([prof.get(t, {}).get("cv", np.nan) for t in turbs])
        dv = delta[horizons.index(FOCUS_H)]
        ok = np.isfinite(cv)
        axc.scatter(cv[ok], dv[ok], s=18, color=S.INNOV["C_endo"], alpha=0.85,
                    edgecolors="none", zorder=3)
        for t, x, v in zip(np.array(turbs)[ok], cv[ok], dv[ok]):
            axc.annotate(f"T{t}", (x, v), xytext=(2.4, 2.4),
                         textcoords="offset points", fontsize=5.2,
                         color="#666666")
        if ok.sum() > 2:
            r = float(np.corrcoef(cv[ok], dv[ok])[0, 1])
            k, c = np.polyfit(cv[ok], dv[ok], 1)
            xx = np.linspace(cv[ok].min(), cv[ok].max(), 20)
            axc.plot(xx, k * xx + c, color="#888888", lw=0.8, ls=(0, (3, 2)),
                     zorder=2)
            axc.text(0.965, 0.075, f"$r$={r:+.2f}", transform=axc.transAxes,
                     ha="right", va="bottom", fontsize=6.2, color="#333333")
        axc.axhline(0, color="#333333", lw=0.6, zorder=2)
        axc.set_xlabel("turbine output variability (CV)", labelpad=1.5)
        axc.set_ylabel(f"MAE change at $h$={FOCUS_H} (%)", labelpad=2)
        axc.grid(True, zorder=0)
        axc.spines["top"].set_visible(False)
        axc.spines["right"].set_visible(False)
        axc.set_title("Gain is not turbine-specific", fontsize=7.0, pad=4)
        S.panel_tag(axc, "c", size=7.0, loc="lower left")

        # ================= (d) expanding-window protocol =================
        axd = fig.add_subplot(gs[2, 0])
        names = list(WINDOWS)
        for i, w in enumerate(names):
            yy = len(names) - 1 - i
            for part in ("train", "valid", "test"):
                lo, hi = WINDOWS[w][part]
                axd.barh(yy, hi - lo, left=lo, height=0.56,
                         color=SPAN_COLOR[part], alpha=0.88,
                         edgecolor="white", linewidth=0.5, zorder=3,
                         label=part.capitalize() if i == 0 else None)
            axd.text(-0.022, yy, w, ha="right", va="center", fontsize=6.5,
                     fontweight="bold", color="#333333")
        axd.set_xlim(0, 1.0)
        # headroom above the top fold so the span legend fits *inside* the
        # axes; anchoring it above the axes pushed the temporal grouping box
        # up until it crossed the spatial one
        axd.set_ylim(-0.55, len(names) + 0.30)
        axd.set_yticks([])
        axd.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axd.set_xlabel("fraction of the chronological series", labelpad=1.5)
        axd.grid(True, axis="x", zorder=0)
        for sp in ("top", "right", "left"):
            axd.spines[sp].set_visible(False)
        axd.legend(loc="upper left", fontsize=5.6, ncol=3,
                   labelspacing=0.15, borderpad=0.26, handlelength=1.0,
                   handletextpad=0.4, columnspacing=0.8, framealpha=0.92)
        axd.set_title("Protocol: scaler and DWT refit inside each fold",
                      fontsize=7.0, pad=4)
        S.panel_tag(axd, "d", size=7.0, loc="outside left", boxed=False)

        # ================= (e) per-fold MAE ==============================
        axe = fig.add_subplot(gs[2, 1])
        wkeys = sorted([k for k in ro.get(D.PROPOSED, {}) if k is not None])
        if wkeys:
            xs = np.arange(len(wkeys))
            pa = [float(np.mean(ro[D.PROPOSED][k])) for k in wkeys]
            pb = [float(np.mean(ro[REF][k])) for k in wkeys]
            sa = [float(np.std(ro[D.PROPOSED][k], ddof=1)) for k in wkeys]
            sb = [float(np.std(ro[REF][k], ddof=1)) for k in wkeys]
            wd = 0.34
            axe.bar(xs - wd / 2, pa, wd, yerr=sa, capsize=1.6,
                    error_kw=dict(lw=0.6), color=S.HERO, alpha=0.88,
                    label="TRDFormer", zorder=3)
            axe.bar(xs + wd / 2, pb, wd, yerr=sb, capsize=1.6,
                    error_kw=dict(lw=0.6), color=D.MODEL_COLOR[REF],
                    alpha=0.88, label=D.PRETTY[REF], zorder=3)
            for x, va, vb in zip(xs, pa, pb):
                d = 100.0 * (va / vb - 1.0)
                axe.text(x, max(va, vb) * 1.055, f"{d:+.1f}%", ha="center",
                         va="bottom", fontsize=6.1, fontweight="bold",
                         color=S.INNOV["C_exo"] if d < 0 else "#B0413E")
            axe.set_xticks(xs)
            axe.set_xticklabels([f"W{k}" for k in wkeys])
            axe.set_ylim(0, max(max(pa), max(pb)) * 1.34)
            axe.set_ylabel(f"MAE at $h$={FOCUS_H} (kW)", labelpad=2)
            axe.grid(True, axis="y", zorder=0)
            axe.spines["top"].set_visible(False)
            axe.spines["right"].set_visible(False)
            axe.legend(loc="upper right", fontsize=5.7, labelspacing=0.18,
                       borderpad=0.26, handlelength=1.1, framealpha=0.92)
        axe.set_title("Per-fold accuracy (3 seeds)", fontsize=7.0, pad=4)
        S.panel_tag(axe, "e", size=7.0, loc="outside left", boxed=False)

        # ================= grouping boxes ================================
        pa_, pb_, pd_ = (axa.get_position(), axb.get_position(),
                         axd.get_position())
        S.group_box(fig, 0.020, pb_.y0 - 0.068, 0.990, pa_.y1 + 0.036,
                    label="1  Across 10 turbines (3 seeds each)",
                    color="#4A4A4A", lw=0.8, size=6.4,
                    label_side="bottom left")
        S.group_box(fig, 0.020, pd_.y0 - 0.050, 0.990, pd_.y1 + 0.034,
                    label="2  Across time: expanding-window validation",
                    color=S.INNOV["B"], lw=0.8, size=6.4,
                    label_side="bottom left")

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
