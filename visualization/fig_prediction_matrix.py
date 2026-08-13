"""
Figure: forecast behaviour as a horizon x model matrix.

REPLACES FIVE LEGACY FIGURES
----------------------------
``fig_panels_h1``, ``fig_panels_h6``, ``fig_panels_h24``,
``fig_prediction_panels_main`` and ``fig_qualitative_4h`` between them spent
four full pages and 36 sub-panels restating one thing: predicted versus
measured power. Their content is reorganised here into a single matrix whose
rows are horizons and columns are models, so the two comparisons a reader
actually wants - "how does accuracy decay with horizon?" (read down a
column) and "who is better at this horizon?" (read across a row) - become
directly visible instead of requiring page-flipping.

Three substantive improvements over the originals:

* **Rows share one wall-clock window.** The legacy panels sliced each
  horizon by *window index*, but window *i* of h=1 and window *i* of h=24
  describe different moments in time (the terminal target step sits at
  ``i + L + h - 1``). Slicing in absolute index instead means every row here
  shows the same 48 hours, which is what makes the rows comparable.
* **The window is chosen to contain a ramp**, not merely to have high
  variance, so the panels exercise the regime where the models actually
  differ.
* **Error distributions sit beside the curves** rather than in a separate
  figure, so a horizon's bias and spread can be read together with its
  trajectory.

The h=24 row is annotated with the fact that DLinear catches up there
(p=0.22, not significant); presenting the one horizon we do not win on is
deliberate.
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

N_POINTS = 288             # 48 h at 10-min resolution
STEP_MIN = 10


def _mae_table() -> dict:
    agg = D.agg_metrics(("mae",))
    return {m: {h: agg[m][h]["mae"][0] for h in agg[m]} for m in agg}


def _pooled_error(model: str, horizon: int, seeds=D.SEEDS) -> np.ndarray:
    """Terminal-step signed error pooled over seeds, kW."""
    chunks = []
    for s in seeds:
        try:
            act, pred = D.load_preds(model, horizon, s)
        except FileNotFoundError:
            continue
        t = act.shape[1] - 1
        chunks.append(pred[:, t] - act[:, t])
    return np.concatenate(chunks) if chunks else np.array([])


def build(out_dir: str = "manuscript/figures/results",
          out_name: str = "fig_prediction_matrix.png",
          seed: int = 42) -> int:
    models = list(D.MATRIX_MODELS)
    horizons = list(D.HORIZONS)
    mae = _mae_table()

    # ---- choose one absolute-time window shared by every row -------------
    lo, hi = D.common_abs_span(horizons, seed=seed)
    _, act12, _ = D.slice_by_abs(D.PROPOSED, 12, seed, lo, hi)
    off = D.pick_ramp_window(act12, N_POINTS)
    win_lo = lo + off
    win_hi = win_lo + N_POINTS
    logger.info("display window: absolute [%d, %d) of [%d, %d)",
                win_lo, win_hi, lo, hi)

    # ---- gather everything up front so layout code stays readable -------
    curves: dict = {}
    for h in horizons:
        for m in models:
            ai, a, p = D.slice_by_abs(m, h, seed, win_lo, win_hi)
            curves[(h, m)] = (ai, a, p)
    errors = {(h, m): _pooled_error(m, h) for h in horizons for m in models}

    ymax = max(np.nanmax(v[1]) for v in curves.values()) * 1.10
    # One error scale *per horizon*. A single shared scale is tempting for
    # cross-row comparison but the h=1 spread is ~10x smaller than h=24, so
    # sharing it collapses the top rows to invisible slivers; the axis
    # numbers carry the cross-row comparison instead.
    #
    # The limit follows the *whisker* extent (q3 + 1.5 IQR), not a
    # percentile of the raw error: rare large excursions push the 99th
    # percentile far beyond the box, which again flattened the boxes to
    # slivers even with per-row scaling.
    def _whisker(e: np.ndarray) -> float:
        q1, q3 = np.percentile(e, [25, 75])
        return max(abs(q3 + 1.5 * (q3 - q1)), abs(q1 - 1.5 * (q3 - q1)))

    emax = {h: max(_whisker(errors[(h, m)]) for m in models
                   if errors[(h, m)].size) * 1.18
            for h in horizons}

    with plt.rc_context(S.rc(base=7.0)):
        fig = plt.figure(figsize=(S.FULL_W, S.FULL_H))
        gs = fig.add_gridspec(
            len(horizons), 6,
            width_ratios=[1, 1, 1, 1, 0.12, 0.72],   # col 4 = gutter
            left=0.108, right=0.968, top=0.910, bottom=0.200,
            hspace=0.34, wspace=0.10,
        )

        tag = iter("abcdefghijklmnopqrst")
        AX: dict = {}          # (horizon, model) -> axes; ("err", h) -> axes

        # ================= column headers =================
        for j, m in enumerate(models):
            ax0 = fig.add_subplot(gs[0, j])
            pos = ax0.get_position()
            ax0.remove()
            is_hero = (m == D.PROPOSED)
            S.band_label(
                fig, (pos.x0 + pos.x1) / 2, 0.945, D.PRETTY[m],
                color=S.HERO if is_hero else "#4A4A4A",
                fill=S.HERO_FILL if is_hero else "white",
                size=7.4, pad=0.34,
            )
        axe0 = fig.add_subplot(gs[0, 5])
        pos_e = axe0.get_position()
        axe0.remove()
        S.band_label(fig, (pos_e.x0 + pos_e.x1) / 2, 0.945,
                     "Error CDF", color="#4A4A4A", fill="white",
                     size=7.4, pad=0.34)

        # ================= the matrix =================
        for i, h in enumerate(horizons):
            hcol = S.HORIZON_COLOR[h]
            row_axes = []
            for j, m in enumerate(models):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
                AX[(h, m)] = ax
                ai, a, p = curves[(h, m)]
                x = (ai - ai[0]) * STEP_MIN / 60.0

                S.shade_error(ax, x, a, p,
                              color=D.MODEL_COLOR[m], alpha=0.28, zorder=1.2)
                ax.plot(x, a, color=S.ACTUAL_COLOR, lw=0.65, ls=(0, (3, 1.6)),
                        zorder=2.4, label="Measured")
                ax.plot(x, p, color=D.MODEL_COLOR[m],
                        lw=1.05 if m == D.PROPOSED else 0.85,
                        zorder=2.6, label="Forecast")

                ax.set_xlim(x[0], x[-1])
                ax.set_ylim(0, ymax)
                ax.grid(True, axis="y", zorder=0)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelbottom=(i == len(horizons) - 1),
                               labelleft=(j == 0))
                if i == len(horizons) - 1:
                    ax.set_xlabel("Time (h)", labelpad=1.5)
                if j == 0:
                    ax.set_yticks([0, 500, 1000, 1500])
                else:
                    ax.set_yticks([0, 500, 1000, 1500])

                if m == D.PROPOSED:
                    # subtle: a heavy frame on a 1.1in panel reads as a
                    # data line, which is exactly what it must not do
                    S.hero_frame(ax, lw=0.75)

                # per-panel accuracy, 10-seed mean so it matches the tables
                v = mae.get(m, {}).get(h)
                if v is not None:
                    ax.text(0.975, 0.955, f"MAE {v:.1f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=6.2,
                            color=D.MODEL_COLOR[m],
                            fontweight="bold" if m == D.PROPOSED else "normal",
                            bbox=dict(boxstyle="round,pad=0.18",
                                      facecolor="white", edgecolor="none",
                                      alpha=0.82), zorder=50)
                S.panel_tag(ax, next(tag), size=6.2)

            # row identity chip
            p0 = row_axes[0].get_position()
            S.band_label(fig, 0.050, (p0.y0 + p0.y1) / 2,
                         S.HORIZON_LABEL[h], color=hcol, fill="white",
                         rotation=90, size=6.9, pad=0.30)

            # ---- absolute-error CDF column ----
            # A box plot was tried here and is actively misleading on this
            # data. 60% of the test set is near-zero and flat, where a
            # linear extrapolator is almost exact, so DLinear's IQR at h=1
            # is 0.6 kW against our 21 kW - the box says DLinear wins.
            # The full distribution says the opposite where it matters: at
            # h=12 DLinear's median |error| is 7.5 kW vs our 31.8, but its
            # 95th percentile is 447 kW vs our 308. The CDF shows both the
            # body and the tail, so the trade is visible instead of hidden.
            axe = fig.add_subplot(gs[i, 5])
            AX[("err", h)] = axe
            xmax = max(np.percentile(np.abs(errors[(h, m)]), 97)
                       for m in models if errors[(h, m)].size)
            for m in models:
                e = np.abs(errors[(h, m)])
                if not e.size:
                    continue
                q = np.linspace(0, 1, 256)
                v = np.quantile(e, q)
                axe.plot(v, q, color=D.MODEL_COLOR[m],
                         lw=1.15 if m == D.PROPOSED else 0.75,
                         alpha=1.0 if m == D.PROPOSED else 0.85,
                         zorder=3 if m == D.PROPOSED else 2)
            axe.axhline(0.9, color="#777777", lw=0.45, ls=":", zorder=1)
            axe.text(0.965, 0.905, "$p_{90}$", transform=axe.transAxes,
                     ha="right", va="bottom", fontsize=5.3, color="#666666")
            axe.set_xlim(0, xmax)
            axe.set_ylim(0, 1.0)
            axe.set_yticks([0, 0.5, 1.0])
            axe.set_yticklabels(["0", ".5", "1"])
            axe.grid(True, axis="both", zorder=0)
            axe.spines["top"].set_visible(False)
            axe.spines["right"].set_visible(False)
            axe.tick_params(labelbottom=True, labelsize=5.8)
            # Labelled "step h" explicitly: the annotated MAE in the curve
            # panels is the table value, averaged over all lead times, while
            # these CDFs describe the terminal step alone, matching the
            # curves. Two definitions in one figure are fine only if both
            # are named.
            axe.set_xlabel("$|$error$|$ at step $h$ (kW)", labelpad=1.2,
                           fontsize=5.9)
            S.panel_tag(axe, next(tag), size=6.2, loc="lower right")

        # ================= shared y label =================
        # x must clear half the rotated glyph height, or the label is cut
        # off by the canvas edge.
        fig.text(0.014, 0.530, "Active power (kW)", rotation=90,
                 va="center", ha="center", fontsize=7.6)

        # No zoom inset here: at 1.1 x 1.4in per cell an inset plus its
        # connector lines cross the very data they are meant to clarify.
        # The magnified ramp view lives in the ramp-analysis figure, which
        # has the room for it.

        # ============ callouts: state the two findings a reader might miss ==
        S.callout(
            AX[(24, "dlinear")],
            "at $h$=24 DLinear\nmatches us ($p$=0.22)",
            xy=(0.55, 0.60), xytext=(0.11, 0.17),
            color="#7A5C00", fill="#FFF8E1", size=5.8, rad=0.20,
        )
        # The CDF crossing is the mechanism behind the ramp results and a
        # reader will not extract it unaided - but a 0.85in-wide panel
        # cannot hold the sentence, so it goes in the full-width strip
        # below the matrix instead of crowding panel (o).
        e_p = D.PROPOSED
        med_p = np.percentile(np.abs(errors[(12, e_p)]), 50)
        med_d = np.percentile(np.abs(errors[(12, "dlinear")]), 50)
        p95_p = np.percentile(np.abs(errors[(12, e_p)]), 95)
        p95_d = np.percentile(np.abs(errors[(12, "dlinear")]), 95)
        fig.text(
            0.53, 0.076,
            f"Read the CDFs at the tail, not the median: at $h$=12 DLinear is "
            f"near-exact more often\n(median $|$error$|$ {med_d:.0f} vs "
            f"{med_p:.0f}\u2009kW) yet its 95th percentile is "
            f"{100*(p95_d/p95_p-1):.0f}% larger ({p95_d:.0f} vs "
            f"{p95_p:.0f}\u2009kW).",
            ha="center", va="bottom", fontsize=6.1, color="#1F4E79",
            linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.34", facecolor="#EEF5FC",
                      edgecolor="#1F4E79", linewidth=0.55),
        )

        # ============ hero column band ============
        # A tinted band behind the proposed model's column makes "which one
        # is ours" readable at a glance, without the heavy per-panel frame
        # that earlier looked like plotted data.
        p_ht = AX[(horizons[0], D.PROPOSED)].get_position()
        p_hb = AX[(horizons[-1], D.PROPOSED)].get_position()
        fig.patches.append(plt.Rectangle(
            (p_ht.x0 - 0.011, p_hb.y0 - 0.033),
            (p_ht.x1 - p_ht.x0) + 0.022,
            (p_ht.y1 - p_hb.y0) + 0.045,
            transform=fig.transFigure, facecolor=S.HERO, alpha=0.045,
            edgecolor=S.HERO, linewidth=0.55, linestyle=(0, (3, 2)),
            zorder=0.15, clip_on=False))

        # ================= grouping boxes =================
        # Labels go *below* the boxes: the row of model-name chips already
        # occupies the strip above, and a top-left label collided with it.
        p_tl = AX[(horizons[0], models[0])].get_position()
        p_br = AX[(horizons[-1], models[-1])].get_position()
        S.group_box(fig, 0.019, p_br.y0 - 0.052, p_br.x1 + 0.013,
                    p_tl.y1 + 0.016,
                    label="1  Forecast vs. measured power",
                    color="#4A4A4A", lw=0.8, size=6.3,
                    label_side="bottom left")
        pe_t = AX[("err", horizons[0])].get_position()
        pe_b = AX[("err", horizons[-1])].get_position()
        S.group_box(fig, pe_t.x0 - 0.020, pe_b.y0 - 0.052, 0.988,
                    pe_t.y1 + 0.016, pad=0.005,
                    label="2  Error distribution", color="#4A4A4A", lw=0.8,
                    size=6.3, label_side="bottom right")

        # ================= legend =================
        handles = [
            plt.Line2D([], [], color=S.ACTUAL_COLOR, lw=0.8,
                       ls=(0, (3, 1.6)), label="Measured power"),
            plt.Line2D([], [], color=S.HERO, lw=1.2, label="Model forecast"),
            plt.Rectangle((0, 0), 1, 1, facecolor=S.HERO, alpha=0.28,
                          edgecolor="none", label="Forecast error"),
        ]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.53, 0.036), ncol=3, fontsize=6.8,
                   frameon=False, handlelength=1.8, columnspacing=1.6)
        fig.text(0.53, 0.010,
                 f"curves and CDFs: forecast issued $h$ steps earlier "
                 f"(terminal step) \u00b7 seed 42, one shared "
                 f"{N_POINTS*STEP_MIN/60:.0f}-h window \u00b7 "
                 f"MAE: 10-seed mean over all lead times, as in the tables",
                 fontsize=5.7, color=S.MUTED, ha="center", va="bottom")

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
