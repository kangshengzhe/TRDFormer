"""
Graphical abstract for the EPSR submission.

SPEC (Guide for authors, Electric Power Systems Research)
---------------------------------------------------------
    "Ensure the image is 531 x 1328 pixels (h x w) or proportionally more,
     and is readable at a size of 5 x 13 cm. Preferred file types for
     graphical abstracts are TIFF, EPS, PDF or MS Office files."
    "Submit your graphical abstract as a separate file."

So the canvas is authored at 13 x 5.2 cm -- 5.118 x 2.047 in, whose 2.5009
aspect matches 1328:531 exactly -- and written as PDF (vector, on the
preferred list) plus a 640 dpi PNG at 3275 x 1310 px, i.e. "proportionally
more" in both dimensions. Authoring at the final physical size is the same
rule ``_style`` enforces for the in-article figures: the point sizes below
are the point sizes a reader sees at 13 cm.

WHY IT LOOKS LIKE THIS
----------------------
An earlier ``graphical_abstract.py`` (now in ``_cleanup_archive/``) drew a
pure block schematic. A reviewer learns nothing from boxes-and-arrows that
the architecture figure does not already say, so two of the three zones here
carry **measured data** instead, and every number is recomputed from the
result tree at build time rather than typed in:

  left    the actual input decomposition -- one real 70 h test window, its
          moving-average trend, and the five db4 sub-bands that become
          variate tokens. Same data path as ``fig_mechanism_strip.py``.
  middle  the only schematic zone, and the smallest: what routes where.
  right   the headline result -- MAE against the strongest baseline at all
          four horizons, plus the ramp-decile margin, from
          ``_data.agg_metrics`` and ``fig_ramp_analysis._ramp_stats``.

Colours are ``_style.INNOV``, so gold/blue/purple/green/red mean the same
here as in the architecture figure and every data figure.

NOTE ON THE GENAI POLICY
------------------------
The journal requires AI use in graphical-abstract production to follow its
GenAI policy. Nothing here is generated imagery: this is a matplotlib plot
of the authors' own measurements, in the same way every other figure in the
paper is produced. The manuscript's existing AI declaration already covers
the authoring of the plotting scripts.

Usage
-----
    python visualization/fig_graphical_abstract.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _data as D                                    # noqa: E402
import _style as S                                   # noqa: E402

logger = logging.getLogger(__name__)

# Elsevier graphical-abstract geometry: 13 cm wide, 1328:531 aspect.
GA_W = 13.0 / 2.54
GA_H = GA_W * 531.0 / 1328.0

CSV = "data/wind/sdwpf_turb1_cleaned_final.csv"
PARTITION = "outputs/manifests/partition_indices_l144_h12.json"
SCALER = "outputs/manifests/scaler.pkl"
FEAT = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]

BANDS = ["D1", "D2", "D3", "D4", "A4"]
BAND_COLOR = {"D1": "#2166AC", "D2": "#4393C3", "D3": "#92C5DE",
              "D4": "#F4A582", "A4": "#B7950B"}
TREND_KERNEL = 25
WIN = (1200, 1620)              # 420 samples x 10 min = 70 h of test set
REF = "dlinear"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Symmetric-padded moving average, matching ``MovingAvgBlock``."""
    pad = (k - 1) // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")[:len(x)]


def _dwt_bands(sig: np.ndarray, wavelet="db4", level=4) -> dict:
    import pywt

    coeffs = pywt.wavedec(sig, wavelet, mode="symmetric", level=level)
    names = ["A4", "D4", "D3", "D2", "D1"]           # wavedec order
    out = {}
    for i, nm in enumerate(names):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        out[nm] = pywt.waverec(z, wavelet, mode="symmetric")[:len(sig)]
    return out


def _signals():
    """The standardised test-set Patv window, its trend, residual, sub-bands."""
    df = pd.read_csv(CSV)
    with open(PARTITION, encoding="utf-8") as fh:
        part = json.load(fh)
    ts, te = part["test"]["start"], part["test"]["end"]

    from data_pipeline.scaling import FeatureScaler

    Z = FeatureScaler.load(SCALER).transform(
        df[FEAT].to_numpy(float)[ts:te]).astype(float)
    y = Z[:, 0]
    trend = _moving_average(y, TREND_KERNEL)
    return y, trend, y - trend, _dwt_bands(y)


# ---------------------------------------------------------------------------
# zones
# ---------------------------------------------------------------------------
def _zone_input(ax_top, ax_bot, y, trend, bands):
    """Left zone: the real signal and the tokens it is decomposed into."""
    w0, w1 = WIN
    t = np.arange(w1 - w0) * 10 / 60.0

    yw = y[w0:w1]
    ax_top.plot(t, yw, color="#4A4A4A", lw=0.5, zorder=3)
    ax_top.plot(t, trend[w0:w1], color=S.INNOV["A"], lw=1.25, zorder=4)
    ax_top.set_xlim(t[0], t[-1])
    # Reserve the upper ~30% as a clear label band. Both traces were labelled
    # in place before, which put "moving-average trend" straight on top of the
    # gold curve it names.
    lo, hi = float(yw.min()), float(yw.max())
    span = hi - lo if hi > lo else 1.0
    ax_top.set_ylim(lo - 0.06 * span, hi + 0.46 * span)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax_top.spines[sp].set_visible(False)
    ax_top.text(0.012, 0.965, "measured power", transform=ax_top.transAxes,
                fontsize=5.5, color="#4A4A4A", va="top", ha="left")
    ax_top.text(0.988, 0.965, "moving-average trend",
                transform=ax_top.transAxes, fontsize=5.5, color=S.INNOV["A"],
                va="top", ha="right", fontweight="bold")

    # five sub-bands, offset on a shared axis and rescaled to equal height so
    # that A4 (97% of the energy) cannot flatten D1-D4 into flat lines.
    gap = 1.0
    for i, nm in enumerate(BANDS[::-1]):
        b = np.asarray(bands[nm][w0:w1], dtype=float)
        span = np.ptp(b)
        b = (b - b.mean()) / (span if span > 0 else 1.0)
        base = i * gap
        ax_bot.axhline(base, color="0.90", lw=0.25, zorder=0)
        ax_bot.plot(t, b + base, color=BAND_COLOR[nm], lw=0.45, zorder=3)
        ax_bot.text(-0.012, base, f"${nm[0]}_{{{nm[1]}}}$",
                    transform=ax_bot.get_yaxis_transform(), ha="right",
                    va="center", fontsize=5.6, color=BAND_COLOR[nm],
                    fontweight="bold")
    ax_bot.set_xlim(t[0], t[-1])
    ax_bot.set_ylim(-gap * 0.7, (len(BANDS) - 1) * gap + gap * 0.7)
    ax_bot.set_yticks([])
    ax_bot.set_xticks([0, 24, 48])
    ax_bot.set_xticklabels(["0", "24", "48 h"], fontsize=5.4)
    ax_bot.tick_params(axis="x", length=1.4, width=0.4, pad=1.2)
    for sp in ("top", "right", "left"):
        ax_bot.spines[sp].set_visible(False)
    ax_bot.spines["bottom"].set_color(S.INNOV["B"])
    ax_bot.spines["bottom"].set_linewidth(1.2)


def _zone_model(ax, alpha_en):
    """Middle zone: the routing, and only the routing."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, key, text, *, fs=5.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.03",
            facecolor=S.INNOV_FILL[key], edgecolor=S.INNOV[key],
            linewidth=0.75, zorder=3, transform=ax.transAxes, clip_on=False))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#1A1A1A", zorder=4,
                transform=ax.transAxes, linespacing=1.25)

    def arrow(p0, p1, *, rad=0.0, lw=0.8):
        ax.add_patch(FancyArrowPatch(
            p0, p1, transform=ax.transAxes, arrowstyle="-|>",
            mutation_scale=5.0, linewidth=lw, color=S.MUTED,
            connectionstyle=f"arc3,rad={rad}", zorder=2, clip_on=False))

    # endogenous / exogenous, side by side, then the gate, then the sum
    box(0.02, 0.735, 0.60, 0.20, "B",
        "5 sub-bands + de-trended $y$\nas variate tokens")
    box(0.02, 0.455, 0.60, 0.20, "C_endo",
        "iTransformer\nvariate attention")
    box(0.66, 0.455, 0.32, 0.20, "C_exo", "LSTM\ncovariates")
    box(0.02, 0.180, 0.96, 0.18, "D",
        rf"adaptive gate   $\alpha_{{\rm en}}$ = {alpha_en:.2f}")
    box(0.02, -0.055, 0.96, 0.16, "A",
        r"$+$ linear trend branch   $\rightarrow$   $\hat{y}$")

    arrow((0.32, 0.735), (0.32, 0.660))
    arrow((0.32, 0.455), (0.32, 0.365))
    arrow((0.82, 0.455), (0.62, 0.365), rad=-0.18)
    arrow((0.50, 0.180), (0.50, 0.108))


def _zone_result(ax, mae, ramp):
    """Right zone: the headline numbers, all recomputed at build time."""
    hs = list(D.HORIZONS)
    x = np.arange(len(hs), dtype=float)
    bw = 0.34

    prop = [mae[D.PROPOSED][h] for h in hs]
    ref = [mae[REF][h] for h in hs]

    ax.bar(x - bw / 2, prop, bw, color=S.HERO_FILL, edgecolor=S.HERO,
           linewidth=0.7, zorder=3, label="TRDFormer")
    ax.bar(x + bw / 2, ref, bw, color="#DCDCDC", edgecolor="#8A8A8A",
           linewidth=0.6, zorder=3, label=D.PRETTY[REF])

    top = max(max(prop), max(ref))
    for i, h in enumerate(hs):
        d = 100.0 * (prop[i] - ref[i]) / ref[i]
        win = d < 0
        ax.text(x[i], max(prop[i], ref[i]) + top * 0.045,
                f"{d:+.1f}%", ha="center", va="bottom", fontsize=5.7,
                color=S.HERO if win else "#7A7A7A",
                fontweight="bold" if win else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels([f"$h$={h}" for h in hs], fontsize=5.7)
    ax.tick_params(axis="both", length=1.5, width=0.4, pad=1.4,
                   labelsize=5.4)
    ax.set_ylabel("test MAE (kW)", fontsize=6.0, labelpad=1.6)
    # 1.30 headroom put the tallest bar's delta label right under the ramp
    # callout box; 1.48 separates them.
    ax.set_ylim(0, top * 1.48)
    ax.grid(True, axis="y", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", fontsize=5.4, frameon=True, framealpha=0.94,
              borderpad=0.22, handlelength=1.0, handletextpad=0.35,
              labelspacing=0.15, borderaxespad=0.25)

    gain = 100.0 * (1.0 - ramp[D.PROPOSED]["ramp"] / ramp[REF]["ramp"])
    ax.text(0.985, 0.965,
            f"ramp decile\n$-${gain:.1f}% MAE",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
            color=S.HERO, fontweight="bold", linespacing=1.2, zorder=10,
            bbox=dict(boxstyle="round,pad=0.26", facecolor="#FFF4F4",
                      edgecolor=S.HERO, linewidth=0.6))
    return gain


# ---------------------------------------------------------------------------
def build(out_dir: str = "manuscript/figures",
          out_name: str = "graphical_abstract.png") -> int:
    y, trend, resid, bands = _signals()

    agg = D.agg_metrics(("mae",))
    mae = {m: {h: agg[m][h]["mae"][0] for h in agg[m]} for m in agg}

    from fig_ramp_analysis import _ramp_stats

    _, _, _, ramp, _ = _ramp_stats()

    gate = D.load_gate(12)
    alpha_en = float(np.mean(gate["alpha_en"])) if "alpha_en" in gate \
        else float(np.mean(next(iter(gate.values()))))

    with plt.rc_context(S.rc(base=6.0)):
        fig = plt.figure(figsize=(GA_W, GA_H))
        gs = fig.add_gridspec(
            2, 3, width_ratios=[1.00, 0.78, 1.30], height_ratios=[1.0, 1.55],
            left=0.052, right=0.988, top=0.865, bottom=0.150,
            wspace=0.30, hspace=0.16,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])
        _zone_input(ax_top, ax_bot, y, trend, bands)

        ax_mid = fig.add_subplot(gs[:, 1])
        _zone_model(ax_mid, alpha_en)

        ax_res = fig.add_subplot(gs[:, 2])
        gain = _zone_result(ax_res, mae, ramp)

        # Zone headers, anchored to the axes they label rather than to
        # hand-tuned figure fractions -- the hard-coded values drifted off
        # their zones and the third one collided with the provenance line.
        for ax_ref, txt, col in (
            (ax_top, "1  Decompose", S.INNOV["B"]),
            (ax_mid, "2  Route asymmetrically", S.INNOV["C_endo"]),
            (ax_res, "3  Forecast", S.HERO),
        ):
            fig.text(ax_ref.get_position().x0, 0.955, txt, ha="left",
                     va="center", fontsize=6.2, fontweight="bold",
                     color="white", zorder=50,
                     bbox=dict(boxstyle="round,pad=0.26", facecolor=col,
                               edgecolor="none"))

        # Provenance sits on the bottom rule, clear of the header row.
        fig.text(0.988, 0.022,
                 "TRDFormer  $\\cdot$  SDWPF Turb1  $\\cdot$  "
                 "1,048 runs  $\\cdot$  10 seeds",
                 ha="right", va="bottom", fontsize=5.2, color=S.MUTED,
                 style="italic", zorder=50)

        S.save_figure(fig, Path(out_dir) / out_name, also_pdf=True)
        plt.close(fig)

    ratio = trend.std() / resid.std()
    logger.info("canvas %.3f x %.3f in (%.2f cm x %.2f cm), aspect %.4f",
                GA_W, GA_H, GA_W * 2.54, GA_H * 2.54, GA_W / GA_H)
    logger.info("trend/residual std ratio %.2fx | alpha_en(h=12) %.3f", ratio,
                alpha_en)
    logger.info("ramp-decile gain %.1f%% | MAE h=12 %.2f vs %s %.2f",
                gain, mae[D.PROPOSED][12], REF, mae[REF][12])
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="manuscript/figures")
    ap.add_argument("--out-name", default="graphical_abstract.png")
    a = ap.parse_args()
    raise SystemExit(build(a.out_dir, a.out_name))
