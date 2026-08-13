"""
Companion strip for Figure 2: the architecture annotated with real data.

WHY A SEPARATE ASSET
--------------------
The architecture itself stays a TikZ drawing (vector, and the only place the
innovation colours are defined), so it cannot be produced by matplotlib. But
a block diagram on its own asks the reader to trust that "DWT sub-bands" and
"trend / residual" mean something; showing the actual tensors that enter each
block turns the diagram into evidence.

This renders the strip that sits directly beneath the diagram in
``main.tex``: one panel per branch entry point, framed in the innovation
colour of the block it feeds.

  (b) what the trend branch computes   - y, its moving average, the residual
  (c) what the endogenous branch gets  - five db4 sub-bands of y
  (d) what the exogenous branch gets   - four standardised covariates

All three panels use the same grammar - traces offset vertically on a common
time axis - so the strip reads as one object. An earlier version gave each
panel its own layout (a legend on the first, a five-row nested gridspec on
the second, overlaid traces on the third); at 2.1in tall the nested rows
collapsed to 0.25in each and the legend overflowed the canvas.

WHAT THE DWT IS ACTUALLY APPLIED TO
-----------------------------------
The standardised Patv series ``y``, NOT the de-trended residual. In the
implementation the two act in *parallel* on the same ``y``:

  data_pipeline/dwt.py         DWT of scaled_*[:, 0] (= standardised Patv),
                               fitted per train/valid/test partition
  models/proposed_v2.py:215-225  patv = x[:, :, 0]; trend = MA(patv)
                               x_residual = cat[(patv - trend), x[:, :, 1:]]
                               -- i.e. channels 1..5 (the sub-bands) pass
                               through UNCHANGED, so they remain sub-bands
                               of the un-de-trended y.

So the reconstruction identity is ``y = A4 + sum(D1..D4)``, and panel (c)
must decompose ``patv``. An earlier version of this script decomposed
``resid`` instead, which drew a tensor that never enters the model -- the
opposite of what this figure exists to show. Panel (b) still shows the
trend/residual split, whose scales differ by roughly 4.5x (std 0.78 vs
0.17); that ratio is the point of doing the split at all.

Panel (c) therefore does overlap Fig. 1(e) in content (both decompose y),
but not in purpose: Fig. 1 argues decomposition is *needed*, this panel
shows the resulting five channels *as the variate tokens the encoder
receives*.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S         # noqa: E402

logger = logging.getLogger(__name__)

CSV = "data/wind/sdwpf_turb1_cleaned_final.csv"
PARTITION = "outputs/manifests/partition_indices_l144_h12.json"
SCALER = "outputs/manifests/scaler.pkl"

FEAT = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
COV = ["Wspd", "Wdir", "Etmp", "Itmp"]

BANDS = ["D1", "D2", "D3", "D4", "A4"]
BAND_COLOR = {"D1": "#2166AC", "D2": "#4393C3", "D3": "#92C5DE",
              "D4": "#F4A582", "A4": "#B7950B"}

#: Moving-average kernel of the trend extractor (models/proposed_v2.py).
TREND_KERNEL = 25
WIN = (1200, 1620)              # 420 samples x 10 min = 70 h of test data


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Symmetric-padded moving average, matching ``MovingAvgBlock``."""
    pad = (k - 1) // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")[:len(x)]


def _dwt_bands(signal: np.ndarray, wavelet="db4", level=4) -> dict:
    import pywt

    coeffs = pywt.wavedec(signal, wavelet, mode="symmetric", level=level)
    names = ["A4", "D4", "D3", "D2", "D1"]          # wavedec order
    out = {}
    for i, nm in enumerate(names):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        out[nm] = pywt.waverec(z, wavelet, mode="symmetric")[:len(signal)]
    return out


def _stack(ax, t, series, *, gap: float = 1.0, unit_scale: bool = True):
    """Plot traces offset on a shared axis, bottom-up, and label each.

    ``unit_scale`` rescales every trace to a common peak-to-peak height so
    that one dominant channel cannot flatten the rest; the vertical axis then
    carries no units, which is why it is left unticked.
    """
    for i, (name, y, color, lw) in enumerate(series):
        y = np.asarray(y, dtype=float)
        if unit_scale:
            span = np.ptp(y)
            y = (y - y.mean()) / (span if span > 0 else 1.0)
        base = i * gap
        ax.axhline(base, color="0.88", lw=0.3, zorder=0)
        ax.plot(t, y + base, color=color, lw=lw, zorder=3)
        ax.text(0.008, base + gap * 0.40, name, transform=
                ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=5.5, color=color, fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", facecolor="white",
                          edgecolor="none", alpha=0.80))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-gap * 0.62, (len(series) - 1) * gap + gap * 0.72)
    ax.set_yticks([])
    ax.set_xlabel("hours", labelpad=1.2)
    ax.grid(True, axis="x", zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)


def _tie_to_block(ax, color: str, title: str) -> None:
    """Colour a panel's baseline and title to match its architecture block.

    A full frame was tried first, but ``_stack`` hides three of the four
    spines, so only the bottom one took the colour and the result read as a
    half-drawn box. A deliberate thick baseline plus a matching bold title is
    the same visual tie with none of the ambiguity.
    """
    ax.spines["bottom"].set_color(color)
    ax.spines["bottom"].set_linewidth(1.6)
    ax.set_title(title, fontsize=6.2, pad=3, color=color, fontweight="bold")


def build(out_dir: str = "manuscript/figures/method",
          out_name: str = "fig_mechanism_strip.png") -> int:
    df = pd.read_csv(CSV)
    with open(PARTITION, encoding="utf-8") as fh:
        part = json.load(fh)
    ts, te = part["test"]["start"], part["test"]["end"]

    from data_pipeline.scaling import FeatureScaler

    scaler = FeatureScaler.load(SCALER)
    Z = scaler.transform(df[FEAT].to_numpy(float)[ts:te]).astype(float)

    w0, w1 = WIN
    t = np.arange(w1 - w0) * 10 / 60.0

    patv = Z[:, 0]
    trend = _moving_average(patv, TREND_KERNEL)
    resid = patv - trend
    # DWT is applied to y itself, in parallel with the trend removal -- see
    # the module docstring. Decomposing `resid` here would plot a tensor the
    # model never sees.
    bands = _dwt_bands(patv)

    with plt.rc_context(S.rc(base=6.8)):
        fig = plt.figure(figsize=(S.FULL_W, 2.05))
        gs = fig.add_gridspec(
            1, 3, width_ratios=[1.0, 1.0, 0.88],
            left=0.028, right=0.990, top=0.800, bottom=0.150, wspace=0.13,
        )

        # ---------- (b) trend / residual split ----------
        axb = fig.add_subplot(gs[0, 0])
        _stack(axb, t, [
            ("residual", resid[w0:w1], S.INNOV["C_endo"], 0.6),
            (f"trend, $k$={TREND_KERNEL}", trend[w0:w1], S.INNOV["A"], 1.1),
            ("$y$", patv[w0:w1], "#555555", 0.6),
        ][::-1], gap=1.25)
        _tie_to_block(axb, S.INNOV["A"], "A  Trend $-$ residual split")
        S.panel_tag(axb, "b", size=6.5, loc="lower right")

        # ---------- (c) sub-bands of y ----------
        axc = fig.add_subplot(gs[0, 1])
        _stack(axc, t, [(f"${nm[0]}_{{{nm[1]}}}$", bands[nm][w0:w1],
                         BAND_COLOR[nm], 0.55) for nm in BANDS][::-1],
               gap=1.0)
        _tie_to_block(axc, S.INNOV["B"], "B  db4 sub-bands of $y$")
        S.panel_tag(axc, "c", size=6.5, loc="lower right")

        # ---------- (d) covariates ----------
        axd = fig.add_subplot(gs[0, 2])
        greens = ["#1B7837", "#5AAE61", "#7FBC9B", "#A6DBA0"]
        _stack(axd, t, [(c, Z[w0:w1, FEAT.index(c)], greens[j], 0.55)
                        for j, c in enumerate(COV)][::-1], gap=1.0)
        _tie_to_block(axd, S.INNOV["C_exo"], "C$_2$  Standardised covariates")
        S.panel_tag(axd, "d", size=6.5, loc="lower right")

        fig.text(0.500, 0.012,
                 "traces rescaled to equal height and offset for display; "
                 "test set, 70 h",
                 ha="center", va="bottom", fontsize=5.3, color=S.MUTED)

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    logger.info("trend std %.3f | residual std %.3f | ratio %.1fx",
                trend.std(), resid.std(), trend.std() / resid.std())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
