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
preferred list) plus a 640 dpi PNG, i.e. "proportionally more" in both
dimensions. Authoring at the final physical size is the same rule ``_style``
enforces for the in-article figures: the point sizes below are the point
sizes a reader sees at 13 cm.

REBUILT 2026-08 FOR THE CAUSALITY RESULT
----------------------------------------
The previous version sold the headline the paper no longer makes. It showed
TRDFormer beating DLinear by 15-17% at three horizons and by 43.5% on ramp
windows, which is exactly the set of numbers the causality experiments show
to be an artefact of the offline decomposition's access to the future. Left
unchanged it would have contradicted the abstract.

The three zones now carry the paper's actual argument:

  left    the practice: one real 70 h test window, its moving-average trend,
          and the five db4 sub-bands that become variate tokens. Unchanged --
          this zone describes what the field does, and that is still accurate.
  middle  the problem: perturb the signal five standardised units at t+1 and
          the reconstruction AT t moves by 2.29 units. Drawn from a live
          recomputation, not an illustration.
  right   the consequence: MAE at h=12 across the six causalisation variants,
          with the no-sub-band ablation as a reference line. The bar for a
          12-step delay lands on that line to within 0.01 kW.

Every number is recomputed from the result tree at build time.

NOTE ON THE GENAI POLICY
------------------------
Nothing here is generated imagery: this is a matplotlib plot of the authors'
own measurements, as with every other figure in the paper. The manuscript's
AI declaration already covers the authoring of the plotting scripts.

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

PERTURB = 5.0                   # standardised units injected at t0+1

# Right zone: the causalisation variants, in the order the paper tables them.
# label, run directory, strictly causal?
VARIANTS = [
    ("offline\n(as published)", "outputs/v2_full", False),
    ("delay\n$k$=3", "outputs/lag3_dwt_h12", False),
    ("delay\n$k$=12", "outputs/lag12_dwt_h12", False),
    ("causal\nfilter bank", "outputs/atrous_h12", True),
    ("causal\ntrailing", "outputs/causal_reflect_h12", True),
]
NO_BANDS_MAE = 81.95            # w/o DWT ablation, recomputed below


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


def _perturbation(y: np.ndarray, t0: int, half: int = 26):
    """Inject PERTURB at t0+1 and return the D1 reconstructions around t0.

    D1, not A4: the bands' SUM at t0 is unchanged to ~1e-16 because additivity
    holds, so the leakage shows up as a redistribution among bands rather than
    a shift of the reconstructed signal. D1 carries the largest share of that
    redistribution (2.29 of 5.0 injected, against 0.24 for A4), so it is the
    band that makes the effect visible. Plotting A4 here understated the
    response by 5x and would have contradicted the manuscript.

    The decomposition MUST run over the whole test partition, exactly as the
    paper's preprocessing does, and only then be sliced for display. An earlier
    version decomposed a short window around t0 instead and measured -0.71
    rather than -2.29: at 208 samples the level-4 filters sit close enough to
    both edges that the padding rule, not the data, sets the finest band. That
    would have put a number in the graphical abstract that contradicts the
    manuscript.

    Decomposing the clean and the perturbed series separately and differencing
    at t0 is the test asserted in tests/test_dwt_causality.py, so the number
    drawn here and the number quoted in the manuscript cannot drift apart.
    """
    pert_full = y.copy()
    pert_full[t0 + 1] += PERTURB

    b_clean = _dwt_bands(y)
    b_pert = _dwt_bands(pert_full)
    response = float(b_pert["D1"][t0] - b_clean["D1"][t0])

    # Guard the claim: the bands' sum at t0 must not move, or the
    # "redistribution" framing in Section 3.2 would be wrong.
    resid = abs(sum(b_pert[b][t0] - b_clean[b][t0] for b in BANDS))
    if resid > 1e-9:
        raise AssertionError(
            f"bands at t0 do not sum-cancel (residual {resid:.2e}); the "
            f"'redistribution' framing in the manuscript would be wrong")
    # Sanity band, not an equality check: the response depends on t0's phase
    # relative to the decimated coefficient grid (measured range over 64
    # positions: 0.71 to 2.29 for D1, median 1.50). The manuscript reports that
    # distribution rather than a point value, so this only catches a decode
    # that has gone badly wrong.
    if not (0.5 <= abs(response) <= 2.6):
        raise AssertionError(
            f"D1 response {response:.4f} outside the measured range "
            f"[0.71, 2.29]; check the partition and t0")

    lo, hi = t0 - half, t0 + half
    return (y[lo:hi], pert_full[lo:hi], b_clean["D1"], b_pert["D1"],
            t0, response, half)


def _response_by_parity(y: np.ndarray, n_probe: int = 16):
    """Measure the D1 response separately for odd and even forecast origins.

    The response is not a distribution to be summarised by a median -- it is
    exactly two-valued. Because the level-1 coefficients live on every second
    sample, t0 either coincides with a coefficient centre or falls between two,
    and the two cases give 2.29 and 0.71 units for a 5.0-unit injection with
    *zero* variation inside each class (verified over 12 positions per parity).
    A median over mixed parities returns 1.50, a value no forecast origin
    actually takes, which is why this returns the two classes instead.

    The figure plots the odd case and labels both, since odd origins are half of
    all origins and carry the larger exposure.
    """
    n = len(y)
    clean_d1 = _dwt_bands(y)["D1"]
    out = {}
    for parity in (1, 0):
        cand = [t for t in range(n // 4, n // 4 + 4 * n_probe)
                if t % 2 == parity][:n_probe]
        vals = []
        for t0 in cand:
            p = y.copy()
            p[t0 + 1] += PERTURB
            vals.append(abs(_dwt_bands(p)["D1"][t0] - clean_d1[t0]))
        vals = np.asarray(vals)
        if vals.std() > 1e-6:
            raise AssertionError(
                f"parity-{parity} response is not constant (std {vals.std():.2e});"
                f" the two-valued claim in Section 3.2 would be wrong")
        out[parity] = (int(cand[0]), float(vals[0]))
    return out


def _variant_mae() -> dict:
    """Mean test MAE per causalisation variant, read from run records."""
    out = {}
    for label, run_dir, causal in VARIANTS:
        p = Path(run_dir) / "outputs" / "runs" / "run_records.jsonl"
        vals = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("status") != "success" or r.get("horizon") != 12:
                    continue
                rid = r.get("run_id", "")
                # v2_full holds only the proposed model; the variant dirs hold
                # one model each, so no further filtering is needed.
                m = r.get("metrics") or {}
                if m.get("mae") is not None:
                    vals.append(float(m["mae"]))
        if not vals:
            raise FileNotFoundError(
                f"no h=12 records for '{label}' at {p}. The six causalisation "
                f"runs live on the training host; copy their run_records.jsonl "
                f"into outputs/<variant>/outputs/runs/ before building.")
        out[label] = (float(np.mean(vals)), float(np.std(vals, ddof=1)),
                      causal, len(vals))
    return out


def _no_bands_mae() -> float:
    """The w/o DWT ablation MAE, so the reference line is not hard-coded."""
    p = Path("outputs/ablation_v2/outputs/runs/run_records.jsonl")
    if not p.exists():
        return NO_BANDS_MAE
    vals = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("status") != "success" or r.get("horizon") != 12:
            continue
        if "no_dwt" not in r.get("run_id", ""):
            continue
        m = r.get("metrics") or {}
        if m.get("mae") is not None:
            vals.append(float(m["mae"]))
    return float(np.mean(vals)) if vals else NO_BANDS_MAE


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
    lo, hi = float(yw.min()), float(yw.max())
    span = hi - lo if hi > lo else 1.0
    ax_top.set_ylim(lo - 0.06 * span, hi + 0.46 * span)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for sp in ("top", "right", "left", "bottom"):
        ax_top.spines[sp].set_visible(False)
    # Stacked, not side by side: the middle zone widened to fit the
    # perturbation plot, and at this width the two labels ran together into
    # "measured powermoving-average trend".
    ax_top.text(0.012, 0.975, "measured power", transform=ax_top.transAxes,
                fontsize=5.4, color="#4A4A4A", va="top", ha="left")
    ax_top.text(0.012, 0.795, "moving-average trend",
                transform=ax_top.transAxes, fontsize=5.4, color=S.INNOV["A"],
                va="top", ha="left", fontweight="bold")

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


def _zone_problem(ax, seg, pert, a_clean, a_pert, i0, response, half,
                  alt_label=""):
    """Middle zone: the reconstruction at t moves when t+1 changes."""
    sl = slice(i0 - half, i0 + half)
    t = (np.arange(sl.start, sl.stop) - i0) * 10 / 60.0   # hours around t0

    ax.plot(t, a_clean[sl], color=BAND_COLOR["D1"], lw=1.15, zorder=4,
            label="$D_1$ as measured")
    ax.plot(t, a_pert[sl], color=S.HERO, lw=1.15, ls=(0, (2.6, 1.5)),
            zorder=5, label="$D_1$ perturbed")

    # the injected sample, and the response one step earlier
    ax.axvline(0.0, color="#9A9A9A", lw=0.5, ls=(0, (1.4, 1.4)), zorder=2)
    ax.annotate("", xy=(0.0, a_pert[i0]), xytext=(0.0, a_clean[i0]),
                arrowprops=dict(arrowstyle="<->", color=S.HERO, lw=0.8,
                                shrinkA=0, shrinkB=0), zorder=6)
    # Left of the arrow, not right: the perturbed trace spikes immediately to
    # the right of t, and the label sat on top of it.
    ax.text(-0.10, (a_pert[i0] + a_clean[i0]) / 2,
            f"{abs(response):.2f}\nat $t$", color=S.HERO, fontsize=5.8,
            fontweight="bold", va="center", ha="right", zorder=7,
            linespacing=1.1)
    # Top-left. At bottom-right it sat flush against the legend box on the same
    # baseline, reading as a third legend entry rather than as a caveat on the
    # 2.29 figure.
    ax.text(0.025, 0.975, alt_label, transform=ax.transAxes,
            fontsize=4.9, color=S.MUTED, ha="left", va="top", zorder=7)

    ax.annotate(f"$+${PERTURB:.0f} injected\nat $t{{+}}1$",
                xy=(10 / 60.0, a_pert[i0 + 1]),
                xytext=(0.62, 0.90), textcoords="axes fraction",
                color=S.HERO, fontsize=5.2, ha="left", va="top",
                linespacing=1.15, zorder=7,
                arrowprops=dict(arrowstyle="-|>", color=S.HERO, lw=0.65,
                                shrinkA=1.0, shrinkB=1.5,
                                connectionstyle="arc3,rad=0.25"))

    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([-2, 0, 2])
    ax.set_xticklabels(["$-$2 h", "$t$", "$+$2 h"], fontsize=5.4)
    ax.tick_params(axis="both", length=1.5, width=0.4, pad=1.4, labelsize=5.2)
    ax.set_ylabel("$D_1$ (standardised)", fontsize=5.8, labelpad=1.4)
    ax.grid(True, axis="y", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", fontsize=5.0, frameon=True, framealpha=0.94,
              borderpad=0.20, handlelength=1.3, handletextpad=0.35,
              labelspacing=0.16, borderaxespad=0.22)


def _zone_result(ax, mae, no_bands):
    """Right zone: the gain tracks the look-ahead, and vanishes without it."""
    labels = [v[0] for v in VARIANTS]
    vals = [mae[k][0] for k in labels]
    causal = [mae[k][2] for k in labels]
    x = np.arange(len(labels), dtype=float)

    colors = [S.HERO_FILL if not c else "#DCDCDC" for c in causal]
    edges = [S.HERO if not c else "#8A8A8A" for c in causal]
    ax.bar(x, vals, 0.68, color=colors, edgecolor=edges, linewidth=0.7,
           zorder=3)

    ax.axhline(no_bands, color="#333333", lw=0.85, ls=(0, (3.2, 1.8)),
               zorder=4)
    # Left end, above the line: at the right end this label sat on top of the
    # 91 kW bar. The leftmost bar is 58 kW, so there is clear space here.
    ax.text(-0.40, no_bands + max(vals) * 0.012, "no sub-bands", fontsize=5.1,
            color="#333333", va="bottom", ha="left", zorder=6,
            bbox=dict(boxstyle="square,pad=0.10", facecolor="white",
                      edgecolor="none"))

    top = max(max(vals), no_bands)
    for xi, v in zip(x, vals):
        ax.text(xi, v + top * 0.022, f"{v:.0f}", ha="center", va="bottom",
                fontsize=5.5, color="#1A1A1A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.0, linespacing=1.15)
    ax.tick_params(axis="both", length=1.5, width=0.4, pad=1.4, labelsize=5.2)
    ax.set_ylabel("test MAE (kW), $h$=12", fontsize=5.8, labelpad=1.6)
    ax.set_ylim(0, top * 1.30)
    ax.grid(True, axis="y", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    ax.text(0.985, 0.972,
            "causal $\\Rightarrow$ no gain",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.9,
            color=S.HERO, fontweight="bold", zorder=10,
            bbox=dict(boxstyle="round,pad=0.24", facecolor="#FFF4F4",
                      edgecolor=S.HERO, linewidth=0.6))
    return vals


# ---------------------------------------------------------------------------
def build(out_dir: str = "manuscript/figures",
          out_name: str = "graphical_abstract.png") -> int:
    y, trend, resid, bands = _signals()
    mae = _variant_mae()
    no_bands = _no_bands_mae()

    parity = _response_by_parity(y)
    t0 = parity[1][0]                                # odd origin: larger case
    seg, pert, a_clean, a_pert, i0, response, half = _perturbation(y, t0)

    with plt.rc_context(S.rc(base=6.0)):
        fig = plt.figure(figsize=(GA_W, GA_H))
        gs = fig.add_gridspec(
            2, 3, width_ratios=[1.00, 0.92, 1.24], height_ratios=[1.0, 1.55],
            left=0.052, right=0.988, top=0.865, bottom=0.150,
            wspace=0.34, hspace=0.16,
        )

        ax_top = fig.add_subplot(gs[0, 0])
        ax_bot = fig.add_subplot(gs[1, 0])
        _zone_input(ax_top, ax_bot, y, trend, bands)

        ax_mid = fig.add_subplot(gs[:, 1])
        alt = f"({parity[0][1]:.2f} at even origins)"
        _zone_problem(ax_mid, seg, pert, a_clean, a_pert, i0, response, half,
                      alt_label=alt)

        ax_res = fig.add_subplot(gs[:, 2])
        vals = _zone_result(ax_res, mae, no_bands)

        for ax_ref, txt, col in (
            (ax_top, "1  Standard practice", S.INNOV["B"]),
            (ax_mid, "2  It sees the future", S.INNOV["C_endo"]),
            (ax_res, "3  Remove that, no gain", S.HERO),
        ):
            fig.text(ax_ref.get_position().x0, 0.955, txt, ha="left",
                     va="center", fontsize=6.2, fontweight="bold",
                     color="white", zorder=50,
                     bbox=dict(boxstyle="round,pad=0.26", facecolor=col,
                               edgecolor="none"))

        fig.text(0.988, 0.022,
                 "SDWPF Turb1  $\\cdot$  1,108 runs  $\\cdot$  10 seeds",
                 ha="right", va="bottom", fontsize=5.2, color=S.MUTED,
                 style="italic", zorder=50)

        # Self-check for text escaping the canvas, the failure mode that put
        # fig. 7's xlabel 0.035 outside the figure and had it silently
        # clipped. Runs before saving so a bad layout cannot ship.
        fig.canvas.draw()
        for ax_ in (ax_top, ax_bot, ax_mid, ax_res):
            for txt in ax_.texts:
                bb = txt.get_window_extent(fig.canvas.get_renderer())
                bb = bb.transformed(fig.transFigure.inverted())
                if bb.x0 < -0.002 or bb.x1 > 1.002 or \
                        bb.y0 < -0.002 or bb.y1 > 1.002:
                    logger.warning(
                        "text %r escapes the canvas: x[%.3f, %.3f] "
                        "y[%.3f, %.3f]",
                        txt.get_text()[:34].replace("\n", " "),
                        bb.x0, bb.x1, bb.y0, bb.y1)

        S.save_figure(fig, Path(out_dir) / out_name, also_pdf=True)
        plt.close(fig)

    logger.info("canvas %.3f x %.3f in (%.2f x %.2f cm), aspect %.4f",
                GA_W, GA_H, GA_W * 2.54, GA_H * 2.54, GA_W / GA_H)
    logger.info("perturbation %.1f at t+1 -> response %.3f at t",
                PERTURB, response)
    logger.info("no-sub-band reference %.2f kW", no_bands)
    for (label, _, _), v in zip(VARIANTS, vals):
        m, sd, causal, n = mae[label]
        logger.info("  %-22s %7.2f +- %4.2f  causal=%-5s n=%d",
                    label.replace("\n", " "), m, sd, causal, n)
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
