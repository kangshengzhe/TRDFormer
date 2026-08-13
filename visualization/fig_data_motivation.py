"""
Figure 1: the data, the physics, and why the architecture is shaped as it is.

REPLACES ``fig01_workflow_overview`` AND ``fig_feature_correlation``
-------------------------------------------------------------------
The old Fig. 1 was a nine-panel workflow poster that ended with training
curves and accuracy bars - results material shown before the method, and
duplicating three later figures. What a first figure should instead do is
*earn* the method: every design choice defended in Section 3 should be
traceable to something visible here.

So the panels are organised as four argument bands, each closing with the
design consequence it implies:

  1  The asset and its signal        -> what the model is given
  2  Covariates carry little         -> the exogenous branch can be light
  3  The signal is multi-scale       -> decompose before modelling
  4  Leakage-free protocol           -> the numbers are trustworthy

Band 3 is where the paper's central claim becomes visual. The butterfly
chart in panel (f) puts energy share and step-to-step-variability share on
mirrored axes, which makes their inversion unmissable: A4 holds 97% of the
energy but 4% of the variability, while D1 and D2 together hold ~1% of the
energy and 84% of the variability. A conventional grouped bar chart lets a
reader miss that; mirroring it does not.

Numbers here are computed the same way the manuscript computes them, which
matters in two places:

* **VIF** uses ``statsmodels.variance_inflation_factor`` on the uncentred
  covariate matrix, reproducing the values quoted in Section 2.3
  (Itmp 20.9, Etmp 13.6). Centring the matrix first - arguably the more
  standard definition - gives ~9.6 for both and would contradict the text.
* **Sub-band energy** is ``sum(x^2)`` on the *standardised* test signal, not
  the variance. On a signal whose mean is -0.39 after scaling, the two
  differ enough to move A4's share from 82% to 97%; the latter is what the
  text states.
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
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _style as S         # noqa: E402

logger = logging.getLogger(__name__)

CSV = "data/wind/sdwpf_turb1_cleaned_final.csv"
PARTITION = "outputs/manifests/partition_indices_l144_h12.json"
SCALER = "outputs/manifests/scaler.pkl"

FEAT = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
COV = ["Wspd", "Wdir", "Etmp", "Itmp"]
UNITS = {"Patv": "kW", "Wspd": "m/s", "Wdir": "deg",
         "Etmp": "$^\\circ$C", "Itmp": "$^\\circ$C"}
CH_COLOR = {"Patv": S.HERO, "Wspd": "#2166AC", "Wdir": "#7F7F7F",
            "Etmp": "#E8A33D", "Itmp": "#B07AA1"}

BAND_ORDER = ["D1", "D2", "D3", "D4", "A4"]
BAND_COLOR = {"D1": "#2166AC", "D2": "#4393C3", "D3": "#92C5DE",
              "D4": "#F4A582", "A4": "#B7950B"}
BAND_ROLE = {"D1": "spikes, 20-40 min", "D2": "gusts, 40-80 min",
             "D3": "ramps, 80-160 min", "D4": "sub-diurnal, 160-320 min",
             "A4": "trend, $>$320 min"}

CUT_IN = 3.0
RATED_KW = 1500.0


# --------------------------------------------------------------------------
def _dwt_bands(signal: np.ndarray, wavelet="db4", level=4) -> dict:
    """Additive reconstruction of each db4 sub-band at full signal length."""
    import pywt

    coeffs = pywt.wavedec(signal, wavelet, mode="symmetric", level=level)
    # wavedec returns [cA_L, cD_L, ..., cD_1]
    names = ["A4", "D4", "D3", "D2", "D1"]
    out = {}
    for i, nm in enumerate(names):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        out[nm] = pywt.waverec(z, wavelet, mode="symmetric")[:len(signal)]
    return out


def _vif(df_cov: pd.DataFrame) -> pd.Series:
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df_cov.to_numpy(float)
    return pd.Series([variance_inflation_factor(X, i)
                      for i in range(X.shape[1])], index=df_cov.columns)


# --------------------------------------------------------------------------
def build(out_dir: str = "manuscript/figures/method",
          out_name: str = "fig_data_motivation.png") -> int:
    df = pd.read_csv(CSV)
    with open(PARTITION, encoding="utf-8") as fh:
        part = json.load(fh)
    n = int(part["n_total_rows"])
    ts, te = part["test"]["start"], part["test"]["end"]

    pear = df[FEAT].corr(method="pearson")
    vif = _vif(df[COV])

    # sub-bands on the standardised test signal, i.e. what the model sees
    from data_pipeline.scaling import FeatureScaler

    scaler = FeatureScaler.load(SCALER)
    sig = scaler.transform(df[FEAT].to_numpy(float)[ts:te]).astype(float)[:, 0]
    bands = _dwt_bands(sig)
    energy = {k: float(np.sum(bands[k] ** 2)) for k in BAND_ORDER}
    dvar = {k: float(np.var(np.diff(bands[k]))) for k in BAND_ORDER}
    e_tot, d_tot = sum(energy.values()), sum(dvar.values())
    e_pct = {k: 100 * energy[k] / e_tot for k in BAND_ORDER}
    d_pct = {k: 100 * dvar[k] / d_tot for k in BAND_ORDER}
    logger.info("A4 energy %.2f%% dvar %.2f%% | D1+D2 energy %.2f%% dvar %.2f%%",
                e_pct["A4"], d_pct["A4"], e_pct["D1"] + e_pct["D2"],
                d_pct["D1"] + d_pct["D2"])

    with plt.rc_context(S.rc(base=6.9)):
        fig = plt.figure(figsize=(S.FULL_W, S.FULL_H))
        outer = fig.add_gridspec(
            4, 2, height_ratios=[1.05, 0.92, 1.34, 0.24],
            width_ratios=[1.16, 1.0],
            left=0.098, right=0.972, top=0.938, bottom=0.062,
            hspace=0.92, wspace=0.34,
        )

        # ================= (a) the five SCADA channels =================
        seg0 = 3000
        segn = 720                       # 5 days at 10-min resolution
        inner = GridSpecFromSubplotSpec(len(FEAT), 1, subplot_spec=outer[0, 0],
                                        hspace=0.22)
        t = np.arange(segn) * 10 / 60.0 / 24.0
        scada_axes = []
        for i, ch in enumerate(FEAT):
            axs = fig.add_subplot(inner[i])
            scada_axes.append(axs)
            axs.plot(t, df[ch].to_numpy()[seg0:seg0 + segn],
                     color=CH_COLOR[ch], lw=0.55)
            axs.set_xlim(t[0], t[-1])
            # unit belongs in the label, not floating at the right edge of
            # the axes where it was clipped and sat on top of the trace
            axs.set_ylabel(f"{ch}\n{UNITS[ch]}", rotation=0, ha="right",
                           va="center", labelpad=3, fontsize=5.8,
                           linespacing=1.05, color=CH_COLOR[ch])
            axs.tick_params(labelsize=5.2, length=1.4)
            axs.set_yticks([])
            for sp in ("top", "right", "left"):
                axs.spines[sp].set_visible(False)
            if i == 0:
                axs.set_title("Five SCADA channels, 5 days",
                              fontsize=6.9, pad=3)
                # extra left offset: the two-line "Patv / kW" label already
                # occupies the default outside-left position
                S.panel_tag(axs, "a", size=6.6, loc="outside left",
                            boxed=False, dx=-0.075)
            if i == len(FEAT) - 1:
                axs.set_xlabel("days", labelpad=1.2, fontsize=6.2)
            else:
                axs.set_xticklabels([])

        # ================= (b) measured power curve =================
        axb = fig.add_subplot(outer[0, 1])
        hb = axb.hexbin(df["Wspd"], df["Patv"], gridsize=42, mincnt=1,
                        cmap="Blues", bins="log", linewidths=0)
        edges = np.arange(0, 26, 1.0)
        mid = 0.5 * (edges[:-1] + edges[1:])
        binned = df.groupby(pd.cut(df["Wspd"], edges), observed=True)["Patv"].mean()
        axb.plot(mid[:len(binned)], binned.to_numpy(), color=S.HERO, lw=1.1,
                 marker="o", ms=1.8, zorder=5, label="binned mean")
        axb.axvline(CUT_IN, color="#555555", lw=0.6, ls=(0, (2.5, 1.6)),
                    zorder=4)
        axb.axhline(RATED_KW, color="#555555", lw=0.6, ls=(0, (2.5, 1.6)),
                    zorder=4)
        axb.text(CUT_IN + 0.35, 1330, "cut-in\n3 m/s", fontsize=5.4,
                 color="#555555", va="top")
        axb.text(20.5, RATED_KW - 60, "rated", fontsize=5.4, color="#555555",
                 ha="right", va="top")
        axb.set_xlim(0, 24)
        axb.set_ylim(-40, 1640)
        axb.set_xlabel("wind speed (m/s)", labelpad=1.5)
        axb.set_ylabel("active power (kW)", labelpad=2)
        axb.grid(True, zorder=0)
        axb.spines["top"].set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.set_title("Measured power curve", fontsize=6.9, pad=3)
        axb.legend(loc="lower right", fontsize=5.5, borderpad=0.24,
                   handlelength=1.2, framealpha=0.92)
        S.panel_tag(axb, "b", size=6.6, loc="upper left")
        cb = fig.colorbar(hb, ax=axb, fraction=0.030, pad=0.015)
        cb.ax.set_title("count", fontsize=5.0, pad=2)
        cb.ax.tick_params(labelsize=4.8, length=1.2)
        cb.outline.set_linewidth(0.3)

        # ================= (c) correlation matrix =================
        # Band 2 gets its own nested gridspec: the outer width ratios are
        # shared by every row, and a square 5x5 matrix placed in the wide
        # left cell left a third of the band empty.
        band2 = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, :],
                                        width_ratios=[0.60, 1.0], wspace=0.30)
        axc = fig.add_subplot(band2[0, 0])
        M = pear.to_numpy().copy()
        mask = np.triu(np.ones_like(M, dtype=bool), k=1)
        Mm = np.ma.array(M, mask=mask)
        im = axc.imshow(Mm, cmap="RdBu_r", vmin=-1, vmax=1)
        axc.set_xticks(range(len(FEAT)))
        axc.set_xticklabels(FEAT, fontsize=6.0, rotation=42, ha="right")
        axc.set_yticks(range(len(FEAT)))
        axc.set_yticklabels(FEAT, fontsize=6.0)
        axc.tick_params(length=0)
        for i in range(len(FEAT)):
            for j in range(i + 1):
                v = M[i, j]
                axc.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=5.7,
                         color="white" if abs(v) > 0.55 else "#1A1A1A",
                         fontweight="bold" if (i, j) in {(1, 0), (4, 3)}
                         else "normal")
        for s in axc.spines.values():
            s.set_visible(False)
        axc.set_title("Pearson correlation", fontsize=6.9, pad=3)
        S.panel_tag(axc, "c", size=6.6, loc="outside left", boxed=False)

        # ================= (d) VIF =================
        axd = fig.add_subplot(band2[0, 1])
        vs = vif.sort_values()
        cols = ["#B2182B" if v > 10 else "#E8A33D" if v > 5 else "#2166AC"
                for v in vs.to_numpy()]
        axd.barh(range(len(vs)), vs.to_numpy(), color=cols, height=0.58,
                 zorder=3)
        for i, v in enumerate(vs.to_numpy()):
            axd.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=6.0,
                     fontweight="bold", color="#1A1A1A")
        axd.axvline(5, color="#E8A33D", lw=0.7, ls=(0, (2.5, 1.6)), zorder=4)
        axd.axvline(10, color="#B2182B", lw=0.7, ls=(0, (2.5, 1.6)), zorder=4)
        # The thresholds are named in the title rather than labelled in the
        # plot: at 5 and 10 the labels landed on the x ticks of the same
        # value, and the bar colours already encode the three severity
        # regimes.
        axd.set_yticks(range(len(vs)))
        axd.set_yticklabels(vs.index, fontsize=6.0)
        axd.tick_params(axis="y", length=0)
        axd.set_xlim(0, max(vs.to_numpy()) * 1.28)
        axd.set_xlabel("variance inflation factor", labelpad=1.5)
        axd.grid(True, axis="x", zorder=0)
        for sp in ("top", "right", "left"):
            axd.spines[sp].set_visible(False)
        axd.set_title("Covariate multicollinearity  "
                      "(dashed: VIF = 5 and 10)", fontsize=6.9, pad=3)
        S.panel_tag(axd, "d", size=6.6, loc="lower right")

        # ================= (e) DWT sub-bands =================
        w0, w1 = 1200, 1800
        inner2 = GridSpecFromSubplotSpec(len(BAND_ORDER), 1,
                                         subplot_spec=outer[2, 0], hspace=0.20)
        tt = np.arange(w1 - w0) * 10 / 60.0
        band_axes = []
        for i, nm in enumerate(BAND_ORDER):
            axe = fig.add_subplot(inner2[i])
            band_axes.append(axe)
            axe.plot(tt, bands[nm][w0:w1], color=BAND_COLOR[nm], lw=0.55)
            axe.axhline(0, color="0.8", lw=0.35, zorder=0)
            axe.set_xlim(tt[0], tt[-1])
            axe.set_ylabel(f"${nm[0]}_{{{nm[1]}}}$", rotation=0, ha="right",
                           va="center", labelpad=3, fontsize=6.4,
                           color=BAND_COLOR[nm])
            axe.set_yticks([])
            axe.tick_params(labelsize=5.2, length=1.4)
            for sp in ("top", "right", "left"):
                axe.spines[sp].set_visible(False)
            axe.text(0.995, 0.92, BAND_ROLE[nm], transform=axe.transAxes,
                     ha="right", va="top", fontsize=5.0, color="#888888")
            if i == 0:
                axe.set_title("db4 sub-bands of standardised power",
                              fontsize=6.9, pad=3)
                S.panel_tag(axe, "e", size=6.6, loc="outside left",
                            boxed=False)
            if i == len(BAND_ORDER) - 1:
                axe.set_xlabel("hours into the test set", labelpad=1.2,
                               fontsize=6.2)
            else:
                axe.set_xticklabels([])

        # ================= (f) butterfly: energy vs variability =========
        axf = fig.add_subplot(outer[2, 1])
        yy = np.arange(len(BAND_ORDER))[::-1]
        for y, nm in zip(yy, BAND_ORDER):
            axf.barh(y, -e_pct[nm], height=0.56, color=BAND_COLOR[nm],
                     alpha=0.92, zorder=3)
            axf.barh(y, d_pct[nm], height=0.56, color=BAND_COLOR[nm],
                     alpha=0.52, zorder=3)
            axf.text(-e_pct[nm] - 3.5, y, f"{e_pct[nm]:.1f}", ha="right",
                     va="center", fontsize=5.7, color="#1A1A1A")
            axf.text(d_pct[nm] + 3.5, y, f"{d_pct[nm]:.1f}", ha="left",
                     va="center", fontsize=5.7, color="#1A1A1A")
        axf.axvline(0, color="#333333", lw=0.7, zorder=4)
        axf.set_yticks(yy)
        axf.set_yticklabels([f"${nm[0]}_{{{nm[1]}}}$" for nm in BAND_ORDER],
                            fontsize=6.4)
        axf.tick_params(axis="y", length=0)
        axf.set_xlim(-128, 128)
        axf.set_xticks([-100, -50, 0, 50, 100])
        axf.set_xticklabels(["100", "50", "0", "50", "100"], fontsize=5.6)
        axf.set_xlabel("share of total (%)", labelpad=1.5)
        axf.text(-64, len(BAND_ORDER) - 0.32, "energy", ha="center",
                 va="bottom", fontsize=6.2, fontweight="bold",
                 color="#1A1A1A")
        axf.text(64, len(BAND_ORDER) - 0.32, "$\\Delta$-variability",
                 ha="center", va="bottom", fontsize=6.2, fontweight="bold",
                 color="#1A1A1A")
        axf.grid(True, axis="x", zorder=0)
        for sp in ("top", "right", "left"):
            axf.spines[sp].set_visible(False)
        axf.set_ylim(-0.62, len(BAND_ORDER) - 0.10)
        S.panel_tag(axf, "f", size=6.6, loc="lower right")

        # ================= (g) leakage-free protocol =================
        axg = fig.add_subplot(outer[3, :])
        spans = [("train", 0, part["train"]["end"], "#4C72B0"),
                 ("valid", part["valid"]["start"], part["valid"]["end"],
                  "#E8A33D"),
                 ("test", part["test"]["start"], part["test"]["end"], S.HERO)]
        for name, lo, hi, col in spans:
            axg.barh(0, (hi - lo) / n, left=lo / n, height=0.52, color=col,
                     alpha=0.90, edgecolor="white", linewidth=0.6, zorder=3)
            axg.text((lo + hi) / 2 / n, 0, f"{name}  {100*(hi-lo)/n:.0f}%",
                     ha="center", va="center", fontsize=5.9, color="white",
                     fontweight="bold", zorder=5)
        axg.set_xlim(0, 1)
        axg.set_ylim(-0.42, 0.42)
        axg.set_yticks([])
        axg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axg.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=5.6)
        axg.set_xlabel(f"chronological position in the {n:,} 10-min records"
                       "  \u00b7  scaler and DWT fitted on train only, "
                       "never refitted downstream", labelpad=1.5, fontsize=6.1)
        for sp in ("top", "right", "left"):
            axg.spines[sp].set_visible(False)
        axg.grid(True, axis="x", zorder=0)
        S.panel_tag(axg, "g", size=6.6, loc="outside left", boxed=False)

        # ================= bands, labels and consequences ==============
        def box(bottom_ax, top_ax, num, title, consequence, color):
            """Enclose one argument band and print the design it implies."""
            y0 = bottom_ax.get_position().y0
            y1 = top_ax.get_position().y1
            S.group_box(fig, 0.020, y0 - 0.030, 0.990, y1 + 0.030,
                        label=f"{num}  {title}", color=color, lw=0.8,
                        size=6.2, label_side="bottom left")
            fig.text(0.988, y0 - 0.0335, consequence, ha="right",
                     va="center", fontsize=6.0, color=color,
                     fontstyle="italic")

        box(scada_axes[-1], scada_axes[0], "1", "The asset and its signal",
            "$\\Rightarrow$ five channels, one target", "#4A4A4A")
        box(axc, axc, "2", "Covariates carry little information",
            "$\\Rightarrow$ a light exogenous encoder suffices",
            S.INNOV["C_exo"])
        box(band_axes[-1], band_axes[0], "3",
            "The signal is multi-scale and non-stationary",
            "$\\Rightarrow$ decompose before modelling", S.INNOV["B"])
        box(axg, axg, "4", "Leakage-free protocol",
            "$\\Rightarrow$ reported numbers are out-of-sample",
            S.INNOV["A"])

        # No prose restatement of band 3's numbers: the butterfly's own bar
        # labels already carry them, and a text box large enough to hold the
        # sentence could only fit by overlapping the protocol bar below.

        S.save_figure(fig, Path(out_dir) / out_name)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import os

    os.chdir(Path(__file__).resolve().parents[1])
    raise SystemExit(build())
