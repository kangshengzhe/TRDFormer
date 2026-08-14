"""Perturbation test: does a change at t+1 move the sub-bands at t?

This is the diagnostic reported in Section 3.2 of the manuscript. It answers
the question a reader should ask of any decomposition-based pipeline: at the
forecast origin, do the features depend on samples that have not happened yet?

METHOD
Decompose the test partition once as the preprocessing does. Then add AMP
standardised units at t0+1, decompose again, and difference the two at t0. A
strictly causal transform gives exactly zero. The offline DWT does not.

TWO FINDINGS WORTH KNOWING BEFORE READING THE OUTPUT

1. The bands' SUM at t0 does not move -- additivity forces the changes to
   cancel to ~1e-16. The leakage is therefore a REDISTRIBUTION among bands,
   invisible to any check on the reconstructed signal and fully visible to a
   model handed the bands as separate input channels. Checking only the
   reconstruction is the natural thing to do and finds nothing.

2. The magnitude is not a distribution to be summarised by a mean. It is
   exactly two-valued in t0's parity, because the level-1 coefficients live on
   every second sample: t0 either coincides with a coefficient centre or falls
   between two. On SDWPF Turb1 with db4/level 4 and AMP=5.0, D1 moves 2.29
   units at odd t0 and 0.71 at even t0, with zero variance inside each class.
   A median over mixed parities returns 1.50, a value no forecast origin
   actually takes -- which is why this script reports the two classes.

Usage
-----
    python tools/probe_perturbation.py

Runnable from anywhere: the block below puts the project root on sys.path and
chdir's into it, since the paths below are repo-relative.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from data_pipeline.scaling import FeatureScaler          # noqa: E402

CSV = "data/wind/sdwpf_turb1_cleaned_final.csv"
PARTITION = "outputs/manifests/partition_indices_l144_h12.json"
SCALER = "outputs/manifests/scaler.pkl"
FEAT = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]

BANDS = ["D1", "D2", "D3", "D4", "A4"]
WAVELET, LEVEL = "db4", 4
AMP = 5.0
LEADS = (1, 2, 3, 5, 10, 20, 50)
N_PER_PARITY = 12


def bands_of(sig: np.ndarray) -> dict:
    """Reconstruct each sub-band to full length, as the preprocessing does."""
    coeffs = pywt.wavedec(sig, WAVELET, mode="symmetric", level=LEVEL)
    names = ["A4"] + [f"D{i}" for i in range(LEVEL, 0, -1)]
    out = {}
    for i, nm in enumerate(names):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        out[nm] = pywt.waverec(z, WAVELET, mode="symmetric")[: len(sig)]
    return out


def main() -> int:
    df = pd.read_csv(CSV)
    y = FeatureScaler.load(SCALER).transform(
        df[FEAT].to_numpy(float))[:, 0].astype(np.float64)
    with open(PARTITION, encoding="utf-8") as fh:
        part = json.load(fh)
    seg = y[part["test"]["start"]: part["test"]["end"]].copy()
    n = len(seg)
    clean = bands_of(seg)
    print(f"test partition {n} samples, {WAVELET} level {LEVEL}, "
          f"injecting {AMP} standardised units\n")

    def response(t0: int, lead: int) -> np.ndarray:
        p = seg.copy()
        p[t0 + lead] += AMP
        pb = bands_of(p)
        return np.array([pb[b][t0] - clean[b][t0] for b in BANDS])

    # --- 1. additivity: the changes must cancel ------------------------------
    t_ref = n // 2 | 1                                # an odd position
    d = response(t_ref, 1)
    print("Per-band change at t0 for an injection at t0+1 (odd t0):")
    print("  " + "  ".join(f"{b}={v:+.4f}" for b, v in zip(BANDS, d)))
    print(f"  sum = {d.sum():+.2e}  <- cancels: the reconstructed signal at t0"
          f" is unchanged")
    assert abs(d.sum()) < 1e-9, "additivity violated"

    # --- 2. two-valued in parity -------------------------------------------
    print("\nD1 response by parity of t0 "
          f"({N_PER_PARITY} positions each):")
    for parity, name in ((1, "odd "), (0, "even")):
        cand = [t for t in range(n // 4, 3 * n // 4)
                if t % 2 == parity][:N_PER_PARITY]
        vals = np.array([abs(response(t, 1)[0]) for t in cand])
        print(f"  t0 {name}: {vals[0]:.4f}  "
              f"(std over positions {vals.std():.2e}, "
              f"{vals[0] / AMP * 100:.0f}% of the injection)")
        assert vals.std() < 1e-6, f"parity-{parity} response is not constant"

    # --- 3. decay with distance --------------------------------------------
    print("\nLargest-band response vs distance of the injection (odd t0):")
    print(f'  {"lead":>5s} {"max|d|":>9s} {"/AMP":>8s}')
    for lead in LEADS:
        d = response(t_ref, lead)
        print(f"  {lead:5d} {np.abs(d).max():9.4f} "
              f"{np.abs(d).max() / AMP:8.4f}")

    print("\nA strictly causal decomposition returns exactly 0.0 for every "
          "row above;\nsee scripts/gen_dwt_imfs_atrous.py for one that does.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
