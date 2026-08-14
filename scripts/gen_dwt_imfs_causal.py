"""Strictly sample-wise causal DWT sub-bands.

WHY THIS EXISTS
---------------
The sub-bands used in the paper are partition-isolated but not sample-wise
causal: within a partition ``pywt.wavedec``/``waverec`` reconstruct from the
whole segment, so the value at t is influenced by samples after t (measured:
a five-unit perturbation at t+1 moves the reconstruction at t by 2.3
standardised units, decaying to 0.09 two steps out). That is honest in the
paper, but it invites a fair objection: the baselines only ever see a causal
lookback, so part of the DWT's +40.8% ablation effect might come from the
lookahead rather than from frequency separation.

This script removes the lookahead so the question can be answered with a
number. For every index t it decomposes a trailing window ending AT t and
keeps only the final sample of each reconstructed band. Nothing after t enters
the feature at t, by construction.

The trailing window is allowed to reach back across a partition boundary.
That is deliberate: reaching backwards uses only data that is already in the
past at inference time, which is exactly what a deployed forecaster has, and
it avoids a confound -- if the window were clipped at the boundary, the first
~W samples of the test partition would carry degenerate short-window features
and the comparison would measure truncation rather than causality.

Output layout is byte-compatible with ``vmd_imfs.npz`` / ``dwt_imfs.npz``:
``all_imfs`` of shape (N, K=5), float32, channel order [D1, D2, D3, D4, A4].

BOUNDARY MODE MATTERS A LOT
---------------------------
Taking the last sample of the window puts every value on the signal extension
boundary, where the padding rule dominates. Measured over 1005 test indices,
correlation between the causal band and the batch band at the same index:

    mode          D1 std   r(D1)   r(D2)   r(A4)
    symmetric     0.0193    0.44    0.49    0.99
    reflect       0.0837    0.74    0.58    0.99
    constant      0.0230    0.71    0.48    0.99
    antireflect   0.0433   -0.67    0.33    0.96
    smooth        0.0060   -0.52    0.41    0.97
    zero          0.2809    0.04    0.32    0.98
    periodic      0.3894    0.00    0.08    0.98
    (batch ref.)  0.0695

``symmetric`` (half-point mirror) makes the signal locally even at the edge,
so its derivative there is ~0 and the finest detail collapses (std 0.019 vs
0.070). ``antireflect``/``smooth`` extrapolate the local slope and invert the
sign of D1. ``reflect`` (whole-point mirror) tracks the batch bands best and
is therefore the default here.

Usage
-----
    python scripts/gen_dwt_imfs_causal.py \
        --out outputs/manifests/dwt_imfs_causal_reflect.npz \
        --window 512 --mode reflect
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.scaling import FeatureScaler          # noqa: E402

FEATURES = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
BAND_ORDER = ["D1", "D2", "D3", "D4", "A4"]


def _wavedec_last(win: np.ndarray, wavelet: str, level: int,
                  mode: str = "reflect") -> np.ndarray:
    """Return the final sample of each reconstructed band for one window.

    ``waverec`` output can be one longer than the input for odd lengths, so the
    reconstruction is trimmed to the window length before the last sample is
    taken.
    """
    import pywt

    n = len(win)
    coeffs = pywt.wavedec(win, wavelet, mode=mode, level=level)
    # wavedec order: [A_L, D_L, ..., D_1]; we want [D1..DL, AL]
    names = ["A4"] + [f"D{i}" for i in range(level, 0, -1)]
    out = {}
    for i, nm in enumerate(names):
        zeroed = [np.zeros_like(c) for c in coeffs]
        zeroed[i] = coeffs[i]
        rec = pywt.waverec(zeroed, wavelet, mode=mode)[:n]
        out[nm] = float(rec[-1])
    return np.array([out[b] for b in BAND_ORDER], dtype=np.float64)


def build(csv_path: str, partition_path: str, scaler_path: str,
          out_path: str, wavelet: str = "db4", level: int = 4,
          window: int = 512, min_window: int = 128,
          mode: str = "reflect") -> np.ndarray:
    df = pd.read_csv(csv_path)
    with open(partition_path, encoding="utf-8") as fh:
        part = json.load(fh)

    # Standardise with the training-fit scaler, exactly as the batch DWT path
    # does -- the decomposition must act on the same signal, or the comparison
    # confounds causality with scaling.
    X = df[FEATURES].to_numpy(dtype=float)
    scaler = FeatureScaler.load(scaler_path)
    y = scaler.transform(X)[:, 0].astype(np.float64)
    N = len(y)

    bounds = {k: (part[k]["start"], part[k]["end"])
              for k in ("train", "valid", "test") if k in part}
    print(f"samples          : {N}")
    print(f"partitions       : "
          + ", ".join(f"{k}[{a}:{b}]" for k, (a, b) in bounds.items()))
    print(f"wavelet / level  : {wavelet} / {level}")
    print(f"trailing window  : {window} (min {min_window})")
    print(f"boundary mode    : {mode}")

    imfs = np.zeros((N, level + 1), dtype=np.float64)
    t0 = time.time()
    for t in range(N):
        lo = max(0, t - window + 1)
        if t - lo + 1 < min_window:                 # very start of the series
            lo = 0
        win = y[lo:t + 1]
        if len(win) < (2 ** level) * 2:             # too short for level-4 db4
            imfs[t, BAND_ORDER.index("A4")] = win[-1]
            continue
        imfs[t] = _wavedec_last(win, wavelet, level, mode)
        if t and t % 5000 == 0:
            el = time.time() - t0
            print(f"  {t:6d}/{N}  {el:6.1f}s  eta {el / t * (N - t):6.1f}s",
                  flush=True)

    print(f"done in {time.time() - t0:.1f}s")

    # A causal decomposition does NOT reconstruct the signal exactly -- each
    # band's value comes from a different window. Report the gap so nobody
    # mistakes this file for the batch one, where the identity holds to ~1e-7.
    resid = np.abs(imfs.sum(axis=1) - y)
    print(f"sum(bands) vs signal: mean {resid.mean():.4f}  max {resid.max():.4f}"
          "   (expected non-zero: each band uses its own trailing window)")
    print(f"per-channel std      : "
          + " ".join(f"{b}={imfs[:, i].std():.4f}"
                     for i, b in enumerate(BAND_ORDER)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, all_imfs=imfs.astype(np.float32))
    print(f"wrote {out_path}  shape={imfs.shape}")
    return imfs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/wind/sdwpf_turb1_cleaned_final.csv")
    ap.add_argument("--partition",
                    default="outputs/manifests/partition_indices_l144_h12.json")
    ap.add_argument("--scaler", default="outputs/manifests/scaler.pkl")
    ap.add_argument("--out", default="outputs/manifests/dwt_imfs_causal.npz")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--min-window", type=int, default=128)
    ap.add_argument("--mode", default="reflect",
                    choices=["reflect", "symmetric", "constant", "antireflect",
                             "smooth", "zero", "periodic", "periodization"],
                    help="pywt signal-extension mode; see module docstring for "
                         "why 'reflect' is the default")
    a = ap.parse_args()
    build(a.csv, a.partition, a.scaler, a.out, a.wavelet, a.level,
          a.window, a.min_window, a.mode)
