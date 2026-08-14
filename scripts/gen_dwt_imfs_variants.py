"""Alternative lookahead-free sub-band designs, for the causality ablation.

WHY MORE THAN ONE VARIANT
-------------------------
The first causal attempt (``gen_dwt_imfs_causal.py``: trailing window, take the
last sample of a symmetric-boundary reconstruction) turned out WORSE than using
no sub-bands at all. That is a suspicious result, and before concluding that the
paper's +40.8% DWT effect depends on lookahead, the implementation itself has to
be ruled out as the cause. It is a weak estimator by construction: the sample
taken is exactly at the boundary, where symmetric extension mirrors the recent
past and drives the edge detail coefficients toward a spurious zero. Measured
consequence: causal D1 std 0.019 against the batch 0.065, correlation 0.268.

Two better-posed designs are provided.

``swt``  Undecimated (stationary) wavelet transform on the trailing window,
         last sample per level. Because SWT does not decimate, every level has a
         coefficient defined at every index, which removes the decimation-phase
         problem the decimated DWT has at an edge.

``lag``  The paper's own batch decomposition, but read k steps in the past, so
         every value the model sees was computed from data at least k steps old.
         This is motivated by the measured decay of the lookahead: a five-unit
         perturbation at t+1 moves the reconstruction at t by 2.3 standardised
         units, 0.094 two steps out, and 0.017 at twenty steps. At k=20 the
         residual contamination is therefore under 1% of the immediate effect,
         while the sub-bands keep the quality of the batch transform. If accuracy
         survives this shift, the gain cannot be attributed to lookahead.

Usage
-----
    python scripts/gen_dwt_imfs_variants.py --variant swt --out <path>
    python scripts/gen_dwt_imfs_variants.py --variant lag --lag 20 --out <path>
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


def _load_signal(csv_path: str, scaler_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    X = df[FEATURES].to_numpy(dtype=float)
    return FeatureScaler.load(scaler_path).transform(X)[:, 0].astype(np.float64)


# ---------------------------------------------------------------------------
def build_swt(y: np.ndarray, wavelet: str, level: int, window: int) -> np.ndarray:
    """Undecimated wavelet transform on a trailing window; last sample per level."""
    import pywt

    N = len(y)
    # SWT requires the length to be a multiple of 2**level.
    W = (window // (2 ** level)) * (2 ** level)
    out = np.zeros((N, level + 1), dtype=np.float64)
    t0 = time.time()
    for t in range(N):
        lo = t - W + 1
        if lo < 0:                       # pad the very start by edge-replication
            win = np.concatenate([np.full(-lo, y[0]), y[0:t + 1]])
        else:
            win = y[lo:t + 1]
        coeffs = pywt.swt(win, wavelet, level=level, trim_approx=False,
                          norm=False)
        # pywt.swt returns [(cA_L, cD_L), ..., (cA_1, cD_1)] coarsest first
        for i, (cA, cD) in enumerate(coeffs):
            lvl = level - i              # coeffs[0] is level `level`
            out[t, BAND_ORDER.index(f"D{lvl}")] = cD[-1]
        out[t, BAND_ORDER.index("A4")] = coeffs[0][0][-1]
        if t and t % 5000 == 0:
            el = time.time() - t0
            print(f"  {t:6d}/{N}  {el:6.1f}s  eta {el / t * (N - t):6.1f}s",
                  flush=True)
    print(f"swt done in {time.time() - t0:.1f}s   (window {W})")
    return out


def build_lag(batch_path: str, lag: int, bounds: dict) -> np.ndarray:
    """Shift the batch sub-bands `lag` steps into the past, per partition.

    Shifting inside each partition keeps the partition-isolation property: no
    value is ever taken from a different partition. The first `lag` rows of each
    partition repeat that partition's first available value.
    """
    d = np.load(batch_path)
    key = "all_imfs" if "all_imfs" in d else list(d.keys())[0]
    b = d[key].astype(np.float64)
    out = np.empty_like(b)
    for name, (a, z) in bounds.items():
        seg = b[a:z]
        shifted = np.empty_like(seg)
        shifted[lag:] = seg[:-lag] if lag else seg
        shifted[:lag] = seg[0]
        out[a:z] = shifted
        print(f"  {name}: [{a}:{z}] shifted by {lag}")
    return out


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=["swt", "lag"], required=True)
    ap.add_argument("--csv", default="data/wind/sdwpf_turb1_cleaned_final.csv")
    ap.add_argument("--partition",
                    default="outputs/manifests/partition_indices_l144_h12.json")
    ap.add_argument("--scaler", default="outputs/manifests/scaler.pkl")
    ap.add_argument("--batch", default="outputs/manifests/vmd_imfs.npz",
                    help="source sub-bands for the 'lag' variant")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--lag", type=int, default=20)
    a = ap.parse_args()

    part = json.load(open(a.partition, encoding="utf-8"))
    bounds = {k: (part[k]["start"], part[k]["end"])
              for k in ("train", "valid", "test") if k in part}

    if a.variant == "swt":
        y = _load_signal(a.csv, a.scaler)
        print(f"samples {len(y)}  variant swt  wavelet {a.wavelet} level {a.level}")
        imfs = build_swt(y, a.wavelet, a.level, a.window)
    else:
        print(f"variant lag  k={a.lag}  source {a.batch}")
        imfs = build_lag(a.batch, a.lag, bounds)

    ts, te = bounds["test"]
    print("per-channel std (test partition): "
          + " ".join(f"{n}={imfs[ts:te, i].std():.4f}"
                     for i, n in enumerate(BAND_ORDER)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, all_imfs=imfs.astype(np.float32))
    print(f"wrote {a.out}  shape={imfs.shape}")
