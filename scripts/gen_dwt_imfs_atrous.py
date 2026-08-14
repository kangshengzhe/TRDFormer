"""Strictly causal multi-scale decomposition that keeps exact additivity.

WHY A THIRD VARIANT
-------------------
Two earlier attempts each fail for a different reason:

1. ``gen_dwt_imfs_causal.py`` (trailing window, keep last sample) IS strictly
   causal, but every value lands on the signal-extension boundary, so the
   padding rule dominates the finest bands. Measured over the test partition,
   D1 std collapses to 0.019 under 'symmetric' against 0.070 for the batch
   bands; even the best mode ('reflect') only reaches r=0.74 on D1. A model
   trained on those features is being punished by boundary artefacts, not by
   the absence of lookahead.

2. ``gen_dwt_imfs_lagged.py`` (shift the paper's bands by k) leaves the
   decomposition byte-identical, but is NOT strictly causal: the batch
   reconstruction at t-k still depends on samples around t-k with an
   influence radius of tens of steps, so band(t-k) can still see y(t+1) and
   beyond. Shifting by 3 cuts the worst-case lookahead roughly six-fold, not
   to zero.

This variant gets both properties at once by dropping the decimation and
using purely causal FIR filtering (the 'a trous' / starlet construction):

    a_0 = y
    a_{j+1}[n] = sum_k h_j[k] * a_j[n-k]      (lfilter: only n-k <= n)
    d_{j+1}    = a_j - a_{j+1}
    output     = [d_1, d_2, d_3, d_4, a_J]

where h_j is the db4 scaling filter with 2^j - 1 zeros inserted between taps,
normalised to unit DC gain.

PROPERTIES
    * strictly causal: lfilter touches no sample after n, at any level.
    * exactly additive: the d_j telescope, so sum(d_j) + a_J == a_0 == y to
      float precision. The runner's additivity guard passes unmodified.
    * no extrapolation anywhere, hence no boundary artefacts. The only
      transient is at the very start of the series, deep inside the training
      partition.

THE HONEST COST: a causal low-pass filter has group delay. At level j the
a trous filter spans (8-1)*2^j + 1 samples, so a_4 lags the true trend by
roughly 56 steps. That lag is intrinsic to causality, not a bug -- an online
system genuinely cannot know the centred trend at t.

Usage
-----
    python scripts/gen_dwt_imfs_atrous.py \
        --out outputs/manifests/dwt_imfs_atrous.npz --level 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import lfilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.scaling import FeatureScaler          # noqa: E402

FEATURES = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
BAND_ORDER = ["D1", "D2", "D3", "D4", "A4"]


def _atrous(h: np.ndarray, j: int) -> np.ndarray:
    """Insert 2**j - 1 zeros between consecutive taps of ``h``."""
    if j == 0:
        return h
    step = 2 ** j
    out = np.zeros((len(h) - 1) * step + 1, dtype=float)
    out[::step] = h
    return out


def build(csv_path: str, partition_path: str, scaler_path: str, out_path: str,
          wavelet: str = "db4", level: int = 4,
          normalize: bool = False) -> np.ndarray:
    import pywt

    df = pd.read_csv(csv_path)
    with open(partition_path, encoding="utf-8") as fh:
        part = json.load(fh)

    X = df[FEATURES].to_numpy(dtype=float)
    y = FeatureScaler.load(scaler_path).transform(X)[:, 0].astype(np.float64)
    N = len(y)

    h = np.asarray(pywt.Wavelet(wavelet).dec_lo, dtype=float)
    h = h / h.sum()                     # unit DC gain -> a_j tracks the level

    print(f"samples          : {N}")
    print(f"scaling filter   : {wavelet} dec_lo, {len(h)} taps, "
          f"normalised sum={h.sum():.6f}")
    print(f"levels           : {level}")

    a = y.copy()
    bands = {}
    for j in range(level):
        hj = _atrous(h, j)
        a_next = lfilter(hj, [1.0], a)
        bands[f"D{j + 1}"] = a - a_next
        span = len(hj)
        print(f"  level {j + 1}: filter span {span:4d} samples "
              f"({span * 10 / 60:5.1f} h), group delay ~{span // 2} steps")
        a = a_next
    bands[f"A{level}"] = a

    imfs = np.stack([bands[b] for b in BAND_ORDER], axis=1)

    # Additivity is structural here (the detail bands telescope), but verify --
    # the runner rejects the file above 1e-4 and a silent failure here would
    # be indistinguishable from a bad decomposition.
    resid = np.abs(imfs.sum(axis=1) - y)
    print(f"\nsum(bands) vs signal : mean {resid.mean():.2e}  "
          f"max {resid.max():.2e}   (must be < 1e-4)")
    assert resid.max() < 1e-6, "additivity broken -- refusing to write"

    # Causality is structural too (lfilter is causal), but assert it by
    # perturbation: change y at a future index and confirm nothing before it
    # moves. This is the exact property the reviewer would ask about.
    probe = N // 2
    y_pert = y.copy()
    y_pert[probe + 1] += 5.0
    a_p = y_pert
    rows = []
    for j in range(level):
        hj = _atrous(h, j)
        a_pn = lfilter(hj, [1.0], a_p)
        rows.append(a_p - a_pn)
        a_p = a_pn
    rows.append(a_p)
    imfs_p = np.stack(rows, axis=1)
    moved = np.abs(imfs_p[probe] - imfs[probe]).max()
    print(f"causality check      : 5.0-unit perturbation at t+1 moves the "
          f"bands at t by {moved:.2e}  (must be 0)")
    assert moved == 0.0, "not causal -- refusing to write"

    ts, te = part["test"]["start"], part["test"]["end"]
    print("\nper-channel std (test partition):")
    for i, b in enumerate(BAND_ORDER):
        print(f"  {b}: {imfs[ts:te, i].std():.4f}")

    if normalize:
        # Undecimated bands carry far more energy per channel than the
        # decimated ones the model was tuned on (D1 std 0.28 vs 0.065). Without
        # this rescaling, a poor result cannot be attributed to the loss of
        # lookahead rather than to a scale mismatch against a learning rate and
        # architecture tuned for the original magnitudes.
        #
        # Statistics come from the TRAIN partition only -- using the full
        # series would leak test information through the scaling constants and
        # invalidate the very comparison this variant exists to make.
        tr_s, tr_e = part["train"]["start"], part["train"]["end"]
        mu = imfs[tr_s:tr_e].mean(axis=0)
        sd = imfs[tr_s:tr_e].std(axis=0)
        sd[sd < 1e-12] = 1.0
        print("\nnormalising per channel using TRAIN statistics "
              "(breaks additivity -- needs --allow-non-additive):")
        for i, b in enumerate(BAND_ORDER):
            print(f"  {b}: mu={mu[i]:+.4f} sd={sd[i]:.4f}")
        imfs = (imfs - mu) / sd
        print("post-normalisation std (test partition): "
              + " ".join(f"{b}={imfs[ts:te, i].std():.3f}"
                         for i, b in enumerate(BAND_ORDER)))
        # Causality survives an affine per-channel map; additivity does not.
        resid2 = np.abs(imfs.sum(axis=1) - y).max()
        print(f"sum(bands) vs signal now max {resid2:.3f} "
              "(expected -- pass --allow-non-additive when training)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, all_imfs=imfs.astype(np.float32))
    print(f"\nwrote {out_path}  shape={imfs.shape}")
    return imfs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="data/wind/sdwpf_turb1_cleaned_final.csv")
    ap.add_argument("--partition",
                    default="outputs/manifests/partition_indices_l144_h12.json")
    ap.add_argument("--scaler", default="outputs/manifests/scaler.pkl")
    ap.add_argument("--out", default="outputs/manifests/dwt_imfs_atrous.npz")
    ap.add_argument("--wavelet", default="db4")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--normalize", action="store_true",
                    help="rescale each band to unit variance using TRAIN "
                         "statistics, so a bad result cannot be blamed on the "
                         "scale mismatch against the decimated bands. Breaks "
                         "additivity: train with --allow-non-additive.")
    a = ap.parse_args()
    build(a.csv, a.partition, a.scaler, a.out, a.wavelet, a.level, a.normalize)
