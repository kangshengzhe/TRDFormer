"""Causality test that changes ONE variable: shift the paper's own sub-bands
back by k steps.

WHY THIS IS BETTER THAN RE-DECOMPOSING
--------------------------------------
``gen_dwt_imfs_causal.py`` builds sample-wise causal bands by decomposing a
trailing window and keeping its last sample. That is genuinely causal, but it
changes three things at once: the lookahead disappears, every value lands on
the signal-extension boundary (so the padding rule dominates -- measured
D1 std 0.019 under 'symmetric' vs 0.070 for the batch bands), and the band
scales shift. If the model then gets worse, the cause is ambiguous.

This script instead reuses the EXACT arrays the paper trains on and only
delays them:

    lagged[t] = batch[t - k]

The decomposition algorithm, wavelet, level, partition isolation and scaling
are untouched. What changes is the alignment: the feature at t is a function
of the signal around t-k, so with a large enough k it cannot depend on any
sample after t.

CHOOSING k
----------
Measured influence of a perturbation at t+lead on the batch reconstruction at
t (5.0 standardised units injected, response in standardised units):

    lead   1      2      20     50
    resp   0.457  0.094  0.017  0.0006

At lag k the feature at t is batch[t-k], whose dependence on y(t+1) is the
lead-(k+1) response. k=3 cuts the worst-case lookahead from 0.457 to roughly
0.05; k=6 to roughly 0.03. Both are an order of magnitude below the value the
paper reports for the unshifted bands.

COST: the model loses k steps of freshness in the sub-band channels. It does
NOT lose the current target -- scaled Patv stays in channel 0 either way, so
y(t) is still available. Only the frequency decomposition is delayed.

ADDITIVITY: sum(lagged[t]) == y(t-k) != y(t), so the runner's additivity guard
must be bypassed with ``--allow-non-additive`` on the training side (the guard
exists to catch VMD modes sitting in the DWT slot, not this deliberate shift).

Usage
-----
    python scripts/gen_dwt_imfs_lagged.py --lag 3 \
        --out outputs/manifests/dwt_imfs_lag3.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.scaling import FeatureScaler          # noqa: E402

FEATURES = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
BAND_ORDER = ["D1", "D2", "D3", "D4", "A4"]


def _load_bands(path: str) -> np.ndarray:
    d = np.load(path)
    key = "imfs" if "imfs" in d else ("all_imfs" if "all_imfs" in d
                                      else list(d.keys())[0])
    return d[key].astype(np.float64)


def build(src: str, out_path: str, lag: int, csv_path: str,
          scaler_path: str, partition_path: str) -> np.ndarray:
    bands = _load_bands(src)
    N, K = bands.shape
    print(f"source           : {src}  shape={bands.shape}")
    print(f"lag              : {lag} steps "
          f"({lag * 10} min at the 10-min SDWPF cadence)")

    # Shift back by `lag`, edge-padding the first `lag` rows with the earliest
    # available frame. Those rows sit deep inside the training partition and
    # are never a forecast origin for the test set, so the padding choice has
    # no bearing on the reported metrics.
    lagged = np.empty_like(bands)
    lagged[lag:] = bands[:N - lag]
    lagged[:lag] = bands[0]

    # Verify the shift is exactly what we claim, and quantify the additivity
    # gap so the training side knows it must pass --allow-non-additive.
    df = pd.read_csv(csv_path)
    y = FeatureScaler.load(scaler_path).transform(
        df[FEATURES].to_numpy(float))[:, 0].astype(np.float64)
    with open(partition_path, encoding="utf-8") as fh:
        part = json.load(fh)
    ts, te = part["test"]["start"], part["test"]["end"]

    assert np.array_equal(lagged[lag:], bands[:N - lag]), "shift is wrong"
    resid_now = np.abs(lagged.sum(axis=1) - y).max()
    resid_lag = np.abs(lagged[lag:].sum(axis=1) - y[:N - lag]).max()
    print(f"sum(lagged[t]) vs y[t]     : max {resid_now:.4f}  "
          "(expected non-zero -- needs --allow-non-additive)")
    print(f"sum(lagged[t]) vs y[t-lag] : max {resid_lag:.2e}  "
          "(must be ~1e-7: proves the bands themselves are untouched)")

    print("per-channel std (test partition):")
    for i, b in enumerate(BAND_ORDER[:K]):
        r = np.corrcoef(lagged[ts:te, i], bands[ts:te, i])[0, 1]
        print(f"  {b}: lagged={lagged[ts:te, i].std():.4f}  "
              f"batch={bands[ts:te, i].std():.4f}  "
              f"corr(lagged,batch)={r:.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, all_imfs=lagged.astype(np.float32))
    print(f"wrote {out_path}  shape={lagged.shape}")
    return lagged


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="outputs/manifests/vmd_imfs.npz",
                    help="the batch DWT bands the paper trains on")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--csv", default="data/wind/sdwpf_turb1_cleaned_final.csv")
    ap.add_argument("--scaler", default="outputs/manifests/scaler.pkl")
    ap.add_argument("--partition",
                    default="outputs/manifests/partition_indices_l144_h12.json")
    a = ap.parse_args()
    out = a.out or f"outputs/manifests/dwt_imfs_lag{a.lag}.npz"
    build(a.src, out, a.lag, a.csv, a.scaler, a.partition)
