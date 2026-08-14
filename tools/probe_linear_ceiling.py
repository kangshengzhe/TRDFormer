"""Linear ceiling: does the sub-band lookahead help ANY model, or only ours?

This is the diagnostic reported in Section 5.5 of the manuscript, released so
the check can be repeated on other pipelines. We recommend running it beside
any ablation of a decomposition-based model: it costs one ridge fit and bounds
what the architecture can possibly be contributing.

The ablation says: with the offline sub-bands 58.18 kW, without them 81.95 kW.
If a plain ridge regression on the same windows closes the same gap, the gap
belongs to the features -- not to the iTransformer/LSTM/KAN architecture.

Protocol mirrors the runner: lookback 144, horizon 12, fit on train only,
score on test, MAE in kW so the numbers sit beside Table 2.

    config A  = 5 raw channels x 144 steps            -> mimics "w/o DWT"
    config B  = A + 5 sub-band channels x 144 steps   -> mimics the full model

Expected output on SDWPF Turb1 (manuscript Section 5.5), with last-value
normalisation enabled:
    A: raw only            106.09 kW   (deep ablation on same inputs: 81.95)
    B: raw + offline bands  58.06 kW   (deep model on same inputs: 58.18)
    B: raw + causal bands  108.36 kW

Usage
-----
    python tools/probe_linear_ceiling.py

Runnable from anywhere: the block below puts the project root on sys.path and
chdir's into it, since the data and manifest paths below are repo-relative.
"""
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from data_pipeline.scaling import FeatureScaler

FEATURES = ['Patv', 'Wspd', 'Wdir', 'Etmp', 'Itmp']
LOOKBACK, HORIZON = 144, 12
SOURCES = {
    'batch (paper)': 'outputs/manifests/vmd_imfs.npz',
    'lag3': 'outputs/manifests/dwt_imfs_lag3.npz',
    'atrous causal': 'outputs/manifests/dwt_imfs_atrous.npz',
    'causal reflect': 'outputs/manifests/dwt_imfs_causal_reflect.npz',
}


def load_bands(path):
    d = np.load(path)
    key = ('imfs' if 'imfs' in d
           else 'all_imfs' if 'all_imfs' in d else list(d.keys())[0])
    return d[key].astype(np.float64)


df = pd.read_csv('data/wind/sdwpf_turb1_cleaned_final.csv')
scaler = FeatureScaler.load('outputs/manifests/scaler.pkl')
raw = df[FEATURES].to_numpy(float)
Z = scaler.transform(raw).astype(np.float64)          # (N, 5)
y = Z[:, 0]

# Recover Patv's sigma without touching scaler internals: transform is affine,
# so a unit step in raw Patv maps to 1/sigma in standardised space.
probe0 = scaler.transform(np.zeros((1, 5)))[0, 0]
probe1 = scaler.transform(np.array([[1.0, 0, 0, 0, 0]]))[0, 0]
sigma = 1.0 / (probe1 - probe0)
print(f'Patv sigma = {sigma:.2f} kW per standardised unit')

part = json.load(open('outputs/manifests/partition_indices_l144_h12.json'))
tr = (part['train']['start'], part['train']['end'])
te = (part['test']['start'], part['test']['end'])


def windows(lo, hi, extra=None, stride=1, inorm=False):
    """Build (X, Y) with X = flattened lookback, Y = next HORIZON targets.

    ``inorm`` subtracts each channel's last in-window value from that channel's
    block, and the target channel's last value from Y (the NLinear trick). On a
    non-stationary series this is what lets a purely linear map be competitive
    -- without it the ridge wastes capacity tracking the level and the "linear
    baseline" is unfairly weak, which would understate how much of the paper's
    margin a linear model can reach.
    """
    # upper bound is hi - HORIZON (exclusive): the last origin must leave room
    # for HORIZON targets at indices origin+1 .. origin+HORIZON <= hi-1
    origins = np.arange(lo + LOOKBACK, hi - HORIZON, stride)
    # index matrices: (n_origins, LOOKBACK) and (n_origins, HORIZON)
    back = origins[:, None] - LOOKBACK + np.arange(LOOKBACK)[None, :]
    fwd = origins[:, None] + np.arange(1, HORIZON + 1)[None, :]
    blocks = [Z[back, c] for c in range(Z.shape[1])]
    if extra is not None:
        blocks += [extra[back, c] for c in range(extra.shape[1])]
    if inorm:
        blocks = [b - b[:, -1:] for b in blocks]
    X = np.concatenate(blocks, axis=1)
    Y = y[fwd]
    anchor = y[origins][:, None] if inorm else None
    return X, Y, anchor


def evaluate(extra, label, inorm=False):
    Xtr, Ytr, atr = windows(*tr, extra=extra, stride=2, inorm=inorm)
    Xte, Yte, ate = windows(*te, extra=extra, stride=1, inorm=inorm)
    if inorm:
        Ytr = Ytr - atr
    m = Ridge(alpha=10.0).fit(Xtr, Ytr)
    pred = m.predict(Xte)
    if inorm:
        pred = pred + ate
    mae = float(np.abs(pred - Yte).mean()) * sigma
    print(f'  {label:34s} dims={Xtr.shape[1]:5d}  MAE={mae:7.2f} kW')
    return mae


print(f'\ntrain origins stride 2, test stride 1, '
      f'lookback {LOOKBACK}, horizon {HORIZON}')
print('\nridge regression, MAE in kW (paper Table 2: full 58.18, '
      'w/o DWT 81.95, DLinear 70.37)\n')

for inorm in (False, True):
    tag = 'with last-value normalisation (NLinear-style)' if inorm else \
          'raw ridge, no normalisation'
    print(f'--- {tag} ---')
    base = evaluate(None, 'A: raw channels only (w/o DWT)', inorm)
    for name, path in SOURCES.items():
        try:
            B = load_bands(path)
        except FileNotFoundError:
            print(f'  {name}: missing, skipped')
            continue
        if B.shape[0] != len(y):
            print(f'  {name}: length mismatch, skipped')
            continue
        mae = evaluate(B, f'B: raw + {name}', inorm)
        print(f'  {"":34s} -> {mae - base:+7.2f} kW vs config A '
              f'({(mae / base - 1) * 100:+.1f}%)')
    print()

print('\nReading: if "raw + batch" beats "raw only" by roughly the same margin')
print('the deep ablation reports (-23.8 kW), then the margin is a property of')
print('the features -- available to a linear model -- and not evidence that')
print('the proposed architecture is what exploits the decomposition.')
