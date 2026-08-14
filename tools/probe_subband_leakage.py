"""Sub-band probe: how much future does a sub-band vector at time t reveal?

This is the diagnostic reported in Section 5.3 of the manuscript, released so
the check can be repeated on other pipelines rather than taken on trust.

The retraining ablations answer "does the model get worse without the
lookahead". This answers the sharper question without training anything: given
ONLY the five sub-band values at time t, how well can a linear map predict
y(t+h)? Measured against the persistence information every model holds anyway
(y(t) itself, in channel 0 regardless of the decomposition), the excess is the
leakage, in the kW units the paper reports.

A linear probe is deliberately weak. Whatever it extracts is a LOWER bound on
what a 3-layer LSTM plus a 4-block iTransformer can extract, so if the probe
already reads the future off the offline bands, the objection is real.

Ridge is fit on the train partition only and scored on test, matching the
paper's protocol so the numbers sit beside Table 2.

Expected output on SDWPF Turb1 (manuscript Table 3), lead-1 gain over the
40.52 kW persistence reference:
    offline bands    -16.61 kW
    causal atrous     -1.50 kW
    causal trailing   -0.68 kW

Usage
-----
    python tools/probe_subband_leakage.py

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
BANDS = ['D1', 'D2', 'D3', 'D4', 'A4']
LEADS = [1, 2, 3, 6, 12]

SOURCES = {
    'batch (paper)': 'outputs/manifests/vmd_imfs.npz',
    'causal symmetric': 'outputs/manifests/dwt_imfs_causal.npz',
    'causal reflect': 'outputs/manifests/dwt_imfs_causal_reflect.npz',
    'lag3': 'outputs/manifests/dwt_imfs_lag3.npz',
    'atrous causal': 'outputs/manifests/dwt_imfs_atrous.npz',
}


def load_bands(path):
    d = np.load(path)
    key = ('imfs' if 'imfs' in d
           else 'all_imfs' if 'all_imfs' in d else list(d.keys())[0])
    return d[key].astype(np.float64)


df = pd.read_csv('data/wind/sdwpf_turb1_cleaned_final.csv')
y = FeatureScaler.load('outputs/manifests/scaler.pkl').transform(
    df[FEATURES].to_numpy(float))[:, 0].astype(np.float64)
part = json.load(open('outputs/manifests/partition_indices_l144_h12.json'))
tr = (part['train']['start'], part['train']['end'])
te = (part['test']['start'], part['test']['end'])

# Patv std in physical units, so R^2 can be read back as an MAE-scale figure
patv = df['Patv'].to_numpy(float)
scale = np.nanstd(patv)
print(f'Patv std = {scale:.1f} kW  (1 standardised unit)')
print(f'train [{tr[0]}:{tr[1]}]  test [{te[0]}:{te[1]}]\n')


def probe(X, lead, label):
    """Fit on train, score on test, predicting y(t+lead) from X(t)."""
    n = len(y)
    idx_tr = np.arange(tr[0], tr[1] - lead)
    idx_te = np.arange(te[0], te[1] - lead)
    m = Ridge(alpha=1.0).fit(X[idx_tr], y[idx_tr + lead])
    pred = m.predict(X[idx_te])
    true = y[idx_te + lead]
    ss_res = float(((true - pred) ** 2).sum())
    ss_tot = float(((true - true.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    mae = float(np.abs(true - pred).mean()) * scale
    return r2, mae


print('Reference: persistence, i.e. predict y(t+h) from y(t) alone.')
print('This information is available to EVERY model -- scaled Patv is')
print('channel 0 whether or not the DWT is enabled.\n')
base = {}
X_persist = y[:, None]
print(f'{"lead":>5s} {"R2":>8s} {"MAE kW":>9s}')
for h in LEADS:
    r2, mae = probe(X_persist, h, 'persist')
    base[h] = (r2, mae)
    print(f'{h:5d} {r2:8.4f} {mae:9.2f}')

print('\nSub-band vector at t -> y(t+h).  "excess" = improvement over')
print('persistence, which is what the DWT channels add beyond y(t).\n')
hdr = (f'{"source":18s} {"lead":>5s} {"R2":>8s} {"MAE kW":>9s} '
       f'{"dR2":>8s} {"dMAE kW":>9s}')
print(hdr)
for name, path in SOURCES.items():
    try:
        B = load_bands(path)
    except FileNotFoundError:
        print(f'{name:18s} -- file missing, skipped')
        continue
    if B.shape[0] != len(y):
        print(f'{name:18s} -- length {B.shape[0]} != {len(y)}, skipped')
        continue
    for h in LEADS:
        r2, mae = probe(B, h, name)
        b_r2, b_mae = base[h]
        print(f'{name:18s} {h:5d} {r2:8.4f} {mae:9.2f} '
              f'{r2 - b_r2:+8.4f} {mae - b_mae:+9.2f}')
    print()

print('Reading: a large positive dR2 at lead 1-2 for the batch bands means')
print('the sub-band vector at t carries information about y(t+1) that y(t)')
print('does not -- i.e. the decomposition looked ahead. A causal variant')
print('should show dR2 ~ 0 or negative there.')
