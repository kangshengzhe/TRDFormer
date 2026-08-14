"""Cross-dataset check: does the leakage finding hold on an independent SCADA
record, not just SDWPF?

WHY THIS EXISTS
----------------
Every other diagnostic in this repository (probe_subband_leakage.py,
probe_linear_ceiling.py, the six causalisation retraining variants) uses
SDWPF. A reviewer's natural question is whether the leakage is a quirk of that
one dataset. This script repeats the two model-free diagnostics -- the sub-band
probe and the linear ceiling -- on a second, independently collected
wind-turbine SCADA record: a different turbine, country, year and data
provider, sharing only the timestamp/power/wind-speed fields common to
essentially every public SCADA release.

THE SAMPLING-GRID TRAP, AND WHY THIS SCRIPT IS SEGMENT-AWARE
-------------------------------------------------------------
pywt.wavedec has no notion of a timestamp: it filters whatever array it is
handed, assuming uniform sampling. This record is NOT uniformly sampled --
2030 of 52560 ten-minute slots are absent (3.86%), across 32 gaps, and three
of those gaps span 520, 546 and 625 rows (up to 4.3 days). Dropping the
missing rows and concatenating what remains splices discontinuous segments
together, and the filter bank then manufactures a step artefact at every
splice. A probe run on such a series would read those artefacts as
"information" and the leakage estimate would be partly self-inflicted.

Interpolating the gaps instead is no better here: linear interpolation across
a 4.3-day hole invents a straight line through a third of a week, which then
enters both the decomposition and the evaluation.

So this script decomposes ONLY WITHIN CONTIGUOUS SEGMENTS. The record is split
at every timestamp discontinuity, segments shorter than MIN_SEGMENT are
dropped, and each surviving segment is decomposed independently. That is
strictly cleaner than the SDWPF protocol (which isolates decomposition per
train/valid/test partition): here it is isolated per contiguous run, so no
filter ever crosses a gap. Segments are then assigned to train/test in
chronological order.

WHAT THIS DELIBERATELY DOES NOT DO
-----------------------------------
It does not retrain the full deep architecture on this second dataset. Both
diagnostics here are model-free (ridge regression, or no model at all), so they
need no architecture or hyperparameter choices and cannot be accused of a
tuning artefact. The manuscript is explicit that the retraining results and the
baseline comparison remain SDWPF-only.

DATASET
-------
Single-turbine SCADA record, Turkey, 2018, 10-minute resolution
(dominodatalab/reference-project-wind-turbine-scada, originally distributed via
Kaggle as "Wind Turbine Scada Dataset"). Columns used:
    Date/Time                 -> timestamp
    LV ActivePower (kW)       -> Patv (target)
    Wind Speed (m/s)          -> Wspd
    Wind Direction (deg)      -> Wdir
No temperature channels exist for this turbine; the manuscript's own ablation
shows the temperature covariates contribute 1.9% of MAE on SDWPF, so their
absence does not confound this check. Theoretical_Power_Curve is dropped: it is
a manufacturer curve evaluated on the same Wspd column, not an independent
sensor, so including it would let the probe read wind speed by proxy.

Usage
-----
    python tools/cross_dataset_leakage_check.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import pandas as pd
import pywt
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

CSV = "data/wind_turkey/T1.csv"
WAVELET, LEVEL = "db4", 4
LOOKBACK, HORIZON = 144, 12
LEADS = (1, 2, 3, 6, 12)
BAND_ORDER = ["D1", "D2", "D3", "D4", "A4"]
WSPD_MIN, WSPD_MAX, CUTIN = 0.0, 25.0, 3.0
STEP = pd.Timedelta(minutes=10)

# A level-4 db4 decomposition needs room: the level-4 filter spans
# (8-1)*2^4+1 = 113 samples, and the linear ceiling additionally needs
# LOOKBACK+HORIZON = 156 samples per window. 512 gives comfortable margin on
# both while still retaining most of the record.
MIN_SEGMENT = 512
TRAIN_FRACTION = 0.8


def load_segments() -> tuple[list[pd.DataFrame], dict]:
    """Clean, then split at every timestamp discontinuity."""
    df = pd.read_csv(CSV).rename(columns={
        "LV ActivePower (kW)": "Patv",
        "Wind Speed (m/s)": "Wspd",
        "Wind Direction (\u00b0)": "Wdir",
    })
    df["timestamp"] = pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M")
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_raw = len(df)

    # Same three physical rules as data_pipeline/cleaning.py, on the columns
    # this turbine has. Rule-flagged rows become NaN and are then dropped,
    # which creates further discontinuities -- handled by the split below.
    df.loc[df["Patv"] < 0, "Patv"] = 0.0
    bad_speed = (df["Wspd"] < WSPD_MIN) | (df["Wspd"] > WSPD_MAX)
    below_cutin = (df["Patv"] > 0) & (df["Wspd"] < CUTIN)
    n_flagged = int((bad_speed | below_cutin).sum())
    df.loc[bad_speed | below_cutin, ["Patv", "Wspd", "Wdir"]] = np.nan
    df = df.dropna(subset=["Patv", "Wspd", "Wdir"]).reset_index(drop=True)

    # Split wherever consecutive timestamps are not exactly one step apart.
    gap_after = df["timestamp"].diff().shift(-1) != STEP
    cut = np.flatnonzero(gap_after.to_numpy())
    bounds = np.concatenate(([0], cut + 1))
    segs = [df.iloc[a:b].reset_index(drop=True)
            for a, b in zip(bounds[:-1], bounds[1:])]
    segs.append(df.iloc[bounds[-1]:].reset_index(drop=True))

    kept = [s for s in segs if len(s) >= MIN_SEGMENT]
    info = {
        "n_raw": n_raw,
        "n_flagged": n_flagged,
        "n_after_rules": len(df),
        "n_segments_total": len(segs),
        "n_segments_kept": len(kept),
        "n_rows_kept": sum(len(s) for s in kept),
    }
    return kept, info


def dwt_offline(y: np.ndarray) -> np.ndarray:
    """Batch DWT over the whole segment -- the paper's offline decomposition."""
    coeffs = pywt.wavedec(y, WAVELET, mode="symmetric", level=LEVEL)
    names = ["A4"] + [f"D{i}" for i in range(LEVEL, 0, -1)]
    out = {}
    for i, nm in enumerate(names):
        z = [np.zeros_like(c) for c in coeffs]
        z[i] = coeffs[i]
        out[nm] = pywt.waverec(z, WAVELET, mode="symmetric")[: len(y)]
    return np.stack([out[b] for b in BAND_ORDER], axis=1)


def _atrous(h: np.ndarray, j: int) -> np.ndarray:
    if j == 0:
        return h
    step = 2 ** j
    out = np.zeros((len(h) - 1) * step + 1, dtype=float)
    out[::step] = h
    return out


def dwt_causal(y: np.ndarray) -> np.ndarray:
    """Strictly causal undecimated filter bank; same maths as
    scripts/gen_dwt_imfs_atrous.py."""
    h = np.asarray(pywt.Wavelet(WAVELET).dec_lo, dtype=float)
    h = h / h.sum()
    a = y.copy()
    bands = {}
    for j in range(LEVEL):
        hj = _atrous(h, j)
        a_next = lfilter(hj, [1.0], a)
        bands[f"D{j + 1}"] = a - a_next
        a = a_next
    bands[f"A{LEVEL}"] = a
    return np.stack([bands[b] for b in BAND_ORDER], axis=1)


def probe_pairs(segs, bands_of, lead, scaler):
    """Collect (features at t, target at t+lead) across segments."""
    X, Y = [], []
    for s in segs:
        z = scaler.transform(s[["Patv", "Wspd", "Wdir"]].to_numpy(float))
        y = z[:, 0]
        B = bands_of(y)
        if len(y) <= lead:
            continue
        X.append(B[: len(y) - lead])
        Y.append(y[lead:])
    return np.concatenate(X), np.concatenate(Y)


def persistence_pairs(segs, lead, scaler):
    X, Y = [], []
    for s in segs:
        z = scaler.transform(s[["Patv", "Wspd", "Wdir"]].to_numpy(float))
        y = z[:, 0]
        if len(y) <= lead:
            continue
        X.append(y[: len(y) - lead, None])
        Y.append(y[lead:])
    return np.concatenate(X), np.concatenate(Y)


def ceiling_windows(segs, bands_of, scaler, stride):
    """Flattened-lookback windows, never crossing a segment boundary."""
    Xs, Ys, As = [], [], []
    for s in segs:
        z = scaler.transform(s[["Patv", "Wspd", "Wdir"]].to_numpy(float))
        y = z[:, 0]
        n = len(y)
        if n < LOOKBACK + HORIZON + 1:
            continue
        extra = bands_of(y) if bands_of is not None else None
        origins = np.arange(LOOKBACK, n - HORIZON, stride)
        back = origins[:, None] - LOOKBACK + np.arange(LOOKBACK)[None, :]
        fwd = origins[:, None] + np.arange(1, HORIZON + 1)[None, :]
        blocks = [z[back, c] for c in range(z.shape[1])]
        if extra is not None:
            blocks += [extra[back, c] for c in range(extra.shape[1])]
        blocks = [b - b[:, -1:] for b in blocks]        # NLinear normalisation
        Xs.append(np.concatenate(blocks, axis=1))
        Ys.append(y[fwd])
        As.append(y[origins][:, None])
    return (np.concatenate(Xs), np.concatenate(Ys), np.concatenate(As))


def main() -> int:
    segs, info = load_segments()
    print(f"raw rows                    : {info['n_raw']}")
    print(f"rows flagged by rules       : {info['n_flagged']}")
    print(f"rows after rules            : {info['n_after_rules']}")
    print(f"contiguous segments         : {info['n_segments_total']}")
    print(f"segments >= {MIN_SEGMENT} rows        : {info['n_segments_kept']}")
    print(f"rows retained               : {info['n_rows_kept']} "
          f"({100 * info['n_rows_kept'] / info['n_raw']:.1f}% of raw)")
    lens = [len(s) for s in segs]
    print(f"segment lengths             : min {min(lens)}, "
          f"median {int(np.median(lens))}, max {max(lens)}")

    # Chronological segment-level split: no segment is shared between fits.
    n_train_seg = max(1, int(round(TRAIN_FRACTION * len(segs))))
    tr_segs, te_segs = segs[:n_train_seg], segs[n_train_seg:]
    print(f"\ntrain segments              : {len(tr_segs)} "
          f"({sum(len(s) for s in tr_segs)} rows)")
    print(f"test segments               : {len(te_segs)} "
          f"({sum(len(s) for s in te_segs)} rows)")
    print(f"train period                : {tr_segs[0]['timestamp'].iloc[0]} "
          f".. {tr_segs[-1]['timestamp'].iloc[-1]}")
    print(f"test period                 : {te_segs[0]['timestamp'].iloc[0]} "
          f".. {te_segs[-1]['timestamp'].iloc[-1]}")

    # Scaler fitted on training segments only.
    train_raw = np.concatenate(
        [s[["Patv", "Wspd", "Wdir"]].to_numpy(float) for s in tr_segs])
    scaler = StandardScaler().fit(train_raw)
    sigma = scaler.scale_[0]
    print(f"Patv sigma                  : {sigma:.2f} kW per standardised unit")

    # Additivity sanity check on a test segment, both decompositions.
    zt = scaler.transform(te_segs[0][["Patv", "Wspd", "Wdir"]].to_numpy(float))
    yt = zt[:, 0]
    for nm, fn in (("offline", dwt_offline), ("causal ", dwt_causal)):
        r = np.abs(fn(yt).sum(axis=1) - yt).max()
        print(f"{nm} additivity (test seg) : max|sum-y| = {r:.2e}")

    print("\n=== Sub-band probe: MAE (kW) for y(t+lead) from bands(t) alone ===")
    print("(fit on train segments, scored on test segments; no network)")
    print(f'{"lead":>5s} {"persist":>9s} {"offline":>9s} {"causal":>9s} '
          f'{"offline gain":>13s} {"causal gain":>12s} {"ratio":>7s}')
    probe_rows = []
    for lead in LEADS:
        res = {}
        for label, fn in (("persist", None),
                          ("offline", dwt_offline),
                          ("causal", dwt_causal)):
            if fn is None:
                Xtr, Ytr = persistence_pairs(tr_segs, lead, scaler)
                Xte, Yte = persistence_pairs(te_segs, lead, scaler)
            else:
                Xtr, Ytr = probe_pairs(tr_segs, fn, lead, scaler)
                Xte, Yte = probe_pairs(te_segs, fn, lead, scaler)
            m = Ridge(alpha=1.0).fit(Xtr, Ytr)
            res[label] = float(np.abs(m.predict(Xte) - Yte).mean()) * sigma
        g_off = res["offline"] - res["persist"]
        g_cau = res["causal"] - res["persist"]
        ratio = g_off / g_cau if abs(g_cau) > 1e-9 else float("inf")
        probe_rows.append((lead, res, g_off, g_cau, ratio))
        print(f"{lead:5d} {res['persist']:9.2f} {res['offline']:9.2f} "
              f"{res['causal']:9.2f} {g_off:+13.2f} {g_cau:+12.2f} "
              f"{ratio:7.1f}x")

    print("\n=== Linear ceiling: ridge on flattened lookback "
          f"(L={LOOKBACK}, H={HORIZON}, NLinear-normalised) ===")
    ceil = {}
    for label, fn in (("raw only (no sub-bands)", None),
                      ("raw + offline sub-bands", dwt_offline),
                      ("raw + causal sub-bands", dwt_causal)):
        Xtr, Ytr, Atr = ceiling_windows(tr_segs, fn, scaler, stride=2)
        Xte, Yte, Ate = ceiling_windows(te_segs, fn, scaler, stride=1)
        m = Ridge(alpha=10.0).fit(Xtr, Ytr - Atr)
        mae = float(np.abs(m.predict(Xte) + Ate - Yte).mean()) * sigma
        ceil[label] = mae
        print(f"  {label:26s} dims={Xtr.shape[1]:5d}  "
              f"n_train={Xtr.shape[0]:6d}  MAE={mae:7.2f} kW")
    base = ceil["raw only (no sub-bands)"]
    for label in ("raw + offline sub-bands", "raw + causal sub-bands"):
        d = ceil[label] - base
        print(f"  {label:26s} -> {d:+7.2f} kW vs raw only "
              f"({100 * (ceil[label] / base - 1):+.1f}%)")

    print("\nReading: the offline row should show a large negative gain and the")
    print("causal row near zero. Decomposition happens strictly within contiguous")
    print("segments, so no filter crosses a sampling gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
