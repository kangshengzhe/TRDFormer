"""Build every artefact needed to TRAIN on the second SCADA record (Turkey 2018).

WHAT THIS PRODUCES
------------------
    outputs/manifests/turkey_data.npz        cleaned matrix + segment bounds
    outputs/manifests/turkey_scaler.pkl      scaler, fitted on TRAIN segments only
    outputs/manifests/turkey_segments.json   split, per-segment row counts
    outputs/manifests/turkey_dwt_offline.npz db4 sub-bands, decomposed PER SEGMENT
    outputs/manifests/turkey_dwt_lag12.npz   the same bands delayed 12 steps

WHY IT IS SEPARATE FROM preprocess_cli.py
-----------------------------------------
``preprocess_cli`` assumes a single contiguous series and writes the SDWPF
manifests that all 1,108 published runs read. Extending it would put those
results at risk for no benefit, so this is a parallel, additive path: nothing
here overwrites an SDWPF artefact.

THE GAP PROBLEM, AND WHY EVERY STEP IS SEGMENT-AWARE
----------------------------------------------------
2030 of 52560 ten-minute slots are missing (3.86%), across 32 gaps, three of
which span 520, 546 and 625 rows (up to 4.3 days). ``pywt.wavedec`` has no
notion of a timestamp: hand it a concatenation of what survives and it will
filter straight across every join, manufacturing a step artefact there. A
probe -- or a model -- would then read those artefacts as signal.

So the record is cut at every discontinuity, segments shorter than
MIN_SEGMENT are dropped, and the DWT is run independently inside each
surviving segment. Training windows are likewise confined to one segment
(see data_pipeline/segmented_windowing.py).

CHANNEL LAYOUT
--------------
This turbine has no temperature channels, so the covariate group is
[Wspd, Wdir] rather than SDWPF's [Wspd, Wdir, Etmp, Itmp]:

    with sub-bands : [Patv, D1, D2, D3, D4, A4, Wspd, Wdir]   -> 6 + 2
    without        : [Patv, Wspd, Wdir]                        -> 1 + 2

Usage
-----
    python scripts/prep_turkey.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import numpy as np
import pandas as pd

from data_pipeline.scaling import FeatureScaler

CSV = Path("data/wind_turkey/T1.csv")
OUT = Path("outputs/manifests")
MIN_SEGMENT = 512          # same threshold as tools/cross_dataset_leakage_check.py
LOOKBACK, HORIZON = 144, 12
LAG = 12
WAVELET, LEVEL = "db4", 4
FEATURES = ["Patv", "Wspd", "Wdir"]
TRAIN_FRAC, VALID_FRAC = 0.8, 0.1


def load_and_clean() -> pd.DataFrame:
    """Read the raw CSV and apply the manuscript's three physical rules."""
    df = pd.read_csv(CSV)
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["Date/Time"], format="%d %m %Y %H:%M"),
        "Patv": pd.to_numeric(df["LV ActivePower (kW)"], errors="coerce"),
        "Wspd": pd.to_numeric(df["Wind Speed (m/s)"], errors="coerce"),
        "Wdir": pd.to_numeric(df["Wind Direction (°)"], errors="coerce"),
    }).sort_values("ts").reset_index(drop=True)

    n0 = len(out)
    out["Patv"] = out["Patv"].clip(lower=0.0)          # negative -> 0
    bad = (out.Wspd < 0) | (out.Wspd > 25) | ((out.Patv > 0) & (out.Wspd < 3))
    out.loc[bad, FEATURES] = np.nan
    out = out.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"raw rows {n0}, after physical rules {len(out)} "
          f"({n0 - len(out)} removed)")
    return out


def split_segments(df: pd.DataFrame):
    """Cut at every timestamp discontinuity; keep segments >= MIN_SEGMENT."""
    gap = df.ts.diff() > pd.Timedelta("10min")
    seg_id = gap.cumsum()
    kept, dropped = [], 0
    for _, g in df.groupby(seg_id):
        if len(g) >= MIN_SEGMENT:
            kept.append(g)
        else:
            dropped += 1
    lens = [len(s) for s in kept]
    print(f"contiguous segments kept {len(kept)} (dropped {dropped} short ones)")
    print(f"  lengths: min {min(lens)}, median {int(np.median(lens))}, "
          f"max {max(lens)}, total {sum(lens)}")
    return kept


def assign_splits(segments):
    """Chronological split by cumulative ROWS, not segment count.

    Splitting by segment count would let a few long segments dominate one side.
    Bounds are returned as half-open [start, end) index pairs into the
    concatenated matrix.
    """
    lens = np.array([len(s) for s in segments])
    cum = np.cumsum(lens) / lens.sum()
    split = np.where(cum <= TRAIN_FRAC, 0,
                     np.where(cum <= TRAIN_FRAC + VALID_FRAC, 1, 2))
    # guarantee every split is non-empty even on an unlucky cumulative curve
    if not (split == 1).any():
        split[max(1, len(split) - 2)] = 1
    if not (split == 2).any():
        split[-1] = 2

    bounds = {0: [], 1: [], 2: []}
    off = 0
    for seg, s in zip(segments, split):
        bounds[int(s)].append((off, off + len(seg)))
        off += len(seg)

    names = {0: "train", 1: "valid", 2: "test"}
    for k in (0, 1, 2):
        rows = sum(b - a for a, b in bounds[k])
        wins = sum(max(b - a - LOOKBACK - HORIZON + 1, 0) for a, b in bounds[k])
        print(f"  {names[k]:5s}: {len(bounds[k]):2d} segments, {rows:6d} rows, "
              f"{wins:6d} windows")
    return bounds


def dwt_per_segment(y: np.ndarray, bounds_all) -> np.ndarray:
    """db4 level-4 DWT run INDEPENDENTLY inside each segment.

    Returns (N, 5) = [D1, D2, D3, D4, A4]; rows outside any segment stay zero.
    Additivity (sum of bands == input) is asserted per segment.
    """
    import pywt

    bands = np.zeros((len(y), LEVEL + 1), dtype=np.float64)
    worst = 0.0
    for a, b in bounds_all:
        seg = y[a:b]
        coeffs = pywt.wavedec(seg, WAVELET, level=LEVEL, mode="symmetric")
        # reconstruct each band to full segment length
        for j in range(LEVEL + 1):
            only = [np.zeros_like(c) for c in coeffs]
            only[j] = coeffs[j]
            rec = pywt.waverec(only, WAVELET, mode="symmetric")[:len(seg)]
            # coeffs[0] is A4; coeffs[1..4] are D4..D1 (coarse -> fine)
            col = LEVEL if j == 0 else LEVEL - j
            bands[a:b, col] = rec
        worst = max(worst, float(np.abs(bands[a:b].sum(axis=1) - seg).max()))
    print(f"  per-segment additivity |sum(bands) - y|: max {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("additivity broken -- decomposition is wrong")
    return bands


def lag_within_segments(bands: np.ndarray, bounds_all, k: int) -> np.ndarray:
    """bands_lagged[t] = bands[t-k], applied INSIDE each segment.

    Lagging across a segment boundary would reintroduce exactly the splice the
    segmentation exists to avoid, so the first k rows of every segment are
    edge-padded from that segment's own first frame.
    """
    out = np.empty_like(bands)
    for a, b in bounds_all:
        seg = bands[a:b]
        out[a + k:b] = seg[:len(seg) - k]
        out[a:a + k] = seg[0]
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("Turkey 2018 -- training artefacts")
    print("=" * 66)

    df = load_and_clean()
    segments = split_segments(df)
    mat = pd.concat(segments, ignore_index=True)
    raw = mat[FEATURES].to_numpy(float)

    print("\nchronological split by rows:")
    bounds = assign_splits(segments)
    bounds_all = sorted(bounds[0] + bounds[1] + bounds[2])

    # ---- scaler: TRAIN segments only -----------------------------------
    train_rows = np.concatenate([raw[a:b] for a, b in bounds[0]], axis=0)
    scaler = FeatureScaler()
    scaler.fit(train_rows)          # returns None, mutates in place
    scaled = scaler.transform(raw).astype(np.float64)
    scaler.save(str(OUT / "turkey_scaler.pkl"))
    print(f"\nscaler fitted on {len(train_rows)} train rows -> "
          f"{OUT / 'turkey_scaler.pkl'}")
    print(f"  Patv sigma = {float(np.nanstd(raw[:, 0])):.2f} kW per unit")

    # ---- DWT inside each segment, on the STANDARDISED target -----------
    print("\ndb4 level-4 DWT, per segment, on standardised Patv:")
    bands = dwt_per_segment(scaled[:, 0], bounds_all)
    lagged = lag_within_segments(bands, bounds_all, LAG)

    np.savez_compressed(OUT / "turkey_dwt_offline.npz",
                        all_imfs=bands.astype(np.float32))
    np.savez_compressed(OUT / "turkey_dwt_lag12.npz",
                        all_imfs=lagged.astype(np.float32))
    np.savez_compressed(OUT / "turkey_data.npz",
                        scaled=scaled.astype(np.float32),
                        raw=raw.astype(np.float32))
    with open(OUT / "turkey_segments.json", "w", encoding="utf-8") as fh:
        json.dump({
            "min_segment": MIN_SEGMENT,
            "lookback": LOOKBACK, "horizon": HORIZON, "lag": LAG,
            "features": FEATURES,
            "patv_sigma_kw": float(np.nanstd(raw[:, 0])),
            "bounds": {"train": bounds[0], "valid": bounds[1],
                       "test": bounds[2]},
        }, fh, indent=2)

    # ---- sanity: the lag must be a pure shift inside every segment ------
    ok = all(
        np.allclose(lagged[a + LAG:b], bands[a:b - LAG])
        for a, b in bounds_all
    )
    print(f"  lag is a pure within-segment shift: {ok}")
    if not ok:
        raise SystemExit("lag construction is wrong")

    print("\nwrote:")
    for f in ("turkey_data.npz", "turkey_scaler.pkl", "turkey_segments.json",
              "turkey_dwt_offline.npz", "turkey_dwt_lag12.npz"):
        p = OUT / f
        print(f"  {p}  ({p.stat().st_size:,} B)")
    print("\nnext: python scripts/run_turkey.py --seeds 42 43 ...")


if __name__ == "__main__":
    main()
