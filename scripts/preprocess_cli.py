"""
Preprocessing CLI for the wind power forecasting pipeline.

Orchestrates the full local preprocessing sequence described in the design
document (§ Data_Pipeline, Innovation A):

    1.  Load CSV from data/wind/sdwpf_turb1_cleaned_final.csv
    2.  Construct timestamps from Day + Tmstamp fields, sort chronologically
    3.  Apply physical_rule_clean (physical-rule outlier filtering)
    4.  Linear interpolation for missing values
    5.  Drop remaining NaN rows
    6.  chronological_split (80 / 10 / 10)
    7.  Clip Patv ≥ 0 before scaling
    8.  Fit FeatureScaler on train only; save scaler.pkl
    9.  Apply VMD on train partition only; persist vmd_params.json + vmd_imfs.npz
    10. Save partition indices JSON
    11. Restrict features to {Patv, Wspd, Wdir, Etmp, Itmp} ONLY

Usage
-----
From the repository root:

    python -m scripts.preprocess_cli                           # defaults
    python -m scripts.preprocess_cli --config configs/dataset/sdwpf_turb1.yaml
    python -m scripts.preprocess_cli --lookback 144 --horizon 6 --vmd-k 5
    python -m scripts.preprocess_cli --vmd-off        # skip VMD (vmd_off ablation)

Arguments
---------
--config PATH          Path to a YAML config file.  Keys under the 'dataset'
                       block override defaults; CLI flags override YAML.
--lookback INT         Lookback window length (default: 144).
--horizon INT          Forecast horizon length (default: 6).
--vmd-k INT            Number of VMD modes K (default: 5; range 3–10).
--vmd-off              Disable VMD entirely (vmd_off ablation).
--no-cleaning          Disable physical-rule outlier cleaning (outlier_off ablation).
--csv-path PATH        Override CSV input path.
--manifest-dir PATH    Directory for all output artifacts (default: outputs/manifests).

Exit codes: 0 = success, 1 = configuration / runtime error.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path when the script is run directly
# (e.g. `python scripts/preprocess_cli.py`).  When run as a module
# (`python -m scripts.preprocess_cli`) this is not strictly necessary.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_pipeline.cleaning import physical_rule_clean
from data_pipeline.manifest import PartitionManifest, VMDParams
from data_pipeline.scaling import FeatureScaler
from data_pipeline.splits import chronological_split, persist_partition_indices
from data_pipeline.vmd import (
    apply_vmd_to_partition,
    fit_vmd_on_train,
    persist_vmd_params,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical feature column order; Patv MUST be at index 0.
FEATURE_COLS: list[str] = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]

# Default paths (relative to repo root)
_DEFAULT_CSV = "data/wind/sdwpf_turb1_cleaned_final.csv"
_DEFAULT_MANIFEST_DIR = "outputs/manifests"
_DEFAULT_LOOKBACK = 144
_DEFAULT_HORIZON = 6
_DEFAULT_VMD_K = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("preprocess_cli")


# ---------------------------------------------------------------------------
# YAML config loader (optional dependency on PyYAML)
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path) -> dict:
    """Load a YAML config file and return the inner 'dataset' dict (or the
    top-level dict if no 'dataset' key exists).

    Raises
    ------
    ImportError  if PyYAML is not installed.
    FileNotFoundError  if the file does not exist.
    """
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load --config files. "
            "Install it with `pip install PyYAML`."
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # Support both `dataset: {...}` and flat dict layouts.
    return raw.get("dataset", raw)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess_cli",
        description="Wind power preprocessing pipeline – Innovation A",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a dataset YAML config file.",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=None,
        metavar="INT",
        help=f"Lookback window length (default: {_DEFAULT_LOOKBACK}).",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        metavar="INT",
        help=f"Forecast horizon length (default: {_DEFAULT_HORIZON}).",
    )
    p.add_argument(
        "--vmd-k",
        dest="vmd_k",
        type=int,
        default=None,
        metavar="INT",
        help=f"Number of VMD modes K (default: {_DEFAULT_VMD_K}; range 3–10).",
    )
    p.add_argument(
        "--vmd-off",
        dest="vmd_off",
        action="store_true",
        default=False,
        help="Disable VMD (vmd_off ablation). IMF channels are not appended.",
    )
    p.add_argument(
        "--no-cleaning",
        dest="no_cleaning",
        action="store_true",
        default=False,
        help="Disable physical-rule outlier cleaning (outlier_off ablation).",
    )
    p.add_argument(
        "--csv-path",
        dest="csv_path",
        default=None,
        metavar="PATH",
        help=f"Override CSV input path (default: {_DEFAULT_CSV}).",
    )
    p.add_argument(
        "--manifest-dir",
        dest="manifest_dir",
        default=None,
        metavar="PATH",
        help=f"Directory for all output artifacts (default: {_DEFAULT_MANIFEST_DIR}).",
    )
    return p


def _resolve_config(args: argparse.Namespace) -> dict:
    """Merge YAML config, then apply CLI overrides.

    Priority (highest → lowest):
        1. Explicit CLI flags
        2. YAML config file (--config)
        3. Hard-coded defaults
    """
    cfg: dict = {
        "csv_path": _DEFAULT_CSV,
        "manifest_dir": _DEFAULT_MANIFEST_DIR,
        "lookback": _DEFAULT_LOOKBACK,
        "horizon": _DEFAULT_HORIZON,
        "vmd_k": _DEFAULT_VMD_K,
        "vmd_off": False,
        "cleaning": True,
        # VMD algorithm parameters (not user-overridable via CLI, but YAML can set them)
        "vmd_alpha": 2000.0,
        "vmd_tau": 0.0,
        "vmd_DC": 0,
        "vmd_init": 1,
        "vmd_tol": 1e-7,
    }

    # ------------------------------------------------------------------
    # Apply YAML overrides
    # ------------------------------------------------------------------
    if args.config:
        yaml_cfg = _load_yaml(args.config)

        if "csv_path" in yaml_cfg:
            cfg["csv_path"] = yaml_cfg["csv_path"]
        if "manifest_dir" in yaml_cfg:
            cfg["manifest_dir"] = yaml_cfg["manifest_dir"]
        if "lookback" in yaml_cfg:
            cfg["lookback"] = int(yaml_cfg["lookback"])
        if "horizon" in yaml_cfg:
            cfg["horizon"] = int(yaml_cfg["horizon"])

        # VMD sub-block
        vmd_yaml = yaml_cfg.get("vmd", {})
        if isinstance(vmd_yaml, dict):
            if "K" in vmd_yaml:
                cfg["vmd_k"] = int(vmd_yaml["K"])
            if "enabled" in vmd_yaml:
                cfg["vmd_off"] = not bool(vmd_yaml["enabled"])
            if "alpha" in vmd_yaml:
                cfg["vmd_alpha"] = float(vmd_yaml["alpha"])
            if "tau" in vmd_yaml:
                cfg["vmd_tau"] = float(vmd_yaml["tau"])
            if "DC" in vmd_yaml:
                cfg["vmd_DC"] = int(vmd_yaml["DC"])
            if "init" in vmd_yaml:
                cfg["vmd_init"] = int(vmd_yaml["init"])
            if "tol" in vmd_yaml:
                cfg["vmd_tol"] = float(vmd_yaml["tol"])

        # Cleaning sub-block
        cleaning_yaml = yaml_cfg.get("cleaning", {})
        if isinstance(cleaning_yaml, dict):
            cfg["cleaning"] = bool(cleaning_yaml.get("physical_rules", True))

    # ------------------------------------------------------------------
    # Apply explicit CLI overrides (they take precedence over YAML)
    # ------------------------------------------------------------------
    if args.csv_path is not None:
        cfg["csv_path"] = args.csv_path
    if args.manifest_dir is not None:
        cfg["manifest_dir"] = args.manifest_dir
    if args.lookback is not None:
        cfg["lookback"] = args.lookback
    if args.horizon is not None:
        cfg["horizon"] = args.horizon
    if args.vmd_k is not None:
        cfg["vmd_k"] = args.vmd_k
    if args.vmd_off:
        cfg["vmd_off"] = True
    if args.no_cleaning:
        cfg["cleaning"] = False

    return cfg


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _load_and_sort(csv_path: str | Path) -> pd.DataFrame:
    """Step 1 & 2: Load CSV, construct timestamps, sort chronologically.

    The CSV contains columns: TurbID, Day, Tmstamp, Wspd, Wdir, Etmp, Itmp, Patv.
    We build a proper datetime index from Day (integer day number) and Tmstamp
    (HH:MM string), sort ascending, then restrict to FEATURE_COLS.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with columns [Patv, Wspd, Wdir, Etmp, Itmp] and a
        DatetimeIndex starting from a reference date (day 1 = 2020-01-01).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("  Raw shape: %s, columns: %s", df.shape, df.columns.tolist())

    # Verify required columns exist
    required = {"Day", "Tmstamp"} | set(FEATURE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Build timestamps: treat Day as 1-indexed integer day offset from a
    # reference date. Tmstamp is expected as "HH:MM".
    reference_date = pd.Timestamp("2020-01-01")
    df["_timestamp"] = pd.to_datetime(
        (reference_date + pd.to_timedelta(df["Day"].astype(int) - 1, unit="D"))
        .astype(str)
        + " "
        + df["Tmstamp"].astype(str),
        format="%Y-%m-%d %H:%M",
    )

    df = df.sort_values("_timestamp").reset_index(drop=True)
    df = df.set_index("_timestamp")
    df.index.name = "timestamp"

    # Restrict to canonical feature columns in the required order
    df = df[FEATURE_COLS].copy()

    log.info("  After timestamp construction and sort: shape %s", df.shape)
    return df


def _clean_and_interpolate(df: pd.DataFrame, *, enable_cleaning: bool) -> pd.DataFrame:
    """Steps 3–5: Physical-rule cleaning, linear interpolation, drop NaN rows."""
    # Step 3: Physical-rule outlier filtering
    df_clean, report = physical_rule_clean(df, enable=enable_cleaning)
    if enable_cleaning:
        log.info(
            "  Cleaning report: clipped_neg_patv=%d, wspd_oor=%d, below_cutin=%d",
            report["n_clipped_negative_patv"],
            report["n_marked_wspd_out_of_range"],
            report["n_marked_below_cutin_with_power"],
        )
    else:
        log.info("  Physical-rule cleaning DISABLED (outlier_off ablation).")

    # Step 4: Linear interpolation for missing values
    n_missing_before = df_clean.isnull().sum().sum()
    df_clean = df_clean.interpolate(method="linear", limit_direction="both")
    n_missing_after = df_clean.isnull().sum().sum()
    log.info(
        "  Interpolation: %d missing values before → %d after",
        n_missing_before,
        n_missing_after,
    )

    # Step 5: Drop remaining NaN rows (e.g. at the boundaries)
    n_before_drop = len(df_clean)
    df_clean = df_clean.dropna()
    n_dropped = n_before_drop - len(df_clean)
    if n_dropped:
        log.info("  Dropped %d rows with remaining NaN values.", n_dropped)

    log.info("  Final shape after cleaning + interpolation + dropna: %s", df_clean.shape)
    return df_clean


def _split_and_scale(
    df: pd.DataFrame,
    lookback: int,
    horizon: int,
    manifest_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """Steps 6–8: Chronological split, clip Patv, fit scaler, save scaler.

    Returns
    -------
    train_arr, valid_arr, test_arr : np.ndarray  (scaled, shape (N_x, 5))
    indices : PartitionIndices
    """
    # Step 6: Chronological split
    indices = chronological_split(df, train_ratio=0.8, test_ratio=0.1,
                                  lookback=lookback, horizon=horizon)
    log.info(
        "  Split: train [%d, %d), valid [%d, %d), test [%d, %d)",
        *indices.train, *indices.valid, *indices.test,
    )

    arr = df.values.astype(np.float64)  # shape (N, 5)

    train_arr = arr[indices.train[0]:indices.train[1]]
    valid_arr = arr[indices.valid[0]:indices.valid[1]]
    test_arr = arr[indices.test[0]:indices.test[1]]

    # Step 7: Clip Patv >= 0 before scaling (Requirement 1.8)
    # Patv is column 0
    for part in (train_arr, valid_arr, test_arr):
        part[:, 0] = np.clip(part[:, 0], 0.0, None)

    # Step 8: Fit FeatureScaler on train only
    scaler = FeatureScaler()
    scaler.fit(train_arr)

    # Save scaler
    scaler_path = manifest_dir / "scaler.pkl"
    scaler.save(str(scaler_path))
    log.info("  Scaler saved to: %s", scaler_path)

    # Transform all partitions
    train_scaled = scaler.transform(train_arr)
    valid_scaled = scaler.transform(valid_arr)
    test_scaled = scaler.transform(test_arr)

    return train_scaled, valid_scaled, test_scaled, indices, scaler


def _run_vmd(
    train_scaled: np.ndarray,
    valid_scaled: np.ndarray,
    test_scaled: np.ndarray,
    vmd_params: VMDParams,
    manifest_dir: Path,
    n_total_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step 9: Fit VMD on train Patv; apply with same params to valid and test.

    IMF channels are appended AFTER the 5 original feature columns in each
    partition array.  Channel order after VMD:
        [Patv, IMF_1, …, IMF_K, Wspd, Wdir, Etmp, Itmp]

    Returns
    -------
    train_vmd, valid_vmd, test_vmd : np.ndarray  (shape (N_x, 5 + K))
    """
    K = vmd_params.K
    log.info("  Running VMD on training partition (K=%d)…", K)

    # Patv is column 0 in all partition arrays
    train_patv = train_scaled[:, 0]
    valid_patv = valid_scaled[:, 0]
    test_patv = test_scaled[:, 0]

    train_imfs = fit_vmd_on_train(train_patv, vmd_params)     # (n_train, K)
    log.info("  Train IMFs shape: %s", train_imfs.shape)

    log.info("  Applying VMD to validation partition…")
    valid_imfs = apply_vmd_to_partition(valid_patv, vmd_params)  # (n_valid, K)

    log.info("  Applying VMD to test partition…")
    test_imfs = apply_vmd_to_partition(test_patv, vmd_params)    # (n_test, K)

    # Assemble channel layout: [Patv, IMF_1…K, Wspd, Wdir, Etmp, Itmp]
    # train_scaled columns: [Patv=0, Wspd=1, Wdir=2, Etmp=3, Itmp=4]
    def _assemble(feat: np.ndarray, imfs: np.ndarray) -> np.ndarray:
        """feat: (N, 5), imfs: (N, K) → (N, 5+K) with IMFs after Patv."""
        patv_col = feat[:, :1]          # (N, 1)
        covariates = feat[:, 1:]        # (N, 4)  [Wspd, Wdir, Etmp, Itmp]
        return np.concatenate([patv_col, imfs, covariates], axis=1)

    train_vmd = _assemble(train_scaled, train_imfs)
    valid_vmd = _assemble(valid_scaled, valid_imfs)
    test_vmd = _assemble(test_scaled, test_imfs)

    # Persist VMD params manifest
    persist_vmd_params(
        manifest_dir,
        vmd_params,
        fit_seed=42,
        fit_n_samples=len(train_scaled),
        imf_shape=(n_total_rows, K),
    )
    log.info("  vmd_params.json saved to: %s", manifest_dir / "vmd_params.json")

    # Persist IMF arrays (train, valid, test concatenated along axis=0 for full
    # dataset; also save individually for convenience)
    imf_path = manifest_dir / "vmd_imfs.npz"
    all_imfs = np.concatenate([train_imfs, valid_imfs, test_imfs], axis=0)
    np.savez_compressed(
        imf_path,
        all_imfs=all_imfs,
        train_imfs=train_imfs,
        valid_imfs=valid_imfs,
        test_imfs=test_imfs,
    )
    log.info("  vmd_imfs.npz saved to: %s  (all_imfs shape: %s)", imf_path, all_imfs.shape)

    return train_vmd, valid_vmd, test_vmd


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cfg: dict) -> None:
    """Execute the full preprocessing pipeline with the given configuration.

    Parameters
    ----------
    cfg : dict
        Resolved configuration dict (see _resolve_config).
    """
    manifest_dir = Path(cfg["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    lookback: int = int(cfg["lookback"])
    horizon: int = int(cfg["horizon"])
    vmd_off: bool = bool(cfg["vmd_off"])
    enable_cleaning: bool = bool(cfg["cleaning"])

    log.info("=" * 60)
    log.info("Wind Power Preprocessing Pipeline")
    log.info("  csv_path     : %s", cfg["csv_path"])
    log.info("  manifest_dir : %s", manifest_dir)
    log.info("  lookback     : %d", lookback)
    log.info("  horizon      : %d", horizon)
    log.info("  vmd_off      : %s", vmd_off)
    log.info("  cleaning     : %s", enable_cleaning)
    if not vmd_off:
        log.info("  vmd_k        : %d", cfg["vmd_k"])
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 & 2: Load and sort
    # ------------------------------------------------------------------
    df = _load_and_sort(cfg["csv_path"])

    # ------------------------------------------------------------------
    # Steps 3–5: Clean, interpolate, drop NaN
    # ------------------------------------------------------------------
    df = _clean_and_interpolate(df, enable_cleaning=enable_cleaning)
    n_total_rows = len(df)

    # ------------------------------------------------------------------
    # Steps 6–8: Split, clip, scale
    # ------------------------------------------------------------------
    log.info("Splitting and scaling…")
    train_scaled, valid_scaled, test_scaled, indices, scaler = _split_and_scale(
        df, lookback, horizon, manifest_dir
    )

    # ------------------------------------------------------------------
    # Step 10: Save partition indices JSON
    # ------------------------------------------------------------------
    manifest_path = persist_partition_indices(indices, n_total_rows, str(manifest_dir))
    log.info("  Partition manifest saved to: %s", manifest_path)

    # ------------------------------------------------------------------
    # Step 9: VMD (train only; then apply to valid/test)
    # ------------------------------------------------------------------
    if not vmd_off:
        log.info("Running VMD…")
        vmd_params = VMDParams(
            K=int(cfg["vmd_k"]),
            alpha=float(cfg["vmd_alpha"]),
            tau=float(cfg["vmd_tau"]),
            DC=int(cfg["vmd_DC"]),
            init=int(cfg["vmd_init"]),
            tol=float(cfg["vmd_tol"]),
        )
        train_final, valid_final, test_final = _run_vmd(
            train_scaled, valid_scaled, test_scaled,
            vmd_params, manifest_dir, n_total_rows,
        )
        n_features_out = train_final.shape[1]
        log.info(
            "  VMD complete. Output channels per partition: %d "
            "(5 original + %d IMF)",
            n_features_out,
            cfg["vmd_k"],
        )
    else:
        log.info("VMD DISABLED – using 5 original features only.")
        train_final = train_scaled
        valid_final = valid_scaled
        test_final = test_scaled
        n_features_out = 5

    # ------------------------------------------------------------------
    # Write summary manifest
    # ------------------------------------------------------------------
    summary = {
        "csv_path": str(cfg["csv_path"]),
        "manifest_dir": str(manifest_dir),
        "lookback": lookback,
        "horizon": horizon,
        "n_total_rows": n_total_rows,
        "n_features_out": n_features_out,
        "vmd_enabled": not vmd_off,
        "vmd_k": cfg["vmd_k"] if not vmd_off else 0,
        "cleaning_enabled": enable_cleaning,
        "feature_cols": FEATURE_COLS,
        "channel_layout": (
            ["Patv"] + [f"IMF_{i+1}" for i in range(cfg["vmd_k"])] + ["Wspd", "Wdir", "Etmp", "Itmp"]
            if not vmd_off
            else FEATURE_COLS
        ),
        "partition": {
            "train": {"start": indices.train[0], "end": indices.train[1],
                      "n_rows": indices.train[1] - indices.train[0]},
            "valid": {"start": indices.valid[0], "end": indices.valid[1],
                      "n_rows": indices.valid[1] - indices.valid[0]},
            "test":  {"start": indices.test[0],  "end": indices.test[1],
                      "n_rows": indices.test[1]  - indices.test[0]},
        },
        "train_shape": list(train_final.shape),
        "valid_shape": list(valid_final.shape),
        "test_shape":  list(test_final.shape),
    }

    summary_path = manifest_dir / f"preprocess_summary_l{lookback}_h{horizon}.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary written to: %s", summary_path)

    log.info("=" * 60)
    log.info("Preprocessing complete.")
    log.info(
        "  Train: %s  |  Valid: %s  |  Test: %s",
        train_final.shape, valid_final.shape, test_final.shape,
    )
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = _resolve_config(args)
        run_pipeline(cfg)
        return 0
    except (FileNotFoundError, ValueError, ImportError) as exc:
        log.error("Configuration / input error: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover
        log.exception("Unexpected error during preprocessing: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
