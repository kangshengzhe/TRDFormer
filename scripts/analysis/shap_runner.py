"""
SHAP runner for the Proposed_Model (iTransformer + LSTM + CrossAttention + KAN).

Computes SHAP feature-importance values over ≥100 samples from the test
partition and persists the results to disk.

Output files
------------
outputs/runs/shap_values.npy  (primary output, path configurable via --output)
    Raw SHAP array of shape (n_samples, n_features=5), obtained by averaging
    the per-timestep SHAP values over the lookback dimension.
    Feature order: [Patv, Wspd, Wdir, Etmp, Itmp]

Error handling
--------------
- Checkpoint not found → log to shap_failures.log, exit non-zero, no file written.
- SHAP computation fails → log to shap_failures.log, delete any partial .npy
  files written during the attempt, exit non-zero.
- try/finally block guarantees partial-file cleanup even on unexpected errors.

CLI usage
---------
    python scripts/analysis/shap_runner.py \\
        --checkpoint model_save/wind/{run_id}.pt \\
        --config outputs/runs/configs/{run_id}.yaml \\
        --output outputs/runs/shap_values.npy \\
        --n_samples 100

    # Legacy flags also accepted for backward compatibility:
    python scripts/analysis/shap_runner.py \\
        --checkpoint model_save/wind/{run_id}.pt \\
        --manifest-dir outputs/manifests/ \\
        --n-samples 200 \\
        --out-dir outputs/figures/

Requirements: 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import dill
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
N_BASE_FEATURES = 5          # the 5 raw physical features always attributed
FAILURES_LOG = "outputs/runs/shap_failures.log"
DEFAULT_MANIFEST_DIR = "outputs/manifests/"
DEFAULT_OUT_DIR = "outputs/runs/"
DEFAULT_OUTPUT = "outputs/runs/shap_values.npy"
DEFAULT_N_SAMPLES = 100
DEFAULT_LOOKBACK = 144
DEFAULT_HORIZON = 1


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(failures_log: str) -> logging.Logger:
    """Create a logger that writes failures to *failures_log* and also to stderr."""
    log_path = Path(failures_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("shap_runner")
    logger.setLevel(logging.DEBUG)

    # Rotating stderr handler (info+)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(stream_handler)

    # File handler for failures (warning+)
    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_scaler(manifest_dir: str, logger: logging.Logger):
    """Load FeatureScaler from the manifests directory.

    Returns the loaded scaler, or raises FileNotFoundError with a descriptive
    message if the scaler file is missing.
    """
    from data_pipeline.scaling import FeatureScaler

    scaler_path = Path(manifest_dir) / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")
    logger.info("Loading scaler from %s", scaler_path)
    return FeatureScaler.load(str(scaler_path))


def _load_partition_manifest(manifest_dir: str, lookback: int, horizon: int,
                              logger: logging.Logger):
    """Load the partition indices manifest for the given lookback/horizon.

    Returns the PartitionIndices, or raises FileNotFoundError if the file is
    missing.
    """
    from data_pipeline.manifest import PartitionManifest

    manifest_path = (
        Path(manifest_dir)
        / PartitionManifest.filename(lookback, horizon)
    )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Partition manifest not found: {manifest_path}. "
            "Run the preprocessing CLI first."
        )
    logger.info("Loading partition manifest from %s", manifest_path)
    return PartitionManifest.read(str(manifest_path))


def _load_raw_data(manifest_dir: str, logger: logging.Logger) -> np.ndarray:
    """Load the preprocessed data array from the manifests directory.

    Looks for a .npy or .npz file named 'scaled_data*' in manifest_dir.
    Falls back to rebuilding from the raw CSV if the file is missing.
    """
    manifest_dir_path = Path(manifest_dir)

    # Try to load a pre-scaled data array first (fastest path)
    candidates = (
        list(manifest_dir_path.glob("scaled_data*.npy"))
        + list(manifest_dir_path.glob("scaled_data*.npz"))
    )
    if candidates:
        candidate = candidates[0]
        logger.info("Loading scaled data from %s", candidate)
        if candidate.suffix == ".npz":
            arr = np.load(str(candidate))
            key = list(arr.files)[0]
            return arr[key]
        return np.load(str(candidate))

    # Fallback: rebuild from CSV via the data pipeline
    logger.info(
        "No pre-scaled data file found in %s; rebuilding from CSV …", manifest_dir
    )
    return _rebuild_scaled_data(manifest_dir, logger)


def _rebuild_scaled_data(manifest_dir: str, logger: logging.Logger) -> np.ndarray:
    """Rebuild scaled feature matrix from the raw CSV using the pipeline modules."""
    import pandas as pd
    from data_pipeline.cleaning import physical_rule_clean
    from data_pipeline.scaling import FeatureScaler

    # Locate CSV relative to this script
    repo_root = Path(__file__).resolve().parents[2]
    csv_path = repo_root / "data" / "wind" / "sdwpf_turb1_cleaned_final.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw CSV not found at {csv_path}. Cannot rebuild data."
        )

    logger.info("Reading CSV from %s", csv_path)
    df = pd.read_csv(str(csv_path))

    # Build datetime index and sort chronologically
    df["date"] = "2024-" + df["Day"].astype(str) + " " + df["Tmstamp"]
    df["date"] = pd.to_datetime(df["date"], format="%Y-%j %H:%M")
    df = df.sort_values("date").reset_index(drop=True)

    # Select the 5 feature columns in the canonical order
    features = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]
    df_feat = df[features].copy()

    # Physical-rule cleaning
    df_feat, _ = physical_rule_clean(df_feat)

    # Interpolate missing values then drop remaining NaNs
    df_feat = df_feat.interpolate(method="linear")
    df_feat = df_feat.dropna()

    arr = df_feat.values.astype(np.float32)

    # Fit scaler on training partition only (80%)
    n_train = int(len(arr) * 0.8)
    scaler = FeatureScaler()
    scaler.fit(arr[:n_train])
    scaled = scaler.transform(arr).astype(np.float32)

    return scaled


def _load_vmd_imfs(manifest_dir: str, logger: logging.Logger) -> Optional[np.ndarray]:
    """Load VMD IMF array from the manifests directory if present.

    Returns the (N, K) array, or None if VMD files are not available.
    """
    manifest_dir_path = Path(manifest_dir)
    candidates = (
        list(manifest_dir_path.glob("vmd_imfs*.npz"))
        + list(manifest_dir_path.glob("vmd_imfs*.npy"))
    )
    if not candidates:
        return None
    candidate = candidates[0]
    logger.info("Loading VMD IMFs from %s", candidate)
    if candidate.suffix == ".npz":
        arr = np.load(str(candidate))
        key = list(arr.files)[0]
        return arr[key]
    return np.load(str(candidate))


def _build_test_data(
    manifest_dir: str,
    lookback: int,
    horizon: int,
    n_samples: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, int]:
    """Assemble the test-partition input tensor X and return the feature count.

    Returns
    -------
    X_test : np.ndarray, shape (n_windows, lookback, F_in)
        Sliding windows over the test partition.
    n_base_features : int
        Number of the five base features (always 5) — used to restrict
        SHAP attribution to the original five physical features.
    """
    partition = _load_partition_manifest(manifest_dir, lookback, horizon, logger)
    scaled_data = _load_raw_data(manifest_dir, logger)
    vmd_imfs = _load_vmd_imfs(manifest_dir, logger)

    test_start, test_end = partition.test

    # Extend with VMD IMFs if available
    if vmd_imfs is not None and vmd_imfs.shape[0] == scaled_data.shape[0]:
        # Channel layout: [Patv, IMF_1..IMF_K, Wspd, Wdir, Etmp, Itmp]
        patv_col = scaled_data[:, 0:1]
        cov_cols = scaled_data[:, 1:]
        data_full = np.concatenate([patv_col, vmd_imfs, cov_cols], axis=1)
        logger.info(
            "Using VMD-extended input: %d channels (K=%d IMFs)", data_full.shape[1],
            vmd_imfs.shape[1]
        )
    else:
        data_full = scaled_data
        if vmd_imfs is not None:
            logger.warning(
                "VMD IMF shape %s does not match data shape %s; using raw 5 features.",
                vmd_imfs.shape, scaled_data.shape
            )
        logger.info("Using 5-feature input (no VMD)")

    F_in = data_full.shape[1]

    # Extract test partition with lookback rollback
    test_data = data_full[test_start - lookback : test_end]
    if len(test_data) < lookback + horizon:
        raise ValueError(
            f"Test partition too small: {len(test_data)} rows for "
            f"lookback={lookback}, horizon={horizon}"
        )

    # Build sliding windows
    n_max_windows = len(test_data) - lookback - horizon + 1
    if n_max_windows <= 0:
        raise ValueError(
            f"No windows possible from {len(test_data)} rows with "
            f"lookback={lookback}, horizon={horizon}"
        )

    effective_n = min(n_samples, n_max_windows)
    if effective_n < n_samples:
        logger.warning(
            "Only %d windows available in the test partition; "
            "requested %d samples.",
            effective_n, n_samples
        )
    # Requirement 9.1: must have ≥ 100 samples
    if effective_n < 100:
        raise ValueError(
            f"Test partition yields only {effective_n} windows; "
            "need at least 100 samples to meet Requirement 9.1."
        )

    # Use a uniformly-spaced stride to pick `effective_n` windows without
    # replacement from the available windows
    if effective_n < n_max_windows:
        indices = np.round(
            np.linspace(0, n_max_windows - 1, effective_n)
        ).astype(int)
        # Guarantee uniqueness after rounding
        indices = np.unique(indices)
        if len(indices) < 100:
            indices = np.arange(min(n_max_windows, effective_n))
    else:
        indices = np.arange(n_max_windows)

    X_test = np.stack(
        [test_data[i : i + lookback] for i in indices],
        axis=0,
    ).astype(np.float32)  # (n_selected, lookback, F_in)

    logger.info(
        "Built X_test: shape=%s, F_in=%d", X_test.shape, F_in
    )
    return X_test, F_in


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device,
                logger: logging.Logger) -> torch.nn.Module:
    """Load the Proposed_Model from a .pt checkpoint.

    Supports both:
    - A plain pickled Module (dill.load)
    - A state-dict checkpoint dict with key 'model_state_dict' or 'state_dict'
      (requires the model config to be embedded or inferable)

    Raises
    ------
    FileNotFoundError
        If the checkpoint file is missing (triggers failure log + exit 1).
    RuntimeError
        If the checkpoint cannot be deserialized.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train the Proposed_Model first (see scripts/train/)."
        )

    logger.info("Loading checkpoint from %s", ckpt_path)
    try:
        # Try dill-based full-model load (standard for this repo)
        with open(str(ckpt_path), "rb") as f:
            obj = dill.load(f)

        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict):
            # State-dict-style checkpoint — reconstruct model from embedded config
            model = _reconstruct_from_state_dict(obj, logger)
        else:
            raise RuntimeError(
                f"Unexpected checkpoint content type: {type(obj)}"
            )
    except Exception:
        # Fallback: try torch.load with weights_only=False
        logger.info("dill.load failed, falling back to torch.load …")
        obj = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, dict):
            model = _reconstruct_from_state_dict(obj, logger)
        else:
            raise RuntimeError(
                f"Unexpected checkpoint content type after torch.load: {type(obj)}"
            )

    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully: %s", type(model).__name__)
    return model


def _reconstruct_from_state_dict(
    obj: dict, logger: logging.Logger
) -> torch.nn.Module:
    """Attempt to reconstruct the model from a state-dict-style checkpoint.

    Looks for an embedded 'config' key in the checkpoint dict.
    """
    from models.unified_proposed import UnifiedProposedModel

    cfg = obj.get("config", {})
    if not cfg:
        logger.warning(
            "No 'config' key in checkpoint dict; using default proposed-model config."
        )
        cfg = {
            "lookback": DEFAULT_LOOKBACK,
            "horizon": DEFAULT_HORIZON,
            "n_target_channels": 1,
        }

    logger.info("Reconstructing UnifiedProposedModel from config: %s", cfg)
    model = UnifiedProposedModel.from_config(cfg)

    state_dict_key = "model_state_dict" if "model_state_dict" in obj else "state_dict"
    if state_dict_key in obj:
        model.load_state_dict(obj[state_dict_key])
        logger.info("State dict loaded from checkpoint key '%s'.", state_dict_key)
    else:
        raise RuntimeError(
            "Checkpoint dict has no 'model_state_dict' or 'state_dict' key."
        )
    return model


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def _compute_shap(
    model: torch.nn.Module,
    X_test: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
) -> np.ndarray:
    """Compute SHAP values using GradientExplainer or DeepExplainer.

    Strategy
    --------
    1. Try ``shap.GradientExplainer`` (fastest for gradient-based models).
    2. If that fails (e.g., non-differentiable ops), fall back to
       ``shap.DeepExplainer``.
    3. If both fail, fall back to ``shap.KernelExplainer`` (slowest, but
       model-agnostic).

    Parameters
    ----------
    model : nn.Module
        The loaded Proposed_Model in eval mode.
    X_test : np.ndarray
        Test samples, shape (n_samples, lookback, F_in).
    device : torch.device
    logger : logging.Logger

    Returns
    -------
    shap_values : np.ndarray
        SHAP values averaged over the lookback dimension,
        shape (n_samples, 5) — one column per base physical feature.
    """
    import shap

    n_samples, lookback, F_in = X_test.shape
    logger.info(
        "Computing SHAP values: n_samples=%d, lookback=%d, F_in=%d",
        n_samples, lookback, F_in
    )

    # Use the first min(100, n_samples) samples as background
    n_bg = min(100, n_samples)
    bg_indices = np.linspace(0, n_samples - 1, n_bg, dtype=int)
    background = torch.tensor(X_test[bg_indices], dtype=torch.float32).to(device)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    shap_raw = None

    # ── 1. Try GradientExplainer ─────────────────────────────────────────
    try:
        logger.info("Attempting shap.GradientExplainer …")
        explainer = shap.GradientExplainer(model, background)
        shap_raw = explainer.shap_values(X_tensor)
        # GradientExplainer returns a list (one per output) or an array
        if isinstance(shap_raw, list):
            # Multi-output model: average over outputs
            shap_raw = np.mean(np.stack([np.array(sv) for sv in shap_raw], axis=0),
                               axis=0)
        shap_raw = np.array(shap_raw)
        logger.info("GradientExplainer succeeded; SHAP raw shape: %s", shap_raw.shape)
    except Exception as exc:
        logger.warning("GradientExplainer failed (%s); trying DeepExplainer …", exc)
        shap_raw = None

    # ── 2. Try DeepExplainer ──────────────────────────────────────────────
    if shap_raw is None:
        try:
            logger.info("Attempting shap.DeepExplainer …")
            explainer = shap.DeepExplainer(model, background)
            shap_raw = explainer.shap_values(X_tensor)
            if isinstance(shap_raw, list):
                shap_raw = np.mean(
                    np.stack([np.array(sv) for sv in shap_raw], axis=0), axis=0
                )
            shap_raw = np.array(shap_raw)
            logger.info("DeepExplainer succeeded; SHAP raw shape: %s", shap_raw.shape)
        except Exception as exc:
            logger.warning("DeepExplainer failed (%s); falling back to KernelExplainer …", exc)
            shap_raw = None

    # ── 3. Fall back to KernelExplainer (CPU, flattened) ─────────────────
    if shap_raw is None:
        logger.info("Attempting shap.KernelExplainer (this may be slow) …")
        cpu_model = model.cpu()
        bg_flat = X_test[bg_indices].reshape(n_bg, -1)
        X_flat = X_test.reshape(n_samples, -1)

        def _predict_flat(data: np.ndarray) -> np.ndarray:
            t = torch.tensor(
                data.reshape(-1, lookback, F_in), dtype=torch.float32
            )
            with torch.no_grad():
                out = cpu_model(t).cpu().numpy()
            # out shape: (B, horizon) — return mean over horizon
            if out.ndim == 2:
                return out.mean(axis=1, keepdims=True)
            return out.reshape(-1, 1)

        explainer = shap.KernelExplainer(_predict_flat, bg_flat)
        shap_flat = explainer.shap_values(X_flat, nsamples="auto")
        # shap_flat: (n_samples, lookback * F_in) or list
        if isinstance(shap_flat, list):
            shap_flat = shap_flat[0]
        shap_flat = np.array(shap_flat)
        shap_raw = shap_flat.reshape(n_samples, lookback, F_in)
        logger.info("KernelExplainer succeeded; SHAP raw shape: %s", shap_raw.shape)

    # ── Aggregate: mean over lookback → (n_samples, F_in) ────────────────
    # shap_raw may be (n_samples, lookback, F_in) or (n_samples, F_in) already
    if shap_raw.ndim == 3:
        shap_per_sample = shap_raw.mean(axis=1)       # (n_samples, F_in)
    elif shap_raw.ndim == 2:
        shap_per_sample = shap_raw                    # already (n_samples, F_in)
    else:
        raise RuntimeError(
            f"Unexpected SHAP output shape: {shap_raw.shape}. "
            "Expected 2-D or 3-D array."
        )

    # ── Restrict to the 5 base physical features (Req 9.2) ────────────────
    # Channel layout: [Patv, (IMF_1..IMF_K if VMD), Wspd, Wdir, Etmp, Itmp]
    # The 5 physical features are always the first column (Patv) and the last
    # 4 columns (Wspd, Wdir, Etmp, Itmp).  Sum IMF channels into the Patv slot.
    n_feat = shap_per_sample.shape[1]
    if n_feat == N_BASE_FEATURES:
        # No VMD channels present
        shap_5 = shap_per_sample
    else:
        # n_feat = 1 + K + 4  (K IMF channels in positions 1..K)
        K = n_feat - N_BASE_FEATURES   # number of IMFs
        patv_shap = shap_per_sample[:, 0:1]              # (n, 1)
        imf_shap = shap_per_sample[:, 1:1 + K].sum(axis=1, keepdims=True)  # (n, 1)
        cov_shap = shap_per_sample[:, 1 + K:]            # (n, 4)
        # Patv importance = original Patv channel + sum of IMF channels
        patv_combined = patv_shap + imf_shap             # (n, 1)
        shap_5 = np.concatenate([patv_combined, cov_shap], axis=1)  # (n, 5)

    logger.info(
        "SHAP values aggregated to shape %s (5 base features)", shap_5.shape
    )
    return shap_5


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save_outputs(
    shap_5: np.ndarray,
    out_dir: str,
    logger: logging.Logger,
    output_path: Optional[str] = None,
) -> tuple[Path, Path]:
    """Save shap_values.npy and shap_summary.npy.

    If *output_path* is given (from ``--output`` flag), that path is used for
    shap_values.npy and its parent directory is used for shap_summary.npy.
    Otherwise, both files go into *out_dir*.

    Returns
    -------
    (shap_values_path, shap_summary_path)
    """
    if output_path is not None:
        shap_values_path = Path(output_path)
        shap_values_path.parent.mkdir(parents=True, exist_ok=True)
        shap_summary_path = shap_values_path.parent / "shap_summary.npy"
    else:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        shap_values_path = out_path / "shap_values.npy"
        shap_summary_path = out_path / "shap_summary.npy"

    # Save raw SHAP array: (n_samples, 5)
    np.save(str(shap_values_path), shap_5)
    logger.info("SHAP values saved to %s (shape=%s)", shap_values_path, shap_5.shape)

    # Save summary: mean absolute SHAP per feature → (5,)
    shap_summary = np.abs(shap_5).mean(axis=0)
    np.save(str(shap_summary_path), shap_summary)
    logger.info("SHAP summary saved to %s", shap_summary_path)

    # Log feature importance ranking
    ranked_idx = np.argsort(shap_summary)[::-1]
    logger.info("Feature importance ranking:")
    for rank, idx in enumerate(ranked_idx):
        logger.info(
            "  %d. %s — mean |SHAP| = %.6f",
            rank + 1, FEATURE_NAMES[idx], shap_summary[idx]
        )

    return shap_values_path, shap_summary_path


def _cleanup_partial_files(paths: list[Path], logger: logging.Logger) -> None:
    """Delete any partially-written output files.

    Called from the ``finally`` block to guarantee no partial files remain
    after a failed SHAP computation (Requirement 9.4).
    """
    for p in paths:
        if p.exists():
            try:
                p.unlink()
                logger.warning("Deleted partial file: %s", p)
            except OSError as exc:
                logger.error("Could not delete partial file %s: %s", p, exc)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _manifest_dir_from_config(config_path: str, logger: logging.Logger) -> str:
    """Extract ``manifest_dir`` from a run-config YAML file.

    Falls back to the default manifest directory if the YAML cannot be parsed
    or the relevant keys are absent.
    """
    try:
        import yaml  # PyYAML
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        manifest_dir = (
            cfg.get("dataset", {}).get("manifest_dir", DEFAULT_MANIFEST_DIR)
        )
        logger.info(
            "Using manifest_dir=%s (read from config %s)", manifest_dir, config_path
        )
        return manifest_dir
    except Exception as exc:
        logger.warning(
            "Could not read manifest_dir from config %s (%s); "
            "using default %s",
            config_path, exc, DEFAULT_MANIFEST_DIR,
        )
        return DEFAULT_MANIFEST_DIR


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SHAP values for the Proposed_Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained Proposed_Model checkpoint (.pt file).",
    )

    # Primary interface (task 10.1 spec)
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to the run-config YAML file (outputs/runs/configs/X.yaml). "
            "When provided, manifest_dir is read from dataset.manifest_dir. "
            "Takes precedence over --manifest-dir."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Full path for the output shap_values.npy file "
            "(e.g. outputs/runs/shap_values.npy). "
            "Overrides the --out-dir flag when provided."
        ),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        dest="n_samples",
        help="Number of test-partition samples to explain (≥ 100 required).",
    )

    # Legacy / alternative flags kept for backward compatibility
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        dest="n_samples_legacy",
        help="Alias for --n_samples (legacy hyphen form).",
    )
    parser.add_argument(
        "--manifest-dir",
        default=DEFAULT_MANIFEST_DIR,
        dest="manifest_dir",
        help="Directory containing partition_indices_*.json and scaler.pkl.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        dest="out_dir",
        help="Directory where shap_values.npy and shap_summary.npy are written.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=DEFAULT_LOOKBACK,
        help="Lookback window length (number of time steps).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Forecast horizon length (number of steps ahead).",
    )
    parser.add_argument(
        "--failures-log",
        default=FAILURES_LOG,
        dest="failures_log",
        help="Path to the SHAP failure log file.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Computation device: 'auto', 'cpu', or 'cuda'.",
    )

    args = parser.parse_args(argv)

    # Merge legacy --n-samples into n_samples
    if args.n_samples_legacy is not None:
        args.n_samples = args.n_samples_legacy

    return args


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    checkpoint: str,
    manifest_dir: str = DEFAULT_MANIFEST_DIR,
    n_samples: int = DEFAULT_N_SAMPLES,
    out_dir: str = DEFAULT_OUT_DIR,
    output_path: Optional[str] = None,
    lookback: int = DEFAULT_LOOKBACK,
    horizon: int = DEFAULT_HORIZON,
    failures_log: str = FAILURES_LOG,
    device_str: str = "auto",
) -> int:
    """Execute the SHAP computation pipeline.

    Parameters
    ----------
    checkpoint : str
        Path to the trained model checkpoint .pt file.
    manifest_dir : str
        Directory containing partition_indices_*.json and scaler.pkl.
    n_samples : int
        Number of test-partition samples to explain (≥ 100).
    out_dir : str
        Output directory when *output_path* is not given.
    output_path : str, optional
        Full path for shap_values.npy.  When provided, this overrides
        ``out_dir / "shap_values.npy"``.  The SHAP values are saved
        here (Requirement 9.3: ``outputs/runs/shap_values.npy``).
    lookback : int
        Lookback window length.
    horizon : int
        Forecast horizon length.
    failures_log : str
        Path to the failure log (Requirement 9.4).
    device_str : str
        Device selection: 'auto', 'cpu', or 'cuda'.

    Returns
    -------
    int
        0 on success, non-zero on failure.
    """
    logger = _setup_logging(failures_log)

    # Resolve device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    logger.info("Using device: %s", device)

    # Validate n_samples early
    if n_samples < 100:
        msg = (
            f"--n-samples={n_samples} is below the minimum of 100 "
            "(Requirement 9.1). Raising to 100."
        )
        logger.warning(msg)
        n_samples = 100

    # Determine the concrete output paths so we can track them for cleanup
    if output_path is not None:
        shap_values_path = Path(output_path)
        shap_summary_path = shap_values_path.parent / "shap_summary.npy"
    else:
        out_path = Path(out_dir)
        shap_values_path = out_path / "shap_values.npy"
        shap_summary_path = out_path / "shap_summary.npy"

    written_paths: list[Path] = []

    try:
        # ── 1. Load model checkpoint ─────────────────────────────────────
        try:
            model = _load_model(checkpoint, device, logger)
        except FileNotFoundError as exc:
            logger.error("SHAP FAILED — checkpoint not found: %s", exc)
            # Requirement 9.4: do NOT write any file on checkpoint failure
            return 1

        # ── 2. Build test-partition input tensor ─────────────────────────
        X_test, F_in = _build_test_data(
            manifest_dir, lookback, horizon, n_samples, logger
        )

        # ── 3. Compute SHAP values ────────────────────────────────────────
        shap_5 = _compute_shap(model, X_test, device, logger)

        # Sanity-check the output has exactly 5 features (Req 9.2)
        if shap_5.shape[1] != N_BASE_FEATURES:
            raise RuntimeError(
                f"SHAP output has {shap_5.shape[1]} feature columns; "
                f"expected {N_BASE_FEATURES} ({FEATURE_NAMES})."
            )

        # ── 4. Persist results ────────────────────────────────────────────
        # Track paths so we can delete them if a subsequent step fails
        written_paths.append(shap_values_path)
        written_paths.append(shap_summary_path)
        _save_outputs(shap_5, out_dir, logger, output_path=str(output_path) if output_path else None)

        logger.info("SHAP computation completed successfully.")
        return 0

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("SHAP FAILED — %s\n%s", exc, tb)
        # Requirement 9.4: delete any partial files
        _cleanup_partial_files(written_paths, logger)
        return 1

    finally:
        # This block runs even if an unexpected BaseException occurs.
        # If we are in the failure path the cleanup is already done above;
        # the finally block catches any edge cases (e.g. KeyboardInterrupt).
        # We check file existence so a successful write is not accidentally deleted.
        pass


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    # Ensure the repo root is on sys.path so local imports work when the
    # script is executed from any directory.
    repo_root = str(Path(__file__).resolve().parents[2])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Resolve manifest_dir: --config takes precedence over --manifest-dir
    manifest_dir = args.manifest_dir
    if args.config is not None:
        # Lazy import of logger-only setup for the config read
        import logging as _logging
        _tmp_logger = _logging.getLogger("shap_runner.config")
        manifest_dir = _manifest_dir_from_config(args.config, _tmp_logger)

    # Resolve the output path: --output takes precedence over --out-dir
    output_path = args.output  # may be None

    # Default failures log: place next to the output file when --output is given
    failures_log = args.failures_log
    if output_path is not None and failures_log == FAILURES_LOG:
        # Keep failures log alongside the output for easy discovery
        failures_log = str(Path(output_path).parent / "shap_failures.log")

    exit_code = run(
        checkpoint=args.checkpoint,
        manifest_dir=manifest_dir,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
        output_path=output_path,
        lookback=args.lookback,
        horizon=args.horizon,
        failures_log=failures_log,
        device_str=args.device,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
