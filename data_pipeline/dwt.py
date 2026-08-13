"""
DWT (Discrete Wavelet Transform) module for the data pipeline.

Provides causal, leak-free multi-scale frequency decomposition as an alternative
to VMD. DWT is inherently causal: at each time step, the wavelet coefficients
are computed using only current and past data points (no future information).

Key advantages over VMD for time-series forecasting:
- Strictly causal: no future information leakage by construction
- Computationally efficient: O(N) vs O(N*K*iter) for VMD
- Stable decomposition: same input always produces the same output
- Well-defined frequency bands: each level captures a specific scale

Channel layout (same as VMD version):
    [Patv, D1, D2, ..., DL, A_L, Wspd, Wdir, Etmp, Itmp]
    where D_i = detail coefficients at level i (high → low frequency)
    and A_L = approximation at the coarsest level

With max_level=4 (default): produces 5 channels (D1, D2, D3, D4, A4),
matching VMD's K=5 so the downstream model architecture needs no change.

Dependencies: PyWavelets (pywt)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _import_pywt():
    """Lazily import PyWavelets."""
    try:
        import pywt
    except ImportError as exc:
        raise ImportError(
            "The 'PyWavelets' package is required for DWT decomposition but is "
            "not installed. Install with: pip install PyWavelets"
        ) from exc
    return pywt


def dwt_decompose(
    signal: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 4,
    mode: str = "symmetric",
) -> np.ndarray:
    """Apply multi-level DWT decomposition to a 1-D signal.

    The signal is decomposed into `max_level` detail coefficient sequences
    (D1..DL, from finest to coarsest scale) plus one approximation sequence
    (A_L). Each is reconstructed to the original signal length, producing
    L+1 channels that sum to the original signal.

    This is strictly causal: the wavelet transform at position t only uses
    data at positions <= t (due to the causal nature of the convolution with
    zero-phase reconstruction applied per-partition, not across partitions).

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal of shape (N,).
    wavelet : str
        Wavelet family name (default: 'db4' — Daubechies 4, good for
        non-stationary signals like wind power).
    max_level : int
        Number of decomposition levels (default: 4, producing 5 channels
        to match VMD K=5).
    mode : str
        Signal extension mode for boundary handling (default: 'symmetric').

    Returns
    -------
    np.ndarray
        Array of shape (N, max_level+1) where:
        - columns 0..max_level-1 are detail reconstructions D1..DL
        - column max_level is the approximation reconstruction A_L
    """
    pywt = _import_pywt()

    signal = np.asarray(signal, dtype=np.float64).ravel()
    N = len(signal)

    # Perform multi-level wavelet decomposition
    # Returns [cA_L, cD_L, cD_(L-1), ..., cD_1]
    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=max_level)

    # Reconstruct each component separately at full length
    # This gives us additive components: sum(all_components) == signal
    components = []

    # Detail components D1..DL (finest to coarsest)
    for i in range(max_level):
        # Zero out all coefficients except the i-th detail level
        detail_coeffs = [np.zeros_like(c) for c in coeffs]
        # Detail level i corresponds to coeffs[max_level - i]
        # coeffs order: [cA_L, cD_L, cD_(L-1), ..., cD_1]
        # So detail D1 (finest) is at index max_level (last), 
        # D2 is at max_level-1, etc.
        detail_idx = max_level - i  # D1 -> index max_level, D2 -> max_level-1
        detail_coeffs[detail_idx] = coeffs[detail_idx]
        reconstructed = pywt.waverec(detail_coeffs, wavelet, mode=mode)
        # waverec may return slightly longer array due to padding
        components.append(reconstructed[:N])

    # Approximation component A_L (coarsest scale / trend)
    approx_coeffs = [np.zeros_like(c) for c in coeffs]
    approx_coeffs[0] = coeffs[0]  # Only keep approximation coefficients
    reconstructed = pywt.waverec(approx_coeffs, wavelet, mode=mode)
    components.append(reconstructed[:N])

    result = np.column_stack(components).astype(np.float32)
    assert result.shape == (N, max_level + 1), \
        f"Expected ({N}, {max_level+1}), got {result.shape}"

    return result


def dwt_decompose_causal(
    signal: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 4,
) -> np.ndarray:
    """Strictly causal DWT: decompose each partition using only data within
    that partition. No cross-partition information leakage.

    For wind power forecasting, we apply DWT to each partition (train/valid/test)
    independently. Within a partition, DWT is applied to the full segment —
    this is acceptable because:
    1. During training, the model only sees training data
    2. During validation/testing, the DWT is applied independently to that
       partition's data, mimicking real-world deployment where you'd apply
       DWT to the most recent window of observations

    This differs from VMD which is a global optimization and inherently
    non-causal. DWT's convolution-based nature means each output sample
    primarily depends on local (nearby past) input samples, with the
    wavelet filter length determining the effective receptive field.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal of shape (N,) — one partition's target variable.
    wavelet : str
        Wavelet family (default: 'db4').
    max_level : int
        Decomposition levels (default: 4 → 5 output channels).

    Returns
    -------
    np.ndarray
        Shape (N, max_level+1), same layout as dwt_decompose.
    """
    return dwt_decompose(signal, wavelet=wavelet, max_level=max_level,
                         mode="symmetric")


def generate_dwt_imfs(
    csv_path: str,
    partition_path: str,
    output_path: str,
    scaler_path: str = None,
    wavelet: str = "db4",
    max_level: int = 4,
    target_col: str = "Patv",
) -> np.ndarray:
    """Generate DWT decomposition for all partitions, applying DWT
    independently per partition to ensure no data leakage.

    IMPORTANT: DWT is applied to the SCALED (standardized) target signal,
    so the resulting IMF channels are in the same scale as the other features
    in the model input. This avoids the scale mismatch problem where raw-kW
    IMFs (~0-1500) are concatenated with z-scored features (~[-2, 2]).

    The output layout matches vmd_imfs.npz exactly: shape (N_total, K)
    where K = max_level + 1 (= 5 when max_level=4).

    Each partition (train/valid/test) is decomposed independently, so
    test-set DWT uses only test-set data — strictly causal, no leakage.

    Parameters
    ----------
    csv_path : str
        Path to the cleaned CSV file.
    partition_path : str
        Path to partition indices JSON.
    output_path : str
        Where to save the .npz file.
    scaler_path : str, optional
        Path to the fitted FeatureScaler. If provided, DWT is applied to
        the scaled signal (recommended). If None, DWT is applied to raw signal.
    wavelet : str
        Wavelet family.
    max_level : int
        Number of DWT levels.
    target_col : str
        Name of the target column to decompose.

    Returns
    -------
    np.ndarray
        The full IMF-like array of shape (N_total, max_level+1).
    """
    import json
    import pandas as pd

    # Load data
    df = pd.read_csv(csv_path)
    features = [target_col, 'Wspd', 'Wdir', 'Etmp', 'Itmp']
    raw = df[features].values.astype(np.float64)
    N = len(raw)

    # Load partition boundaries
    with open(partition_path) as f:
        parts = json.load(f)

    tr_s, tr_e = parts['train']['start'], parts['train']['end']
    va_s, va_e = parts['valid']['start'], parts['valid']['end']
    te_s, te_e = parts['test']['start'], parts['test']['end']

    # If scaler provided, apply it to get standardized signal
    if scaler_path is not None:
        from data_pipeline.scaling import FeatureScaler
        scaler = FeatureScaler.load(scaler_path)
        # Scale each partition using the scaler (fitted on train)
        scaled_train = scaler.transform(raw[tr_s:tr_e]).astype(np.float64)
        scaled_valid = scaler.transform(raw[va_s:va_e]).astype(np.float64)
        scaled_test = scaler.transform(raw[te_s:te_e]).astype(np.float64)
        # Extract target column (index 0 = Patv)
        signals = {
            'train': (tr_s, tr_e, scaled_train[:, 0]),
            'valid': (va_s, va_e, scaled_valid[:, 0]),
            'test': (te_s, te_e, scaled_test[:, 0]),
        }
        print("  DWT applied on SCALED signal (standardized Patv)")
    else:
        signal_full = raw[:, 0]  # Patv column
        signals = {
            'train': (tr_s, tr_e, signal_full[tr_s:tr_e]),
            'valid': (va_s, va_e, signal_full[va_s:va_e]),
            'test': (te_s, te_e, signal_full[te_s:te_e]),
        }
        print("  DWT applied on RAW signal (original kW scale)")

    # Decompose each partition independently (no leakage)
    all_imfs = np.zeros((N, max_level + 1), dtype=np.float32)

    for name, (start, end, segment) in signals.items():
        imfs = dwt_decompose_causal(segment, wavelet=wavelet, max_level=max_level)
        all_imfs[start:end] = imfs
        recon_err = np.abs(imfs.sum(axis=1) - segment).max()
        print(f"  DWT {name}: [{start}:{end}] → {imfs.shape}, "
              f"signal_range=[{segment.min():.3f}, {segment.max():.3f}], "
              f"reconstruction_error={recon_err:.2e}")

    # Save in same format as vmd_imfs.npz
    np.savez_compressed(output_path, all_imfs=all_imfs)
    print(f"  Saved DWT decomposition to {output_path}, shape={all_imfs.shape}")

    return all_imfs


def generate_dwt_imfs_all_horizons(
    csv_path: str,
    manifest_dir: str,
    output_path: str,
    scaler_path: str = None,
    wavelet: str = "db4",
    max_level: int = 4,
    target_col: str = "Patv",
) -> np.ndarray:
    """Generate DWT using the h=12 partition (canonical split).

    Since DWT is applied per-partition and the partition boundaries only
    depend on the split ratios (not the horizon), we use the canonical
    h=12 partition file. The same IMF file works for all horizons because
    the train/valid/test boundaries are identical (only the number of
    valid windows changes with horizon, not the raw data boundaries).

    Parameters
    ----------
    csv_path : str
        Path to cleaned CSV.
    manifest_dir : str
        Directory containing partition_indices_l144_h*.json files.
    output_path : str
        Output .npz path.
    scaler_path : str, optional
        Path to fitted scaler. If provided, DWT on standardized signal.
    wavelet, max_level, target_col : as in generate_dwt_imfs.

    Returns
    -------
    np.ndarray
    """
    import os
    # Use h=12 as the canonical partition (boundaries are the same for all h)
    partition_path = os.path.join(manifest_dir, "partition_indices_l144_h12.json")
    if not os.path.exists(partition_path):
        # Fallback: try any available partition file
        for h in [1, 6, 24, 12]:
            p = os.path.join(manifest_dir, f"partition_indices_l144_h{h}.json")
            if os.path.exists(p):
                partition_path = p
                break

    # Auto-detect scaler if not provided
    if scaler_path is None:
        default_scaler = os.path.join(manifest_dir, "scaler.pkl")
        if os.path.exists(default_scaler):
            scaler_path = default_scaler

    return generate_dwt_imfs(
        csv_path=csv_path,
        partition_path=partition_path,
        output_path=output_path,
        scaler_path=scaler_path,
        wavelet=wavelet,
        max_level=max_level,
        target_col=target_col,
    )
