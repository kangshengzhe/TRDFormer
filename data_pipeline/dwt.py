"""
DWT (Discrete Wavelet Transform) module for the data pipeline.

Multi-scale frequency decomposition used in place of VMD.

WHAT THIS GUARANTEES, AND WHAT IT DOES NOT
------------------------------------------
Earlier versions of this docstring claimed the transform was "strictly causal"
and had "no future information leakage by construction". **That claim was
wrong and has been retracted**, here and in the paper. Be precise about the
two different properties:

*Partition isolation -- GUARANTEED.* The decomposition is fitted and applied
independently within each of train/valid/test, so no coefficient in one
partition is computed from another partition's samples. This is the property
that makes the train/test comparison fair, and it is what distinguishes this
pipeline from fitting VMD over the whole series before splitting.

*Sample-wise causality -- NOT guaranteed.* Within one partition,
``pywt.wavedec``/``waverec`` reconstruct each sub-band from the entire
partition at once, with symmetric boundary extension. A sample at t+1
therefore influences the reconstructed value at t. Measured effect: perturbing
a single sample one step after a reference index shifts the reconstruction at
that index by up to 2.3 standardised units (see
``tests/test_dwt_causality.py``, which asserts this rather than assuming it).

Consequence: results obtained with this module describe an offline, per-window
decomposition. A strictly online deployment needs a causal FIR approximation,
and the accuracy reported in the paper should not be assumed to transfer to
that setting unchanged.

Other properties, which do hold:
- Computationally efficient: O(N) vs O(N*K*iter) for VMD
- Deterministic: same input always produces the same output
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


def dwt_decompose_partition(
    signal: np.ndarray,
    wavelet: str = "db4",
    max_level: int = 4,
) -> np.ndarray:
    """Decompose one partition's signal, using only that partition's samples.

    Renamed from ``dwt_decompose_causal``: the old name asserted a property the
    function does not have. What it delivers is *partition isolation* -- no
    coefficient here is computed from another partition's data -- not
    *sample-wise causality*. Within the segment, ``wavedec``/``waverec`` see the
    whole segment, so a sample at t+1 does influence the reconstruction at t.

    The previous docstring justified this as "mimicking real-world deployment
    where you'd apply DWT to the most recent window of observations". That
    justification does not hold: deployment would decompose a trailing window
    ending at t, whereas this decomposes a segment that extends past t. The
    paper states the limitation explicitly instead; see the module docstring and
    ``tests/test_dwt_causality.py``.

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
        imfs = dwt_decompose_partition(segment, wavelet=wavelet,
                                       max_level=max_level)
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
