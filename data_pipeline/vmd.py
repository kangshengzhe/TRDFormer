"""
VMD (Variational Mode Decomposition) module for the data pipeline.

Provides:
- fit_vmd_on_train: Run VMD on the training partition's target signal, returning IMFs.
- apply_vmd_to_partition: Re-run VMD with same parameters on val/test partitions.
- persist_vmd_params: Persist the VMD parameters used to ``vmd_params.json``.

The key insight is that ``vmdpy`` does not have a ``transform`` API, so we re-run
VMD with the same fixed parameters on each partition. We never RE-TUNE the
parameters on validation/test data — we just reuse the exact same K, alpha, tau,
DC, init, tol that were used on the training partition. This satisfies the
requirement that VMD parameters are not recomputed using validation or test data
(Requirement 1.16).

The ``vmdpy`` dependency is imported lazily inside the functions that need it so
that this module can always be imported (e.g. on environments where ``vmdpy`` is
not installed). ``vmdpy`` should be listed in ``requirements-local.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from data_pipeline.manifest import VMDParams, VMDManifest

# The VMD library this module is built against. Persisted into the manifest so
# the decomposition can be reproduced with the matching library version.
VMD_LIBRARY = "vmdpy==0.2"

# Allowable range for the number of IMF modes K (Requirement 1.15).
K_MIN = 3
K_MAX = 10


def _validate_params(params: VMDParams) -> None:
    """Validate VMD parameters.

    Raises
    ------
    ValueError
        If ``K`` is not in the inclusive range [3, 10].
    """
    if not (K_MIN <= params.K <= K_MAX):
        raise ValueError(
            f"VMD parameter K must be in [{K_MIN}, {K_MAX}], got {params.K}"
        )


def _import_vmd():
    """Lazily import the ``VMD`` callable from ``vmdpy``.

    Returns
    -------
    Callable
        The ``vmdpy.VMD`` function.

    Raises
    ------
    ImportError
        With a clear, actionable message if ``vmdpy`` is not installed.
    """
    try:
        from vmdpy import VMD  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without vmdpy
        raise ImportError(
            "The 'vmdpy' package is required for VMD decomposition but is not "
            "installed. Install it with `pip install vmdpy==0.2` (it is listed "
            "in requirements-local.txt). VMD runs on the local CPU stage of the "
            "pipeline."
        ) from exc
    return VMD


def _match_length(imfs: np.ndarray, target_len: int) -> np.ndarray:
    """Resize the IMF array along the time axis to ``target_len`` rows.

    ``vmdpy`` mirrors/extends the signal internally and returns an output that
    is truncated to an even length, so an odd-length input of length N yields an
    output of length N-1. This restores the IMF array to exactly match the
    number of input samples by linear interpolation (when the lengths differ)
    or by a plain truncation/identity when they already align.

    Parameters
    ----------
    imfs : np.ndarray
        IMF array of shape (n_out, K).
    target_len : int
        Desired number of rows (the length of the original input signal).

    Returns
    -------
    np.ndarray
        IMF array of shape (target_len, K).
    """
    n_out, k = imfs.shape
    if n_out == target_len:
        return imfs

    if n_out > target_len:
        # Output longer than input: truncate the trailing samples.
        return imfs[:target_len, :]

    # Output shorter than input (the common odd-length case): interpolate each
    # IMF channel back onto the original sample grid so the row count matches.
    src_grid = np.linspace(0.0, 1.0, num=n_out)
    dst_grid = np.linspace(0.0, 1.0, num=target_len)
    resized = np.empty((target_len, k), dtype=imfs.dtype)
    for j in range(k):
        resized[:, j] = np.interp(dst_grid, src_grid, imfs[:, j])
    return resized


def _run_vmd(signal: np.ndarray, params: VMDParams) -> np.ndarray:
    """Run VMD on a 1-D signal using the given parameters.

    Parameters
    ----------
    signal : np.ndarray
        1-D array of the target variable (e.g., Patv).
    params : VMDParams
        VMD configuration parameters.

    Returns
    -------
    np.ndarray
        Array of shape (len(signal), K) containing the K IMF components,
        transposed from vmdpy's native (K, n_samples) output and resized so the
        number of rows matches the length of the input signal.
    """
    VMD = _import_vmd()

    # vmdpy.VMD returns (u, u_hat, omega) where:
    #   u: (K, n_samples) — the K mode functions
    #   u_hat: frequency-domain representation
    #   omega: center frequencies
    u, _, _ = VMD(
        signal,
        params.alpha,
        params.tau,
        params.K,
        params.DC,
        params.init,
        params.tol,
    )

    # Transpose from (K, n_samples) to (n_samples, K).
    imfs = np.asarray(u).T

    # vmdpy may truncate odd-length signals to an even length; restore the
    # original sample count so the IMF channels align with the input rows.
    return _match_length(imfs, target_len=signal.shape[0])


def _as_1d(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``arr`` to a contiguous 1-D float64 array.

    Raises
    ------
    ValueError
        If the input cannot be represented as a 1-D signal.
    """
    out = np.asarray(arr, dtype=np.float64)
    if out.ndim == 2 and 1 in out.shape:
        out = out.ravel()
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {np.asarray(arr).shape}")
    if out.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return out


def fit_vmd_on_train(target_train: np.ndarray, params: VMDParams) -> np.ndarray:
    """Apply VMD to the training partition's target variable.

    This is the "fitting" step — we run VMD on the training data using the
    fixed VMD hyperparameters. The parameters are not learned from the data;
    they are the authoritative configuration reused for all other partitions.

    Parameters
    ----------
    target_train : np.ndarray
        1-D array of shape (n_train,) containing the target variable values
        from the training partition.
    params : VMDParams
        VMD configuration parameters. K must be in [3, 10].

    Returns
    -------
    np.ndarray
        Array of shape (n_train, K) containing the K IMF components for the
        training partition.

    Raises
    ------
    ValueError
        If K is not in [3, 10] or if ``target_train`` is not a non-empty 1-D
        signal.
    """
    _validate_params(params)
    signal = _as_1d(target_train, "target_train")
    return _run_vmd(signal, params)


def apply_vmd_to_partition(
    target_partition: np.ndarray, params: VMDParams
) -> np.ndarray:
    """Apply VMD to a non-training partition (validation or test) using the
    same parameters that were used on the training partition.

    Since ``vmdpy`` does not expose a ``transform`` API, we re-run VMD with the
    identical parameters. The critical constraint (Requirement 1.16) is that we
    never re-tune the parameters — we reuse K, alpha, tau, DC, init, tol exactly
    as supplied, so no information is fitted from validation or test data.

    Parameters
    ----------
    target_partition : np.ndarray
        1-D array of shape (n_samples,) containing the target variable values
        from the validation or test partition.
    params : VMDParams
        VMD configuration parameters (same as used in ``fit_vmd_on_train``).

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, K) containing the K IMF components for the
        given partition.

    Raises
    ------
    ValueError
        If K is not in [3, 10] or if ``target_partition`` is not a non-empty
        1-D signal.
    """
    _validate_params(params)
    signal = _as_1d(target_partition, "target_partition")
    return _run_vmd(signal, params)


def persist_vmd_params(
    path: str | Path,
    params: VMDParams,
    *,
    fit_seed: int = 42,
    fit_n_samples: int = 0,
    imf_shape: tuple[int, int] = (0, 0),
) -> Path:
    """Persist the VMD parameters to ``vmd_params.json`` via the manifest module.

    This records the number of modes K, the penalty term alpha, the tolerance,
    the maximum iterations, and additional reproducibility metadata so the
    decomposition can be reproduced (Requirement 1.17).

    Parameters
    ----------
    path : str or Path
        Directory (uses the default ``vmd_params.json`` filename) or a full
        file path.
    params : VMDParams
        The VMD configuration parameters that were used.
    fit_seed : int
        Seed used when fitting VMD on the training partition.
    fit_n_samples : int
        Number of training samples used for VMD fitting.
    imf_shape : tuple[int, int]
        Shape of the full IMF array (n_total_samples, K).

    Returns
    -------
    Path
        The path to the written manifest file.
    """
    _validate_params(params)
    return VMDManifest.write(
        path,
        params,
        library=VMD_LIBRARY,
        fit_seed=fit_seed,
        fit_n_samples=fit_n_samples,
        imf_shape=imf_shape,
    )
