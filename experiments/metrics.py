"""
experiments/metrics.py
======================
Standalone Metric_Evaluator module.

Computes the five evaluation metrics defined in Requirements 6.1–6.3 and 6.7
on denormalized kW values:

    MAE   — Mean Absolute Error
    RMSE  — Root Mean Squared Error
    R²    — Coefficient of Determination
    MBE   — Mean Bias Error  (positive → overestimate)
    sMAPE — Symmetric Mean Absolute Percentage Error  [0, 200 %]

Usage
-----
    from experiments.metrics import compute_metrics

    metrics = compute_metrics(actual_kw, predicted_kw)
    # {'mae': ..., 'rmse': ..., 'r2': ..., 'mbe': ..., 'smape': ...}
"""

import numpy as np


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, R², MBE, and sMAPE on denormalized kW values.

    Inputs of shape ``(n_samples,)`` or ``(n_samples, n_horizons)`` are both
    accepted; both arrays are flattened to 1-D before any computation.

    Parameters
    ----------
    actual : array-like
        Ground-truth active-power values in kW.
    predicted : array-like
        Predicted active-power values in kW.

    Returns
    -------
    dict
        Keys: ``'mae'``, ``'rmse'``, ``'r2'``, ``'mbe'``, ``'smape'``.

        * All values are plain Python ``float``.
        * ``smape`` is expressed as a percentage in ``[0, 200]``.
        * Samples where **both** ``actual`` and ``predicted`` are exactly
          0 kW are excluded from the sMAPE denominator (Requirement 6.7).

    Raises
    ------
    ValueError
        If ``actual`` and ``predicted`` have a different number of elements
        after flattening.

    Examples
    --------
    >>> import numpy as np
    >>> from experiments.metrics import compute_metrics
    >>> a = np.array([100.0, 200.0, 0.0])
    >>> p = np.array([ 90.0, 210.0, 0.0])
    >>> m = compute_metrics(a, p)
    >>> round(m['mae'], 4)
    10.0
    >>> round(m['mbe'], 4)   # predicted > actual on avg? no: 90+210+0 vs 100+200+0
    0.0
    """
    a = np.asarray(actual, dtype=float).flatten()
    p = np.asarray(predicted, dtype=float).flatten()

    if a.shape != p.shape:
        raise ValueError(
            f"actual and predicted must contain the same number of elements; "
            f"got shapes {np.asarray(actual).shape} and {np.asarray(predicted).shape} "
            f"({a.size} vs {p.size} elements after flattening)."
        )

    # ── MAE ─────────────────────────────────────────────────────────────────
    mae = float(np.mean(np.abs(a - p)))

    # ── RMSE ─────────────────────────────────────────────────────────────────
    rmse = float(np.sqrt(np.mean((a - p) ** 2)))

    # ── R² ───────────────────────────────────────────────────────────────────
    ss_res = np.sum((a - p) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    # Guard against degenerate constant target (ss_tot == 0)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else float("nan")

    # ── MBE ──────────────────────────────────────────────────────────────────
    # Positive MBE means the model overestimates on average.
    mbe = float(np.mean(p - a))

    # ── sMAPE ────────────────────────────────────────────────────────────────
    # Formula: 200 * |a - p| / (|a| + |p|), averaged over valid samples.
    # "Valid" = NOT (a == 0 AND p == 0).  (Requirement 6.7)
    both_zero = (a == 0.0) & (p == 0.0)
    denom = np.abs(a) + np.abs(p)

    # After excluding both-zero pairs the denominator cannot be 0, but we add
    # a tiny floor for numerical safety (e.g. near-zero floating-point noise).
    denom_safe = np.where(both_zero, 1.0, np.maximum(denom, 1e-12))
    per_sample = 200.0 * np.abs(a - p) / denom_safe

    # Zero out excluded samples so they don't contribute to the sum.
    per_sample = np.where(both_zero, 0.0, per_sample)

    n_valid = int(np.sum(~both_zero))
    smape = float(np.sum(per_sample) / n_valid) if n_valid > 0 else 0.0

    return {
        "mae":   mae,
        "rmse":  rmse,
        "r2":    r2,
        "mbe":   mbe,
        "smape": smape,
    }
