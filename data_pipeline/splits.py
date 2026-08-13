"""
Chronological train/validation/test splitting for time-series data.

Splits are performed WITHOUT shuffling, preserving temporal order:
- First 80% (truncated to int) → training partition
- Final 10% (truncated to int) → test partition
- Remaining middle rows → validation partition

The function returns a PartitionIndices dataclass (defined in manifest.py)
and optionally persists the indices to a JSON manifest.
"""

from __future__ import annotations

import pandas as pd

from data_pipeline.manifest import PartitionIndices, PartitionManifest


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
    *,
    lookback: int,
    horizon: int,
) -> PartitionIndices:
    """Split a DataFrame chronologically into train/valid/test partitions.

    The split is deterministic and does NOT shuffle the data. Row ordering
    in the DataFrame is assumed to already be chronological.

    Parameters
    ----------
    df : pd.DataFrame
        The preprocessed DataFrame (already cleaned, interpolated, dropna'd).
    train_ratio : float
        Fraction of rows allocated to training (default 0.8).
    test_ratio : float
        Fraction of rows allocated to testing (default 0.1).
    lookback : int
        Number of past time steps used as model input.
    horizon : int
        Number of future steps the model predicts.

    Returns
    -------
    PartitionIndices
        Dataclass with half-open [start, end) boundaries for each partition,
        plus the lookback and horizon values used.

    Raises
    ------
    ValueError
        If ratios are invalid (sum > 1, negative, or zero rows result).
    """
    n = len(df)

    if train_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio and test_ratio must be positive.")
    if train_ratio + test_ratio > 1.0:
        raise ValueError(
            f"train_ratio + test_ratio = {train_ratio + test_ratio} exceeds 1.0"
        )
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive integers.")

    # Truncate to integer counts
    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    n_valid = n - n_train - n_test

    if n_train < 1 or n_test < 1 or n_valid < 1:
        raise ValueError(
            f"Partition sizes too small: train={n_train}, valid={n_valid}, "
            f"test={n_test} from n={n} rows."
        )

    # Half-open intervals [start, end)
    train_start = 0
    train_end = n_train

    valid_start = n_train
    valid_end = n_train + n_valid

    test_start = n_train + n_valid
    test_end = n

    return PartitionIndices(
        train=(train_start, train_end),
        valid=(valid_start, valid_end),
        test=(test_start, test_end),
        lookback=lookback,
        horizon=horizon,
    )


def persist_partition_indices(
    indices: PartitionIndices,
    n_total_rows: int,
    manifest_dir: str,
) -> str:
    """Persist partition indices to JSON via the PartitionManifest module.

    Parameters
    ----------
    indices : PartitionIndices
        The computed partition boundaries.
    n_total_rows : int
        Total number of rows in the dataset after preprocessing.
    manifest_dir : str
        Directory where the manifest JSON file will be written.

    Returns
    -------
    str
        The path to the written manifest file.
    """
    filepath = PartitionManifest.write(manifest_dir, indices, n_total_rows)
    return str(filepath)
