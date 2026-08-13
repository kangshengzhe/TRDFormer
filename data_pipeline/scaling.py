"""
Feature scaling for the data pipeline.

Provides a FeatureScaler class that wraps sklearn's StandardScaler:
- Fits ONLY on the training partition (no data leakage).
- Applies the same transform to train/valid/test identically.
- Exposes a dedicated inverse_transform_target method for the target
  column (Patv, index 0), enabling denormalization of model predictions
  back to original kW scale.
- Supports persistence via pickle (dill) for reproducibility.
"""

from __future__ import annotations

import os
from pathlib import Path

import dill
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """StandardScaler wrapper fitted only on training data.

    The target column (Patv) is at index 0. The scaler operates on all
    feature columns jointly but exposes a dedicated inverse transform
    for the target column alone, which is needed to convert model
    predictions back to the original kW scale.

    Usage
    -----
    >>> scaler = FeatureScaler()
    >>> scaler.fit(train_array)          # shape (n_train, n_features)
    >>> scaled_train = scaler.transform(train_array)
    >>> scaled_valid = scaler.transform(valid_array)
    >>> # After model produces predictions in normalized scale:
    >>> preds_kw = scaler.inverse_transform_target(preds_normalized)
    """

    TARGET_COL_INDEX: int = 0

    def __init__(self) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the scaler has been fitted on training data."""
        return self._is_fitted

    def fit(self, train: np.ndarray) -> None:
        """Fit the scaler on the training partition only.

        Parameters
        ----------
        train : np.ndarray
            Training data array of shape (n_samples, n_features).
            Column 0 is assumed to be the target (Patv).

        Raises
        ------
        ValueError
            If train is not a 2D array or has fewer than 1 feature.
        """
        if train.ndim != 2:
            raise ValueError(
                f"Expected 2D array for fitting, got shape {train.shape}"
            )
        if train.shape[1] < 1:
            raise ValueError("Training data must have at least 1 feature column.")

        self._scaler.fit(train)
        self._is_fitted = True

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler to transform features.

        Parameters
        ----------
        x : np.ndarray
            Data array of shape (n_samples, n_features). Must have the
            same number of features as the training data used in fit().

        Returns
        -------
        np.ndarray
            Scaled array of the same shape.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        return self._scaler.transform(x)

    def inverse_transform_target(self, y_norm: np.ndarray) -> np.ndarray:
        """Inverse-transform only the target column (Patv, index 0).

        This is used to convert model predictions from the normalized
        scale back to the original kW scale.

        Parameters
        ----------
        y_norm : np.ndarray
            Normalized target values. Can be 1D (n_samples,) or 2D
            (n_samples, n_horizons).

        Returns
        -------
        np.ndarray
            Target values in the original kW scale, same shape as input.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")

        # Extract the mean and scale for the target column
        target_mean = self._scaler.mean_[self.TARGET_COL_INDEX]
        target_scale = self._scaler.scale_[self.TARGET_COL_INDEX]

        # Inverse transform: y_original = y_norm * scale + mean
        return y_norm * target_scale + target_mean

    def save(self, path: str) -> None:
        """Persist the scaler state to a file using dill.

        Parameters
        ----------
        path : str
            File path where the scaler will be saved (typically .pkl).

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")

        filepath = Path(path)
        os.makedirs(filepath.parent, exist_ok=True)

        with open(filepath, "wb") as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "FeatureScaler":
        """Load a persisted scaler from a file.

        Parameters
        ----------
        path : str
            File path to the saved scaler (typically .pkl).

        Returns
        -------
        FeatureScaler
            The deserialized scaler instance.

        Raises
        ------
        FileNotFoundError
            If the specified path does not exist.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {path}")

        with open(filepath, "rb") as f:
            scaler = dill.load(f)

        if not isinstance(scaler, cls):
            raise TypeError(
                f"Loaded object is not a FeatureScaler: {type(scaler)}"
            )

        return scaler
