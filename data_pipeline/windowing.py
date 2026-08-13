"""
Windowed time-series dataset for the wind power forecasting pipeline.

Provides:
- WindowedSeriesDataset: a torch.utils.data.Dataset that produces fixed-length
  (lookback, F_in) input windows and (horizon,) target sequences from a
  pre-scaled feature matrix.

Channel layout (column 0 is ALWAYS Patv – the target variable):
    VMD on:  [Patv, IMF_1, …, IMF_K, Wspd, Wdir, Etmp, Itmp]
    VMD off: [Patv, Wspd, Wdir, Etmp, Itmp]

The dataset returns:
    x : FloatTensor of shape (lookback, F_in)   – model input window
    y : FloatTensor of shape (horizon,)          – target-column future steps

The target column is always index 0 (Patv). The caller is responsible for
assembling the data matrix with the correct channel layout BEFORE passing it
to this class – this class only slices windows from whatever is given.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowedSeriesDataset(Dataset):
    """Sliding-window dataset over a pre-scaled feature matrix.

    Given a 2-D numpy array `data` with shape ``(N, F_in)``, this dataset
    produces ``N - lookback - horizon + 1`` non-overlapping (in start-position)
    sliding windows.

    Each item ``(x, y)`` is:
        x : FloatTensor, shape ``(lookback, F_in)``   — all features for the
            input window ending at the step before the forecast period.
        y : FloatTensor, shape ``(horizon,)``          — the target column
            (column index 0, i.e. Patv) for the ``horizon`` steps immediately
            following the input window.

    Parameters
    ----------
    data : np.ndarray
        Pre-scaled feature matrix of shape ``(N, F_in)`` where column 0 is
        always the target variable (Patv).
        Channel layout when VMD is enabled:
            [Patv, IMF_1, …, IMF_K, Wspd, Wdir, Etmp, Itmp]
        Channel layout when VMD is disabled:
            [Patv, Wspd, Wdir, Etmp, Itmp]
    lookback : int
        Number of past time steps used as model input (must be ≥ 1).
    horizon : int
        Number of future steps the model must predict (must be ≥ 1).

    Raises
    ------
    ValueError
        If ``lookback`` or ``horizon`` are not positive integers, or if
        ``data`` does not have at least ``lookback + horizon`` rows.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 5).astype(np.float32)
    >>> ds = WindowedSeriesDataset(data, lookback=144, horizon=6)
    >>> len(ds)
    851
    >>> x, y = ds[0]
    >>> x.shape, y.shape
    (torch.Size([144, 5]), torch.Size([6,]))
    """

    TARGET_COL: int = 0  # Patv is always column 0

    def __init__(
        self,
        data: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be a positive integer, got {lookback}")
        if horizon < 1:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")

        data = np.asarray(data, dtype=np.float32)

        if data.ndim != 2:
            raise ValueError(
                f"data must be a 2-D array of shape (N, F_in), got shape {data.shape}"
            )

        n_rows, n_features = data.shape
        if n_rows < lookback + horizon:
            raise ValueError(
                f"data has only {n_rows} rows but lookback + horizon = "
                f"{lookback + horizon}. Need at least {lookback + horizon} rows."
            )
        if n_features < 1:
            raise ValueError("data must have at least 1 feature column.")

        self._data = torch.from_numpy(data)  # shape (N, F_in), float32
        self._lookback = lookback
        self._horizon = horizon
        self._n_samples = n_rows - lookback - horizon + 1

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of (x, y) windows in the dataset."""
        return self._n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a single (x, y) window pair.

        Parameters
        ----------
        idx : int
            Window index in ``[0, len(self) - 1]``.

        Returns
        -------
        x : torch.FloatTensor, shape ``(lookback, F_in)``
            All features for the input window.
        y : torch.FloatTensor, shape ``(horizon,)``
            Target column (Patv, index 0) values for the forecast period.
        """
        if idx < 0 or idx >= self._n_samples:
            raise IndexError(
                f"Index {idx} is out of range for dataset of length {self._n_samples}"
            )

        start = idx
        end_x = start + self._lookback           # exclusive end of input window
        end_y = end_x + self._horizon             # exclusive end of target window

        x = self._data[start:end_x, :]           # (lookback, F_in)
        y = self._data[end_x:end_y, self.TARGET_COL]  # (horizon,)

        return x, y

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def lookback(self) -> int:
        """Input sequence length."""
        return self._lookback

    @property
    def horizon(self) -> int:
        """Forecast horizon length."""
        return self._horizon

    @property
    def n_features(self) -> int:
        """Number of input feature channels (F_in)."""
        return self._data.shape[1]
