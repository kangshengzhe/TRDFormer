"""Windowed dataset over a series that is broken into disjoint segments.

WHY THIS EXISTS
---------------
``WindowedSeriesDataset`` slides a window over one contiguous matrix, which is
correct for SDWPF (a single unbroken 10-min series) and is what all 1,108 runs
in the paper use. It is NOT correct for a record with sampling gaps: a window
that straddles a gap splices two moments that are days apart and presents the
join as if it were a real transition.

The second SCADA record in the paper (Turkey, 2018) has 32 gaps, three of them
longer than three days. Section 5.3 of the manuscript therefore decomposes it
strictly inside contiguous segments. To *train* on that record the same rule
has to hold for the windows: no window may cross a segment boundary.

This module adds that as a SEPARATE class rather than a flag on the existing
one, so the SDWPF code path is untouched and cannot regress.

WHAT IT GUARANTEES
------------------
For segment bounds [(s, e), ...] over a full-length matrix, a window starting
at ``t`` is emitted only if

    s <= t  and  t + lookback + horizon <= e

for some single segment. Every (x, y) pair is therefore drawn from one
uninterrupted stretch of 10-minute samples.

Usage
-----
    ds = SegmentedWindowedDataset(data, bounds, lookback=144, horizon=12)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SegmentedWindowedDataset(Dataset):
    """Sliding windows that never cross a segment boundary.

    Parameters
    ----------
    data : np.ndarray
        Pre-scaled feature matrix, shape ``(N, F_in)``, column 0 the target.
        Rows outside the given segments are allowed to be present (they are
        simply never used as part of any window).
    segment_bounds : sequence of (int, int)
        Half-open ``[start, end)`` index pairs into ``data``. Must be
        non-overlapping; they need not be adjacent.
    lookback, horizon : int
        Window geometry, same meaning as in ``WindowedSeriesDataset``.

    Raises
    ------
    ValueError
        If the geometry is invalid, the bounds overlap or run outside the
        matrix, or no segment is long enough to yield a single window.
    """

    TARGET_COL: int = 0

    def __init__(
        self,
        data: np.ndarray,
        segment_bounds,
        lookback: int,
        horizon: int,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D (N, F_in), got {data.shape}")
        n_rows = data.shape[0]

        bounds = [(int(a), int(b)) for a, b in segment_bounds]
        if not bounds:
            raise ValueError("segment_bounds is empty")
        for a, b in bounds:
            if not (0 <= a < b <= n_rows):
                raise ValueError(
                    f"segment ({a}, {b}) is outside a matrix of {n_rows} rows"
                )
        # Overlap check: sort by start and verify each begins at or after the
        # previous end. Overlapping segments would emit duplicate windows and
        # silently inflate the training set.
        ordered = sorted(bounds)
        for (a0, b0), (a1, _) in zip(ordered, ordered[1:]):
            if a1 < b0:
                raise ValueError(
                    f"segments overlap: ({a0}, {b0}) and ({a1}, ...)"
                )

        need = lookback + horizon
        starts: list[int] = []
        for a, b in ordered:
            last_start = b - need          # inclusive
            if last_start >= a:
                starts.extend(range(a, last_start + 1))

        if not starts:
            raise ValueError(
                f"no segment is long enough for lookback+horizon={need}; "
                f"longest segment has {max(b - a for a, b in ordered)} rows"
            )

        self._data = torch.from_numpy(data)
        self._starts = np.asarray(starts, dtype=np.int64)
        self._lookback = lookback
        self._horizon = horizon
        self._bounds = ordered

    def __len__(self) -> int:
        return int(self._starts.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"index {idx} out of range for {len(self)} windows")
        start = int(self._starts[idx])
        end_x = start + self._lookback
        end_y = end_x + self._horizon
        x = self._data[start:end_x, :]
        y = self._data[end_x:end_y, self.TARGET_COL]
        return x, y

    # -- introspection, used by the runner's logging ------------------------

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def n_features(self) -> int:
        return int(self._data.shape[1])

    @property
    def n_segments(self) -> int:
        return len(self._bounds)

    @property
    def window_starts(self) -> np.ndarray:
        """Global start index of every emitted window (for auditing)."""
        return self._starts.copy()
