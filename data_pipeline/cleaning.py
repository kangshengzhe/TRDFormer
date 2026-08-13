"""
Physical-rule outlier cleaning for SDWPF wind power data.

Applies domain-specific filtering rules to remove physically impossible
or highly suspicious samples from the SCADA-derived wind turbine signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Physical constants for the turbine
_WSPD_MIN: float = 0.0       # Minimum valid wind speed (m/s)
_WSPD_MAX: float = 25.0      # Maximum valid wind speed (m/s)
_CUTIN_SPEED: float = 3.0    # Turbine cut-in wind speed (m/s)

# Column names in canonical order
_FEATURE_COLS: list[str] = ["Patv", "Wspd", "Wdir", "Etmp", "Itmp"]


def physical_rule_clean(
    df: pd.DataFrame,
    *,
    enable: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Apply physical-rule outlier filters to wind power data.

    Rules applied in order:
    1. Clip Patv < 0 to 0 (negative active power is physically impossible).
    2. Mark entire row as NaN where Wspd is outside [0, 25] m/s.
    3. Mark entire row as NaN where Patv > 0 and Wspd < 3 m/s (below cut-in).

    When a row is "marked NaN", all 5 feature columns are set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least the columns {Patv, Wspd, Wdir, Etmp, Itmp}.
    enable : bool, default True
        If False (outlier_off ablation), returns a copy of the DataFrame unchanged
        with zero counts in the report dict.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - cleaned_df: A modified copy of the input (original is not mutated).
        - report: Dict with keys:
            'n_clipped_negative_patv': int
            'n_marked_wspd_out_of_range': int
            'n_marked_below_cutin_with_power': int
    """
    # Always work on a copy to avoid mutating the original
    cleaned = df.copy()

    # Zero-count report for the disabled case
    zero_report = {
        "n_clipped_negative_patv": 0,
        "n_marked_wspd_out_of_range": 0,
        "n_marked_below_cutin_with_power": 0,
    }

    if not enable:
        return cleaned, zero_report

    # --- Rule 1: Clip negative Patv to 0 ---
    negative_mask = cleaned["Patv"] < 0
    n_clipped = int(negative_mask.sum())
    cleaned.loc[negative_mask, "Patv"] = 0.0

    # --- Rule 2: Mark NaN where Wspd outside [0, 25] m/s ---
    wspd_out_of_range = (cleaned["Wspd"] < _WSPD_MIN) | (cleaned["Wspd"] > _WSPD_MAX)
    n_wspd_oor = int(wspd_out_of_range.sum())
    cleaned.loc[wspd_out_of_range, _FEATURE_COLS] = np.nan

    # --- Rule 3: Mark NaN where Patv > 0 and Wspd < cut-in speed ---
    # Note: After rule 1, Patv >= 0 everywhere (non-NaN rows).
    # After rule 2, some rows may already be NaN. We only consider
    # rows that are still valid (non-NaN Patv and Wspd).
    below_cutin_with_power = (cleaned["Patv"] > 0) & (cleaned["Wspd"] < _CUTIN_SPEED)
    n_below_cutin = int(below_cutin_with_power.sum())
    cleaned.loc[below_cutin_with_power, _FEATURE_COLS] = np.nan

    report = {
        "n_clipped_negative_patv": n_clipped,
        "n_marked_wspd_out_of_range": n_wspd_oor,
        "n_marked_below_cutin_with_power": n_below_cutin,
    }

    return cleaned, report
