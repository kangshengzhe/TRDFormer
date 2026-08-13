"""Unit tests for data_pipeline.cleaning.physical_rule_clean."""

import numpy as np
import pandas as pd
import pytest

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_pipeline.cleaning import physical_rule_clean


def _make_df(patv, wspd, wdir=None, etmp=None, itmp=None):
    """Helper to build a DataFrame with the 5 canonical columns."""
    n = len(patv)
    if wdir is None:
        wdir = [180.0] * n
    if etmp is None:
        etmp = [20.0] * n
    if itmp is None:
        itmp = [25.0] * n
    return pd.DataFrame({
        "Patv": patv,
        "Wspd": wspd,
        "Wdir": wdir,
        "Etmp": etmp,
        "Itmp": itmp,
    })


class TestPhysicalRuleCleanDisabled:
    """Tests for enable=False (outlier_off ablation)."""

    def test_returns_copy_unchanged(self):
        df = _make_df(patv=[-5.0, 100.0, 50.0], wspd=[30.0, 2.0, 10.0])
        cleaned, report = physical_rule_clean(df, enable=False)

        # DataFrame should be identical to original
        pd.testing.assert_frame_equal(cleaned, df)

        # Report should have all zeros
        assert report["n_clipped_negative_patv"] == 0
        assert report["n_marked_wspd_out_of_range"] == 0
        assert report["n_marked_below_cutin_with_power"] == 0

    def test_does_not_mutate_original(self):
        df = _make_df(patv=[-5.0, 100.0], wspd=[10.0, 10.0])
        original_copy = df.copy()
        physical_rule_clean(df, enable=False)
        pd.testing.assert_frame_equal(df, original_copy)


class TestRule1ClipNegativePatv:
    """Tests for Rule 1: Clip Patv < 0 to 0."""

    def test_negative_patv_clipped_to_zero(self):
        df = _make_df(patv=[-10.0, -0.1, 0.0, 50.0], wspd=[10.0, 10.0, 10.0, 10.0])
        cleaned, report = physical_rule_clean(df)

        assert cleaned["Patv"].iloc[0] == 0.0
        assert cleaned["Patv"].iloc[1] == 0.0
        assert cleaned["Patv"].iloc[2] == 0.0
        assert cleaned["Patv"].iloc[3] == 50.0
        assert report["n_clipped_negative_patv"] == 2

    def test_zero_patv_not_counted_as_clipped(self):
        df = _make_df(patv=[0.0, 100.0], wspd=[10.0, 10.0])
        _, report = physical_rule_clean(df)
        assert report["n_clipped_negative_patv"] == 0


class TestRule2WspdOutOfRange:
    """Tests for Rule 2: Mark NaN where Wspd outside [0, 25] m/s."""

    def test_wspd_above_25_marks_all_cols_nan(self):
        df = _make_df(patv=[100.0, 200.0], wspd=[25.0, 25.1])
        cleaned, report = physical_rule_clean(df)

        # First row is fine (Wspd == 25 is within range)
        assert not cleaned.iloc[0].isna().any()
        # Second row is all NaN
        assert cleaned.iloc[1].isna().all()
        assert report["n_marked_wspd_out_of_range"] == 1

    def test_wspd_below_zero_marks_all_cols_nan(self):
        df = _make_df(patv=[100.0, 200.0], wspd=[-0.1, 5.0])
        cleaned, report = physical_rule_clean(df)

        assert cleaned.iloc[0].isna().all()
        assert not cleaned.iloc[1].isna().any()
        assert report["n_marked_wspd_out_of_range"] == 1

    def test_boundary_values_valid(self):
        """Wspd = 0 and Wspd = 25 are both within the valid range."""
        df = _make_df(patv=[0.0, 100.0], wspd=[0.0, 25.0])
        cleaned, report = physical_rule_clean(df)

        assert not cleaned.iloc[0].isna().any()
        assert not cleaned.iloc[1].isna().any()
        assert report["n_marked_wspd_out_of_range"] == 0


class TestRule3BelowCutinWithPower:
    """Tests for Rule 3: Mark NaN where Patv > 0 and Wspd < 3 m/s."""

    def test_positive_patv_below_cutin_marked_nan(self):
        df = _make_df(patv=[50.0, 100.0], wspd=[2.9, 3.0])
        cleaned, report = physical_rule_clean(df)

        # First row: Patv > 0 and Wspd < 3 → NaN
        assert cleaned.iloc[0].isna().all()
        # Second row: Wspd == 3.0, so not below cut-in → valid
        assert not cleaned.iloc[1].isna().any()
        assert report["n_marked_below_cutin_with_power"] == 1

    def test_zero_patv_below_cutin_not_marked(self):
        """If Patv == 0 and Wspd < 3, no action (rule requires Patv > 0)."""
        df = _make_df(patv=[0.0], wspd=[1.0])
        cleaned, report = physical_rule_clean(df)

        assert not cleaned.iloc[0].isna().any()
        assert report["n_marked_below_cutin_with_power"] == 0

    def test_originally_negative_patv_clipped_then_not_flagged(self):
        """Patv was -5 (clipped to 0 by Rule 1), so Rule 3 should NOT flag it."""
        df = _make_df(patv=[-5.0], wspd=[2.0])
        cleaned, report = physical_rule_clean(df)

        # Rule 1 clips to 0, Rule 3 checks Patv > 0 → won't trigger
        assert not cleaned.iloc[0].isna().any()
        assert cleaned["Patv"].iloc[0] == 0.0
        assert report["n_clipped_negative_patv"] == 1
        assert report["n_marked_below_cutin_with_power"] == 0


class TestRuleOrdering:
    """Tests verifying the correct ordering of rules."""

    def test_wspd_oor_row_not_counted_in_rule3(self):
        """A row already NaN'd by Rule 2 shouldn't be counted by Rule 3."""
        # Wspd = -1 is out of range (Rule 2), and Patv = 50 > 0 with Wspd < 3
        # But since Rule 2 fires first, the row is already NaN
        df = _make_df(patv=[50.0], wspd=[-1.0])
        cleaned, report = physical_rule_clean(df)

        assert cleaned.iloc[0].isna().all()
        assert report["n_marked_wspd_out_of_range"] == 1
        # After Rule 2 NaNs the row, Rule 3 sees NaN for Patv/Wspd → condition is False
        assert report["n_marked_below_cutin_with_power"] == 0


class TestOriginalNotMutated:
    """Ensure the original DataFrame is never mutated."""

    def test_original_unchanged_after_cleaning(self):
        df = _make_df(patv=[-5.0, 100.0, 30.0], wspd=[30.0, 1.5, 10.0])
        original_copy = df.copy()
        physical_rule_clean(df, enable=True)
        pd.testing.assert_frame_equal(df, original_copy)


class TestReportDictKeys:
    """Verify the report dict always has the required keys."""

    def test_report_keys_present(self):
        df = _make_df(patv=[100.0], wspd=[10.0])
        _, report = physical_rule_clean(df)

        assert "n_clipped_negative_patv" in report
        assert "n_marked_wspd_out_of_range" in report
        assert "n_marked_below_cutin_with_power" in report
