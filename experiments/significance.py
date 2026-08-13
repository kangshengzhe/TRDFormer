"""Statistical significance testing for model comparisons.

Provides paired t-test utilities to determine whether the difference in
performance between two models is statistically significant across multiple
independent runs (seeds).

Validates: Requirements 6.4, 6.5, 6.6, 6.8
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import ttest_rel

from reproducibility.records import RunRecord


def paired_significance(
    model_a_records: list[RunRecord],
    model_b_records: list[RunRecord],
    metric: str = "mae",
) -> float:
    """Run a paired t-test on per-seed metric values between two models.

    Records are matched by seed: both lists are sorted by ``RunRecord.seed``
    before extracting metric values so that the i-th entry of model A is
    compared to the i-th entry of model B at the same seed.

    Args:
        model_a_records: List of RunRecords for the first model (e.g., the
            proposed model). Must have the same length as *model_b_records*
            and at least 5 entries.
        model_b_records: List of RunRecords for the second model (e.g., a
            baseline). Must have the same length as *model_a_records* and at
            least 5 entries.
        metric: The metric key to extract from ``RunRecord.metrics``.
            Defaults to ``'mae'``.

    Returns:
        The two-tailed p-value from ``scipy.stats.ttest_rel``.

    Raises:
        ValueError: If the two lists have different lengths, if fewer than 5
            records are provided, or if the sorted seed sequences do not match
            between the two lists.
    """
    if len(model_a_records) != len(model_b_records):
        raise ValueError(
            f"model_a_records and model_b_records must have the same length, "
            f"got {len(model_a_records)} vs {len(model_b_records)}."
        )

    n = len(model_a_records)
    if n < 5:
        raise ValueError(
            f"At least 5 records are required for a paired significance test, "
            f"got {n}."
        )

    # Sort both lists by seed so matching is deterministic.
    sorted_a = sorted(model_a_records, key=lambda r: r.seed)
    sorted_b = sorted(model_b_records, key=lambda r: r.seed)

    seeds_a = [r.seed for r in sorted_a]
    seeds_b = [r.seed for r in sorted_b]
    if seeds_a != seeds_b:
        raise ValueError(
            f"Seeds must match between the two record lists after sorting. "
            f"Got seeds_a={seeds_a}, seeds_b={seeds_b}."
        )

    values_a = [r.metrics[metric] for r in sorted_a]
    values_b = [r.metrics[metric] for r in sorted_b]

    _, p_value = ttest_rel(values_a, values_b)
    return float(p_value)


def is_significant(p_value: float, threshold: float = 0.05) -> bool:
    """Return True when *p_value* is below *threshold*.

    Args:
        p_value: The p-value returned by :func:`paired_significance`.
        threshold: Significance level. Defaults to 0.05.

    Returns:
        ``True`` if ``p_value < threshold``, ``False`` otherwise.
    """
    return p_value < threshold


def compute_significance_table(
    proposed_records: list[RunRecord],
    baseline_records_dict: dict[str, list[RunRecord]],
    metric: str = "mae",
) -> pd.DataFrame:
    """Compute a table of p-values for proposed vs each baseline at each horizon.

    For every (baseline, horizon) pair the function calls
    :func:`paired_significance` on the subset of records matching that horizon,
    then assembles a DataFrame indexed by baseline model name with one column
    per horizon.

    If a particular (baseline, horizon) pair cannot be tested (e.g., fewer
    than 5 matched seeds, mismatched seeds, or missing metric key), the
    corresponding cell is set to ``float('nan')``.

    Args:
        proposed_records: All RunRecords for the proposed model across all
            horizons and seeds.
        baseline_records_dict: Mapping from baseline model name to its list of
            RunRecords (across all horizons and seeds).
        metric: Metric key used for comparison. Defaults to ``'mae'``.

    Returns:
        A :class:`pandas.DataFrame` indexed by baseline model name (rows) with
        columns for each unique horizon found in *proposed_records*. Each cell
        contains the p-value (float) for that (baseline, horizon) comparison,
        or ``nan`` if the test could not be run.
    """
    # Discover all horizons present in proposed_records.
    horizons = sorted({r.horizon for r in proposed_records})

    rows: dict[str, dict[int, float]] = {}

    for baseline_name, baseline_records in baseline_records_dict.items():
        row: dict[int, float] = {}
        for horizon in horizons:
            proposed_h = [r for r in proposed_records if r.horizon == horizon]
            baseline_h = [r for r in baseline_records if r.horizon == horizon]
            try:
                p_val = paired_significance(proposed_h, baseline_h, metric=metric)
            except (ValueError, KeyError):
                p_val = float("nan")
            row[horizon] = p_val
        rows[baseline_name] = row

    df = pd.DataFrame.from_dict(rows, orient="index", columns=horizons)
    df.index.name = "baseline_model"
    return df
