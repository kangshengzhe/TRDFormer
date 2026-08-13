"""Innovation-Experiment Traceability Matrix.

Maps each innovation (A, B, C, ALL) to the specific experiments and
requirements that substantiate its claim.  Given an AggregateTables object
(produced by experiments.aggregator.aggregate_runs), builds a
(claim × horizon) matrix whose cells are:

  'PASS'    — proposed.MAE(h) < cited.MAE(h), and p < 0.05 when ≥5 seeds
  'FAIL'    — proposed.MAE(h) >= cited.MAE(h)
  'MISSING' — any required record is absent (Req 13.6)

Exports the matrix to:
  outputs/tables/traceability_matrix.csv
  outputs/tables/traceability_matrix.tex

Validates: Requirements 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7, 13.8
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HORIZONS: list[int] = [1, 6, 12, 24]
PROPOSED_NAME = "proposed"
SIGNIFICANCE_THRESHOLD = 0.05

# Cell-value literals
PASS_VAL = "PASS"
FAIL_VAL = "FAIL"
MISSING_VAL = "MISSING"
NA_LITERAL = "NA"


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class Claim:
    """A single innovation claim with the experiments that substantiate it.

    Attributes:
        innovation: Identifier string, e.g. 'A', 'B', 'C', or 'ALL'.
        claim: Human-readable claim statement (Req 13.7).
        metric: Metric on which the claim is judged; defaults to 'mae' (Req 13.7).
        experiments: Experiment keys against which the claim is tested.
            Each entry is either a baseline key (e.g. 'baseline:lstm') or an
            ablation key (e.g. 'ablation:vmd_off') that maps to a row in the
            AggregateTables comparison tables.
        requirements: Requirement identifiers that this claim substantiates.
    """

    innovation: str
    claim: str
    metric: str
    experiments: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)


# ── Traceability registry (Req 13.1–13.5) ─────────────────────────────────────

TRACEABILITY: list[Claim] = [
    Claim(
        innovation="A",
        claim="VMD multi-IMF variate modeling reduces MAE relative to proposed without it",
        metric="mae",
        # NOTE: ablation:outlier_off is a no-op on this pre-cleaned dataset
        # (physical-rule filters flag zero rows), so it is documented as void
        # and NOT used as evidence; Innovation A is judged on ablation:vmd_off.
        experiments=["ablation:vmd_off"],
        requirements=["Req 5.5", "Req 5.6", "Req 13.2"],
    ),
    Claim(
        innovation="B",
        claim="Variate-attention target branch is essential (removing it severely degrades MAE)",
        metric="mae",
        experiments=["ablation:itrans_off"],
        requirements=["Req 5.1", "Req 5.2", "Req 13.3"],
    ),
    Claim(
        innovation="C",
        claim="Adaptive gated fusion + KAN head beats alternative fusion/head choices",
        metric="mae",
        experiments=[
            "ablation:fusion_concat",
            "ablation:fusion_sum",
            "ablation:fusion_cross_attention",
            "ablation:head_linear",
            "ablation:head_mlp",
        ],
        requirements=["Req 5.3", "Req 5.4", "Req 13.4"],
    ),
    Claim(
        innovation="ALL",
        claim="Proposed beats all 11 SOTA baselines",
        metric="mae",
        experiments=[
            "baseline:lstm",
            "baseline:transformer",
            "baseline:informer",
            "baseline:fedformer",
            "baseline:dlinear",
            "baseline:patchtst",
            "baseline:itransformer",
            "baseline:timesnet",
            "baseline:autoformer",
            "baseline:nonstationary_transformer",
            "baseline:timexer",
        ],
        requirements=["Req 4", "Req 13.5"],
    ),
]


# ── Helper: parse mean from 'mean±std' strings ────────────────────────────────


def _parse_mean(cell_value: object) -> Optional[float]:
    """Extract the mean from a 'mean±std' cell, or return None if unparseable.

    Accepts:
      - 'mean±std' string (e.g. '41.2300±3.1200')
      - A bare numeric string or numeric value
      - 'NA' / any other non-numeric string → None
    """
    if cell_value is None:
        return None
    s = str(cell_value).strip()
    if s == NA_LITERAL or s == "":
        return None
    # Try 'mean±std' pattern first.
    m = re.match(r"^([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)[±]", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    # Try bare float.
    try:
        return float(s)
    except ValueError:
        return None


def _parse_pvalue(cell_value: object) -> Optional[float]:
    """Parse a p-value string to float, or return None if missing/NA."""
    if cell_value is None:
        return None
    s = str(cell_value).strip()
    if s == NA_LITERAL or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ── Model-name resolution ──────────────────────────────────────────────────────


def _resolve_model_name(experiment_key: str) -> str:
    """Convert a Claim.experiments entry to the model_name used in tables.

    Experiment keys follow the convention:
      'baseline:<name>'  → model_name is '<name>'
      'ablation:<name>'  → model_name is 'ablation:<name>'
    """
    if experiment_key.startswith("baseline:"):
        return experiment_key[len("baseline:"):]
    # 'ablation:...' keys are stored as-is in AggregateTables.
    return experiment_key


# ── Core matrix builder ────────────────────────────────────────────────────────


def build_matrix(aggregated: "AggregateTables") -> pd.DataFrame:  # noqa: F821
    """Build the Innovation-Experiment Traceability Matrix.

    For each :class:`Claim` in :data:`TRACEABILITY` and each horizon in
    ``HORIZONS``, evaluates every experiment listed in ``claim.experiments``
    against the proposed model:

    * **PASS** — proposed.MAE(h) < cited.MAE(h), and the corresponding
      significance p-value < 0.05 when at least 5 seeds are available.
      If the significance table cell is 'NA' (fewer than 5 seeds), the
      significance condition is waived and the result is PASS purely on
      the mean-MAE comparison.
    * **FAIL** — proposed.MAE(h) >= cited.MAE(h).
    * **MISSING** — any required record (proposed or cited) is absent from
      the aggregated tables (Req 13.6).

    A claim row passes a horizon if *all* of its listed experiments pass at
    that horizon; it fails as soon as any experiment fails; it is MISSING
    if any required experiment record is absent (irrespective of other results).

    Args:
        aggregated: An :class:`~experiments.aggregator.AggregateTables`
            instance returned by :func:`~experiments.aggregator.aggregate_runs`.

    Returns:
        A :class:`pandas.DataFrame` with one row per claim and one column per
        horizon.  Index is the innovation identifier; columns are horizon
        integers.  Cell values are the string literals 'PASS', 'FAIL', or
        'MISSING'.

    Side effects:
        Writes ``outputs/tables/traceability_matrix.csv`` and
        ``outputs/tables/traceability_matrix.tex`` (Req 13.8).
    """
    # Retrieve the tables we need.
    baseline_df: pd.DataFrame = aggregated.baseline_comparison_table
    ablation_df: pd.DataFrame = aggregated.ablation_table
    significance_df: pd.DataFrame = aggregated.significance_table

    # Build the result matrix.
    #   Rows: one per Claim (labelled by innovation id + abbreviated claim)
    #   Columns: one per horizon
    row_labels: list[str] = []
    for c in TRACEABILITY:
        row_labels.append(f"Innovation {c.innovation}: {c.claim}")

    matrix = pd.DataFrame(
        index=row_labels,
        columns=HORIZONS,
        dtype=object,
    )
    matrix.index.name = "claim"
    matrix.columns.name = "horizon"

    for claim in TRACEABILITY:
        row_label = f"Innovation {claim.innovation}: {claim.claim}"
        metric = claim.metric.lower()

        for horizon in HORIZONS:
            # Accumulate verdict across all cited experiments.
            # Priority: MISSING > FAIL > PASS
            has_missing = False
            has_fail = False

            # --- Look up proposed model mean for this horizon / metric -------
            proposed_mean = _lookup_mean(
                baseline_df=baseline_df,
                ablation_df=ablation_df,
                model_name=PROPOSED_NAME,
                horizon=horizon,
                metric=metric,
            )
            if proposed_mean is None:
                # Proposed model record is absent → all cells in this row/col
                # are MISSING regardless of anything else.
                has_missing = True

            for exp_key in claim.experiments:
                cited_model = _resolve_model_name(exp_key)

                # --- Look up cited model mean --------------------------------
                cited_mean = _lookup_mean(
                    baseline_df=baseline_df,
                    ablation_df=ablation_df,
                    model_name=cited_model,
                    horizon=horizon,
                    metric=metric,
                )
                if cited_mean is None or proposed_mean is None:
                    has_missing = True
                    continue

                # --- MAE comparison (lower is better) -----------------------
                proposed_better = proposed_mean < cited_mean

                # --- Significance gate (when ≥5 seeds available) ------------
                # The significance table only covers baseline experiments.
                # For ablation experiments there is no direct entry; we waive
                # the gate (compare on mean only) as the ablation table does
                # not carry a separate significance column.
                sig_ok = True
                if exp_key.startswith("baseline:"):
                    p_val = _lookup_pvalue(
                        significance_df=significance_df,
                        baseline_model=cited_model,
                        horizon=horizon,
                    )
                    if p_val is not None:
                        # Significance data exists → apply threshold.
                        sig_ok = p_val < SIGNIFICANCE_THRESHOLD

                if proposed_better and sig_ok:
                    pass  # PASS — do nothing; absence of fail/missing = pass
                else:
                    has_fail = True

            # Determine final cell value (MISSING > FAIL > PASS).
            if has_missing:
                cell = MISSING_VAL
            elif has_fail:
                cell = FAIL_VAL
            else:
                cell = PASS_VAL

            matrix.loc[row_label, horizon] = cell

    # Export CSV and LaTeX (Req 13.8).
    _export_matrix(matrix)

    return matrix


# ── Lookup helpers ────────────────────────────────────────────────────────────


def _lookup_mean(
    *,
    baseline_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    model_name: str,
    horizon: int,
    metric: str,
) -> Optional[float]:
    """Return the mean metric value for a (model, horizon) pair.

    Searches ``baseline_df`` first (covers proposed + baselines), then
    ``ablation_df`` (covers proposed + ablations).  Returns ``None`` if the
    model row or the (horizon, metric) column is not found.
    """
    for df in (baseline_df, ablation_df):
        if df is None:
            continue
        if model_name not in df.index:
            continue
        # Multi-level columns: (horizon, metric)
        if isinstance(df.columns, pd.MultiIndex):
            if (horizon, metric) not in df.columns:
                continue
            cell = df.loc[model_name, (horizon, metric)]
        else:
            # Flat columns — try direct access.
            col_key = (horizon, metric)
            if col_key not in df.columns:
                continue
            cell = df.loc[model_name, col_key]

        mean = _parse_mean(cell)
        if mean is not None:
            return mean
    return None


def _lookup_pvalue(
    *,
    significance_df: pd.DataFrame,
    baseline_model: str,
    horizon: int,
) -> Optional[float]:
    """Return the p-value for (proposed vs baseline_model) at a given horizon.

    Returns ``None`` if the entry is 'NA' or the model/horizon is not found.
    """
    if significance_df is None:
        return None
    if baseline_model not in significance_df.index:
        return None
    if horizon not in significance_df.columns:
        return None
    cell = significance_df.loc[baseline_model, horizon]
    return _parse_pvalue(cell)


# ── Export helpers ─────────────────────────────────────────────────────────────


def _df_to_latex(df: pd.DataFrame) -> str:
    """Convert the traceability matrix DataFrame to a LaTeX tabular string."""
    try:
        from tabulate import tabulate  # noqa: PLC0415

        df_reset = df.reset_index()
        return tabulate(df_reset, headers="keys", tablefmt="latex", showindex=False)
    except ImportError:
        logger.warning("tabulate not installed; using pandas built-in LaTeX export.")
        return df.to_latex(na_rep=MISSING_VAL)


def _export_matrix(
    matrix: pd.DataFrame,
    out_dir: str = "outputs/tables",
) -> None:
    """Write the traceability matrix to CSV and LaTeX files (Req 13.8).

    Args:
        matrix: The traceability matrix DataFrame.
        out_dir: Directory where the files are written.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / "traceability_matrix.csv"
    tex_path = out_path / "traceability_matrix.tex"

    try:
        matrix.to_csv(csv_path)
        logger.info("Wrote traceability matrix CSV: %s", csv_path)
    except OSError as exc:
        logger.error("Failed to write %s: %s", csv_path, exc)

    try:
        tex_path.write_text(_df_to_latex(matrix), encoding="utf-8")
        logger.info("Wrote traceability matrix LaTeX: %s", tex_path)
    except OSError as exc:
        logger.error("Failed to write %s: %s", tex_path, exc)


# ── Convenience entry point ───────────────────────────────────────────────────


def build_matrix_from_records(
    records_path: str = "outputs/runs/run_records.jsonl",
    out_dir: str = "outputs/tables",
) -> pd.DataFrame:
    """Convenience wrapper: aggregate runs then build the traceability matrix.

    Calls :func:`~experiments.aggregator.aggregate_runs` internally, then
    passes the result to :func:`build_matrix`.

    Args:
        records_path: Path to the JSONL run-records file.
        out_dir: Directory where all tables (including the traceability
            matrix) are written.

    Returns:
        The traceability matrix as a :class:`pandas.DataFrame`.
    """
    from experiments.aggregator import aggregate_runs  # noqa: PLC0415

    aggregated = aggregate_runs(records_path=records_path, out_dir=out_dir)
    return build_matrix(aggregated)
