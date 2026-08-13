"""Result aggregation: consolidate run_records.jsonl into paper-ready tables.

Reads the append-only JSONL run log produced by experiments/runner.py and
produces four structured tables:

  1. baseline_comparison_table  — Proposed model vs 8 SOTA baselines
  2. ablation_table             — Proposed model vs 8 ablation variants
  3. statistical_summary_table  — Mean ± std across seeds for every (model, horizon)
  4. significance_table         — p-values from paired t-tests (proposed vs each baseline)

Each table is exported as both CSV and LaTeX into outputs/tables/.
Missing (model, horizon) cells are filled with the literal string 'NA'.
On write error, tables fall back to outputs/tables/_recovery/ with a UTC
timestamp suffix and the event is logged to outputs/runs/aggregator_recovery.log.

Validates: Requirements 3.5, 4.7, 4.8, 5.10, 11.1, 11.2, 11.3, 11.4, 11.5,
           11.6, 11.7
"""

from __future__ import annotations

import json
import logging
import os
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from experiments.matrix import PROPOSED, BASELINES, ABLATIONS

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
# BASELINES and ABLATIONS are imported from experiments.matrix (single
# source of truth) rather than duplicated here, so that adding a new
# baseline/ablation variant (e.g. 'ablation:fusion_gated') only requires
# updating matrix.py.

PROPOSED_NAME = PROPOSED[0]

METRICS: list[str] = ["mae", "rmse", "r2", "mbe", "smape"]

HORIZONS: list[int] = [1, 6, 12, 24]

NA_LITERAL = "NA"

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class AggregateTables:
    """Container for all four aggregated tables."""

    baseline_comparison_table: pd.DataFrame
    ablation_table: pd.DataFrame
    statistical_summary_table: pd.DataFrame
    significance_table: pd.DataFrame


# ── Core aggregation helpers ──────────────────────────────────────────────────


def _load_records(records_path: str) -> list[dict]:
    """Load run records from a JSONL file.

    Returns an empty list (and logs a warning) if the file does not exist,
    rather than raising, so that the aggregator can still produce skeleton
    tables filled with 'NA'.
    """
    p = Path(records_path)
    if not p.exists():
        logger.warning("run_records.jsonl not found at %s; producing empty tables.", records_path)
        return []

    records: list[dict] = []
    with open(p, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                records.append(rec)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)
    return records


def _group_records(
    records: list[dict],
) -> dict[tuple[str, int], list[dict]]:
    """Group valid (status='success') records by (model_name, horizon).

    Records with unexpected schema (missing keys) are skipped with a warning.
    """
    groups: dict[tuple[str, int], list[dict]] = {}
    skipped = 0
    for rec in records:
        try:
            model_name: str = rec["model_name"]
            horizon: int = int(rec["horizon"])
            status: str = rec.get("status", "success")
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if status != "success":
            continue

        key = (model_name, horizon)
        groups.setdefault(key, []).append(rec)

    if skipped:
        logger.warning("Skipped %d records due to schema issues.", skipped)
    return groups


def _mean_std_str(values: list[float]) -> str:
    """Return 'mean±std' string rounded to 4 decimal places."""
    if not values:
        return NA_LITERAL
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def _mean_val(values: list[float]) -> float:
    """Return the mean, or NaN if the list is empty."""
    if not values:
        return float("nan")
    return float(np.mean(values))


# ── Table 1 & 2 — comparison tables (baselines / ablations) ──────────────────


def _build_comparison_table(
    groups: dict[tuple[str, int], list[dict]],
    model_rows: list[str],
    horizons: list[int] = HORIZONS,
    metrics: list[str] = METRICS,
) -> pd.DataFrame:
    """Build a wide-form comparison table with multi-level columns.

    Columns: (horizon, metric), rows: model names.
    Cell values: 'mean±std' string, or 'NA' if no data exists.
    """
    col_tuples = [(h, m) for h in horizons for m in metrics]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["horizon", "metric"])
    df = pd.DataFrame(index=model_rows, columns=columns, dtype=object)
    df.index.name = "model"

    for model in model_rows:
        for horizon in horizons:
            key = (model, horizon)
            recs = groups.get(key, [])
            for metric in metrics:
                values: list[float] = []
                for rec in recs:
                    try:
                        v = rec["metrics"][metric]
                        if v is not None and np.isfinite(float(v)):
                            values.append(float(v))
                    except (KeyError, TypeError, ValueError):
                        pass
                df.loc[model, (horizon, metric)] = _mean_std_str(values)

    return df


# ── Table 3 — statistical summary table ──────────────────────────────────────


def _build_statistical_summary_table(
    groups: dict[tuple[str, int], list[dict]],
    all_models: list[str],
    horizons: list[int] = HORIZONS,
    metrics: list[str] = METRICS,
) -> pd.DataFrame:
    """Build a detailed statistical summary table.

    Rows are (model, horizon) pairs.
    Columns are (metric, stat) where stat ∈ {mean, std, n_seeds}.
    Cells are numeric strings or 'NA'.
    """
    stat_cols = [(m, s) for m in metrics for s in ["mean", "std", "n_seeds"]]
    columns = pd.MultiIndex.from_tuples(stat_cols, names=["metric", "stat"])
    row_tuples = [(m, h) for m in all_models for h in horizons]
    index = pd.MultiIndex.from_tuples(row_tuples, names=["model", "horizon"])
    df = pd.DataFrame(index=index, columns=columns, dtype=object)

    for model in all_models:
        for horizon in horizons:
            key = (model, horizon)
            recs = groups.get(key, [])
            for metric in metrics:
                values: list[float] = []
                for rec in recs:
                    try:
                        v = rec["metrics"][metric]
                        if v is not None and np.isfinite(float(v)):
                            values.append(float(v))
                    except (KeyError, TypeError, ValueError):
                        pass

                if not values:
                    df.loc[(model, horizon), (metric, "mean")] = NA_LITERAL
                    df.loc[(model, horizon), (metric, "std")] = NA_LITERAL
                    df.loc[(model, horizon), (metric, "n_seeds")] = NA_LITERAL
                else:
                    arr = np.array(values, dtype=float)
                    df.loc[(model, horizon), (metric, "mean")] = f"{float(np.mean(arr)):.4f}"
                    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                    df.loc[(model, horizon), (metric, "std")] = f"{std:.4f}"
                    df.loc[(model, horizon), (metric, "n_seeds")] = str(len(values))

    return df


# ── Table 4 — significance table ──────────────────────────────────────────────


def _build_significance_table(
    groups: dict[tuple[str, int], list[dict]],
    baselines: list[str] = BASELINES,
    horizons: list[int] = HORIZONS,
    metric: str = "mae",
) -> pd.DataFrame:
    """Build a p-value table: proposed vs each baseline at each horizon.

    Uses scipy.stats.ttest_rel on per-seed metric values.
    Cells are the p-value as a string (6 decimal places) or 'NA' if fewer
    than 5 matched seeds are available.
    """
    try:
        from scipy.stats import ttest_rel  # noqa: PLC0415
    except ImportError:
        logger.error("scipy is not installed; significance_table will be all NA.")

        df = pd.DataFrame(NA_LITERAL, index=baselines, columns=horizons)
        df.index.name = "baseline_model"
        df.columns.name = "horizon"
        return df

    rows: dict[str, dict[int, str]] = {}

    for baseline in baselines:
        row: dict[int, str] = {}
        for horizon in horizons:
            proposed_recs = groups.get((PROPOSED_NAME, horizon), [])
            baseline_recs = groups.get((baseline, horizon), [])

            # Extract per-seed values for proposed and baseline.
            proposed_by_seed: dict[int, float] = {}
            for rec in proposed_recs:
                try:
                    seed = int(rec["seed"])
                    v = rec["metrics"][metric]
                    if v is not None and np.isfinite(float(v)):
                        proposed_by_seed[seed] = float(v)
                except (KeyError, TypeError, ValueError):
                    pass

            baseline_by_seed: dict[int, float] = {}
            for rec in baseline_recs:
                try:
                    seed = int(rec["seed"])
                    v = rec["metrics"][metric]
                    if v is not None and np.isfinite(float(v)):
                        baseline_by_seed[seed] = float(v)
                except (KeyError, TypeError, ValueError):
                    pass

            # Match seeds present in both.
            common_seeds = sorted(set(proposed_by_seed) & set(baseline_by_seed))

            if len(common_seeds) < 5:
                row[horizon] = NA_LITERAL
            else:
                vals_a = [proposed_by_seed[s] for s in common_seeds]
                vals_b = [baseline_by_seed[s] for s in common_seeds]
                try:
                    _, p_val = ttest_rel(vals_a, vals_b)
                    row[horizon] = f"{float(p_val):.6f}"
                except Exception:  # noqa: BLE001
                    row[horizon] = NA_LITERAL

        rows[baseline] = row

    df = pd.DataFrame.from_dict(rows, orient="index", columns=horizons)
    df.index.name = "baseline_model"
    df.columns.name = "horizon"
    return df


# ── File write helpers ─────────────────────────────────────────────────────────


def _recovery_path(out_dir: Path, stem: str, suffix: str) -> Path:
    """Return a timestamped recovery path under out_dir/_recovery/."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    recovery_dir = out_dir / "_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    return recovery_dir / f"{stem}_{ts}{suffix}"


def _log_recovery(log_path: Path, message: str) -> None:
    """Append a recovery event to the aggregator recovery log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {message}\n")
    except OSError:
        logger.warning("Could not write to recovery log at %s", log_path)


def _df_to_latex(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a LaTeX tabular string using tabulate if available,
    falling back to pandas' built-in to_latex method."""
    try:
        from tabulate import tabulate  # noqa: PLC0415

        # Flatten multi-level columns for tabulate compatibility.
        if isinstance(df.columns, pd.MultiIndex):
            flat_cols = [f"{c[0]}_{c[1]}" for c in df.columns]
            df_flat = df.copy()
            df_flat.columns = flat_cols
        else:
            df_flat = df.copy()

        if isinstance(df_flat.index, pd.MultiIndex):
            df_flat = df_flat.reset_index()
        else:
            df_flat = df_flat.reset_index()

        return tabulate(df_flat, headers="keys", tablefmt="latex", showindex=False)
    except ImportError:
        logger.warning("tabulate not installed; using pandas built-in LaTeX export.")
        return df.to_latex(na_rep=NA_LITERAL)


def _write_table(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    recovery_log: Path,
) -> None:
    """Write a table as both CSV and LaTeX.

    On error, attempt to write to the _recovery/ subdirectory and log the
    fallback.  On total failure, log and continue so other tables are not
    blocked.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix, writer in [
        (".csv", lambda p: df.to_csv(p)),
        (".tex", lambda p: p.write_text(_df_to_latex(df), encoding="utf-8")),
    ]:
        primary = out_dir / f"{name}{suffix}"
        try:
            writer(primary)
            logger.info("Wrote %s", primary)
        except OSError as exc:
            # Primary write failed — attempt recovery path.
            msg = (
                f"Failed to write {primary}: {exc}. "
                f"Attempting fallback to _recovery/."
            )
            logger.warning(msg)
            _log_recovery(recovery_log, msg)
            recovery = _recovery_path(out_dir, name, suffix)
            try:
                writer(recovery)
                recovery_msg = f"Fallback succeeded: wrote {recovery}"
                logger.info(recovery_msg)
                _log_recovery(recovery_log, recovery_msg)
            except OSError as exc2:
                fail_msg = f"Recovery write also failed for {name}{suffix}: {exc2}"
                logger.error(fail_msg)
                _log_recovery(recovery_log, fail_msg)


# ── Public API ────────────────────────────────────────────────────────────────


def aggregate_runs(
    records_path: str = "outputs/runs/run_records.jsonl",
    out_dir: str = "outputs/tables/",
) -> AggregateTables:
    """Read run_records.jsonl, build four tables, and export to CSV + LaTeX.

    Args:
        records_path: Path to the JSONL file containing RunRecord entries.
        out_dir: Directory where CSV and LaTeX files are written.

    Returns:
        An :class:`AggregateTables` instance holding all four DataFrames.

    Side effects:
        * Creates ``out_dir`` if it does not exist.
        * Writes ``{name}.csv`` and ``{name}.tex`` for each table.
        * On write error, writes to ``out_dir/_recovery/`` and appends to
          ``outputs/runs/aggregator_recovery.log``.
    """
    out_path = Path(out_dir)
    recovery_log = Path("outputs/runs/aggregator_recovery.log")

    # ── Load and group records ────────────────────────────────────────────
    raw_records = _load_records(records_path)
    groups = _group_records(raw_records)

    all_models = [PROPOSED_NAME] + BASELINES + ABLATIONS

    # ── Build tables ──────────────────────────────────────────────────────

    # 1. Baseline comparison table: proposed + 8 baselines
    baseline_rows = [PROPOSED_NAME] + BASELINES
    baseline_comparison_table = _build_comparison_table(
        groups,
        model_rows=baseline_rows,
        horizons=HORIZONS,
        metrics=METRICS,
    )

    # 2. Ablation table: proposed + 8 ablation variants
    ablation_rows = [PROPOSED_NAME] + ABLATIONS
    ablation_table = _build_comparison_table(
        groups,
        model_rows=ablation_rows,
        horizons=HORIZONS,
        metrics=METRICS,
    )

    # 3. Statistical summary table: all models
    statistical_summary_table = _build_statistical_summary_table(
        groups,
        all_models=all_models,
        horizons=HORIZONS,
        metrics=METRICS,
    )

    # 4. Significance table: proposed vs baselines
    significance_table = _build_significance_table(
        groups,
        baselines=BASELINES,
        horizons=HORIZONS,
        metric="mae",
    )

    # ── Export tables ────────────────────────────────────────────────────
    _write_table(baseline_comparison_table, out_path, "baseline_comparison_table", recovery_log)
    _write_table(ablation_table, out_path, "ablation_table", recovery_log)
    _write_table(statistical_summary_table, out_path, "statistical_summary_table", recovery_log)
    _write_table(significance_table, out_path, "significance_table", recovery_log)

    return AggregateTables(
        baseline_comparison_table=baseline_comparison_table,
        ablation_table=ablation_table,
        statistical_summary_table=statistical_summary_table,
        significance_table=significance_table,
    )
