"""
scripts/aggregate_cli.py
========================
Command-line entry point for result aggregation.

Reads run_records.jsonl, builds all four paper-ready tables, optionally
builds the innovation-experiment traceability matrix, and optionally
triggers the visualization suite to render figures.

Usage
-----
    python scripts/aggregate_cli.py [options]

Options
-------
    --records-path  PATH   Path to run_records.jsonl
                           (default: outputs/runs/run_records.jsonl)
    --out-dir       DIR    Directory for tables and figures
                           (default: outputs/)
    --figures              Also render prediction-curve and error-distribution
                           figures for each horizon
    --scaler-path   PATH   Path to scaler.pkl, required when --figures is set
    --run-id        ID     run_id whose preds.npz is used for --figures;
                           if omitted, the most recent 'proposed' run is used
    --no-traceability      Skip building the traceability matrix
    --verbose              Enable DEBUG-level logging

Exit codes
----------
    0   All requested outputs produced successfully (or best-effort on
        partial failures — individual table/figure errors are logged but
        do not abort the whole aggregation).
    1   A fatal error prevented any output (e.g., missing records file
        AND strict mode, or an unhandled exception).

Requirements: 14.6, 14.9, 14.10
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve project root and add to sys.path
# ---------------------------------------------------------------------------
# Layout: <project_root>/scripts/aggregate_cli.py
_SCRIPT_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _SCRIPT_FILE.parent.parent  # <project_root>/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set working directory so that relative paths in configs resolve correctly.
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("aggregate_cli")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = [1, 6, 12, 24]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_proposed_run_id(records_path: str) -> str | None:
    """Scan run_records.jsonl and return the run_id of the most recent
    successful 'proposed' run, or None if none is found."""
    import json

    p = Path(records_path)
    if not p.is_file():
        return None

    latest = None
    with open(p, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if (
                    rec.get("model_name") == "proposed"
                    and rec.get("status") == "success"
                ):
                    latest = rec.get("run_id")
            except (json.JSONDecodeError, TypeError):
                pass
    return latest


def _resolve_preds_path(
    records_path: str,
    run_id: str | None,
    out_dir: str,
) -> str | None:
    """Return the path to a preds.npz file for the given run_id.

    If run_id is None, falls back to the latest 'proposed' run.
    Returns None if no suitable preds file can be located.
    """
    if run_id is None:
        run_id = _find_latest_proposed_run_id(records_path)
        if run_id:
            logger.info("Using most recent proposed run for figures: %s", run_id)
        else:
            logger.warning(
                "No successful 'proposed' run found in %s; skipping figures.",
                records_path,
            )
            return None

    # Preds files live in outputs/runs/ relative to project root.
    preds_path = Path(out_dir) / "runs" / f"{run_id}_preds.npz"
    if preds_path.is_file():
        return str(preds_path)

    # Also try the default location regardless of out_dir.
    default_preds = Path("outputs/runs") / f"{run_id}_preds.npz"
    if default_preds.is_file():
        return str(default_preds)

    logger.warning("Predictions file not found for run_id=%s; skipping figures.", run_id)
    return None


# ---------------------------------------------------------------------------
# Step 1: Aggregate runs → 4 tables
# ---------------------------------------------------------------------------

def _step_aggregate(records_path: str, tables_dir: str) -> object | None:
    """Call aggregate_runs and return the AggregateTables result.

    Returns None on failure.
    """
    try:
        from experiments.aggregator import aggregate_runs
        tables = aggregate_runs(records_path=records_path, out_dir=tables_dir)
        logger.info("Aggregation complete. Tables written to: %s", tables_dir)
        return tables
    except Exception as exc:  # noqa: BLE001
        logger.error("Aggregation failed: %s", exc)
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Step 2: Build traceability matrix
# ---------------------------------------------------------------------------

def _step_traceability(records_path: str, tables_dir: str) -> bool:
    """Build the innovation-experiment traceability matrix.

    Returns True on success, False on failure.
    """
    try:
        from experiments.traceability import build_matrix_from_records
        matrix = build_matrix_from_records(
            records_path=records_path,
            out_dir=tables_dir,
        )
        n_pass = (matrix == "PASS").sum().sum()
        n_fail = (matrix == "FAIL").sum().sum()
        n_missing = (matrix == "MISSING").sum().sum()
        logger.info(
            "Traceability matrix built: PASS=%d FAIL=%d MISSING=%d",
            n_pass,
            n_fail,
            n_missing,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Traceability matrix failed: %s", exc)
        logger.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Step 3: Trigger visualizations
# ---------------------------------------------------------------------------

def _step_figures(
    preds_path: str,
    scaler_path: str,
    figures_dir: str,
) -> dict[int, int]:
    """Render prediction-curve and error-distribution figures for each horizon.

    Returns a dict {horizon: return_code} (0 = success, 1 = failure).
    """
    results: dict[int, int] = {}

    for horizon in HORIZONS:
        # Prediction curve
        try:
            from visualization.prediction_curve import plot_prediction_curve
            rc_pred = plot_prediction_curve(
                preds_path=preds_path,
                scaler_path=scaler_path,
                horizon=horizon,
                out_dir=figures_dir,
            )
            if rc_pred != 0:
                logger.warning("prediction_curve returned non-zero for h=%d", horizon)
        except Exception as exc:  # noqa: BLE001
            logger.error("prediction_curve failed for h=%d: %s", horizon, exc)
            rc_pred = 1

        # Error distribution
        try:
            from visualization.error_distribution import plot_error_distribution
            rc_err = plot_error_distribution(
                preds_path=preds_path,
                scaler_path=scaler_path,
                horizon=horizon,
                out_dir=figures_dir,
            )
            if rc_err != 0:
                logger.warning("error_distribution returned non-zero for h=%d", horizon)
        except Exception as exc:  # noqa: BLE001
            logger.error("error_distribution failed for h=%d: %s", horizon, exc)
            rc_err = 1

        results[horizon] = max(rc_pred, rc_err)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(
    records_path: str,
    tables_dir: str,
    tables_ok: bool,
    traceability_ok: bool | None,
    figure_results: dict[int, int] | None,
) -> None:
    """Print a structured summary of generated outputs."""
    sep = "=" * 60
    print()
    print(sep)
    print("  Aggregation Summary")
    print(sep)
    print(f"  Records source : {records_path}")
    print(f"  Tables dir     : {tables_dir}")
    print()

    if tables_ok:
        print("  Tables (CSV + LaTeX)")
        table_names = [
            "baseline_comparison_table",
            "ablation_table",
            "statistical_summary_table",
            "significance_table",
        ]
        tables_path = Path(tables_dir)
        for name in table_names:
            csv_path = tables_path / f"{name}.csv"
            tex_path = tables_path / f"{name}.tex"
            csv_status = "✓" if csv_path.is_file() else "✗"
            tex_status = "✓" if tex_path.is_file() else "✗"
            print(f"    [{csv_status}] {name}.csv")
            print(f"    [{tex_status}] {name}.tex")
        print()
    else:
        print("  Tables : FAILED (see logs above)")
        print()

    if traceability_ok is None:
        print("  Traceability matrix : SKIPPED (--no-traceability)")
    elif traceability_ok:
        tm_csv = Path(tables_dir) / "traceability_matrix.csv"
        tm_tex = Path(tables_dir) / "traceability_matrix.tex"
        csv_status = "✓" if tm_csv.is_file() else "✗"
        tex_status = "✓" if tm_tex.is_file() else "✗"
        print("  Traceability matrix")
        print(f"    [{csv_status}] traceability_matrix.csv")
        print(f"    [{tex_status}] traceability_matrix.tex")
    else:
        print("  Traceability matrix : FAILED (see logs above)")
    print()

    if figure_results is None:
        print("  Figures : SKIPPED (use --figures to generate)")
    else:
        print("  Figures")
        all_ok = all(rc == 0 for rc in figure_results.values())
        for h, rc in sorted(figure_results.items()):
            status = "✓" if rc == 0 else "✗"
            print(f"    [{status}] horizon h={h:2d}")
        if not all_ok:
            print("    (some figures failed — check logs above)")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run aggregation pipeline, return exit code.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    int
        0 on success (or partial success with warnings), 1 on fatal failure.
    """
    parser = argparse.ArgumentParser(
        prog="python scripts/aggregate_cli.py",
        description=(
            "Aggregate experiment run records into paper-ready tables,\n"
            "traceability matrix, and (optionally) visualizations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--records-path",
        default="outputs/runs/run_records.jsonl",
        metavar="PATH",
        help=(
            "Path to run_records.jsonl "
            "(default: outputs/runs/run_records.jsonl)"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/",
        metavar="DIR",
        help=(
            "Base output directory. Tables go to <out-dir>/tables/, "
            "figures to <out-dir>/figures/ "
            "(default: outputs/)"
        ),
    )
    parser.add_argument(
        "--figures",
        action="store_true",
        default=False,
        help=(
            "Also render prediction-curve and error-distribution figures "
            "for each horizon. Requires --scaler-path."
        ),
    )
    parser.add_argument(
        "--scaler-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to scaler.pkl. Required when --figures is set. "
            "(default: outputs/manifests/scaler.pkl)"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help=(
            "run_id of the preds.npz to use for figures. "
            "If omitted, the most recent successful 'proposed' run is used."
        ),
    )
    parser.add_argument(
        "--no-traceability",
        action="store_true",
        default=False,
        help="Skip building the innovation-experiment traceability matrix.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve directories
    out_dir = str(Path(args.out_dir).resolve())
    tables_dir = str(Path(out_dir) / "tables")
    figures_dir = str(Path(out_dir) / "figures")
    records_path = str(Path(args.records_path).resolve()
                       if not Path(args.records_path).is_absolute()
                       else Path(args.records_path))

    logger.info("Records: %s", records_path)
    logger.info("Tables dir: %s", tables_dir)

    # ── Step 1: Aggregate runs → 4 tables ────────────────────────────────
    tables_result = _step_aggregate(records_path=records_path, tables_dir=tables_dir)
    tables_ok = tables_result is not None

    # ── Step 2: Traceability matrix ───────────────────────────────────────
    traceability_ok: bool | None = None
    if not args.no_traceability:
        traceability_ok = _step_traceability(
            records_path=records_path,
            tables_dir=tables_dir,
        )
    # else: stays None → shown as SKIPPED

    # ── Step 3: Figures (optional) ────────────────────────────────────────
    figure_results: dict[int, int] | None = None
    if args.figures:
        # Resolve scaler path
        scaler_path = args.scaler_path or str(
            Path(out_dir) / "manifests" / "scaler.pkl"
        )

        preds_path = _resolve_preds_path(
            records_path=records_path,
            run_id=args.run_id,
            out_dir=out_dir,
        )

        if preds_path is None or not Path(scaler_path).is_file():
            if preds_path is None:
                logger.warning("No preds file found; skipping figure generation.")
            if not Path(scaler_path).is_file():
                logger.warning(
                    "Scaler not found at %s; skipping figure generation.",
                    scaler_path,
                )
            figure_results = {h: 1 for h in HORIZONS}
        else:
            logger.info("Generating figures using preds: %s", preds_path)
            figure_results = _step_figures(
                preds_path=preds_path,
                scaler_path=scaler_path,
                figures_dir=figures_dir,
            )

    # ── Print summary ─────────────────────────────────────────────────────
    _print_summary(
        records_path=records_path,
        tables_dir=tables_dir,
        tables_ok=tables_ok,
        traceability_ok=traceability_ok,
        figure_results=figure_results,
    )

    # ── Exit code ─────────────────────────────────────────────────────────
    # Fatal only if the core aggregation itself failed.
    # Individual table-write errors or figure errors are warnings,
    # not fatal failures (they are logged and shown in the summary).
    if not tables_ok:
        logger.error("Core aggregation failed; exiting with code 1.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
