"""CLI wrapper for the Optuna-based hyperparameter tuning module.

Usage
-----
Run from the repository root (``iTansformer_LSTM_CA_KAN-master/``):

    python -m scripts.tune.tune_cli \\
        --config configs/experiment/proposed_h6_seed42.yaml \\
        --study-name wind_proposed_h6 \\
        --n-trials 100 \\
        --horizon 6

The script:
1. Loads the base YAML configuration from ``--config``.
2. Loads the search space from ``--search-space`` (a YAML or JSON file) or
   uses a sensible default search space for the Proposed_Model.
3. Calls :func:`experiments.tuner.tune` and prints the resulting
   :class:`~experiments.tuner.BestTrialRecord` to stdout.

All output artefacts (best-trial JSON) are written to ``outputs/runs/`` by
default (as ``tuning_{study_name}.json``); use ``--records-dir`` to override.

Exit codes
----------
* ``0`` — tuning succeeded.
* ``1`` — tuning stalled (TuningStallError).
* ``2`` — bad arguments or config parse error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Default search space for the Proposed_Model
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_SPACE: dict = {
    "dim_embed": {
        "type": "categorical",
        "choices": [32, 64, 128, 256],
    },
    "depth_itrans": {
        "type": "int",
        "low": 1,
        "high": 6,
    },
    "heads_itrans": {
        "type": "categorical",
        "choices": [2, 4, 6, 8],
    },
    "dim_lstm": {
        "type": "categorical",
        "choices": [32, 64, 128, 256],
    },
    "depth_lstm": {
        "type": "int",
        "low": 1,
        "high": 4,
    },
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-3,
        "log": True,
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128, 256],
    },
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tune_cli",
        description="Optuna hyperparameter tuning for the wind-power Proposed_Model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the base YAML run config (e.g., configs/experiment/proposed_h6_seed42.yaml).",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        metavar="NAME",
        help=(
            "Optuna study name.  Defaults to 'tune_{model_name}_h{horizon}' "
            "derived from the loaded config."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        metavar="N",
        help="Total number of Optuna trials.",
    )
    parser.add_argument(
        "--min-trials",
        type=int,
        default=20,
        metavar="N",
        help="Minimum successful trials before TuningStallError is raised.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        metavar="H",
        help=(
            "Forecast horizon override (steps).  "
            "If omitted, the value from the config file is used."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Base random seed passed to the Optuna TPE sampler.",
    )
    parser.add_argument(
        "--search-space",
        default=None,
        metavar="PATH",
        help=(
            "Path to a YAML or JSON file defining the search space.  "
            "If omitted, a built-in default search space is used."
        ),
    )
    parser.add_argument(
        "--storage",
        default=None,
        metavar="URL",
        help=(
            "Optuna storage URL for study persistence "
            "(e.g., 'sqlite:///db.sqlite3').  "
            "Defaults to in-memory storage."
        ),
    )
    parser.add_argument(
        "--records-dir",
        default="outputs/runs",
        metavar="DIR",
        help=(
            "Directory where the BestTrialRecord JSON is saved.  "
            "The file is named 'tuning_{study_name}.json'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    filepath = Path(path)
    if not filepath.exists():
        print(f"[tune_cli] ERROR: file not found: {filepath}", file=sys.stderr)
        sys.exit(2)
    with open(filepath, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _load_search_space(path: str | None) -> dict:
    """Load the search space from *path* (YAML or JSON) or return the default."""
    if path is None:
        return DEFAULT_SEARCH_SPACE
    filepath = Path(path)
    if not filepath.exists():
        print(f"[tune_cli] ERROR: search-space file not found: {filepath}", file=sys.stderr)
        sys.exit(2)
    with open(filepath, "r", encoding="utf-8") as fh:
        if filepath.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(fh) or {}
        return json.load(fh)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, invoke :func:`~experiments.tuner.tune`, and return exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Set up logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Load base config.
    base_cfg = _load_yaml(args.config)

    # Resolve horizon: CLI flag wins over config file value.
    horizon: int
    if args.horizon is not None:
        horizon = args.horizon
    else:
        # Try common config nesting patterns.
        horizon = (
            base_cfg.get("horizon")
            or (base_cfg.get("dataset") or {}).get("horizon")
            or 6
        )
    horizon = int(horizon)

    # Resolve study name.
    if args.study_name:
        study_name = args.study_name
    else:
        model_name = base_cfg.get("model_name", "proposed")
        study_name = f"tune_{model_name}_h{horizon}"

    # Load search space.
    search_space = _load_search_space(args.search_space)

    logging.getLogger(__name__).info(
        "Starting tuning | study='%s' n_trials=%d min_trials=%d horizon=%d seed=%d",
        study_name,
        args.n_trials,
        args.min_trials,
        horizon,
        args.seed,
    )

    # Import here so the CLI module can be imported without all heavy deps.
    from experiments.tuner import tune, TuningStallError  # noqa: PLC0415

    try:
        record = tune(
            study_name=study_name,
            search_space=search_space,
            base_cfg=base_cfg,
            n_trials=args.n_trials,
            min_trials=args.min_trials,
            horizon=horizon,
            seed=args.seed,
            records_dir=args.records_dir,
            storage=args.storage,
        )
    except TuningStallError as exc:
        print(f"\n[tune_cli] STALL: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"\n[tune_cli] ARGUMENT ERROR: {exc}", file=sys.stderr)
        return 2

    # Print summary to stdout.
    print("\n=== Tuning complete ===")
    print(f"  Study name      : {record.study_name}")
    print(f"  Best trial #    : {record.best_trial_number}")
    print(f"  Best val MAE    : {record.best_val_mae_kw:.4f} kW")
    print(f"  Trials total    : {record.n_trials_total}")
    print(f"  Trials succeeded: {record.n_trials_successful}")
    print(f"  Trials failed   : {record.n_trials_failed}")
    print(f"  Completed at    : {record.completed_at}")
    print(f"  Record saved to : outputs/runs/tuning_{record.study_name}.json")
    print("\nBest hyperparameters:")
    for k, v in record.best_params.items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
