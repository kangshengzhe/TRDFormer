"""
scripts/train/train_cli.py
==========================
Command-line entry point for a single experiment run.

Usage
-----
    python -m scripts.train.train_cli --config path/to/run_config.yaml

The script:
  1. Resolves the project root and inserts it into sys.path so that all
     package imports work regardless of the working directory.
  2. Loads the YAML config file and constructs a RunConfig.
  3. Calls experiments.runner.run(cfg) to execute the full train/eval loop.
  4. Prints a human-readable metrics summary to stdout.
  5. Exits with code 0 on success or code 1 on any failure.

Environment portability
-----------------------
The same script works unmodified on both:
  - Local CPU  : ``runtime.device: auto`` → selects cpu
  - Kaggle GPU : ``runtime.device: auto`` → selects cuda

The only change required between environments is the YAML config file
(or the runtime/ sub-config it references).  In particular, Kaggle runs
must set ``runtime.tsl_root`` and ``runtime.out_dir`` to the Kaggle paths
(e.g. /kaggle/input/... and /kaggle/working), which is done via
``configs/runtime/kaggle_gpu.yaml``.

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
# Layout: <project_root>/scripts/train/train_cli.py
_SCRIPT_DIR = Path(__file__).resolve().parent   # scripts/train/
_SCRIPTS_DIR = _SCRIPT_DIR.parent               # scripts/
_PROJECT_ROOT = _SCRIPTS_DIR.parent             # <project_root>/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Set working directory to project root so that relative paths in YAML configs
# resolve correctly (e.g. "outputs/runs/", "data/wind/...").
os.chdir(_PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("train_cli")


# ---------------------------------------------------------------------------
# YAML config → RunConfig
# ---------------------------------------------------------------------------

def _load_run_config(config_path: str):
    """Load a YAML file and return a populated RunConfig.

    The YAML file is expected to contain top-level keys that directly
    correspond to RunConfig fields:
        run_id, model_name, seed, lookback, horizon,
        train, model, ablation, runtime, dataset

    Missing optional keys are filled with empty dicts so that downstream
    code can safely call .get() on them.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to the YAML config file.

    Returns
    -------
    RunConfig

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required top-level field is absent.
    """
    import yaml
    from experiments.runner import RunConfig

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")

    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError(f"Config file is empty or not valid YAML: {p}")

    # Required top-level fields
    required = ("run_id", "model_name", "seed", "horizon")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(
            f"Config file {p} is missing required keys: {missing}"
        )

    cfg = RunConfig(
        run_id=str(raw["run_id"]),
        model_name=str(raw["model_name"]),
        seed=int(raw["seed"]),
        lookback=int(raw.get("lookback", 144)),
        horizon=int(raw["horizon"]),
        train=dict(raw.get("train") or {}),
        model=dict(raw.get("model") or {}),
        ablation=dict(raw.get("ablation") or {}),
        runtime=dict(raw.get("runtime") or {}),
        dataset=dict(raw.get("dataset") or {}),
    )
    return cfg


# ---------------------------------------------------------------------------
# Metrics summary printer
# ---------------------------------------------------------------------------

def _print_metrics_summary(record) -> None:
    """Print a formatted metrics summary to stdout.

    Parameters
    ----------
    record : RunRecord
        The RunRecord returned by experiments.runner.run().
    """
    sep = "=" * 60
    print(sep)
    print(f"  Run complete: {record.run_id}")
    print(sep)
    print(f"  Model       : {record.model_name}")
    print(f"  Horizon     : {record.horizon} steps")
    print(f"  Lookback    : {record.lookback} steps")
    print(f"  Seed        : {record.seed}")
    print(f"  Device      : {record.env.get('device', 'unknown') if isinstance(record.env, dict) else getattr(record.env, 'device', 'unknown')}")
    print(f"  Wall clock  : {record.wall_clock_seconds:.1f}s")
    print()
    print("  TEST METRICS (de-normalised kW)")
    print("  " + "-" * 40)
    metrics = record.metrics if isinstance(record.metrics, dict) else {}
    for key in ("mae", "rmse", "r2", "mbe", "smape"):
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            print(f"    {key.upper():<8}: {val:.4f}")
        else:
            print(f"    {key.upper():<8}: {val}")
    print()
    print("  VAL METRICS (de-normalised kW)")
    print("  " + "-" * 40)
    val_metrics = record.val_metrics if isinstance(record.val_metrics, dict) else {}
    for key in ("mae", "rmse", "r2", "mbe", "smape"):
        val = val_metrics.get(key, "N/A")
        if isinstance(val, float):
            print(f"    {key.upper():<8}: {val:.4f}")
        else:
            print(f"    {key.upper():<8}: {val}")
    print()
    print(f"  Checkpoint  : {record.checkpoint}")
    print(f"  Record file : outputs/runs/run_records.jsonl")
    print(f"  Status      : {record.status}")
    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run experiment, return exit code.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="python -m scripts.train.train_cli",
        description=(
            "Run a single wind-power forecasting experiment from a YAML config.\n"
            "\n"
            "Works identically on local CPU and Kaggle GPU — device selection\n"
            "is handled by the 'runtime.device' field in the config (default: auto)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to the run config YAML file.",
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

    # ── Load config ───────────────────────────────────────────────────────
    try:
        cfg = _load_run_config(args.config)
        logger.info("Loaded config: run_id=%s, model=%s, horizon=%d, seed=%d",
                    cfg.run_id, cfg.model_name, cfg.horizon, cfg.seed)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        return 1

    # ── Run experiment ────────────────────────────────────────────────────
    try:
        from experiments.runner import run
        record = run(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.error("Experiment failed: %s", exc)
        logger.debug(traceback.format_exc())
        return 1

    # ── Print summary ─────────────────────────────────────────────────────
    try:
        _print_metrics_summary(record)
    except Exception as exc:  # noqa: BLE001
        # Don't let a formatting error mask a successful run
        logger.warning("Could not print metrics summary: %s", exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
