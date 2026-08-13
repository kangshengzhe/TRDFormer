"""Experiment utilities for the wind-power forecasting workflow."""

# Aggregator is always importable (only needs numpy + pandas).
from experiments.aggregator import (
    aggregate_runs,
    AggregateTables,
)

# Significance testing is always importable (only needs scipy + pandas).
from experiments.significance import (
    paired_significance,
    is_significant,
    compute_significance_table,
)

# Tuner is always importable (only needs optuna, which is a listed dep).
from experiments.tuner import (
    BestTrialRecord,
    TuningStallError,
    tune,
)

# Runner has heavier deps (dill, torch, …).  Import lazily so that modules
# that only need significance/tuner don't fail on missing GPU deps.
try:
    from experiments.runner import (
        RunConfig,
        run,
        InvalidHorizonError,
        InsufficientWindowError,
        UnknownModelError,
    )
    _runner_available = True
except ImportError:
    _runner_available = False

__all__ = [
    # Aggregator
    "aggregate_runs",
    "AggregateTables",
    # Significance
    "paired_significance",
    "is_significant",
    "compute_significance_table",
    # Tuner
    "BestTrialRecord",
    "TuningStallError",
    "tune",
]

if _runner_available:
    __all__ += [
        "RunConfig",
        "run",
        "InvalidHorizonError",
        "InsufficientWindowError",
        "UnknownModelError",
    ]
