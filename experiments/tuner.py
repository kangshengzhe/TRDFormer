"""Optuna-based hyperparameter tuning module for the wind-power forecasting workflow.

This module exposes the ``tune`` function, which runs an Optuna study that
minimises validation MAE (on the denormalized kW scale) for the Proposed_Model.
The test partition is **never** accessed during tuning.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import optuna
from optuna.exceptions import TrialPruned

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class BestTrialRecord:
    """Summary of the best trial found in an Optuna study.

    Attributes:
        study_name:          Name of the Optuna study.
        best_trial_number:   The Optuna trial number of the best trial.
        best_val_mae_kw:     Best validation MAE, expressed in kW (denormalized).
        best_params:         Hyperparameter dict for the best trial.
        search_space:        The full search-space definition passed to :func:`tune`.
        n_trials_total:      Total number of trials configured (including failed ones).
        n_trials_successful: Number of trials that finished without error.
        n_trials_failed:     Number of trials that failed (OOM, non-finite loss, etc.).
        completed_at:        ISO 8601 timestamp when the record was persisted.
    """

    study_name: str
    best_trial_number: int
    best_val_mae_kw: float
    best_params: dict
    search_space: dict
    n_trials_total: int
    n_trials_successful: int
    n_trials_failed: int
    completed_at: str

    def to_dict(self) -> dict:
        """Return a plain dict suitable for JSON serialization."""
        return asdict(self)


class TuningStallError(Exception):
    """Raised when the study stalls: >50 % trials fail AND successful < min_trials."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_params(trial: optuna.Trial, search_space: dict) -> dict:
    """Draw hyper-parameter values from *search_space* for *trial*.

    The search-space is a dict mapping param_name → a descriptor dict::

        {
            'type':    'float' | 'int' | 'categorical',
            'low':     <numeric>,          # for float/int
            'high':    <numeric>,          # for float/int
            'log':     <bool>,             # for float/int; default False
            'choices': [...],              # for categorical
        }

    Args:
        trial:        The current Optuna trial.
        search_space: Mapping of parameter names to their search descriptors.

    Returns:
        Dict mapping each parameter name to its sampled value.
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"Unknown search-space type '{ptype}' for parameter '{name}'. "
                "Expected 'float', 'int', or 'categorical'."
            )
    return params


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Return a new dict that is *base* with *overrides* applied (shallow-merge
    at the top level; nested dicts are merged one level deep).
    """
    merged = copy.deepcopy(base)
    for key, val in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def _apply_params_to_cfg(base_cfg: dict, params: dict) -> dict:
    """Overlay sampled *params* onto *base_cfg*.

    Params that match a key inside ``base_cfg['model']`` sub-dict are placed
    there; params that match top-level training keys (``learning_rate``,
    ``batch_size``, …) land in ``base_cfg['train']``; anything else goes at
    the top level.

    This keeps the config structure valid for the runner.
    """
    cfg = copy.deepcopy(base_cfg)

    model_keys = set((cfg.get("model") or {}).keys())
    train_keys = set((cfg.get("train") or {}).keys())

    for k, v in params.items():
        if k in model_keys:
            cfg.setdefault("model", {})[k] = v
        elif k in train_keys:
            cfg.setdefault("train", {})[k] = v
        else:
            cfg[k] = v

    return cfg


def _cfg_to_dict(base_cfg) -> dict:
    """Normalize *base_cfg* to a plain dict, whether it's a RunConfig or dict."""
    # If it's already a dict, just return a deep copy.
    if isinstance(base_cfg, dict):
        return copy.deepcopy(base_cfg)

    # If it's a RunConfig dataclass, convert via dataclasses.asdict.
    try:
        from dataclasses import asdict as _asdict, fields as _fields
        _fields(base_cfg)  # will raise TypeError if not a dataclass
        return _asdict(base_cfg)
    except TypeError:
        pass

    # Fallback: try __dict__.
    return copy.deepcopy(vars(base_cfg))


def _run_trial_cfg(cfg: dict, horizon: int, seed: int) -> float:
    """Execute one trial and return the **denormalized validation MAE** in kW.

    Constructs a ``RunConfig`` from the merged config dict and calls
    ``experiments.runner.run``.  The runner always trains on train+valid
    partitions; the test partition is not evaluated here.

    Returns:
        Validation MAE in kW (float, finite, > 0).

    Raises:
        TrialPruned: If the trial should be skipped (OOM, non-finite loss, etc.).
        RuntimeError: For unexpected errors not related to resource limits.
    """
    try:
        from experiments.runner import run, RunConfig  # type: ignore

        # Build RunConfig from the merged dict.
        # RunConfig is a plain dataclass; construct keyword-by-keyword.
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["seed"] = seed
        cfg_copy["horizon"] = horizon

        run_cfg = RunConfig(
            run_id=cfg_copy.get("run_id", f"tune_trial_s{seed}_h{horizon}"),
            model_name=cfg_copy.get("model_name", "proposed"),
            seed=cfg_copy["seed"],
            lookback=cfg_copy.get("lookback", 144),
            horizon=cfg_copy["horizon"],
            train=cfg_copy.get("train", {}),
            model=cfg_copy.get("model", {}),
            ablation=cfg_copy.get("ablation", {}),
            runtime=cfg_copy.get("runtime", {}),
            dataset=cfg_copy.get("dataset", {}),
        )

        record = run(run_cfg)

        val_mae = record.val_metrics.get("mae")
        if val_mae is None or not math.isfinite(val_mae):
            raise TrialPruned("Non-finite validation MAE returned by runner.")
        return float(val_mae)

    except TrialPruned:
        raise
    except MemoryError as exc:
        logger.warning("Trial failed with OOM: %s", exc)
        raise TrialPruned(f"OOM: {exc}") from exc
    except _torch_oom_exception() as exc:  # type: ignore[misc]
        logger.warning("Trial failed with CUDA OOM: %s", exc)
        raise TrialPruned(f"CUDA OOM: {exc}") from exc
    except Exception as exc:
        msg = str(exc)
        if "out of memory" in msg.lower() or "oom" in msg.lower():
            logger.warning("Trial OOM: %s", exc)
            raise TrialPruned(f"OOM: {exc}") from exc
        raise


def _torch_oom_exception():
    """Return the RuntimeError subclass used by PyTorch for OOM errors.

    We import lazily so the module can be imported even without torch.
    """
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.OutOfMemoryError
    except (ImportError, AttributeError):
        # Older PyTorch or no CUDA: fall back to a class that is never raised.
        return type("_NeverRaised", (BaseException,), {})


def _persist_best_trial_record(record: BestTrialRecord, records_dir: str) -> str:
    """Write *record* to ``tuning_{study_name}.json`` and return the file path.

    Overwrites any existing file for the same study so the records file always
    reflects the latest (best) result.

    Args:
        record:      The BestTrialRecord to persist.
        records_dir: Directory in which to save the JSON file.

    Returns:
        Absolute path to the written JSON file.
    """
    path = Path(records_dir)
    path.mkdir(parents=True, exist_ok=True)
    filename = f"tuning_{record.study_name}.json"
    filepath = path / filename
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("Best trial record saved to %s", filepath)
    return str(filepath)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune(
    study_name: str,
    search_space: dict,
    base_cfg: Union[dict, Any],
    n_trials: int = 100,
    min_trials: int = 20,
    horizon: int = 6,
    seed: int = 42,
    records_dir: str = "outputs/runs",
    storage: str | None = None,
) -> BestTrialRecord:
    """Run an Optuna study to minimise validation MAE for the Proposed_Model.

    The study explores the hyper-parameters defined in *search_space*, applies
    them on top of *base_cfg*, then calls the experiment runner for each trial.
    **Only the validation partition is used; the test partition is never
    accessed.**

    Search-space specification
    --------------------------
    *search_space* maps ``param_name`` → a descriptor dict::

        {
            'type':    'float' | 'int' | 'categorical',
            'low':     <numeric>,          # required for float / int
            'high':    <numeric>,          # required for float / int
            'log':     <bool>,             # optional; default False
            'choices': [<value>, ...],     # required for categorical
        }

    Failure handling
    ----------------
    Trials that fail with OOM or non-finite loss are logged, pruned, and
    counted.  If **more than 50 % of total trials fail** *and* the number of
    successfully completed trials is **less than** ``min_trials``, the study is
    aborted with :class:`TuningStallError`.

    Args:
        study_name:   Unique name for the Optuna study.
        search_space: Hyper-parameter search space (see above).
        base_cfg:     Baseline run configuration (RunConfig dataclass or plain
                      dict) to overlay with sampled params.  The structure must
                      match the layout expected by ``experiments.runner.RunConfig``.
        n_trials:     Total number of Optuna trials to run.  Must be >=
                      ``min_trials``.  Defaults to 100.
        min_trials:   Minimum number of *successful* trials required before the
                      study is considered valid.  Defaults to 20.
        horizon:      Forecast horizon (in steps) used for each trial.
                      Defaults to 6.
        seed:         Random seed for the TPE sampler.  Defaults to 42.
        records_dir:  Directory to persist the :class:`BestTrialRecord` JSON.
                      Defaults to ``'outputs/runs'``.
        storage:      Optional Optuna storage URL (e.g., ``sqlite:///db.sqlite3``).
                      Defaults to ``None`` (in-memory).

    Returns:
        A :class:`BestTrialRecord` containing the best configuration and
        validation MAE.

    Raises:
        TuningStallError: If >50 % of trials fail and < ``min_trials`` succeed.
        ValueError:       If ``n_trials < min_trials``.
    """
    if n_trials < min_trials:
        raise ValueError(
            f"n_trials ({n_trials}) must be >= min_trials ({min_trials})."
        )

    # Normalize base_cfg to a plain dict so we can safely copy/mutate it.
    base_cfg_dict = _cfg_to_dict(base_cfg)

    n_failed = 0
    n_completed = 0

    # Suppress Optuna's own logging so our logger controls verbosity.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=(storage is not None),
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal n_failed, n_completed

        # --- sample hyper-parameters ---
        params = _sample_params(trial, search_space)
        cfg = _apply_params_to_cfg(base_cfg_dict, params)

        logger.info(
            "Trial %d | params: %s",
            trial.number,
            json.dumps(params, ensure_ascii=False),
        )

        # --- run trial ---
        try:
            val_mae = _run_trial_cfg(cfg, horizon=horizon, seed=seed + trial.number)
        except TrialPruned as exc:
            n_failed += 1
            logger.warning(
                "Trial %d PRUNED — reason: %s  (failed=%d, completed=%d)",
                trial.number,
                exc,
                n_failed,
                n_completed,
            )
            # Stall check: performed every time a trial fails.
            _check_stall(
                n_trials=n_trials,
                n_failed=n_failed,
                n_completed=n_completed,
                min_trials=min_trials,
                study_name=study_name,
            )
            raise  # propagate so Optuna marks this trial as PRUNED

        # Non-finite guard (extra safety net beyond the runner's own check).
        if not math.isfinite(val_mae):
            n_failed += 1
            reason = f"Non-finite val_mae={val_mae}"
            logger.warning(
                "Trial %d PRUNED — reason: %s  (failed=%d, completed=%d)",
                trial.number,
                reason,
                n_failed,
                n_completed,
            )
            _check_stall(
                n_trials=n_trials,
                n_failed=n_failed,
                n_completed=n_completed,
                min_trials=min_trials,
                study_name=study_name,
            )
            raise TrialPruned(reason)

        n_completed += 1
        logger.info(
            "Trial %d COMPLETED — val_mae=%.4f kW  (failed=%d, completed=%d)",
            trial.number,
            val_mae,
            n_failed,
            n_completed,
        )
        return val_mae

    # Run the study; stall checks are embedded inside `objective`.
    study.optimize(
        objective,
        n_trials=n_trials,
        catch=(Exception,),
        callbacks=[],
    )

    # Final stall check after all trials finish.
    _check_stall(
        n_trials=n_trials,
        n_failed=n_failed,
        n_completed=n_completed,
        min_trials=min_trials,
        study_name=study_name,
    )

    # If every trial was pruned, best_trial raises ValueError; surface clearly.
    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise TuningStallError(
            f"Study '{study_name}' has no completed trials; cannot determine best config."
        ) from exc

    completed_at = datetime.now(timezone.utc).isoformat()

    record = BestTrialRecord(
        study_name=study_name,
        best_trial_number=best_trial.number,
        best_val_mae_kw=float(best_trial.value),
        best_params=dict(best_trial.params),
        search_space=search_space,
        n_trials_total=n_trials,
        n_trials_successful=n_completed,
        n_trials_failed=n_failed,
        completed_at=completed_at,
    )

    _persist_best_trial_record(record, records_dir=records_dir)

    logger.info(
        "Tuning complete | study='%s' best_val_mae_kw=%.4f kW "
        "trial=%d completed=%d failed=%d",
        study_name,
        record.best_val_mae_kw,
        record.best_trial_number,
        n_completed,
        n_failed,
    )
    return record


# ---------------------------------------------------------------------------
# Internal stall check helper
# ---------------------------------------------------------------------------

def _check_stall(
    *,
    n_trials: int,
    n_failed: int,
    n_completed: int,
    min_trials: int,
    study_name: str,
) -> None:
    """Raise :class:`TuningStallError` when the stall condition is met.

    Condition: *more than 50 % of configured trials have failed* AND the
    number of successfully completed trials is *less than* ``min_trials``.

    Args:
        n_trials:    Total configured trial count.
        n_failed:    Trials that failed so far (pruned / OOM / non-finite).
        n_completed: Trials that completed successfully so far.
        min_trials:  Minimum required successful trials.
        study_name:  Study name (for the error message).

    Raises:
        TuningStallError: When the stall condition is met.
    """
    if n_failed > n_trials * 0.5 and n_completed < min_trials:
        raise TuningStallError(
            f"Study '{study_name}' stalled: {n_failed}/{n_trials} trials failed "
            f"(>{50:.0f}%) and only {n_completed} succeeded "
            f"(required: {min_trials})."
        )
