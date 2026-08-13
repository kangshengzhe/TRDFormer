"""
experiments/matrix.py

Experiment matrix definition and GPU budget tracking for the wind-power
forecasting paper workflow.

Materialises per-run merged YAML configs for every (model, horizon, seed)
cell in the Cartesian product, tracks cumulative GPU hours against the
weekly budget, and defers runs that would exceed the budget.

Priority-based shedding order when over budget:
    1. SHAP run
    2. Optuna tuning runs
    3. Optional visualisations
    4. Reduce seeds from 5 down to min_seeds (default 3)

Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 14.7, 14.8
"""

from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants (Requirements 12.1, 6.4)
# ---------------------------------------------------------------------------

HORIZONS: list[int] = [1, 6, 12, 24]
SEEDS: list[int] = [42, 43, 44, 45, 46]  # >= 5 by Req 6.4

PROPOSED: list[str] = ["proposed"]

BASELINES: list[str] = [
    # ── 原始 8 个 ────────────────────────────────────────────────────────
    "lstm",
    "transformer",
    "informer",
    "fedformer",
    "dlinear",
    "patchtst",
    "itransformer",
    "timesnet",
    # ── 扩展 3 个（更强的对比基线）────────────────────────────────────────
    "autoformer",                # 序列分解 + 自相关，与 VMD 分解形成对比
    "nonstationary_transformer", # 专为非平稳序列设计，与风电非平稳特性直接相关
    "timexer",                   # 外生变量感知 Transformer，与创新点 B/C 直接竞争
]

ABLATIONS: list[str] = [
    "ablation:itrans_off",
    "ablation:lstm_off",
    "ablation:fusion_concat",
    "ablation:fusion_sum",
    "ablation:fusion_cross_attention",  # fixed CrossAttention (former default)
    "ablation:head_linear",
    "ablation:head_mlp",
    "ablation:vmd_off",
    "ablation:outlier_off",
]

# Cardinality reference (informational, auto-derived from the lists above):
#   Total GPU runs = (len(PROPOSED) + len(BASELINES) + len(ABLATIONS))
#                   × len(HORIZONS) × len(SEEDS)

# ---------------------------------------------------------------------------
# Priority-shedding categories (Requirement 12.5, 12.6)
# ---------------------------------------------------------------------------

# Items that can be shed before cutting seeds, in priority order (shed first → last)
SHEDDABLE_OPTIONAL: list[str] = ["shap", "optuna", "visualization_extra"]

# Mandatory run groups that must always complete if budget allows
MANDATORY_GROUPS: list[str] = [
    "proposed",
    "baselines",
    "ablations",
    "significance_tests",
    "statistical_summary",
    "visualization_core",
]

# ---------------------------------------------------------------------------
# Helper: deep-merge two nested dicts (right wins on scalar conflicts)
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *base* deep-merged with *override*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ---------------------------------------------------------------------------
# Helper: resolve ablation-prefixed model names to YAML config paths
# ---------------------------------------------------------------------------


def _ablation_name_to_yaml_key(model_name: str) -> str:
    """Strip the 'ablation:' prefix to get the YAML filename stem."""
    if model_name.startswith("ablation:"):
        return model_name[len("ablation:"):]
    return model_name


# ---------------------------------------------------------------------------
# ExperimentMatrix
# ---------------------------------------------------------------------------


class ExperimentMatrix:
    """Materialise per-run YAML configs and manage GPU budget.

    Parameters
    ----------
    base_config_path:
        Path to the base dataset YAML (e.g. ``configs/dataset/sdwpf_turb1.yaml``).
    runtime_config_path:
        Path to the runtime YAML (e.g. ``configs/runtime/kaggle_gpu.yaml``).  The
        ``runtime.gpu_time_budget_hours`` key is read from here (default 30).
    out_dir:
        Root output directory for run configs, budget file, and deferred list
        (default ``outputs/runs``).
    model_configs_dir:
        Directory containing ``proposed.yaml`` and a ``baselines/`` sub-directory
        (default ``configs/model``).
    ablation_configs_dir:
        Directory containing per-ablation YAML files (default ``configs/ablation``).
    min_seeds:
        Minimum number of seeds to retain when shedding under budget pressure
        (default 3).  Must be >= 1.
    """

    def __init__(
        self,
        base_config_path: str,
        runtime_config_path: str,
        out_dir: str = "outputs/runs",
        model_configs_dir: str = "configs/model",
        ablation_configs_dir: str = "configs/ablation",
        min_seeds: int = 3,
    ) -> None:
        self.base_config_path = Path(base_config_path)
        self.runtime_config_path = Path(runtime_config_path)
        self.out_dir = Path(out_dir)
        self.model_configs_dir = Path(model_configs_dir)
        self.ablation_configs_dir = Path(ablation_configs_dir)
        self.min_seeds = max(1, min_seeds)

        # Load the two base YAML blocks
        self._base_cfg: dict = self._load_yaml(self.base_config_path)
        self._runtime_cfg: dict = self._load_yaml(self.runtime_config_path)

        # GPU budget (hours); falls back to 30 if key is absent
        runtime_block = self._runtime_cfg.get("runtime", {})
        self._budget_hours: float = float(
            runtime_block.get("gpu_time_budget_hours", 30.0)
        )

        # Paths for output artifacts
        self._configs_dir = self.out_dir / "configs"
        self._budget_file = self.out_dir / "budget.json"
        self._deferred_file = self.out_dir / "deferred.jsonl"

        # Ensure directories exist
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def materialize(
        self,
        *,
        models: list[str] | None = None,
        horizons: list[int] | None = None,
        seeds: list[int] | None = None,
        active_seeds: list[int] | None = None,
    ) -> list[dict]:
        """Generate merged YAML for each (model, horizon, seed) cell.

        The method respects priority-based shedding: if *active_seeds* is
        provided it overrides the default ``SEEDS`` list (used when the budget
        tracker has already decided to reduce seeds).

        Parameters
        ----------
        models:
            Override the default combined model list.  Defaults to
            ``PROPOSED + BASELINES + ABLATIONS``.
        horizons:
            Override the default ``HORIZONS``.
        seeds:
            Override the default ``SEEDS``.
        active_seeds:
            Subset of seeds to actually materialise (for budget shedding).
            Defaults to *seeds*.

        Returns
        -------
        list[dict]
            Each element is the merged config dict for one run.  Side effect:
            saves ``outputs/runs/configs/{run_id}.yaml`` for each config.
        """
        models = models if models is not None else (PROPOSED + BASELINES + ABLATIONS)
        horizons = horizons if horizons is not None else HORIZONS
        seeds = seeds if seeds is not None else SEEDS
        active_seeds = active_seeds if active_seeds is not None else seeds

        run_configs: list[dict] = []

        for model_name in models:
            model_cfg = self._load_model_config(model_name)

            for horizon in horizons:
                for seed in active_seeds:
                    run_id = self._make_run_id(model_name, horizon, seed)

                    # Merge: base_dataset ← runtime ← model_specific
                    merged = _deep_merge(self._base_cfg, self._runtime_cfg)
                    merged = _deep_merge(merged, model_cfg)

                    # Inject run-level overrides
                    merged.setdefault("dataset", {})["horizon"] = horizon
                    merged["run_id"] = run_id
                    merged["model_name"] = model_name
                    merged["seed"] = seed

                    # Persist the merged YAML
                    config_path = self._configs_dir / f"{run_id}.yaml"
                    self._save_yaml(merged, config_path)
                    logger.debug("Materialised run config: %s", config_path)

                    run_configs.append(merged)

        logger.info(
            "Materialised %d run configs to %s", len(run_configs), self._configs_dir
        )
        return run_configs

    def materialize_with_shedding(
        self,
        *,
        estimated_hours_per_run: float = 0.25,
        shed_shap: bool = True,
        shed_optuna: bool = True,
        shed_extra_viz: bool = True,
    ) -> dict[str, Any]:
        """Materialise configs applying priority-based shedding if needed.

        Evaluates available budget and progressively sheds optional items
        (SHAP → Optuna → extra visualisations → reduce seeds) until the
        estimated GPU hours fit within the remaining budget.

        Parameters
        ----------
        estimated_hours_per_run:
            Wall-clock GPU hours assumed per training run (used for estimation).
        shed_shap:
            Allow shedding SHAP as first priority.
        shed_optuna:
            Allow shedding Optuna as second priority.
        shed_extra_viz:
            Allow shedding optional visualisations as third priority.

        Returns
        -------
        dict with keys:
            ``run_configs``  – list of materialised run config dicts
            ``active_seeds`` – seeds actually used
            ``shed_items``   – list of item names that were shed
            ``warnings``     – list of human-readable warning strings
        """
        used_hours = self._read_used_hours()
        remaining = self._budget_hours - used_hours

        all_models = PROPOSED + BASELINES + ABLATIONS
        active_seeds = list(SEEDS)
        shed_items: list[str] = []
        warnings: list[str] = []

        # Estimate total GPU runs (before optional items)
        n_runs = len(all_models) * len(HORIZONS) * len(active_seeds)
        estimated_total = n_runs * estimated_hours_per_run

        # --- Shedding pass 1: SHAP (Req 12.5 priority 1) ---
        if estimated_total > remaining and shed_shap:
            shed_items.append("shap")
            msg = (
                f"Budget pressure: SHAP run deferred "
                f"(estimated_total={estimated_total:.1f}h > remaining={remaining:.1f}h)"
            )
            warnings.append(msg)
            logger.warning(msg)
            self._record_deferred_optional("shap", "budget_pressure", reason=msg)

        # --- Shedding pass 2: Optuna (Req 12.5 priority 2) ---
        if estimated_total > remaining and shed_optuna:
            shed_items.append("optuna")
            msg = (
                f"Budget pressure: Optuna tuning deferred "
                f"(estimated_total={estimated_total:.1f}h > remaining={remaining:.1f}h)"
            )
            warnings.append(msg)
            logger.warning(msg)
            self._record_deferred_optional("optuna", "budget_pressure", reason=msg)

        # --- Shedding pass 3: optional charts (Req 12.5 priority 3) ---
        if estimated_total > remaining and shed_extra_viz:
            shed_items.append("visualization_extra")
            msg = (
                f"Budget pressure: optional visualisations deferred "
                f"(estimated_total={estimated_total:.1f}h > remaining={remaining:.1f}h)"
            )
            warnings.append(msg)
            logger.warning(msg)
            self._record_deferred_optional(
                "visualization_extra", "budget_pressure", reason=msg
            )

        # --- Shedding pass 4: reduce seeds (Req 12.6) ---
        while estimated_total > remaining and len(active_seeds) > self.min_seeds:
            dropped_seed = active_seeds.pop()
            n_runs = len(all_models) * len(HORIZONS) * len(active_seeds)
            estimated_total = n_runs * estimated_hours_per_run
            msg = (
                f"Budget pressure: seed {dropped_seed} shed; "
                f"active_seeds now {active_seeds} "
                f"(estimated_total={estimated_total:.1f}h, remaining={remaining:.1f}h)"
            )
            warnings.append(msg)
            logger.warning(msg)
            shed_items.append(f"seed:{dropped_seed}")

        if estimated_total > remaining:
            msg = (
                f"WARNING: estimated GPU hours ({estimated_total:.1f}h) still exceed "
                f"remaining budget ({remaining:.1f}h) after all shedding. "
                f"Proceeding with {len(active_seeds)} seeds; "
                f"excess runs will be checked individually via check_budget()."
            )
            warnings.append(msg)
            logger.warning(msg)

        run_configs = self.materialize(active_seeds=active_seeds)

        return {
            "run_configs": run_configs,
            "active_seeds": active_seeds,
            "shed_items": shed_items,
            "warnings": warnings,
        }

    def check_budget(self, estimated_hours: float) -> bool:
        """Return True if a run can proceed within the GPU budget.

        Parameters
        ----------
        estimated_hours:
            Expected GPU wall-clock hours for the candidate run.

        Returns
        -------
        bool
            ``True`` if ``gpu_used_hours + estimated_hours ≤ gpu_time_budget_hours``.
        """
        used = self._read_used_hours()
        can_proceed = (used + estimated_hours) <= self._budget_hours
        if not can_proceed:
            logger.info(
                "Budget check failed: used=%.2f + estimated=%.2f > budget=%.2f",
                used,
                estimated_hours,
                self._budget_hours,
            )
        return can_proceed

    def record_run_time(self, run_id: str, wall_clock_hours: float) -> None:
        """Update budget.json after a run completes.

        Increments the cumulative ``gpu_used_hours`` counter and appends
        a per-run entry to the ``run_times`` list inside ``budget.json``.

        Parameters
        ----------
        run_id:
            Unique run identifier (as in the run config YAML).
        wall_clock_hours:
            Actual GPU wall-clock time consumed by the run.
        """
        budget = self._read_budget()

        budget["gpu_used_hours"] = budget.get("gpu_used_hours", 0.0) + wall_clock_hours
        budget.setdefault("run_times", []).append(
            {
                "run_id": run_id,
                "wall_clock_hours": wall_clock_hours,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        self._write_budget(budget)
        logger.debug(
            "Recorded run %s: %.4f hours (cumulative: %.4f / %.4f)",
            run_id,
            wall_clock_hours,
            budget["gpu_used_hours"],
            self._budget_hours,
        )

    def defer_run(self, run_config: dict, reason: str) -> None:
        """Append *run_config* to ``deferred.jsonl`` with a reason string.

        Deferred runs can be resumed in a later week using the persisted
        configuration record (Requirement 14.8).

        Parameters
        ----------
        run_config:
            The full merged run config dict (as produced by ``materialize``).
        reason:
            Human-readable string describing why the run was deferred.
        """
        record = {
            "run_id": run_config.get("run_id", "<unknown>"),
            "model_name": run_config.get("model_name"),
            "horizon": run_config.get("dataset", {}).get("horizon"),
            "seed": run_config.get("seed"),
            "reason": reason,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
            "config": run_config,
        }
        with open(self._deferred_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Deferred run %s (reason: %s)", record["run_id"], reason
        )

    def get_budget_status(self) -> dict[str, Any]:
        """Return a summary of the current GPU budget status.

        Returns
        -------
        dict with keys:
            ``gpu_time_budget_hours``  – configured maximum
            ``gpu_used_hours``         – cumulative hours consumed
            ``gpu_remaining_hours``    – budget - used
            ``run_count``              – number of runs recorded
        """
        budget = self._read_budget()
        used = budget.get("gpu_used_hours", 0.0)
        return {
            "gpu_time_budget_hours": self._budget_hours,
            "gpu_used_hours": used,
            "gpu_remaining_hours": max(0.0, self._budget_hours - used),
            "run_count": len(budget.get("run_times", [])),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_run_id(self, model_name: str, horizon: int, seed: int) -> str:
        """Build a deterministic, human-readable run identifier."""
        # Sanitise model name: replace ':' with '_' for filesystem safety
        safe_model = model_name.replace(":", "_")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
        return f"{safe_model}_h{horizon}_seed{seed}_{ts}"

    def _load_model_config(self, model_name: str) -> dict:
        """Load the per-model YAML (proposed, baseline, or ablation)."""
        if model_name == "proposed":
            path = self.model_configs_dir / "proposed.yaml"
        elif model_name.startswith("ablation:"):
            key = _ablation_name_to_yaml_key(model_name)
            path = self.ablation_configs_dir / f"{key}.yaml"
        else:
            # Baseline
            path = self.model_configs_dir / "baselines" / f"{model_name}.yaml"

        if not path.exists():
            logger.warning(
                "Model config not found for '%s' at %s; using empty dict.",
                model_name,
                path,
            )
            return {}

        cfg = self._load_yaml(path)
        # Tag the model name inside the config for downstream consumers
        cfg.setdefault("model", {})["name"] = model_name
        return cfg

    def _load_yaml(self, path: Path) -> dict:
        """Load and parse a YAML file, returning an empty dict on error."""
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            logger.warning("YAML file not found: %s", path)
            return {}
        except yaml.YAMLError as exc:
            logger.error("Failed to parse YAML %s: %s", path, exc)
            return {}

    def _save_yaml(self, data: dict, path: Path) -> None:
        """Serialise *data* to YAML at *path*, creating parent dirs as needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False, sort_keys=False)

    def _read_budget(self) -> dict:
        """Read the current budget.json, returning defaults if absent."""
        if self._budget_file.exists():
            try:
                with open(self._budget_file, encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read budget file %s: %s", self._budget_file, exc)
        return {
            "gpu_used_hours": 0.0,
            "gpu_time_budget_hours": self._budget_hours,
            "run_times": [],
        }

    def _write_budget(self, budget: dict) -> None:
        """Atomically write the budget dict to budget.json."""
        tmp_path = self._budget_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(budget, fh, indent=2, ensure_ascii=False)
        # Atomic rename (works on same filesystem)
        tmp_path.replace(self._budget_file)

    def _read_used_hours(self) -> float:
        """Return the current cumulative GPU hours from budget.json."""
        return float(self._read_budget().get("gpu_used_hours", 0.0))

    def _record_deferred_optional(
        self, item_name: str, reason_code: str, *, reason: str
    ) -> None:
        """Append a deferred optional-item entry to deferred.jsonl."""
        record = {
            "item_name": item_name,
            "reason_code": reason_code,
            "reason": reason,
            "deferred_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._deferred_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Module-level helper (used by the Kaggle notebook Cell 2)
# ---------------------------------------------------------------------------


def load_run_config(config_path: str) -> dict:
    """Load a merged YAML run config from disk.

    Parameters
    ----------
    config_path:
        Path to the ``.yaml`` file produced by ``ExperimentMatrix.materialize``.

    Returns
    -------
    dict
        The merged run configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the file cannot be parsed as valid YAML.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Run config not found: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse run config {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}, got {type(data).__name__}"
        )
    return data
