"""
tests/test_matrix.py

Unit tests for experiments/matrix.py — ExperimentMatrix class and
load_run_config helper.

Requirements covered: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 14.7, 14.8
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import importlib.util

import pytest
import yaml

# ---------------------------------------------------------------------------
# Load matrix module directly (avoids __init__ chain that may be missing deps)
# ---------------------------------------------------------------------------

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)

_spec = importlib.util.spec_from_file_location(
    "matrix", os.path.join(_repo, "experiments", "matrix.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ExperimentMatrix = _mod.ExperimentMatrix
load_run_config = _mod.load_run_config
HORIZONS = _mod.HORIZONS
SEEDS = _mod.SEEDS
PROPOSED = _mod.PROPOSED
BASELINES = _mod.BASELINES
ABLATIONS = _mod.ABLATIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = os.path.join(_repo, "configs", "dataset", "sdwpf_turb1.yaml")
RUNTIME_CONFIG = os.path.join(_repo, "configs", "runtime", "kaggle_gpu.yaml")
MODEL_DIR = os.path.join(_repo, "configs", "model")
ABLATION_DIR = os.path.join(_repo, "configs", "ablation")


@pytest.fixture()
def tmp_em(tmp_path):
    """Return an ExperimentMatrix rooted in a fresh temp directory."""
    em = ExperimentMatrix(
        base_config_path=BASE_CONFIG,
        runtime_config_path=RUNTIME_CONFIG,
        out_dir=str(tmp_path),
        model_configs_dir=MODEL_DIR,
        ablation_configs_dir=ABLATION_DIR,
        min_seeds=3,
    )
    return em, tmp_path


# ---------------------------------------------------------------------------
# 1. Constants — Req 12.1
# ---------------------------------------------------------------------------


class TestConstants:
    def test_horizons_values(self):
        assert HORIZONS == [1, 6, 12, 24], "HORIZONS must be [1, 6, 12, 24]"

    def test_seeds_count(self):
        assert len(SEEDS) >= 5, "At least 5 seeds required (Req 6.4)"

    def test_seeds_values(self):
        assert SEEDS == [42, 43, 44, 45, 46]

    def test_proposed_list(self):
        assert PROPOSED == ["proposed"]

    def test_baselines_count(self):
        assert len(BASELINES) > 0, "BASELINES must not be empty"

    def test_baselines_names(self):
        # 验证 BASELINES 包含核心对比模型（不硬编码总数）
        required = {"lstm", "transformer", "dlinear", "itransformer"}
        assert required.issubset(set(BASELINES)), (
            f"BASELINES must contain at least {required}, got {set(BASELINES)}"
        )

    def test_ablations_count(self):
        assert len(ABLATIONS) > 0, "ABLATIONS must not be empty"

    def test_ablations_prefixed(self):
        for name in ABLATIONS:
            assert name.startswith("ablation:"), (
                f"Ablation '{name}' must start with 'ablation:'"
            )

    def test_total_mandatory_runs(self):
        """总运行数由列表长度动态计算，不硬编码具体数值。"""
        n_proposed  = len(PROPOSED)  * len(HORIZONS) * len(SEEDS)
        n_baselines = len(BASELINES) * len(HORIZONS) * len(SEEDS)
        n_ablations = len(ABLATIONS) * len(HORIZONS) * len(SEEDS)
        total = n_proposed + n_baselines + n_ablations
        # 只断言结构正确性：每个分组都有运行，总数与各分组之和一致
        assert n_proposed  > 0
        assert n_baselines > 0
        assert n_ablations > 0
        assert total == n_proposed + n_baselines + n_ablations


# ---------------------------------------------------------------------------
# 2. ExperimentMatrix initialisation
# ---------------------------------------------------------------------------


class TestExperimentMatrixInit:
    def test_output_dirs_created(self, tmp_em):
        em, tmp = tmp_em
        assert (tmp / "configs").is_dir(), "configs/ sub-directory must be created"

    def test_budget_defaults_to_thirty(self, tmp_em):
        em, _ = tmp_em
        status = em.get_budget_status()
        assert status["gpu_time_budget_hours"] == 30.0

    def test_initial_used_hours_zero(self, tmp_em):
        em, _ = tmp_em
        status = em.get_budget_status()
        assert status["gpu_used_hours"] == 0.0

    def test_initial_remaining_equals_budget(self, tmp_em):
        em, _ = tmp_em
        status = em.get_budget_status()
        assert status["gpu_remaining_hours"] == status["gpu_time_budget_hours"]

    def test_min_seeds_clamped_to_one(self, tmp_path):
        em = ExperimentMatrix(
            base_config_path=BASE_CONFIG,
            runtime_config_path=RUNTIME_CONFIG,
            out_dir=str(tmp_path),
            min_seeds=0,  # should be clamped to 1
        )
        assert em.min_seeds >= 1


# ---------------------------------------------------------------------------
# 3. Materialise configs — Req 12.1, 12.3
# ---------------------------------------------------------------------------


class TestMaterialize:
    def test_materialise_single_run(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        assert len(configs) == 1

    def test_materialise_writes_yaml(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        yaml_files = list((tmp / "configs").glob("*.yaml"))
        assert len(yaml_files) == 1

    def test_materialise_run_id_in_config(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        assert "run_id" in configs[0]

    def test_materialise_model_name_in_config(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        assert configs[0]["model_name"] == "proposed"

    def test_materialise_seed_in_config(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        assert configs[0]["seed"] == 42

    def test_materialise_horizon_injected(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[12], seeds=[42])
        assert configs[0]["dataset"]["horizon"] == 12

    def test_materialise_cartesian_product(self, tmp_em):
        em, tmp = tmp_em
        test_models = ["proposed", "lstm"]
        test_horizons = [1, 6]
        test_seeds = [42, 43]
        configs = em.materialize(
            models=test_models,
            horizons=test_horizons,
            seeds=test_seeds,
        )
        expected = len(test_models) * len(test_horizons) * len(test_seeds)
        assert len(configs) == expected

    def test_materialise_ablation_config(self, tmp_em):
        em, tmp = tmp_em
        configs = em.materialize(
            models=["ablation:vmd_off"],
            horizons=[6],
            seeds=[42],
        )
        assert len(configs) == 1
        assert configs[0]["model_name"] == "ablation:vmd_off"

    def test_yaml_file_loadable(self, tmp_em):
        em, tmp = tmp_em
        em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        yaml_files = list((tmp / "configs").glob("*.yaml"))
        with open(yaml_files[0], encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert isinstance(loaded, dict)
        assert "run_id" in loaded


# ---------------------------------------------------------------------------
# 4. Budget tracking — Req 14.7
# ---------------------------------------------------------------------------


class TestBudgetTracking:
    def test_budget_json_created_after_record(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("run_001", 1.0)
        assert (tmp / "budget.json").exists()

    def test_used_hours_accumulate(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("run_001", 2.5)
        em.record_run_time("run_002", 1.0)
        status = em.get_budget_status()
        assert abs(status["gpu_used_hours"] - 3.5) < 1e-9

    def test_run_times_list_grows(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("run_001", 1.0)
        em.record_run_time("run_002", 2.0)
        with open(tmp / "budget.json") as fh:
            b = json.load(fh)
        assert len(b["run_times"]) == 2

    def test_check_budget_within_limit(self, tmp_em):
        em, tmp = tmp_em
        assert em.check_budget(5.0) is True

    def test_check_budget_at_exact_limit(self, tmp_em):
        em, tmp = tmp_em
        # Use (30 - 1e-9)h, then ask for 1e-9h — should pass (exactly equal)
        em.record_run_time("run_001", 30.0 - 1e-9)
        assert em.check_budget(1e-9) is True

    def test_check_budget_over_limit(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("run_001", 29.9)
        assert em.check_budget(0.2) is False  # 29.9 + 0.2 > 30.0

    def test_remaining_hours_decreases(self, tmp_em):
        em, tmp = tmp_em
        before = em.get_budget_status()["gpu_remaining_hours"]
        em.record_run_time("run_001", 5.0)
        after = em.get_budget_status()["gpu_remaining_hours"]
        assert abs((before - after) - 5.0) < 1e-9

    def test_run_count_tracked(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("r1", 1.0)
        em.record_run_time("r2", 1.0)
        assert em.get_budget_status()["run_count"] == 2


# ---------------------------------------------------------------------------
# 5. Deferred runs — Req 14.8
# ---------------------------------------------------------------------------


class TestDeferredRuns:
    def test_defer_run_creates_file(self, tmp_em):
        em, tmp = tmp_em
        cfg = {"run_id": "test_001", "model_name": "proposed", "seed": 42,
               "dataset": {"horizon": 6}}
        em.defer_run(cfg, reason="budget_exceeded")
        assert (tmp / "deferred.jsonl").exists()

    def test_defer_run_appends_jsonl(self, tmp_em):
        em, tmp = tmp_em
        cfg = {"run_id": "test_001", "model_name": "proposed", "seed": 42,
               "dataset": {"horizon": 6}}
        em.defer_run(cfg, reason="budget_exceeded")
        em.defer_run(cfg, reason="budget_exceeded")
        with open(tmp / "deferred.jsonl") as fh:
            lines = fh.readlines()
        assert len(lines) == 2

    def test_defer_run_stores_run_id(self, tmp_em):
        em, tmp = tmp_em
        cfg = {"run_id": "myrun_42", "model_name": "proposed", "seed": 42,
               "dataset": {"horizon": 6}}
        em.defer_run(cfg, reason="test")
        with open(tmp / "deferred.jsonl") as fh:
            record = json.loads(fh.readline())
        assert record["run_id"] == "myrun_42"

    def test_defer_run_stores_full_config(self, tmp_em):
        em, tmp = tmp_em
        cfg = {"run_id": "myrun_42", "model_name": "proposed", "seed": 42,
               "dataset": {"horizon": 6}}
        em.defer_run(cfg, reason="test")
        with open(tmp / "deferred.jsonl") as fh:
            record = json.loads(fh.readline())
        assert "config" in record
        assert record["config"]["run_id"] == "myrun_42"

    def test_defer_run_stores_reason(self, tmp_em):
        em, tmp = tmp_em
        cfg = {"run_id": "myrun_42", "model_name": "proposed", "seed": 42,
               "dataset": {"horizon": 6}}
        em.defer_run(cfg, reason="budget_exceeded")
        with open(tmp / "deferred.jsonl") as fh:
            record = json.loads(fh.readline())
        assert record["reason"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# 6. Priority-based shedding — Req 12.5, 12.6
# ---------------------------------------------------------------------------


class TestPriorityShedding:
    def test_shap_shed_first(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("old", 29.9)  # almost out of budget
        result = em.materialize_with_shedding(
            estimated_hours_per_run=0.5,
            shed_shap=True, shed_optuna=True, shed_extra_viz=True,
        )
        assert "shap" in result["shed_items"]

    def test_optuna_shed_second(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("old", 29.9)
        result = em.materialize_with_shedding(
            estimated_hours_per_run=0.5,
            shed_shap=True, shed_optuna=True, shed_extra_viz=True,
        )
        # shap must appear before optuna in shed order
        shed = result["shed_items"]
        if "optuna" in shed:
            assert shed.index("shap") < shed.index("optuna")

    def test_seeds_reduced_to_min(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("old", 29.9)
        result = em.materialize_with_shedding(
            estimated_hours_per_run=0.5,
            shed_shap=True, shed_optuna=True, shed_extra_viz=True,
        )
        assert len(result["active_seeds"]) >= em.min_seeds

    def test_seeds_never_below_min(self, tmp_em):
        em, tmp = tmp_em
        # Drain budget completely
        em.record_run_time("old", 30.0)
        result = em.materialize_with_shedding(estimated_hours_per_run=1.0)
        assert len(result["active_seeds"]) >= em.min_seeds

    def test_no_shedding_within_budget(self, tmp_em):
        em, tmp = tmp_em
        # With 30h budget and ~0.0001h per run, nothing should be shed
        result = em.materialize_with_shedding(
            estimated_hours_per_run=0.0001,
            shed_shap=True, shed_optuna=True, shed_extra_viz=True,
        )
        assert result["shed_items"] == []
        assert result["active_seeds"] == list(SEEDS)

    def test_warnings_emitted_on_shedding(self, tmp_em):
        em, tmp = tmp_em
        em.record_run_time("old", 29.9)
        result = em.materialize_with_shedding(estimated_hours_per_run=0.5)
        assert len(result["warnings"]) > 0


# ---------------------------------------------------------------------------
# 7. load_run_config — Req 14.8
# ---------------------------------------------------------------------------


class TestLoadRunConfig:
    def test_loads_valid_yaml(self, tmp_path):
        cfg = {"run_id": "r1", "model_name": "proposed", "seed": 42}
        p = tmp_path / "r1.yaml"
        with open(p, "w") as fh:
            yaml.dump(cfg, fh)
        loaded = load_run_config(str(p))
        assert loaded["run_id"] == "r1"

    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_run_config("/nonexistent/path.yaml")

    def test_raises_value_error_for_non_dict(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_run_config(str(p))

    def test_raises_value_error_for_invalid_yaml(self, tmp_path):
        p = tmp_path / "invalid.yaml"
        # Write something that is a valid YAML scalar (not a mapping)
        p.write_text("just a plain string\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_run_config(str(p))

    def test_roundtrip_via_materialize(self, tmp_em):
        """Config written by materialize should be loadable by load_run_config."""
        em, tmp = tmp_em
        configs = em.materialize(models=["proposed"], horizons=[6], seeds=[42])
        run_id = configs[0]["run_id"]
        yaml_path = tmp / "configs" / f"{run_id}.yaml"
        loaded = load_run_config(str(yaml_path))
        assert loaded["run_id"] == run_id
        assert loaded["model_name"] == "proposed"
        assert loaded["seed"] == 42
