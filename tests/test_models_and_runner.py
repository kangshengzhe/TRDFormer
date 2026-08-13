"""
tests/test_models_and_runner.py

Unit tests for:
- models/unified_proposed.py  (UnifiedProposedModel + ablation variants)
- models/iTransformer_LSTM.py (iTransformer_LSTM base model)
- experiments/runner.py       (RunConfig validation, custom exceptions)
- experiments/metrics.py      (compute_metrics)

These tests verify that models can be instantiated, their forward pass
produces the correct output shape, ablation switches work as intended,
and the runner correctly rejects invalid configurations.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest
import torch

# Ensure repo root is on sys.path
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)

from models.unified_proposed import UnifiedProposedModel
from models.iTransformer_LSTM import iTransformer_LSTM
from models.LSTM import LSTM
from experiments.metrics import compute_metrics
from experiments.runner import (
    RunConfig,
    InvalidHorizonError,
    InsufficientWindowError,
    _validate_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(batch: int, lookback: int, n_channels: int) -> torch.Tensor:
    """Create a random float32 tensor (B, lookback, n_channels)."""
    return torch.randn(batch, lookback, n_channels)


# ---------------------------------------------------------------------------
# 1. iTransformer_LSTM — base model
# ---------------------------------------------------------------------------


class TestITransformerLSTM:
    """Verify the base model constructs and runs forward without errors."""

    def test_default_forward_shape(self):
        """Default config: (B, lookback, 5) → (B, horizon)."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6), f"Expected (2, 6), got {out.shape}"

    def test_horizon_1_forward(self):
        """Single-step prediction."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=1,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
        )
        x = _make_input(4, 24, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_fusion_concat(self):
        """fusion_type='concat' runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            fusion_type="concat",
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_fusion_sum(self):
        """fusion_type='sum' runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            fusion_type="sum",
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_fusion_gated(self):
        """fusion_type='gated' (adaptive soft-gating, SCGF-style) runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            fusion_type="gated",
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_head_linear(self):
        """head_type='linear' runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            head_type="linear",
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_head_mlp(self):
        """head_type='mlp' runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            head_type="mlp",
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_itrans_off(self):
        """use_itransformer=False (itrans_off ablation) runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            use_itransformer=False,
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_lstm_off(self):
        """use_lstm=False (lstm_off ablation) runs without error."""
        model = iTransformer_LSTM(
            input_size=5,
            length_pre=6,
            dim_lstm=32,
            depth_lstm=1,
            length_input=24,
            dim_embed=32,
            depth=1,
            heads=2,
            use_lstm=False,
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_invalid_fusion_type_raises(self):
        with pytest.raises(AssertionError):
            iTransformer_LSTM(
                input_size=5, length_pre=6, dim_lstm=32, depth_lstm=1,
                length_input=24, dim_embed=32, depth=1, heads=2,
                fusion_type="invalid_fusion",
            )

    def test_invalid_head_type_raises(self):
        with pytest.raises(AssertionError):
            iTransformer_LSTM(
                input_size=5, length_pre=6, dim_lstm=32, depth_lstm=1,
                length_input=24, dim_embed=32, depth=1, heads=2,
                head_type="invalid_head",
            )


# ---------------------------------------------------------------------------
# 2. UnifiedProposedModel — forward shape
# ---------------------------------------------------------------------------


class TestUnifiedProposedModelShape:
    """Check that UnifiedProposedModel produces (B, horizon) output."""

    def test_vmd_off_forward_shape(self):
        """VMD disabled: 1 target + 4 covariate channels."""
        model = UnifiedProposedModel(
            lookback=24,
            horizon=6,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(3, 24, 5)  # 1 + 4 = 5 channels
        out = model(x)
        assert out.shape == (3, 6), f"Expected (3, 6), got {out.shape}"

    def test_vmd_on_forward_shape(self):
        """VMD enabled with K=3 IMFs: 1+3=4 target channels + 4 covariate = 8 total."""
        model = UnifiedProposedModel(
            lookback=24,
            horizon=6,
            n_target_channels=4,   # Patv + 3 IMFs
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(3, 24, 8)  # 4 + 4 = 8 channels
        out = model(x)
        assert out.shape == (3, 6)

    def test_horizon_1_vmd_off(self):
        model = UnifiedProposedModel(
            lookback=24,
            horizon=1,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 1)

    def test_horizon_24(self):
        model = UnifiedProposedModel(
            lookback=24,
            horizon=24,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 24)

    def test_batch_size_one(self):
        model = UnifiedProposedModel(
            lookback=24,
            horizon=6,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(1, 24, 5)
        out = model(x)
        assert out.shape == (1, 6)

    def test_larger_batch(self):
        model = UnifiedProposedModel(
            lookback=24,
            horizon=6,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        x = _make_input(16, 24, 5)
        out = model(x)
        assert out.shape == (16, 6)


# ---------------------------------------------------------------------------
# 3. UnifiedProposedModel — ablation switches
# ---------------------------------------------------------------------------


class TestUnifiedProposedModelAblations:
    """Verify all ablation switches produce valid (B, horizon) output."""

    _BASE_KWARGS = dict(
        lookback=24,
        horizon=6,
        n_target_channels=1,
        n_covariate_channels=4,
        dim_embed=32,
        depth_itrans=1,
        heads_itrans=2,
        dim_lstm=32,
        depth_lstm=1,
    )

    def _run(self, **kwargs) -> torch.Tensor:
        kw = {**self._BASE_KWARGS, **kwargs}
        model = UnifiedProposedModel(**kw)
        x = _make_input(2, kw["lookback"], kw["n_target_channels"] + kw["n_covariate_channels"])
        return model(x)

    def test_itrans_off(self):
        out = self._run(use_itransformer=False)
        assert out.shape == (2, 6)

    def test_lstm_off(self):
        out = self._run(use_lstm=False)
        assert out.shape == (2, 6)

    def test_fusion_concat(self):
        out = self._run(fusion_type="concat")
        assert out.shape == (2, 6)

    def test_fusion_sum(self):
        out = self._run(fusion_type="sum")
        assert out.shape == (2, 6)

    def test_head_linear(self):
        out = self._run(head_type="linear")
        assert out.shape == (2, 6)

    def test_head_mlp(self):
        out = self._run(head_type="mlp")
        assert out.shape == (2, 6)


# ---------------------------------------------------------------------------
# 4. UnifiedProposedModel — from_config
# ---------------------------------------------------------------------------


class TestUnifiedProposedModelFromConfig:
    def test_from_config_basic(self):
        cfg = {
            "lookback": 24,
            "horizon": 6,
            "n_target_channels": 1,
            "n_covariate_channels": 4,
            "dim_embed": 32,
            "depth_itrans": 1,
            "heads_itrans": 2,
            "dim_lstm": 32,
            "depth_lstm": 1,
        }
        model = UnifiedProposedModel.from_config(cfg)
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_from_config_missing_lookback_raises(self):
        with pytest.raises(KeyError):
            UnifiedProposedModel.from_config({"horizon": 6, "n_target_channels": 1})

    def test_from_config_missing_horizon_raises(self):
        with pytest.raises(KeyError):
            UnifiedProposedModel.from_config({"lookback": 24, "n_target_channels": 1})

    def test_from_config_ignores_extra_keys(self):
        """Extra keys in the config dict should be silently ignored."""
        cfg = {
            "lookback": 24,
            "horizon": 6,
            "n_target_channels": 1,
            "unknown_key": "should_be_ignored",
            "another_extra": 999,
        }
        model = UnifiedProposedModel.from_config(cfg)
        x = _make_input(2, 24, 5)
        out = model(x)
        assert out.shape == (2, 6)

    def test_from_config_uses_defaults_for_optional_keys(self):
        """Optional keys should fall back to constructor defaults."""
        cfg = {
            "lookback": 24,
            "horizon": 6,
            "n_target_channels": 1,
        }
        # Should not raise — all missing keys use defaults
        model = UnifiedProposedModel.from_config(cfg)
        assert model is not None


# ---------------------------------------------------------------------------
# 5. LSTM baseline model
# ---------------------------------------------------------------------------


class TestLSTMModel:
    def test_forward_shape(self):
        model = LSTM(input_size=5, hidden_size=32, num_layers=2, output_size=6)
        x = _make_input(4, 24, 5)
        out = model(x)
        assert out.shape == (4, 6)

    def test_horizon_1(self):
        model = LSTM(input_size=5, hidden_size=32, num_layers=1, output_size=1)
        x = _make_input(4, 24, 5)
        out = model(x)
        assert out.shape == (4, 1)

    def test_horizon_24(self):
        model = LSTM(input_size=5, hidden_size=32, num_layers=1, output_size=24)
        x = _make_input(2, 48, 5)
        out = model(x)
        assert out.shape == (2, 24)


# ---------------------------------------------------------------------------
# 6. Runner config validation
# ---------------------------------------------------------------------------


def _make_run_config(**overrides) -> RunConfig:
    """Build a minimal valid RunConfig for testing _validate_config."""
    defaults = dict(
        run_id="test_run_001",
        model_name="proposed",
        seed=42,
        lookback=24,
        horizon=6,
        train={"batch_size": 32, "epochs": 2, "learning_rate": 1e-4},
        model={},
        ablation={},
        runtime={"device": "cpu", "execution_location": "local_cpu"},
        dataset={"csv_path": "dummy.csv", "scaler_path": "dummy.pkl",
                 "partition_path": ""},
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


class TestRunnerConfigValidation:
    """Tests for _validate_config (horizon range check, Req 3.6)."""

    def test_valid_horizon_passes(self):
        for h in [1, 6, 12, 24]:
            cfg = _make_run_config(horizon=h)
            _validate_config(cfg)  # should not raise

    def test_horizon_zero_raises(self):
        cfg = _make_run_config(horizon=0)
        with pytest.raises(InvalidHorizonError):
            _validate_config(cfg)

    def test_horizon_25_raises(self):
        cfg = _make_run_config(horizon=25)
        with pytest.raises(InvalidHorizonError):
            _validate_config(cfg)

    def test_horizon_negative_raises(self):
        cfg = _make_run_config(horizon=-1)
        with pytest.raises(InvalidHorizonError):
            _validate_config(cfg)

    def test_horizon_boundary_1_passes(self):
        cfg = _make_run_config(horizon=1)
        _validate_config(cfg)

    def test_horizon_boundary_24_passes(self):
        cfg = _make_run_config(horizon=24)
        _validate_config(cfg)

    def test_invalid_horizon_error_message(self):
        cfg = _make_run_config(horizon=30)
        with pytest.raises(InvalidHorizonError, match="horizon must be in"):
            _validate_config(cfg)


class TestRunnerExceptions:
    """Verify custom exception types are properly defined and inherit correctly."""

    def test_invalid_horizon_error_is_value_error(self):
        assert issubclass(InvalidHorizonError, ValueError)

    def test_insufficient_window_error_is_value_error(self):
        assert issubclass(InsufficientWindowError, ValueError)

    def test_invalid_horizon_error_can_be_raised(self):
        with pytest.raises(InvalidHorizonError):
            raise InvalidHorizonError("test message")

    def test_insufficient_window_error_can_be_raised(self):
        with pytest.raises(InsufficientWindowError):
            raise InsufficientWindowError("test message")


# ---------------------------------------------------------------------------
# 7. compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Verify metric computation correctness (Req 6.1–6.3, 6.7)."""

    def test_returns_all_keys(self):
        a = np.array([100.0, 200.0, 150.0])
        p = np.array([110.0, 190.0, 160.0])
        m = compute_metrics(a, p)
        assert set(m.keys()) == {"mae", "rmse", "r2", "mbe", "smape"}

    def test_perfect_predictions_mae_zero(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a.copy())
        assert m["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_rmse_zero(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a.copy())
        assert m["rmse"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_r2_one(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a.copy())
        assert m["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_perfect_predictions_mbe_zero(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a.copy())
        assert m["mbe"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_predictions_smape_zero(self):
        a = np.array([100.0, 200.0, 300.0])
        m = compute_metrics(a, a.copy())
        assert m["smape"] == pytest.approx(0.0, abs=1e-9)

    def test_known_mae(self):
        """MAE of [10, 20] errors = 15."""
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 220.0])
        m = compute_metrics(a, p)
        assert m["mae"] == pytest.approx(15.0, abs=1e-9)

    def test_known_mbe_positive_overestimate(self):
        """MBE = mean(pred - actual): pred always 10 above → MBE = +10."""
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 210.0])
        m = compute_metrics(a, p)
        assert m["mbe"] == pytest.approx(10.0, abs=1e-9)

    def test_known_mbe_negative_underestimate(self):
        a = np.array([100.0, 200.0])
        p = np.array([90.0, 190.0])
        m = compute_metrics(a, p)
        assert m["mbe"] == pytest.approx(-10.0, abs=1e-9)

    def test_smape_excludes_both_zero_samples(self):
        """Req 6.7: samples where actual=0 AND predicted=0 must be excluded."""
        # Only sample 1 is valid (a=100, p=200)
        # Sample 2 is (0, 0) — should be excluded from sMAPE
        a = np.array([100.0, 0.0])
        p = np.array([200.0, 0.0])
        m = compute_metrics(a, p)
        # sMAPE for (100, 200): 200 * |100 - 200| / (100 + 200) = 200/3 ≈ 66.67
        expected_smape = 200.0 * 100.0 / 300.0
        assert m["smape"] == pytest.approx(expected_smape, abs=1e-6)

    def test_smape_all_both_zero_returns_zero(self):
        a = np.array([0.0, 0.0])
        p = np.array([0.0, 0.0])
        m = compute_metrics(a, p)
        assert m["smape"] == 0.0

    def test_smape_range_is_0_to_200(self):
        """sMAPE must always be in [0, 200]."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1000, size=500)
        p = rng.uniform(0, 1000, size=500)
        m = compute_metrics(a, p)
        assert 0.0 <= m["smape"] <= 200.0

    def test_mismatched_sizes_raise_value_error(self):
        a = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0])
        with pytest.raises(ValueError):
            compute_metrics(a, p)

    def test_2d_input_flattened(self):
        """compute_metrics should accept 2-D arrays."""
        a = np.array([[100.0, 200.0], [300.0, 400.0]])
        p = np.array([[110.0, 190.0], [310.0, 390.0]])
        m = compute_metrics(a, p)
        # MAE = mean([10, 10, 10, 10]) = 10
        assert m["mae"] == pytest.approx(10.0, abs=1e-9)

    def test_r2_degenerate_constant_target(self):
        """R² when all actuals are constant (ss_tot == 0) should be NaN."""
        a = np.array([100.0, 100.0, 100.0])
        p = np.array([110.0, 90.0, 105.0])
        m = compute_metrics(a, p)
        assert np.isnan(m["r2"])


# ---------------------------------------------------------------------------
# 8. Model parameter counts (sanity checks)
# ---------------------------------------------------------------------------


class TestModelParameters:
    def test_unified_proposed_has_parameters(self):
        model = UnifiedProposedModel(
            lookback=24,
            horizon=6,
            n_target_channels=1,
            n_covariate_channels=4,
            dim_embed=32,
            depth_itrans=1,
            heads_itrans=2,
            dim_lstm=32,
            depth_lstm=1,
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0, "Model must have trainable parameters"

    def test_lstm_has_parameters(self):
        model = LSTM(input_size=5, hidden_size=32, num_layers=2, output_size=6)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_itrans_off_ablation_has_fewer_params_than_full(self):
        """Removing iTransformer branch should reduce parameter count."""
        full = UnifiedProposedModel(
            lookback=24, horizon=6, n_target_channels=1, n_covariate_channels=4,
            dim_embed=64, depth_itrans=2, heads_itrans=4, dim_lstm=64, depth_lstm=2,
        )
        ablated = UnifiedProposedModel(
            lookback=24, horizon=6, n_target_channels=1, n_covariate_channels=4,
            dim_embed=64, depth_itrans=2, heads_itrans=4, dim_lstm=64, depth_lstm=2,
            use_itransformer=False,
        )
        full_params = sum(p.numel() for p in full.parameters())
        ablated_params = sum(p.numel() for p in ablated.parameters())
        assert ablated_params < full_params, (
            f"Ablation should reduce params: full={full_params}, ablated={ablated_params}"
        )

    def test_lstm_off_ablation_has_fewer_params_than_full(self):
        """Removing LSTM branch should reduce parameter count."""
        full = UnifiedProposedModel(
            lookback=24, horizon=6, n_target_channels=1, n_covariate_channels=4,
            dim_embed=64, depth_itrans=2, heads_itrans=4, dim_lstm=64, depth_lstm=2,
        )
        ablated = UnifiedProposedModel(
            lookback=24, horizon=6, n_target_channels=1, n_covariate_channels=4,
            dim_embed=64, depth_itrans=2, heads_itrans=4, dim_lstm=64, depth_lstm=2,
            use_lstm=False,
        )
        full_params = sum(p.numel() for p in full.parameters())
        ablated_params = sum(p.numel() for p in ablated.parameters())
        assert ablated_params < full_params
