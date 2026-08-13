"""Unit tests for data_pipeline.vmd.

These tests require the optional ``vmdpy`` dependency. If it is not installed
(e.g. on an environment without the local-CPU preprocessing stack), the whole
module is skipped gracefully via ``pytest.importorskip``.
"""

import json
import os
import sys

import numpy as np
import pytest

# Ensure the project root is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip the entire module if vmdpy is unavailable.
pytest.importorskip("vmdpy")

from data_pipeline.manifest import VMDParams, VMDManifest
from data_pipeline.vmd import (
    fit_vmd_on_train,
    apply_vmd_to_partition,
    persist_vmd_params,
    K_MIN,
    K_MAX,
)


def _make_signal(n: int, seed: int = 0) -> np.ndarray:
    """Build a synthetic multi-component 1-D signal of length ``n``."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0.0, 1.0, n)
    slow = np.sin(2 * np.pi * 3 * t)
    fast = 0.5 * np.sin(2 * np.pi * 25 * t)
    noise = 0.05 * rng.randn(n)
    return (slow + fast + noise).astype(np.float64)


class TestFitVmdOnTrain:
    """Tests for fit_vmd_on_train."""

    def test_returns_shape_n_by_k(self):
        signal = _make_signal(256)
        params = VMDParams(K=5)
        imfs = fit_vmd_on_train(signal, params)
        assert imfs.shape == (256, 5)

    def test_default_k_is_five(self):
        assert VMDParams().K == 5

    def test_custom_k_changes_channel_count(self):
        signal = _make_signal(256)
        imfs = fit_vmd_on_train(signal, VMDParams(K=4))
        assert imfs.shape == (256, 4)

    def test_odd_length_input_preserved(self):
        """vmdpy truncates odd-length inputs; output must match input length."""
        signal = _make_signal(255)
        imfs = fit_vmd_on_train(signal, VMDParams(K=5))
        assert imfs.shape == (255, 5)

    def test_reconstruction_approximates_signal(self):
        """Summing the K IMFs should approximately reconstruct the input."""
        signal = _make_signal(512)
        imfs = fit_vmd_on_train(signal, VMDParams(K=5))
        reconstruction = imfs.sum(axis=1)
        # VMD does not perfectly reconstruct, but the residual energy should be
        # small relative to the signal energy.
        residual = np.linalg.norm(reconstruction - signal)
        signal_norm = np.linalg.norm(signal)
        assert residual / signal_norm < 0.2


class TestApplyVmdToPartition:
    """Tests for apply_vmd_to_partition."""

    def test_returns_shape_n_by_k(self):
        signal = _make_signal(300, seed=1)
        imfs = apply_vmd_to_partition(signal, VMDParams(K=5))
        assert imfs.shape == (300, 5)

    def test_same_params_same_output(self):
        """Re-running with identical params on identical data is deterministic."""
        signal = _make_signal(256, seed=2)
        params = VMDParams(K=5)
        a = apply_vmd_to_partition(signal, params)
        b = apply_vmd_to_partition(signal, params)
        np.testing.assert_allclose(a, b)


class TestParameterValidation:
    """Tests for K range validation (Requirement 1.15)."""

    @pytest.mark.parametrize("k", [K_MIN, 5, K_MAX])
    def test_valid_k_accepted(self, k):
        signal = _make_signal(128)
        imfs = fit_vmd_on_train(signal, VMDParams(K=k))
        assert imfs.shape == (128, k)

    @pytest.mark.parametrize("k", [0, 1, 2, 11, 20])
    def test_invalid_k_rejected(self, k):
        signal = _make_signal(128)
        with pytest.raises(ValueError):
            fit_vmd_on_train(signal, VMDParams(K=k))

    def test_invalid_k_rejected_on_partition(self):
        signal = _make_signal(128)
        with pytest.raises(ValueError):
            apply_vmd_to_partition(signal, VMDParams(K=2))

    def test_non_1d_input_rejected(self):
        signal = _make_signal(128).reshape(8, 16)
        with pytest.raises(ValueError):
            fit_vmd_on_train(signal, VMDParams(K=5))

    def test_empty_input_rejected(self):
        with pytest.raises(ValueError):
            fit_vmd_on_train(np.array([]), VMDParams(K=5))


class TestPersistVmdParams:
    """Tests for persisting VMD parameters to vmd_params.json."""

    def test_writes_default_filename_in_dir(self, tmp_path):
        params = VMDParams(K=5)
        out = persist_vmd_params(
            tmp_path, params, fit_seed=42, fit_n_samples=1000, imf_shape=(1200, 5)
        )
        assert out.name == "vmd_params.json"
        assert out.exists()

    def test_persisted_values_roundtrip(self, tmp_path):
        params = VMDParams(K=6, alpha=1500.0, tau=0.1, DC=1, init=0, tol=1e-6, max_iter=400)
        out = persist_vmd_params(
            tmp_path, params, fit_seed=7, fit_n_samples=2048, imf_shape=(3000, 6)
        )
        loaded, meta = VMDManifest.read(out)
        assert loaded == params
        assert meta["fit_seed"] == 7
        assert meta["fit_n_samples"] == 2048
        assert meta["imf_shape"] == (3000, 6)

    def test_persists_required_fields(self, tmp_path):
        """Requirement 1.17: K, alpha, tol, max_iter must be persisted."""
        out = persist_vmd_params(tmp_path, VMDParams(K=5))
        with open(out, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("K", "alpha", "tol", "max_iter", "library"):
            assert key in data

    def test_invalid_k_not_persisted(self, tmp_path):
        with pytest.raises(ValueError):
            persist_vmd_params(tmp_path, VMDParams(K=2))
