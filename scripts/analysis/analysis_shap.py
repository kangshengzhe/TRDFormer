"""
Wind Power SHAP Analysis — thin wrapper that delegates to shap_runner.py.

This module previously contained training, evaluation, and SHAP computation
code inline.  Those responsibilities have been extracted into dedicated
modules:

    - Training / evaluation  → experiments/runner.py
    - SHAP computation       → scripts/analysis/shap_runner.py

This file is retained for backward compatibility and as a convenient
entry-point that mirrors the original CLI surface.

Usage (original style — still works):
    python scripts/analysis/analysis_shap.py \\
        --checkpoint model_save/wind/X.pt \\
        --config outputs/runs/configs/X.yaml \\
        --output outputs/runs/shap_values.npy \\
        --n_samples 100

    # Or with legacy flags:
    python scripts/analysis/analysis_shap.py \\
        --checkpoint model_save/wind/X.pt \\
        --manifest-dir outputs/manifests/ \\
        --n-samples 200 \\
        --out-dir outputs/figures/

Requirements: 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so that both local packages and
# shap_runner itself can be imported correctly.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Delegate entirely to shap_runner
from scripts.analysis.shap_runner import main  # noqa: E402


if __name__ == "__main__":
    main()
