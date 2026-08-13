"""
baselines/

Baseline model harness for the wind-power-forecasting experiment.

Public API
----------
TSLForecastWrapper
    Wraps any Time-Series-Library (TSL) Model class so it satisfies our
    unified forward contract: (B, lookback, 5) → (B, horizon).

MODEL_REGISTRY
    Dict mapping model name strings to factory callables.
    ``factory(cfg: dict) -> nn.Module``

get_model(name, cfg)
    Look up *name* in MODEL_REGISTRY and build the model from *cfg*.
    Handles ``'ablation:<variant>'`` prefix transparently.

make_tsl_configs(model_cfg, dataset_cfg)
    Convert our YAML config dicts to a TSL-compatible ``argparse.Namespace``.
"""

from baselines.tsl_adapter import TSLForecastWrapper  # noqa: F401
from baselines.registry import MODEL_REGISTRY, get_model  # noqa: F401
from baselines.tsl_configs import make_tsl_configs  # noqa: F401

__all__ = [
    "TSLForecastWrapper",
    "MODEL_REGISTRY",
    "get_model",
    "make_tsl_configs",
]
