"""
baselines/registry.py

MODEL_REGISTRY maps model name strings to factory callables.

Each factory has the signature::

    factory(cfg: dict) -> nn.Module

where ``cfg`` is the *merged* run config dict (containing at minimum the
``model`` and ``dataset`` sub-dicts).

The registry is the single place that connects a model name string
(as written in YAML / experiment matrix) to a concrete ``nn.Module``
instance.  The experiment runner calls::

    model = get_model(cfg['model_name'], cfg)

and receives a ready-to-train module.

Supported model names
---------------------
proposed                 – UnifiedProposedModel (iTransformer + LSTM + CA + KAN)
lstm                     – models.LSTM (senior-repo recurrent baseline)
transformer              – Time-Series-Library Transformer
informer                 – Time-Series-Library Informer
fedformer                – Time-Series-Library FEDformer
dlinear                  – Time-Series-Library DLinear
patchtst                 – Time-Series-Library PatchTST
itransformer             – Time-Series-Library iTransformer
timesnet                 – Time-Series-Library TimesNet
autoformer               – Time-Series-Library Autoformer
nonstationary_transformer – Time-Series-Library Nonstationary_Transformer
timexer                  – Time-Series-Library TimeXer

Ablation names (prefix "ablation:")
------------------------------------
ablation:<name>  – strips prefix, loads the ablation YAML from
                   ``configs/ablation/<name>.yaml``, merges ablation
                   overrides into the proposed model config, then delegates
                   to ``build_proposed``.

Requirements: 4.1, 4.5
"""

from __future__ import annotations

import copy
from typing import Callable

import torch.nn as nn

from baselines.tsl_adapter import TSLForecastWrapper
from baselines.tsl_configs import make_tsl_configs


# ---------------------------------------------------------------------------
# Factory: proposed model (UnifiedProposedModel)
# ---------------------------------------------------------------------------

def build_proposed(cfg: dict) -> nn.Module:
    """
    Build the UnifiedProposedModel from the merged config dict.

    Reads from ``cfg['model']`` (dim_embed, depth_itrans, heads_itrans,
    dim_lstm, depth_lstm) and ``cfg['ablation']`` (use_itransformer,
    use_lstm, fusion_type, head_type) and ``cfg['dataset']`` (lookback,
    horizon, vmd.K when vmd.enabled).

    Falls back gracefully to ``iTransformer_LSTM`` when
    ``models/unified_proposed.py`` is not yet present.
    """
    model_cfg   = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    ablation_cfg = cfg.get("ablation", {})

    lookback: int = int(dataset_cfg.get("lookback", dataset_cfg.get("seq_len", 144)))
    horizon:  int = int(dataset_cfg.get("horizon",  dataset_cfg.get("pred_len", 6)))

    vmd_cfg = dataset_cfg.get("vmd", {})
    vmd_on  = bool(vmd_cfg.get("enabled", True))
    K       = int(vmd_cfg.get("K", 5)) if vmd_on else 0
    n_target_channels   = 1 + K        # Patv + K IMFs  (or just Patv)
    n_covariate_channels = 4            # Wspd, Wdir, Etmp, Itmp

    # Try to import the thin UnifiedProposedModel wrapper; fall back to the
    # directly-refactored iTransformer_LSTM when unified_proposed.py is not
    # yet available (e.g., during incremental development).
    try:
        from models.unified_proposed import UnifiedProposedModel  # type: ignore
        return UnifiedProposedModel(
            lookback=lookback,
            horizon=horizon,
            n_target_channels=n_target_channels,
            n_covariate_channels=n_covariate_channels,
            use_itransformer=bool(ablation_cfg.get("use_itransformer", True)),
            use_lstm=bool(ablation_cfg.get("use_lstm", True)),
            fusion_type=str(ablation_cfg.get("fusion_type", "gated")),
            head_type=str(ablation_cfg.get("head_type", "kan")),
            dim_embed=int(model_cfg.get("dim_embed", 128)),
            depth_itrans=int(model_cfg.get("depth_itrans", 4)),
            heads_itrans=int(model_cfg.get("heads_itrans", 6)),
            dim_lstm=int(model_cfg.get("dim_lstm", 128)),
            depth_lstm=int(model_cfg.get("depth_lstm", 3)),
        )
    except ImportError:
        # Fallback: use the directly refactored iTransformer_LSTM
        from models.iTransformer_LSTM import iTransformer_LSTM  # type: ignore
        return iTransformer_LSTM(
            input_size=n_target_channels + n_covariate_channels,
            length_pre=horizon,
            dim_lstm=int(model_cfg.get("dim_lstm", 128)),
            depth_lstm=int(model_cfg.get("depth_lstm", 3)),
            length_input=lookback,
            dim_embed=int(model_cfg.get("dim_embed", 128)),
            depth=int(model_cfg.get("depth_itrans", 4)),
            heads=int(model_cfg.get("heads_itrans", 6)),
            use_itransformer=bool(ablation_cfg.get("use_itransformer", True)),
            use_lstm=bool(ablation_cfg.get("use_lstm", True)),
            fusion_type=str(ablation_cfg.get("fusion_type", "gated")),
            head_type=str(ablation_cfg.get("head_type", "kan")),
        )


# ---------------------------------------------------------------------------
# Factory: senior-repo LSTM baseline
# ---------------------------------------------------------------------------

def build_senior_lstm(cfg: dict) -> nn.Module:
    """
    Build the LSTM from ``models/LSTM.py`` (the senior-repo recurrent baseline).

    Constructor signature::

        LSTM(input_size, hidden_size, num_layers, output_size, bidirectional=False)

    Mapped from cfg:
        input_size  = 5  (always 5 raw features; no IMF channels for baselines)
        hidden_size = cfg['model']['d_model']  (default 128)
        num_layers  = cfg['model']['n_layers'] (default 3)
        output_size = cfg['dataset']['horizon'] or cfg['dataset']['pred_len']
        bidirectional = cfg['model'].get('bidirectional', False)
    """
    from models.LSTM import LSTM  # type: ignore

    model_cfg   = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    hidden_size   = int(model_cfg.get("d_model",  model_cfg.get("hidden_size", 128)))
    num_layers    = int(model_cfg.get("n_layers", model_cfg.get("num_layers", 3)))
    horizon       = int(dataset_cfg.get("horizon", dataset_cfg.get("pred_len", 6)))
    bidirectional = bool(model_cfg.get("bidirectional", False))

    return LSTM(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=horizon,
        bidirectional=bidirectional,
    )


# ---------------------------------------------------------------------------
# Factory generator: TSL models
# ---------------------------------------------------------------------------

def build_tsl_model(tsl_class_name: str) -> Callable[[dict], nn.Module]:
    """
    Return a factory callable that builds a ``TSLForecastWrapper`` for the
    named TSL model class.

    Parameters
    ----------
    tsl_class_name : str
        The name of the TSL model, matching the filename stem under
        ``Time-Series-Library/models/`` (e.g. ``'Transformer'``,
        ``'DLinear'``, ``'iTransformer'``).

    Returns
    -------
    Callable[[dict], nn.Module]
        A factory ``factory(cfg) -> TSLForecastWrapper``.
    """
    def _factory(cfg: dict) -> nn.Module:
        model_cfg   = cfg.get("model", {})
        dataset_cfg = cfg.get("dataset", {})
        configs = make_tsl_configs(model_cfg, dataset_cfg)
        configs.model = tsl_class_name
        return TSLForecastWrapper(tsl_class_name, configs)

    _factory.__name__ = f"build_{tsl_class_name.lower()}"
    _factory.__qualname__ = f"build_{tsl_class_name.lower()}"
    return _factory


# ---------------------------------------------------------------------------
# MODEL_REGISTRY
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable[[dict], nn.Module]] = {
    "proposed":                  build_proposed,
    "lstm":                      build_senior_lstm,
    "transformer":               build_tsl_model("Transformer"),
    "informer":                  build_tsl_model("Informer"),
    "fedformer":                 build_tsl_model("FEDformer"),
    "dlinear":                   build_tsl_model("DLinear"),
    "patchtst":                  build_tsl_model("PatchTST"),
    "itransformer":              build_tsl_model("iTransformer"),
    "timesnet":                  build_tsl_model("TimesNet"),
    # --- 扩展的 3 个基线 ---
    "autoformer":                build_tsl_model("Autoformer"),
    "nonstationary_transformer": build_tsl_model("Nonstationary_Transformer"),
    "timexer":                   build_tsl_model("TimeXer"),
}


# ---------------------------------------------------------------------------
# Public helper: get_model
# ---------------------------------------------------------------------------

def get_model(name: str, cfg: dict) -> nn.Module:
    """
    Look up a model name in MODEL_REGISTRY and build it from *cfg*.

    Parameters
    ----------
    name : str
        A key in MODEL_REGISTRY (e.g. ``'proposed'``, ``'lstm'``,
        ``'transformer'``) **or** an ablation name in the form
        ``'ablation:<variant_name>'``.

    cfg : dict
        The merged run config dict (must contain at minimum ``'model'``
        and ``'dataset'`` sub-dicts).

    Returns
    -------
    nn.Module
        A ready-to-train model instance.

    Raises
    ------
    KeyError
        If *name* (after stripping any ``'ablation:'`` prefix) is not in
        MODEL_REGISTRY and is not a recognised ablation variant.

    Notes
    -----
    **Ablation handling**

    When *name* starts with ``'ablation:'`` the prefix is stripped and the
    remainder is used as an ablation variant name.  The function loads the
    corresponding ablation YAML (``configs/ablation/<variant>.yaml``),
    merges its ``ablation`` block **on top of** the one already in *cfg*,
    and delegates to ``build_proposed``.  The caller's ``cfg`` is never
    mutated — a deep copy is used internally.
    """
    if name.startswith("ablation:"):
        variant = name[len("ablation:"):]
        return _build_ablation(variant, cfg)

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model name '{name}'.  "
            f"Available: {available}.  "
            "For ablation variants prefix with 'ablation:', e.g. 'ablation:vmd_off'."
        )

    return MODEL_REGISTRY[name](cfg)


# ---------------------------------------------------------------------------
# Internal: ablation builder
# ---------------------------------------------------------------------------

_ABLATION_DEFAULTS: dict[str, dict] = {
    "itrans_off":    {"ablation": {"use_itransformer": False}},
    "lstm_off":      {"ablation": {"use_lstm": False}},
    "fusion_concat": {"ablation": {"fusion_type": "concat"}},
    "fusion_sum":    {"ablation": {"fusion_type": "sum"}},
    "head_linear":   {"ablation": {"head_type": "linear"}},
    "head_mlp":      {"ablation": {"head_type": "mlp"}},
    "vmd_off":       {"dataset":  {"vmd": {"enabled": False}}},
    "outlier_off":   {"dataset":  {"cleaning": {"physical_rules": False}}},
}


def _build_ablation(variant: str, base_cfg: dict) -> nn.Module:
    """
    Merge ablation overrides into a deep copy of *base_cfg* and build
    the proposed model with those overrides applied.

    First tries to load a YAML file at ``configs/ablation/<variant>.yaml``
    for explicit overrides; if the file is not found falls back to the
    hard-coded ``_ABLATION_DEFAULTS`` table.

    Parameters
    ----------
    variant : str
        The ablation variant name (the part after ``'ablation:'``).
    base_cfg : dict
        The unmodified run config dict.

    Returns
    -------
    nn.Module
    """
    cfg = copy.deepcopy(base_cfg)

    # ── Try to load YAML overrides ────────────────────────────────────────
    yaml_overrides: dict = {}
    try:
        import os
        import yaml  # PyYAML; present in both environments

        # Resolve path relative to the repo root (two levels up from baselines/).
        _here = os.path.dirname(os.path.abspath(__file__))
        _repo_root = os.path.dirname(_here)
        ablation_path = os.path.join(_repo_root, "configs", "ablation", f"{variant}.yaml")

        if os.path.isfile(ablation_path):
            with open(ablation_path, "r", encoding="utf-8") as fh:
                yaml_overrides = yaml.safe_load(fh) or {}
    except Exception:  # pragma: no cover – optional dependency missing
        pass

    # ── Fall back to hard-coded defaults if YAML had nothing useful ───────
    if not yaml_overrides and variant in _ABLATION_DEFAULTS:
        yaml_overrides = _ABLATION_DEFAULTS[variant]

    if not yaml_overrides:
        known = ", ".join(sorted(_ABLATION_DEFAULTS.keys()))
        raise KeyError(
            f"Unknown ablation variant '{variant}'.  "
            f"Known variants: {known}."
        )

    # ── Deep merge: yaml_overrides on top of cfg ──────────────────────────
    _deep_merge(cfg, yaml_overrides)

    return build_proposed(cfg)


def _deep_merge(base: dict, overrides: dict) -> None:
    """Recursively merge *overrides* into *base* in-place."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
