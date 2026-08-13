"""
baselines/tsl_configs.py

Converts our merged YAML config dict (or sub-dicts) to an
``argparse.Namespace`` suitable for any Time-Series-Library (TSL) model.

The TSL models read their hyper-parameters directly from a Namespace object
that is conventionally populated by argparse in the TSL training scripts.
Rather than reproducing that argparse boilerplate, this module maps our
YAML config structure to the same Namespace, providing sensible defaults for
every key so that missing YAML entries never cause an AttributeError inside a
TSL model's __init__ or forward.

Public API
----------
make_tsl_configs(model_cfg, dataset_cfg) -> argparse.Namespace
    Build a TSL-compatible Namespace from our model and dataset config dicts.

Requirements: 4.1, 4.5
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any


# ---------------------------------------------------------------------------
# Default values for every TSL config key we are aware of.
# These are the safe, broadly-applicable defaults; model-specific YAMLs
# can override any of them through the model_cfg argument.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    # ── Task / problem shape ─────────────────────────────────────────────
    "task_name": "long_term_forecast",
    "is_training": 1,
    "model_id": "wind",
    "model": "Transformer",          # overridden by registry factories

    # ── Data ─────────────────────────────────────────────────────────────
    "data": "custom",
    "root_path": "./data",
    "data_path": "sdwpf_turb1_cleaned_final.csv",
    "features": "MS",                # multivariate → single-step target
    "target": "Patv",
    "freq": "t",                     # 't' = minute-level (10-min SCADA)
    "checkpoints": "./model_save/wind/",

    # ── Sequence lengths (set by make_tsl_configs from dataset_cfg) ──────
    "seq_len": 144,
    "label_len": 72,
    "pred_len": 6,

    # ── Input / output channels (always 5 raw features for baselines) ────
    "enc_in": 5,
    "dec_in": 5,
    "c_out": 5,

    # ── Model architecture ───────────────────────────────────────────────
    "d_model": 128,
    "d_ff": 256,
    "e_layers": 2,
    "d_layers": 1,
    "n_heads": 8,
    "factor": 3,
    "moving_avg": 25,
    "distil": True,
    "dropout": 0.1,
    "embed": "timeF",
    "activation": "gelu",
    "output_attention": False,
    "channel_independence": 0,       # iTransformer: 0 = channel-mixing (default)

    # ── PatchTST specific ────────────────────────────────────────────────
    "patch_len": 16,
    "stride": 8,

    # ── TimesNet specific ────────────────────────────────────────────────
    "top_k": 5,
    "num_kernels": 6,

    # ── FEDformer specific ───────────────────────────────────────────────
    "modes": 64,
    "mode_select": "random",
    "version": "Fourier",

    # ── Training (not normally used inside the model, but some models
    #    reference batch_size for internal init) ──────────────────────────
    "batch_size": 128,
    "learning_rate": 1e-4,
    "num_workers": 0,
    "use_amp": False,
    "use_gpu": True,
    "gpu": 0,
    "use_multi_gpu": False,
    "devices": "0",

    # ── Nonstationary Transformer specific ──────────────────────────────
    # p_hidden_dims / p_hidden_layers: projector MLP that de-stationaries
    # the series.  Two hidden layers of width 256 is the paper default.
    "p_hidden_dims": [256, 256],
    "p_hidden_layers": 2,

    # ── TimeXer specific ─────────────────────────────────────────────────
    # use_norm: apply instance normalisation inside the model (recommended)
    "use_norm": True,

    # ── Misc ─────────────────────────────────────────────────────────────
    "des": "Exp",
    "itr": 1,
    "patience": 10,
    "inverse": False,
    "do_predict": False,
    "individual": False,             # DLinear: channel-independent heads
    "revin": 1,                      # PatchTST: use RevIN by default
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "class_strategy": "projection",
    "target_root_path": "./data",
    "target_data_path": "sdwpf_turb1_cleaned_final.csv",
    "num_class": 1,                  # classification tasks (unused here)
}


def make_tsl_configs(model_cfg: dict, dataset_cfg: dict) -> Namespace:
    """
    Build an ``argparse.Namespace`` suitable for any TSL model.

    Parameters
    ----------
    model_cfg : dict
        The ``model`` sub-dict from our merged YAML config.  Recognised keys
        (all optional — defaults are used when absent):

        d_model, d_ff, e_layers, d_layers, n_heads, factor, moving_avg,
        dropout, embed, freq, activation,
        patch_len, stride,          # PatchTST
        top_k, num_kernels,         # TimesNet
        modes, mode_select, version # FEDformer

    dataset_cfg : dict
        The ``dataset`` sub-dict from our merged YAML config.  The following
        keys are read:

        lookback  (or seq_len)  → Namespace.seq_len
        horizon   (or pred_len) → Namespace.pred_len
        label_len               → Namespace.label_len  (default: lookback // 2)

    Returns
    -------
    argparse.Namespace
        A Namespace containing every key in ``_DEFAULTS``, updated with
        values from ``model_cfg`` and the derived sequence-length keys.

    Notes
    -----
    * ``task_name`` is always set to ``'long_term_forecast'``.
    * ``enc_in``, ``dec_in``, ``c_out`` are always set to 5 (the number of
      raw features fed to baseline models — no VMD IMF channels).
    * ``label_len`` follows the TSL convention of ``seq_len // 2`` when not
      explicitly provided.
    """
    # ------------------------------------------------------------------ #
    # Start from the global defaults and layer in model_cfg overrides.   #
    # ------------------------------------------------------------------ #
    cfg: dict[str, Any] = dict(_DEFAULTS)

    # Keys we accept directly from model_cfg
    _model_keys = {
        "d_model", "d_ff", "e_layers", "d_layers", "n_heads", "factor",
        "moving_avg", "dropout", "embed", "freq", "activation",
        "patch_len", "stride",
        "top_k", "num_kernels",
        "modes", "mode_select", "version",
        "distil", "output_attention", "channel_independence",
        "individual", "revin", "affine", "subtract_last",
        "decomposition", "kernel_size",
        "class_strategy",
        # Nonstationary Transformer
        "p_hidden_dims", "p_hidden_layers",
        # TimeXer
        "use_norm",
    }
    for key in _model_keys:
        if key in model_cfg:
            cfg[key] = model_cfg[key]

    # ------------------------------------------------------------------ #
    # Derive sequence-length fields from dataset_cfg.                    #
    # ------------------------------------------------------------------ #
    # Accept both 'lookback' (our convention) and 'seq_len' (TSL convention).
    lookback: int = int(
        dataset_cfg.get("lookback", dataset_cfg.get("seq_len", cfg["seq_len"]))
    )
    horizon: int = int(
        dataset_cfg.get("horizon", dataset_cfg.get("pred_len", cfg["pred_len"]))
    )
    label_len: int = int(
        dataset_cfg.get("label_len", lookback // 2)
    )

    cfg["seq_len"] = lookback
    cfg["pred_len"] = horizon
    cfg["label_len"] = label_len

    # ------------------------------------------------------------------ #
    # Enforce the fixed channel counts for baselines.                    #
    # ------------------------------------------------------------------ #
    cfg["task_name"] = "long_term_forecast"
    cfg["enc_in"] = 5
    cfg["dec_in"] = 5
    cfg["c_out"] = 5

    return Namespace(**cfg)
