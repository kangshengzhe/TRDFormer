"""
baselines/tsl_adapter.py

Wraps TSL model classes (copied into models/tsl/) so they satisfy our
unified forward contract:

    x:       (B, lookback, 5)   – 5 raw features, no IMF channels
    returns: (B, horizon)       – predicted Patv (column 0)

The TSL model files live in models/tsl/ inside this repo, so there are
no sys.path tricks or external directory dependencies at runtime.

Requirements: 4.1, 4.2, 4.3
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn


class TSLForecastWrapper(nn.Module):
    """Wrap a TSL Model class to satisfy our forward contract.

    Parameters
    ----------
    tsl_model_name : str
        Name matching the file stem under models/tsl/
        (e.g. 'Transformer', 'DLinear', 'iTransformer').
    configs : argparse.Namespace
        Must contain at minimum: task_name, seq_len, pred_len, label_len,
        enc_in, dec_in, c_out.  Build via tsl_configs.make_tsl_configs().
    """

    def __init__(self, tsl_model_name: str, configs: Namespace) -> None:
        super().__init__()

        # Import the Model class from our local copy in models/tsl/
        try:
            import importlib
            module = importlib.import_module(f"models.tsl.{tsl_model_name}")
        except ImportError as exc:
            raise ImportError(
                f"Could not import models/tsl/{tsl_model_name}.py — "
                f"make sure the file was copied from Time-Series-Library. "
                f"Original error: {exc}"
            ) from exc

        if not hasattr(module, "Model"):
            raise AttributeError(
                f"models/tsl/{tsl_model_name}.py does not expose a 'Model' class."
            )
        ModelClass = module.Model

        # Validate required config attributes
        for attr in ("task_name", "seq_len", "pred_len", "label_len",
                     "enc_in", "dec_in", "c_out"):
            if not hasattr(configs, attr):
                raise ValueError(
                    f"configs is missing required attribute '{attr}'. "
                    "Build configs via baselines.tsl_configs.make_tsl_configs()."
                )

        self.tsl_model_name: str = tsl_model_name
        self.seq_len: int    = configs.seq_len
        self.pred_len: int   = configs.pred_len
        self.label_len: int  = configs.label_len
        self.enc_in: int     = configs.enc_in

        # Time-mark dimension based on freq / embed type
        _freq_to_mark_dim = {
            't': 5, 'h': 4, 's': 6,
            'd': 3, 'b': 3, 'w': 2, 'm': 1, 'a': 1,
        }
        _embed = getattr(configs, 'embed', 'timeF')
        _freq  = getattr(configs, 'freq',  'h')
        self._mark_dim: int = (
            _freq_to_mark_dim.get(_freq, 4) if _embed == 'timeF' else 5
        )

        self.model: nn.Module = ModelClass(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, lookback, 5)  – 5 raw features [Patv, Wspd, Wdir, Etmp, Itmp]
        returns : (B, horizon) – predicted Patv, normalised scale
        """
        B, lookback, n_feat = x.shape
        device, dtype = x.device, x.dtype

        # TimeXer (features='MS') treats the LAST column of x_enc as the
        # endogenous (target) variable and all preceding columns as exogenous.
        # Our column order is [Patv(0), Wspd, Wdir, Etmp, Itmp(4)], so Patv
        # must be moved to the last position before passing to TimeXer.
        if self.tsl_model_name == "TimeXer":
            # [Wspd, Wdir, Etmp, Itmp, Patv]
            x = torch.cat([x[:, :, 1:], x[:, :, :1]], dim=-1)

        x_mark_enc = torch.zeros(B, lookback,
                                 self._mark_dim, device=device, dtype=dtype)
        x_dec_in   = torch.zeros(B, self.label_len + self.pred_len,
                                 n_feat, device=device, dtype=dtype)
        x_mark_dec = torch.zeros(B, self.label_len + self.pred_len,
                                 self._mark_dim, device=device, dtype=dtype)

        out = self.model(x, x_mark_enc, x_dec_in, x_mark_dec)
        # out: (B, pred_len, c_out)
        # For TimeXer (MS mode): the single output channel corresponds to Patv
        # (the endogenous target), which is now at the last input position.
        # For all other models: slice column 0 (Patv in original order).
        if self.tsl_model_name == "TimeXer":
            return out[:, -self.pred_len:, 0]  # only 1 output channel in MS mode
        return out[:, -self.pred_len:, 0]

    def __repr__(self) -> str:
        return (
            f"TSLForecastWrapper(model={self.tsl_model_name}, "
            f"seq_len={self.seq_len}, pred_len={self.pred_len})"
        )
