"""
UnifiedProposedModel — thin wrapper around iTransformer_LSTM.

Exposes a clean, config-friendly interface for the proposed model and all
six ablation variants.  The heavy lifting (branch logic, fusion, head) is
done entirely by iTransformer_LSTM; this class only translates the
high-level channel layout description into the parameters that
iTransformer_LSTM expects.

Channel layout convention (matches data_pipeline/windowing.py):
    x[:, :, :n_target_channels]   → target branch (iTransformer)
        vmd_off: 1 channel  (Patv)
        vmd_on : 1+K channels (Patv + K IMFs)
    x[:, :, n_target_channels:]   → covariate branch (LSTM)
        always 4 channels (Wspd, Wdir, Etmp, Itmp)

Mapping to iTransformer_LSTM constructor:
    input_size   = n_target_channels + n_covariate_channels
    length_input = lookback
    length_pre   = horizon
    depth        = depth_itrans
    heads        = heads_itrans

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.iTransformer_LSTM import iTransformer_LSTM


class UnifiedProposedModel(nn.Module):
    """
    Proposed model (and all ablation variants) expressed in terms of the
    data-pipeline channel layout rather than the raw ``iTransformer_LSTM``
    parameter names.

    Parameters
    ----------
    lookback : int
        Length of the input window (time steps).
    horizon : int
        Number of future time steps to predict (= ``length_pre``).
    n_target_channels : int
        Number of channels fed to the iTransformer branch.
        1 when VMD is off, 1+K when VMD is on.
    n_covariate_channels : int
        Number of channels fed to the LSTM branch (default 4: Wspd, Wdir,
        Etmp, Itmp).
    use_itransformer : bool
        Enable the iTransformer (target) branch.  Set False for the
        ``itrans_off`` ablation.
    use_lstm : bool
        Enable the LSTM (covariate) branch.  Set False for the
        ``lstm_off`` ablation.
    fusion_type : str
        One of ``'cross_attention'`` (default), ``'concat'``, ``'sum'``.
    head_type : str
        One of ``'kan'`` (default), ``'linear'``, ``'mlp'``.
    dim_embed : int
        Embedding dimension for the iTransformer branch and fusion layer.
    depth_itrans : int
        Number of iTransformer layers (``depth`` in iTransformer_block).
    heads_itrans : int
        Number of attention heads in the iTransformer.
    dim_lstm : int
        Hidden size of the LSTM.
    depth_lstm : int
        Number of LSTM layers.
    """

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        n_target_channels: int,
        n_covariate_channels: int = 4,
        use_itransformer: bool = True,
        use_lstm: bool = True,
        fusion_type: str = "gated",
        head_type: str = "kan",
        dim_embed: int = 128,
        depth_itrans: int = 4,
        heads_itrans: int = 6,
        dim_lstm: int = 128,
        depth_lstm: int = 3,
    ) -> None:
        super().__init__()

        self._n_target = n_target_channels
        self._n_covariate = n_covariate_channels

        # iTransformer_LSTM treats the first n_target_channels columns as
        # independent *variates* fed to the iTransformer branch (Patv plus
        # K IMFs when VMD is on) and the remaining n_covariate_channels as
        # the LSTM branch input.  Each target channel gets its own variate
        # token; the iTransformer's variate-attention layers let IMFs and
        # Patv attend to each other, and only the Patv token is carried
        # forward to fusion.  This avoids collapsing multiple target
        # channels through an untrained linear bottleneck before the model
        # has a chance to use them.

        self._inner = iTransformer_LSTM(
            input_size=n_target_channels + n_covariate_channels,
            length_pre=horizon,
            dim_lstm=dim_lstm,
            depth_lstm=depth_lstm,
            length_input=lookback,
            dim_embed=dim_embed,
            depth=depth_itrans,
            heads=heads_itrans,
            n_target_channels=n_target_channels,
            use_itransformer=use_itransformer,
            use_lstm=use_lstm,
            fusion_type=fusion_type,
            head_type=head_type,
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, lookback, n_target_channels + n_covariate_channels)
            Channels are laid out as:
                0 .. n_target_channels-1   : target + IMFs
                n_target_channels ..       : covariates (Wspd, Wdir, Etmp, Itmp)

        Returns
        -------
        Tensor, shape (B, horizon)  — normalised scale.
        """
        # Target channels are passed through unchanged; iTransformer_LSTM
        # treats them as n_target_channels independent variates internally.
        return self._inner(x)  # (B, horizon)

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "UnifiedProposedModel":
        """
        Build a ``UnifiedProposedModel`` from a flat config dict.

        The dict may contain any subset of the constructor keyword arguments;
        unrecognised keys are silently ignored so that a full merged
        experiment YAML can be passed directly.

        Expected keys (all optional, fall back to constructor defaults):
            lookback, horizon, n_target_channels, n_covariate_channels,
            use_itransformer, use_lstm, fusion_type, head_type,
            dim_embed, depth_itrans, heads_itrans, dim_lstm, depth_lstm

        Raises
        ------
        KeyError
            If ``lookback``, ``horizon``, or ``n_target_channels`` are absent
            (these have no sensible defaults).
        """
        return cls(
            lookback=cfg["lookback"],
            horizon=cfg["horizon"],
            n_target_channels=cfg["n_target_channels"],
            n_covariate_channels=cfg.get("n_covariate_channels", 4),
            use_itransformer=cfg.get("use_itransformer", True),
            use_lstm=cfg.get("use_lstm", True),
            fusion_type=cfg.get("fusion_type", "gated"),
            head_type=cfg.get("head_type", "kan"),
            dim_embed=cfg.get("dim_embed", 128),
            depth_itrans=cfg.get("depth_itrans", 4),
            heads_itrans=cfg.get("heads_itrans", 6),
            dim_lstm=cfg.get("dim_lstm", 128),
            depth_lstm=cfg.get("depth_lstm", 3),
        )
