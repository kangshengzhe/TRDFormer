"""
ProposedModelV2 — Enhanced model with:
1. Trend-Residual Decomposition (like DLinear/FiLM)
2. Instance Normalization (RevIN)

Key insight: DLinear is strong because it captures the linear trend directly.
Our deep model is strong at capturing non-linear patterns but weak at trends.
Solution: let a linear branch handle the trend, and the deep model handles
the non-linear residual. This is additive: final_pred = trend_pred + residual_pred.

This approach is well-established:
- DLinear (AAAI 2023): moving_avg decomposition + linear projection
- N-BEATS (ICLR 2020): trend-seasonality-residual decomposition
- FiLM (NeurIPS 2022): frequency-enhanced decomposition
- FITS (ICLR 2024): direct frequency interpolation for trend

Architecture:
    Input x: (B, L, F_in)
        ↓
    [RevIN normalization]
        ↓
    ┌─────────────────────┐    ┌─────────────────────────────────────┐
    │ Trend Branch        │    │ Residual Branch                     │
    │ (simple linear)     │    │ (iTransformer + LSTM + Gated + KAN) │
    │                     │    │                                     │
    │ MA → Linear(L→H)   │    │ x - trend → deep model → residual  │
    └─────────────────────┘    └─────────────────────────────────────┘
        ↓                              ↓
    trend_pred                   residual_pred
        ↓                              ↓
        └──────────── + ───────────────┘
                       ↓
                  final_pred
                       ↓
              [RevIN denormalization]
                       ↓
                  output (B, H)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.iTransformer_LSTM import iTransformer_LSTM


class MovingAvgBlock(nn.Module):
    """Moving average for trend extraction (from DLinear paper)."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        # Use avg_pool1d for efficiency
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1,
                                padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L) — single channel time series
        Returns: (B, L) — trend component (same length via padding)
        """
        # Pad front and back to keep length
        front = x[:, :1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        x_padded = torch.cat([front, x, end], dim=1)  # (B, L + kernel-1)
        # avg_pool1d expects (B, C, L)
        trend = self.avg(x_padded.unsqueeze(1)).squeeze(1)  # (B, L)
        return trend


class RevIN(nn.Module):
    """Reversible Instance Normalization (from RevIN paper, Kim et al. 2022).
    Normalizes each sample independently, stores stats for denormalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        """
        x: (B, L, F) or (B, F)
        mode: 'norm' for normalization, 'denorm' for denormalization
        """
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True)  # (B, 1, F) or (B, 1)
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt()
            x = (x - self._mean) / self._std
            if self.affine and x.dim() == 3:
                x = x * self.weight + self.bias
            return x
        elif mode == "denorm":
            if self.affine and x.dim() == 3:
                x = (x - self.bias) / self.weight
            x = x * self._std + self._mean
            return x
        else:
            raise ValueError(f"Unknown mode '{mode}'")


class ProposedModelV2(nn.Module):
    """
    Enhanced proposed model with trend-residual decomposition.

    The model decomposes the forecasting task into:
    1. Trend prediction: simple linear projection of the moving average trend
    2. Residual prediction: deep model (iTransformer + LSTM + Gated + KAN)
       operating on the de-trended signal

    Final prediction = trend_linear(trend) + deep_model(residual)

    Parameters
    ----------
    lookback : int
        Input window length.
    horizon : int
        Forecast horizon.
    n_target_channels : int
        Number of target channels (1 + K IMFs when DWT is on).
    n_covariate_channels : int
        Number of covariate channels (4: Wspd, Wdir, Etmp, Itmp).
    trend_kernel : int
        Moving average kernel size for trend extraction (default: 25).
    use_revin : bool
        Whether to use reversible instance normalization (default: True).
    use_itransformer, use_lstm, fusion_type, head_type : model switches
    dim_embed, depth_itrans, heads_itrans, dim_lstm, depth_lstm : hyperparams
    """

    def __init__(
        self,
        *,
        lookback: int,
        horizon: int,
        n_target_channels: int,
        n_covariate_channels: int = 4,
        trend_kernel: int = 25,
        use_revin: bool = True,
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
        self._lookback = lookback
        self._horizon = horizon
        self._use_revin = use_revin

        # ── Trend extraction (applied to Patv channel only) ──────────────
        self.trend_extractor = MovingAvgBlock(kernel_size=trend_kernel)
        # Linear projection of trend: (B, lookback) → (B, horizon)
        self.trend_linear = nn.Linear(lookback, horizon)
        # Initialize with small weights to avoid large initial trend predictions
        nn.init.xavier_uniform_(self.trend_linear.weight, gain=0.1)
        nn.init.zeros_(self.trend_linear.bias)

        # ── Optional RevIN (applied to all input channels) ───────────────
        total_channels = n_target_channels + n_covariate_channels
        if use_revin:
            self.revin = RevIN(num_features=total_channels)
        else:
            self.revin = None

        # ── Deep residual branch (original architecture) ─────────────────
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, F_in) where F_in = n_target + n_covariate
            Channel layout: [Patv, IMF1..IMFK, Wspd, Wdir, Etmp, Itmp]

        Returns
        -------
        (B, horizon) — predicted Patv values (normalised scale)
        """
        B, L, F = x.shape

        # ── 1. Instance normalization (RevIN - normalize per sample) ─────
        if self.revin is not None:
            # Store mean/std for denorm later; normalize input
            x = self.revin(x, mode="norm")

        # ── 2. Extract trend from Patv (channel 0) after normalization ───
        patv = x[:, :, 0]  # (B, L)
        trend = self.trend_extractor(patv)  # (B, L)
        trend_pred = self.trend_linear(trend)  # (B, horizon)

        # ── 3. Compute residual input (de-trended Patv, rest unchanged) ──
        # Use in-place subtraction to avoid clone overhead
        x_residual = torch.cat([
            (patv - trend).unsqueeze(-1),  # (B, L, 1) de-trended Patv
            x[:, :, 1:],                    # (B, L, F-1) IMFs + covariates
        ], dim=-1)  # (B, L, F)

        # ── 4. Deep model on residual ────────────────────────────────────
        residual_pred = self._inner(x_residual)  # (B, horizon)

        # ── 5. Combine: final = trend + residual ─────────────────────────
        output = trend_pred + residual_pred

        return output

    @classmethod
    def from_config(cls, cfg: dict) -> "ProposedModelV2":
        """Build from config dict (same interface as UnifiedProposedModel)."""
        return cls(
            lookback=cfg["lookback"],
            horizon=cfg["horizon"],
            n_target_channels=cfg["n_target_channels"],
            n_covariate_channels=cfg.get("n_covariate_channels", 4),
            trend_kernel=cfg.get("trend_kernel", 25),
            use_revin=cfg.get("use_revin", True),
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
