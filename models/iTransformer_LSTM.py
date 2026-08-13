"""
iTransformer_LSTM with ablation switches.

Target-channel handling (n_target_channels)
--------------------------------------------
When VMD decomposition is enabled, the target branch receives
``1 + K`` channels (Patv + K IMFs) instead of a single channel.  These are
fed to the iTransformer branch as ``n_target_channels`` independent
*variates* (``num_variates=n_target_channels``), exactly as the iTransformer
paper intends: each IMF gets its own token and attends to every other IMF
and to Patv through the variate-attention layers.  After the iTransformer
stack, only the Patv token (index 0) is kept and passed on to fusion — it
now carries information attended over all IMFs, without ever being
collapsed through an untrained linear "bottleneck".

(Earlier iteration used a randomly-initialised ``nn.Linear(n_target, 1)``
to compress all target channels into one scalar *before* the iTransformer
branch even saw them.  That projection's output scale/orientation was
essentially arbitrary at initialisation and drove large seed-to-seed
variance; it has been removed.)

When VMD is disabled, ``n_target_channels=1`` and this reduces exactly to
the original single-variate behaviour.

Ablation parameters
-------------------
use_itransformer : bool (default True)
    When False, remove the iTransformer branch and replace it with a 1×1
    projection on the LSTM output so the downstream fusion/head sees the
    same (B, 1, dim_embed) shape.

use_lstm : bool (default True)
    When False, remove the LSTM branch and replace it with a 1×1 projection
    on the iTransformer output so the cross-attention key/value tensors keep
    the same (B, L, dim_lstm) shape expected by CrossAttention.

fusion_type : str  {'cross_attention', 'concat', 'sum', 'gated'}  (default 'cross_attention')
    - 'cross_attention': original CrossAttention(x1, x2) fusion.
    - 'concat': concatenate x1 and x2 along the last dim, then project back
      to dim_embed so the head receives the same shape.
    - 'sum': element-wise sum of x1 and x2 (both must already be dim_embed).
    - 'gated': adaptive soft-gating fusion (a la SCGF in Zhang et al.,
      "A wind power forecasting method based on GWS-STNet", Energy 2026,
      and the learnable gated fusion in WD-SGformer, Energy 2025). The two
      branch outputs are concatenated and passed through a small FC +
      Softmax to produce per-sample fusion weights (alpha_itrans,
      alpha_lstm), which are then used for a weighted combination
      alpha_itrans * x1 + alpha_lstm * x2. Unlike a fixed CrossAttention
      structure, the weights are input-dependent and let the model decide,
      per sample, how much to rely on each branch — this directly targets
      the seed-to-seed instability observed with fixed cross_attention
      fusion on low-channel-count datasets.

head_type : str  {'kan', 'linear', 'mlp'}  (default 'kan')
    - 'kan': original KAN([dim_embed, length_pre]) head.
    - 'linear': single nn.Linear(dim_embed, length_pre).
    - 'mlp': two-layer MLP: Linear(dim_embed, dim_embed) → ReLU →
             Linear(dim_embed, length_pre).

Backward compatibility
----------------------
Calling  iTransformer_LSTM(input_size=5, length_pre=1, …)  with no ablation
arguments reproduces the original model exactly.
"""

from models import iTransformer_block, CrossAttention
import torch
import torch.nn as nn
from models.KAN import KAN


class iTransformer_LSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        length_pre: int = 1,
        dim_lstm: int = 128,
        depth_lstm: int = 3,
        length_input: int = 48,
        dim_embed: int = 128,
        depth: int = 4,
        heads: int = 6,
        n_target_channels: int = 1,
        # ── ablation switches ────────────────────────────────────────────
        use_itransformer: bool = True,
        use_lstm: bool = True,
        fusion_type: str = "cross_attention",   # 'cross_attention'|'concat'|'sum'
        head_type: str = "kan",                 # 'kan'|'linear'|'mlp'
    ):
        super(iTransformer_LSTM, self).__init__()

        assert fusion_type in ("cross_attention", "concat", "sum", "gated"), (
            f"fusion_type must be one of 'cross_attention', 'concat', 'sum', "
            f"'gated'; got '{fusion_type}'"
        )
        assert head_type in ("kan", "linear", "mlp"), (
            f"head_type must be one of 'kan', 'linear', 'mlp'; got '{head_type}'"
        )

        self.use_itransformer = use_itransformer
        self.use_lstm = use_lstm
        self.fusion_type = fusion_type
        self.head_type = head_type
        self.length_pre = length_pre
        self.dim_embed = dim_embed
        self.dim_lstm = dim_lstm
        self.n_target_channels = n_target_channels

        # ── Branch A: iTransformer (target variate branch) ───────────────
        # Each target channel (Patv + K IMFs, when VMD is on) is treated as
        # an independent variate.  The variate-attention layers let every
        # IMF attend to Patv and to every other IMF; only the Patv token
        # (index 0) is kept afterwards and passed on to fusion.  When VMD
        # is off, n_target_channels == 1 and this is exactly the original
        # single-variate model.
        if use_itransformer:
            self.model1 = iTransformer_block(
                num_variates=n_target_channels,
                lookback_len=length_input,
                pred_length=length_pre,
                dim=dim_embed,
                depth=depth,
                heads=heads,
                num_tokens_per_variate=1,
                use_reversible_instance_norm=True,
            )
        else:
            # When iTransformer is off: project LSTM output (B, L, dim_lstm)
            # → (B, 1, dim_embed) so the fusion stage sees the same shape.
            # This 1×1 projection collapses the time dimension to a single token.
            self.itrans_proj = nn.Linear(dim_lstm, dim_embed)

        # ── Branch B: LSTM (covariate branch) ────────────────────────────
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=input_size - n_target_channels,
                hidden_size=dim_lstm,
                num_layers=depth_lstm,
                batch_first=True,
                bidirectional=False,
            )
        else:
            # When LSTM is off: project iTransformer output (B, 1, dim_embed)
            # → (B, L, dim_lstm) so CrossAttention sees the same key/value shape.
            # We expand across the time dimension using a learnable projection.
            self.lstm_proj = nn.Linear(dim_embed, dim_lstm)
            self._length_input = length_input  # needed for shape expansion

        # ── Fusion layer ─────────────────────────────────────────────────
        if fusion_type == "cross_attention":
            # Original: query=x1 (B,1,dim_embed), key/value from x2 (B,L,dim_lstm)
            self.cross = CrossAttention(dim=dim_embed, lenth=dim_lstm)

        elif fusion_type == "concat":
            # x1: (B, 1, dim_embed)  x2: (B, 1, dim_embed) after projection
            # concat → (B, 1, dim_embed*2), project back to (B, 1, dim_embed)
            # For concat we first need x2 at the same token dimension as x1.
            # We take the last hidden state of LSTM: x2[:, -1:, :] (B,1,dim_lstm)
            # then project to dim_embed.
            self.concat_x2_proj = nn.Linear(dim_lstm, dim_embed)
            self.concat_fusion_proj = nn.Linear(dim_embed * 2, dim_embed)

        elif fusion_type == "sum":
            # x1: (B, 1, dim_embed)  x2 last hidden: (B, 1, dim_lstm) → dim_embed
            self.sum_x2_proj = nn.Linear(dim_lstm, dim_embed)

        elif fusion_type == "gated":
            # Project x2's last token to dim_embed (same as 'sum'/'concat'),
            # then compute per-sample softmax gate weights from the
            # concatenation of both (already-aligned) branch summaries.
            self.gated_x2_proj = nn.Linear(dim_lstm, dim_embed)
            self.gate_fc = nn.Linear(dim_embed * 2, 2)

        # ── Prediction head ──────────────────────────────────────────────
        if head_type == "kan":
            self.head = KAN([dim_embed, length_pre])
        elif head_type == "linear":
            self.head = nn.Linear(dim_embed, length_pre)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(dim_embed, dim_embed),
                nn.ReLU(),
                nn.Linear(dim_embed, length_pre),
            )

    # ─────────────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, L, F_in)
            Channels 0 .. n_target_channels-1 are the target variates
            (Patv, and K IMFs when VMD is on).  Remaining channels are the
            covariates fed to the LSTM branch.

        Returns
        -------
        Tensor, shape (B, length_pre)
        """
        n_tgt = self.n_target_channels

        # ── Branch B: LSTM on covariates ─────────────────────────────────
        if self.use_lstm:
            x2, _ = self.lstm(x[:, :, n_tgt:])   # (B, L, dim_lstm)
        # else: x2 will be synthesised after branch A runs.

        # ── Branch A: iTransformer on target variate(s) ───────────────────
        if self.use_itransformer:
            x1_all = self.model1(x[:, :, :n_tgt])   # (B, n_tgt, dim_embed)
            x1 = x1_all[:, 0:1, :]                   # keep Patv token only
        # else: x1 will be synthesised below.

        # ── Handle disabled branches (1×1 projection fallback) ───────────
        if not self.use_itransformer:
            # x2 is available; collapse time → (B, 1, dim_embed)
            # Use last hidden state of LSTM as the single token.
            x1 = self.itrans_proj(x2[:, -1:, :])   # (B, 1, dim_embed)

        if not self.use_lstm:
            # x1 is available (B, 1, dim_embed); expand to (B, L, dim_lstm)
            x2_token = self.lstm_proj(x1)           # (B, 1, dim_lstm)
            x2 = x2_token.expand(-1, self._length_input, -1)  # (B, L, dim_lstm)

        # ── Fusion ───────────────────────────────────────────────────────
        if self.fusion_type == "cross_attention":
            fused = self.cross(x1, x2)   # (B, 1, dim_embed)

        elif self.fusion_type == "concat":
            # Use last token of x2 as a summary vector
            x2_last = self.concat_x2_proj(x2[:, -1:, :])   # (B, 1, dim_embed)
            fused = self.concat_fusion_proj(
                torch.cat([x1, x2_last], dim=-1)             # (B, 1, dim_embed*2)
            )                                                 # (B, 1, dim_embed)

        elif self.fusion_type == "sum":
            x2_last = self.sum_x2_proj(x2[:, -1:, :])       # (B, 1, dim_embed)
            fused = x1 + x2_last                             # (B, 1, dim_embed)

        elif self.fusion_type == "gated":
            # Align x2 to dim_embed, then derive input-dependent gate
            # weights (alpha_itrans, alpha_lstm) via FC + Softmax on the
            # concatenated branch summaries (both are already single tokens
            # of shape (B, 1, dim_embed), so no pooling is needed).
            x2_last = self.gated_x2_proj(x2[:, -1:, :])      # (B, 1, dim_embed)
            gate_logits = self.gate_fc(
                torch.cat([x1, x2_last], dim=-1)              # (B, 1, dim_embed*2)
            )                                                  # (B, 1, 2)
            alpha = torch.softmax(gate_logits, dim=-1)         # (B, 1, 2)
            alpha_itrans = alpha[..., 0:1]                     # (B, 1, 1)
            alpha_lstm = alpha[..., 1:2]                       # (B, 1, 1)
            fused = alpha_itrans * x1 + alpha_lstm * x2_last   # (B, 1, dim_embed)

        # ── Prediction head ──────────────────────────────────────────────
        output = self.head(fused)    # (B, 1, length_pre)
        return output[:, 0, :]       # (B, length_pre)
