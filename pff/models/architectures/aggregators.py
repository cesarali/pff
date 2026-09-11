# -----------------------------------------------------------------------------
# Aggregators
# -----------------------------------------------------------------------------

from typing import Annotated

import torch
from torch import Tensor, nn
from torchtyping import TensorType


# ────────────────────────────────────────────────────────────────────────────────
#  Mean‑pool aggregator  ( ⨑_i z_ci )
# ────────────────────────────────────────────────────────────────────────────────
class MeanStudyAggregator(nn.Module):
    """Simple mean‑pool over individuals, robust to masking."""

    def forward(
        self,
        z_ci: Annotated[Tensor, TensorType["B", "I", "Z"]],  # [B, I, Z]
        mask_individuals: Annotated[Tensor, TensorType["B", "I"]] | None = None,
    ) -> Annotated[Tensor, TensorType["B", "Z"]]:
        if mask_individuals is None:
            return z_ci.mean(dim=1)  # no mask

        mask = mask_individuals.bool().unsqueeze(-1)  # [B, I, 1]
        masked_sum = (z_ci * mask).sum(dim=1)  # [B, Z]
        denom = mask.sum(dim=1).clamp(min=1)  # avoid ÷0
        return masked_sum / denom  # [B, Z]


# ────────────────────────────────────────────────────────────────────────────────
#  Attention aggregator  (global learned query)
# ────────────────────────────────────────────────────────────────────────────────
class AttentionStudyAggregator(nn.Module):
    """
    Aggregate {z_ci}_i with a learned global query.
    NaN‑safe: fully‑masked rows are skipped and set to zero (or to a learned
    default if you prefer – see comment below).
    """

    def __init__(self, latent_dim: int, num_heads: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))  # [1,1,Z]
        self.attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        z_ci: Annotated[Tensor, TensorType["B", "I", "Z"]],  # [B, I, Z]
        mask_individuals: Annotated[Tensor, TensorType["B", "I"]] | None = None,
    ) -> Annotated[Tensor, TensorType["B", "Z"]]:
        B, I, Z = z_ci.shape
        device = z_ci.device

        # ── 1.  Determine which batch rows have ≥ 1 valid individual
        if mask_individuals is None:
            valid_batch = torch.ones(B, dtype=torch.bool, device=device)
            key_padding_mask = None
        else:
            mask = mask_individuals.bool()  # [B, I]
            valid_batch = mask.any(dim=1)  # [B]
            key_padding_mask = ~mask

        # ── 2.  Allocate output tensor (zero for “empty” studies)
        z_out = z_ci.new_zeros(B, Z)  # default

        # ── 3.  Run attention only on the valid rows
        if valid_batch.any():
            q = self.query.expand(valid_batch.sum(), 1, Z)  # [n,1,Z] # type: ignore
            z_valid, _ = self.attn(
                q,
                z_ci[valid_batch],  # key / value
                z_ci[valid_batch],
                key_padding_mask=None
                if key_padding_mask is None
                else key_padding_mask[valid_batch],
            )
            z_out[valid_batch] = z_valid.squeeze(1)  # [n,Z]

        # If you prefer a learned fallback instead of zero:
        # z_out[~valid_batch] = self.empty_token

        return z_out  # [B, Z]
