"""Simplified template classes for new PK models.

These classes are stripped of VAE-specific attributes and use only a single loss.
Based on ContextVAEPK but simplified for general use.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torchtyping import TensorType

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import (
    NewBasePKModel,
)
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)
from pff.models.utils.loss_utils import MultiHeadLoss


class FlowMatchingForwardOutputs(AbstractForwardOutputs):
    """Simplified forward outputs collector for Flow Matching PK models.
    Contains minimal schema with single reconstruction head and single RMSE loss.
    """

    HEAD_SCHEMAS = {"reconstruction": ["prediction", "target", "mask"]}
    LOSS_SCHEMAS = {"reconstruction": ["rmse"]}  # Single RMSE loss

    def __init__(
        self,
        *,
        loss_multihead: Optional[Callable[[List[torch.Tensor]], Tuple[torch.Tensor, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the simplified output container."""
        super().__init__(loss_multihead=loss_multihead, device=device)
        self.stats: Optional[TensorType["B", 1]] = None

    def _compute_total_loss(self, flat_losses: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Compute total loss - simplified to single RMSE loss."""
        if not flat_losses:
            return None
        if self.loss_multihead is None:
            return super()._compute_total_loss(flat_losses)

        # Single loss case - just the RMSE
        device = self._infer_device()
        zero = torch.zeros((), device=device)
        rmse_loss = flat_losses.get("rmse", zero)

        # For compatibility with MultiHeadLoss expecting a list
        total_loss, _ = self.loss_multihead([rmse_loss])
        return total_loss


class FlowMatchingPK(NewBasePKModel):
    """Flow Matching PK model with single RMSE loss.

    Simplified PK model for flow matching with:
    - Basic encoder-decoder architecture from BasePKModel
    - Study-level aggregation for context information
    - Single RMSE loss computation
    - Flow matching interpolation path
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        z_dim = self.encoder.zi_latent_dim
        self.mu_s_layer = nn.Linear(z_dim, z_dim)
        self.logvar_s_layer = nn.Linear(z_dim, z_dim)
        self.aggregator = MeanStudyAggregator()
        self.loss_multihead = MultiHeadLoss(mode="learnable", number_of_losses=1)
        self.sigma = 0.1

    def forward(
        self, databatch_list: Sequence[AICMECompartmentsDataBatch]
    ) -> FlowMatchingForwardOutputs:
        """Run forward passes over every permutation and aggregate results."""

        outputs_collector = FlowMatchingForwardOutputs(loss_multihead=self.loss_multihead)

        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    def _reshape_time_like(self, t, state):
        if isinstance(t, (float, int)):
            return t
        else:
            return t.reshape(-1, *([1] * (state.ndim - 1)))

    def _study_latent(
        self, db: AICMECompartmentsDataBatch, use_target: bool = False
    ) -> Tuple[TensorType["B", "Z"], TensorType["B", 1]]:
        """Encode study observations into latent z_s."""
        if use_target:
            X_raw, T_raw, M = self._include_target(db)
            if db.mask_individuals is not None:
                B = db.mask_individuals.size(0)
                I_t = db.target_obs.shape[1]
                mask_ind = torch.cat(
                    [
                        db.mask_individuals,
                        torch.ones(B, I_t, dtype=torch.bool, device=db.mask_individuals.device),
                    ],
                    dim=1,
                )
            else:
                mask_ind = None
        else:
            X_raw, T_raw, M = db.context_obs, db.context_obs_time, db.context_obs_mask
            mask_ind = db.mask_individuals

        # Scaling statistics from context only
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)
        X_s, T_s = self.scaler.forward(X_raw, T_raw, stats)

        if use_target:
            dose = torch.cat([db.context_dosing_amounts, db.target_dosing_amounts], dim=1)
            route = torch.cat([db.context_dosing_route_types, db.target_dosing_route_types], dim=1)
        else:
            dose = db.context_dosing_amounts
            route = db.context_dosing_route_types

        z_ci = self.encoder(X_s, T_s, M, dose, route, mask_ind)
        z_s_agg = self.aggregator(z_ci, mask_ind)

        mu_s = self.mu_s_layer(z_s_agg)
        logvar_s = self.logvar_s_layer(z_s_agg)

        if self.model_config.network.study_latent_deterministic:
            z_s = mu_s
        else:
            eps = torch.randn_like(mu_s)
            z_s = mu_s + eps * torch.exp(0.5 * logvar_s)

        return z_s, stats

    def _include_target(
        self, db: AICMECompartmentsDataBatch
    ) -> Tuple[
        TensorType["B", "Ic+It", "T", 1],
        TensorType["B", "Ic+It", "T", 1],
        TensorType["B", "Ic+It", "T"],
    ]:
        """Concatenate context and target observations."""
        X = torch.cat([db.context_obs, db.target_obs], dim=1)  # [B, Ic+It, T, 1]
        T = torch.cat([db.context_obs_time, db.target_obs_time], dim=1)  # [B, Ic+It, T, 1]
        M = torch.cat([db.context_obs_mask, db.target_obs_mask], dim=1)  # [B, Ic+It, T]
        return X, T, M

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> FlowMatchingForwardOutputs:
        """Flow matching reconstruction with RMSE loss."""

        # Context encoding
        z_s, stats = self._study_latent(db, use_target=False)

        # Target data
        X1_raw, T_raw, M = db.target_obs, db.target_obs_time, db.target_obs_mask
        X1, T = self.scaler.forward(X1_raw, T_raw, stats)
        B = X1_raw.shape[0]

        outputs = FlowMatchingForwardOutputs(
            loss_multihead=self.loss_multihead,
            device=X1_raw.device,
        )
        outputs.stats = stats

        # Flow matching: sample interpolation path
        eps = torch.randn_like(X1)  # Noise
        X0 = torch.randn_like(X1)  # Source
        tau = torch.rand((B,))  # Time ~ U[0,1]
        tau_ = self._reshape_time_like(tau, state=X1)
        Xtau = tau_ * X1 + (1 - tau_) * X0
        Xtau += self.sigma * eps

        # Target vector field (true direction from source to target)
        utau = X1 - X0
        vtau = Xtau  # TODO: Replace with actual vector field network

        rmse_dict = self.masked_rmse_loss(vtau, utau, M)

        # Store outputs
        outputs.update_head(
            "reconstruction",
            {
                "prediction": vtau,
                "target": utau,
                "mask": M,
            },
        )

        outputs.update_losses(
            "reconstruction",
            {
                "rmse": rmse_dict["rmse"],  # Single RMSE loss
            },
        )

        return outputs


__all__ = ["FlowMatchingForwardOutputs", "FlowMatchingPK"]
