from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
from torchtyping import TensorType, patch_typeguard

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.models.amortized_inference.deprecated.generative_pk import (
    BasePKModel,
    PredictiveMixin,
)
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)

patch_typeguard()


# ──────────────────────────────────────────────────────────────────────
#  Permutation-aware outputs (prediction-only)
# ──────────────────────────────────────────────────────────────────────
from typing import Dict


class PredictionForwardOutputs(AbstractForwardOutputs):
    """
    Container for permutation-level *prediction* heads and losses.

    Heads:
        prediction.mean    : [B, I, Tr, 1] (raw space)
        prediction.logvar  : [B, I, Tr, 1] (scaled space)
        prediction.target  : [B, I, Tr, 1] (raw space)
        prediction.mask    : [B, I, Tr]
    Losses:
        prediction.pred_loss, pred_rmse, pred_log_rmse, pred_r2, pred_log_r2, rmse
    """

    HEAD_SCHEMAS = {"prediction": ["mean", "logvar", "target", "mask"]}
    LOSS_SCHEMAS = {
        "prediction": ["pred_loss", "pred_rmse", "pred_log_rmse", "pred_r2", "pred_log_r2", "rmse"]
    }

    def __init__(
        self,
        *,
        loss_multihead: Optional[Callable[[List[torch.Tensor]], Tuple[torch.Tensor, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        # We ignore any provided loss_multihead and rely on our custom _compute_total_loss
        super().__init__(loss_multihead=None, device=device)

    def _compute_total_loss(
        self,
        flat_losses: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Override the default behaviour (sum of all losses) so that only
        `pred_loss` drives optimisation. All other entries in LOSS_SCHEMAS
        are treated as metrics only.
        """
        if not flat_losses:
            return None
        # Only use pred_loss as the training objective
        return flat_losses.get("pred_loss", None)


# ──────────────────────────────────────────────────────────────────────
#  Prediction-only PK model (uses BasePKModel + PredictiveMixin)
# ──────────────────────────────────────────────────────────────────────
class PredictionPK(BasePKModel, PredictiveMixin):
    """
    Prediction-only Neural-Process style PK model.

    • Uses BasePKModel for encoder/decoder/scaler + shared Lightning hooks.
    • Aggregates permutation outputs like AICMEPK, but only with a prediction head.
    • No custom training/validation hooks or manual logging here.
    """

    def __init__(self, model_config: NodePKExperimentConfig):
        super().__init__(model_config)
        z_dim = self.encoder.zi_latent_dim
        agg = model_config.network.aggregator_type
        if agg == "mean":
            self.aggregator = MeanStudyAggregator()
        elif agg == "attention":
            self.aggregator = AttentionStudyAggregator(
                z_dim, model_config.network.aggregator_num_heads
            )
        else:
            raise ValueError(f"Unknown aggregator_type '{agg}'.")

    # ------------------------------------------------------------------
    # Forward over a list of permutations (matches AICMEPK style)
    # ------------------------------------------------------------------
    def forward(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch] | AICMECompartmentsDataBatch,
        return_forward_report: bool = False,
    ) -> PredictionForwardOutputs:
        # Accept a single batch or a list of permutation-batches
        if isinstance(databatch_list, AICMECompartmentsDataBatch):
            databatch_list = [databatch_list]

        collector = PredictionForwardOutputs()
        for batch in databatch_list:
            outputs = self._forward_prediction(batch)
            collector.add(outputs)
        aggregated = collector.reduce()

        # Attach per-substance metrics so BasePKModel can log them
        if return_forward_report:
            self.forward_report(databatch_list, aggregated)  # sets .per_substance
        return aggregated

    # ------------------------------------------------------------------
    # Internals: one-permutation prediction head
    # ------------------------------------------------------------------
    def _forward_prediction(self, db: AICMECompartmentsDataBatch) -> PredictionForwardOutputs:
        """
        Predict **remainder** for known individuals.

        Training defaults to the ``target_*`` block (mirrors
        ``sample_individual_prediction``) while still computing scaling
        statistics from the context block. If ``target_*`` fields are missing,
        the method falls back to the context remainder for compatibility with
        prediction-only workloads.
        """
        # Prefer target remainder/observations for training; fall back to context for inference-only
        X_obs_raw = db.target_obs if db.target_obs is not None else db.context_obs  # [B, I, To, 1]
        T_obs_raw = (
            db.target_obs_time if db.target_obs_time is not None else db.context_obs_time
        )  # [B, I, To, 1]
        M_obs = (
            db.target_obs_mask if db.target_obs_mask is not None else db.context_obs_mask
        )  # [B, I, To]

        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim
            if db.target_rem_sim is not None
            else db.context_rem_sim,  # [B, I, Tr, 1]
            db.target_rem_sim_time
            if db.target_rem_sim_time is not None
            else db.context_rem_sim_time,  # [B, I, Tr, 1]
            db.target_rem_sim_mask
            if db.target_rem_sim_mask is not None
            else db.context_rem_sim_mask,  # [B, I, Tr]
        )
        dose = (
            db.target_dosing_amounts
            if db.target_dosing_amounts is not None
            else db.context_dosing_amounts
        )  # [B, I, D]
        route = (
            db.target_dosing_route_types
            if db.target_dosing_route_types is not None
            else db.context_dosing_route_types
        )  # [B, I, R]

        # Always take scaling stats from **context** (AICME convention)
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        # Scale inputs
        X_obs_s, T_obs_s = self.scaler.forward(X_obs_raw, T_obs_raw, stats)
        Xrem_s, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)

        # Individual latent from the *observed* block used for prediction
        z_i = self.encoder(X_obs_s, T_obs_s, M_obs, dose, route)  # [B, I, Z]
        z_s = None

        # Init = last valid obs of the observed block
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)

        B, I, _ = z_i.shape
        BI = B * I
        dose_feat = dose.unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)
        route_feat = route.float().unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)
        init_state = init_s.view(BI, 1, -1)
        last_t_feat = last_t_s.view(BI, 1, -1)
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)

        mean_s, logvar_s, H_s, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=last_t_feat,
        )  # [BI, Tr, 1] each

        # Reshape + inverse-scale for heads
        mean_s = mean_s.view(B, I, -1, 1)
        logvar_s = logvar_s.view(B, I, -1, 1).clamp_(-100.0, 100.0)
        mean_raw, _ = self.scaler.inverse(mean_s, Trem_s, stats)

        # Losses in scaled space, like AICMEPK
        loss_dict = self.compute_loss(mean_s, logvar_s, Xrem_s, Mrem, H=H_s)
        rmse = self.masked_rmse_loss(mean_raw, Xrem_raw, Mrem)["rmse"]
        pred_losses = {
            "pred_loss": loss_dict["loss"],
            "pred_rmse": rmse,
            "pred_log_rmse": self.masked_log_rmse_loss(mean_raw, Xrem_raw, Mrem)["rmse"],
            "pred_r2": self.masked_r2_score(mean_raw, Xrem_raw, Mrem),
            "pred_log_r2": self.masked_log_r2_score(mean_raw, Xrem_raw, Mrem),
            "rmse": rmse,
        }

        out = PredictionForwardOutputs(device=mean_s.device)
        out.update_head(
            "prediction",
            {"mean": mean_raw, "logvar": logvar_s, "target": Xrem_raw, "mask": Mrem},
        )
        out.update_losses("prediction", pred_losses)
        return out

    # ------------------------------------------------------------------
    # PredictiveMixin requirement (used by BasePKModel for previews)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ) -> Tuple[
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["B", "It", "Tr", 1],
        TensorType["B", "It", "Tr"],
    ]:
        """
        Draw S predictive trajectories for known individuals.
        Prefers target remainder; falls back to context remainder.
        """
        db = databatch
        X_obs_raw, T_obs_raw, M_obs = db.target_obs, db.target_obs_time, db.target_obs_mask
        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim,
            db.target_rem_sim_time,
            db.target_rem_sim_mask,
        )
        dose, route = db.target_dosing_amounts, db.target_dosing_route_types

        # Context stats (AICME convention)
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        # Individual latent from observed block
        X_obs_s, T_obs_s = self.scaler.forward(X_obs_raw, T_obs_raw, stats)
        z_i = self.encoder(X_obs_s, T_obs_s, M_obs, dose, route)  # [B, I, Z]
        z_s = None  # default deterministic; replace with sampled if you add logvar

        # Init from last observation
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)
        _, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)

        B, I, _ = z_i.shape
        BI = B * I
        dose_feat = dose.unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)
        route_feat = route.float().unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)
        init_state = init_s.view(BI, 1, -1)
        last_t_feat = last_t_s.view(BI, 1, -1)
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)

        # Sampling over study latent only (deterministic z_i)
        samples: List[torch.Tensor] = []
        for _ in range(sample_size):
            mean_s, _, _, _, _ = self.decoder(
                init_state,
                decode_t,
                z_s,
                z_i,
                dose=dose_feat,
                route=route_feat,
                first_t_s=last_t_feat,
            )
            mean_s = mean_s.view(B, I, -1, 1)
            mean_raw, time_raw = self.scaler.inverse(mean_s, Trem_s, stats)
            samples.append(mean_raw)

        return (
            torch.stack(samples, dim=0),  # [S,B,I,Tr,1]
            time_raw.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1),  # [S,B,I,Tr,1]
            Xrem_raw,  # [B,I,Tr,1]
            Mrem,  # [B,I,Tr]
        )

    # ------------------------------------------------------------------
    # Attach per-substance metrics so BasePKModel can log them centrally
    # ------------------------------------------------------------------
    def forward_report(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        outputs_collector: AbstractForwardOutputs,
    ) -> None:
        outputs_collector.per_substance = self.forward_report_per_substance(
            databatch_list, outputs_collector
        )
