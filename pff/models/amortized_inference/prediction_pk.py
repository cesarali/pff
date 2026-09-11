"""Logging-free prediction PK model built on the new base stack."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torchtyping import TensorType, patch_typeguard

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.amortized_inference.generative_pk import (
    NewBasePKModel,
    NewPredictiveMixin,
)
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)
from pff.models.architectures.encoders_pk import ContextEncoderModule

patch_typeguard()


# ---------------------------------------------------------------------
# Prediction-only forward outputs
# ---------------------------------------------------------------------
class PredictionForwardOutputs(AbstractForwardOutputs):
    """Container for permutation-level prediction heads and losses.

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
        # Ignore any provided loss_multihead and rely on custom _compute_total_loss.
        super().__init__(loss_multihead=None, device=device)

    def _compute_total_loss(
        self,
        flat_losses: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Only ``pred_loss`` drives optimisation; other entries are metrics."""
        if not flat_losses:
            return None
        return flat_losses.get("pred_loss", None)


# ---------------------------------------------------------------------
# Prediction-only PK model (logging-free)
# ---------------------------------------------------------------------
class PredictionPK(NewBasePKModel, NewPredictiveMixin):
    """Prediction-only Neural-Process style PK model (logging-free)."""

    encoder: ContextEncoderModule

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)
        if self.encoder is None:
            raise ValueError("PredictionPK requires a configured individual encoder.")
        self.encoder = cast(ContextEncoderModule, self.encoder)
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

    def build_visualization_callback(self):
        """Build predictive callbacks while preserving one cross-repo task entry."""

        train_cfg = getattr(getattr(self, "model_config", None), "train", None)
        callbacks_scheduler = getattr(train_cfg, "callbacks_scheduler", None)
        if not callbacks_scheduler:
            return []

        from pff.models.amortized_inference.aicme import _expand_empirical_scheduler_tasks
        from pff.training.callbacks.scheduler import BaseSchedulerCallback
        from pff.training.callbacks.task_registry import TASK_REGISTRY

        mix_cfg = getattr(getattr(self, "model_config", None), "mix_data", None)
        empirical_datasets = list(getattr(mix_cfg, "test_empirical_datasets", []) or [])
        model_label = getattr(self.model_config, "name_str", self.__class__.__name__)
        raw_scheduler = (
            asdict(callbacks_scheduler)
            if is_dataclass(callbacks_scheduler)
            else dict(callbacks_scheduler)
        )
        expanded_cfg = _expand_empirical_scheduler_tasks(
            raw_scheduler,
            empirical_datasets=empirical_datasets,
            model_label=str(model_label),
        )
        return [BaseSchedulerCallback.from_config(cfg=expanded_cfg, registry=TASK_REGISTRY)]

    def generate(
        self,
        batch: AICMECompartmentsDataBatch,
        num_samples: int = 1,
    ):
        """Return the predictive-only scheduler payload used by ``PredictionPK``."""

        from pff.training.callbacks.pk_tasks import build_predictive_task_samples

        return build_predictive_task_samples(self, batch, num_samples=int(num_samples))

    # ------------------------------------------------------------------
    # Forward over a list of permutations (matches PredictionPK logic)
    # ------------------------------------------------------------------
    def forward(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch] | AICMECompartmentsDataBatch,
        return_forward_report: bool = False,
    ) -> PredictionForwardOutputs:
        # Accept a single batch or a list of permutation-batches.
        if isinstance(databatch_list, AICMECompartmentsDataBatch):
            databatch_list = [databatch_list]

        collector = PredictionForwardOutputs()
        for batch in databatch_list:
            outputs = self._forward_prediction(batch)
            collector.add(outputs)
        aggregated = collector.reduce()

        # Accepted for API parity with PredictionPK; callbacks own evaluation.
        _ = return_forward_report
        return aggregated

    # ------------------------------------------------------------------
    # Internals: one-permutation prediction head
    # ------------------------------------------------------------------
    def _forward_prediction(self, db: AICMECompartmentsDataBatch) -> PredictionForwardOutputs:
        """Predict remainder for known individuals."""
        # Prefer target remainder/observations for training; fall back to context for inference-only.
        X_obs_raw = db.target_obs if db.target_obs is not None else db.context_obs  # [B,I,To,1]
        T_obs_raw = (
            db.target_obs_time if db.target_obs_time is not None else db.context_obs_time
        )  # [B,I,To,1]
        M_obs = (
            db.target_obs_mask if db.target_obs_mask is not None else db.context_obs_mask
        )  # [B,I,To]

        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim
            if db.target_rem_sim is not None
            else db.context_rem_sim,  # [B,I,Tr,1]
            db.target_rem_sim_time
            if db.target_rem_sim_time is not None
            else db.context_rem_sim_time,  # [B,I,Tr,1]
            db.target_rem_sim_mask
            if db.target_rem_sim_mask is not None
            else db.context_rem_sim_mask,  # [B,I,Tr]
        )
        dose = (
            db.target_dosing_amounts
            if db.target_dosing_amounts is not None
            else db.context_dosing_amounts
        )  # [B,I,D]
        route = (
            db.target_dosing_route_types
            if db.target_dosing_route_types is not None
            else db.context_dosing_route_types
        )  # [B,I,R]

        # Always take scaling stats from context (AICME convention).
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        # Scale inputs.
        X_obs_s, T_obs_s = self.scaler.forward(X_obs_raw, T_obs_raw, stats)  # [B,I,To,1]
        Xrem_s, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)  # [B,I,Tr,1]

        # Individual latent from the observed block used for prediction.
        z_i = self.encoder(X_obs_s, T_obs_s, M_obs, dose, route)  # [B,I,Z]
        z_s = None

        # Init = last valid obs of the observed block.
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B,I,1,1]

        B, I, _ = z_i.shape
        BI = B * I
        dose_feat = dose.unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)  # [BI,1,D]
        route_feat = route.float().unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)  # [BI,1,R]
        init_state = init_s.view(BI, 1, -1)  # [BI,1,1]
        last_t_feat = last_t_s.view(BI, 1, -1)  # [BI,1,1]
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)  # [BI,Tr,1]

        mean_s, logvar_s, H_s, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=last_t_feat,
        )  # mean/logvar: [BI,Tr,1], H_s: [BI,Tr,p]

        # Reshape + inverse-scale for heads.
        mean_s = mean_s.view(B, I, -1, 1)  # [B,I,Tr,1]
        logvar_s = logvar_s.view(B, I, -1, 1).clamp_(-100.0, 100.0)  # [B,I,Tr,1]
        mean_raw, _ = self.scaler.inverse(mean_s, Trem_s, stats)  # [B,I,Tr,1]

        # Losses in scaled space, like PredictionPK.
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
    # Predictive sampling
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
        """Draw S predictive trajectories for known individuals."""
        db = databatch
        X_obs_raw, T_obs_raw, M_obs = db.target_obs, db.target_obs_time, db.target_obs_mask
        # X_obs_raw/T_obs_raw: [B,It,To,1], M_obs: [B,It,To]
        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim,
            db.target_rem_sim_time,
            db.target_rem_sim_mask,
        )  # [B,It,Tr,1], [B,It,Tr,1], [B,It,Tr]
        dose, route = db.target_dosing_amounts, db.target_dosing_route_types  # [B,It,D], [B,It,R]

        # Context stats (AICME convention).
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        # Individual latent from observed block.
        X_obs_s, T_obs_s = self.scaler.forward(X_obs_raw, T_obs_raw, stats)  # [B,It,To,1]
        z_i = self.encoder(X_obs_s, T_obs_s, M_obs, dose, route)  # [B,It,Z]
        z_s = None

        # Init from last observation.
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B,It,1,1]
        _, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)  # [B,It,Tr,1]

        B, I, _ = z_i.shape
        BI = B * I
        dose_feat = dose.unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)  # [BI,1,D]
        route_feat = route.float().unsqueeze(-1).unsqueeze(-1).view(BI, 1, -1)  # [BI,1,R]
        init_state = init_s.view(BI, 1, -1)  # [BI,1,1]
        last_t_feat = last_t_s.view(BI, 1, -1)  # [BI,1,1]
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)  # [BI,Tr,1]

        # Sampling over study latent only (deterministic z_i).
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
            )  # [BI,Tr,1]
            mean_s = mean_s.view(B, I, -1, 1)  # [B,It,Tr,1]
            mean_raw, time_raw = self.scaler.inverse(mean_s, Trem_s, stats)  # [B,It,Tr,1]
            samples.append(mean_raw)

        return (
            torch.stack(samples, dim=0),  # [S,B,It,Tr,1]
            time_raw.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1),  # [S,B,It,Tr,1]
            Xrem_raw,  # [B,It,Tr,1]
            Mrem,  # [B,It,Tr]
        )


__all__ = ["PredictionForwardOutputs", "PredictionPK"]
