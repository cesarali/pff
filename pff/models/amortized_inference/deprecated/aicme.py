"""Implementation of a Neural Process style pharmacokinetic model.

All tensor shapes in this module follow the conventions:
``B`` – batch size, ``I`` – number of individuals, ``T`` – time dimension and
``Z`` – latent dimension.  Comments throughout the code explicitly annotate the
shape of intermediate tensors using this notation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
)
from pff.models.amortized_inference.deprecated.generative_pk import (
    BasePKModel,
    GenerativeMixin,
    PredictiveMixin,
)
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures import get_decoder
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)
from pff.models.architectures.encoders_pk import get_individual_encoder
from pff.models.utils.loss_utils import MultiHeadLoss
from pff.utils.tensors_operations import gather_distinct_times_per_substance

patch_typeguard()


class AICMEForwardOutputs(AbstractForwardOutputs):
    """Declarative forward outputs for :class:`AICMEPK`."""

    HEAD_SCHEMAS = {
        "reconstruction": ["mean", "logvar", "target", "mask"],
        "prediction": ["mean", "logvar", "target", "mask"],
    }

    LOSS_SCHEMAS = {
        "reconstruction": [
            "recon_loss",
            "kl_s",
            "kl_i",
            "kl_init",
            "kl_zs_zsN",
            "rmse_norm",
            "r2_norm",
            "log_rmse_norm",
            "log_r2_norm",
            "rmse",
            "log_rmse",
            "r2",
            "log_r2",
            "init_rmse",
        ],
        "prediction": [
            "pred_loss",
            "kl_zi_ziN",
            "pred_rmse",
            "pred_log_rmse",
            "pred_r2",
            "pred_log_r2",
        ],
        "regularization": ["invariance"],
    }

    def __init__(
        self,
        *,
        loss_multihead: Optional[MultiHeadLoss] = None,
        use_invariance_loss: bool = True,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(loss_multihead=loss_multihead, device=device)
        self.use_invariance_loss = use_invariance_loss
        self.z_sN: Optional[TensorType["B", "Z"]] = None
        self.stats: Optional[TensorType["B", 1]] = None
        self.per_substance: Optional[Dict[str, Dict[str, float]]] = None
        self.prediction_permutations: List[Dict[str, torch.Tensor]] = []

    def attach_prediction(
        self,
        prediction: Dict[str, torch.Tensor],
        prediction_losses: Dict[str, torch.Tensor],
    ) -> None:
        """Attach predictive outputs and their losses to the container."""

        self.update_head("prediction", prediction)
        self.update_losses("prediction", prediction_losses)

    def aggregate_losses(self) -> Dict[str, Dict[str, torch.Tensor]]:
        aggregated = super().aggregate_losses()
        device = self._infer_device()
        reg_scope = aggregated.setdefault("regularization", {})
        if self.use_invariance_loss:
            latents = [item.z_sN for item in self.items if item.z_sN is not None]
            if latents:
                stack = torch.stack(latents, dim=0)
                reg_scope["invariance"] = stack.var(dim=0, unbiased=False).mean()
            else:
                reg_scope["invariance"] = torch.zeros((), device=device)
        else:
            reg_scope["invariance"] = torch.zeros((), device=device)
        return aggregated

    def _compute_total_loss(self, flat_losses: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not flat_losses:
            return None
        if self.loss_multihead is None:
            return super()._compute_total_loss(flat_losses)
        device = self._infer_device()
        zero = torch.zeros((), device=device)

        def _get(name: str) -> torch.Tensor:
            return flat_losses.get(name, zero)

        total_loss, _ = self.loss_multihead(
            [
                _get("recon_loss"),
                _get("kl_s"),
                _get("kl_i"),
                _get("kl_init"),
                _get("kl_zs_zsN"),
                _get("kl_zi_ziN"),
                _get("invariance"),
                _get("init_rmse"),
                _get("pred_loss"),
            ]
        )
        return total_loss

    def _new_like(self) -> "AICMEForwardOutputs":
        return type(self)(
            loss_multihead=self.loss_multihead,
            use_invariance_loss=self.use_invariance_loss,
            device=self.device,
        )

    def reduce(self) -> "AICMEForwardOutputs":
        reduced = super().reduce()
        reduced = cast(AICMEForwardOutputs, reduced)
        reduced.prediction_permutations = [
            dict(pred) for pred in reduced.prediction_permutations if pred is not None
        ]
        return reduced


def kl_divergence_gaussians(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """Element-wise KL divergence ``KL(q||p)`` for diagonal Gaussians."""
    return 0.5 * (
        logvar_p - logvar_q + (logvar_q.exp() + (mu_q - mu_p).pow(2)) / logvar_p.exp() - 1
    )


class AICMEPK(BasePKModel, PredictiveMixin, GenerativeMixin):
    """
    Amortized In Context Mixed Effects Model for PK data.

    The model expects target observation strategies that already split past and
    future observations (``split_past_future=True``).
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        if not model_config.target_observations.split_past_future:
            raise ValueError("AICMEPK requires target_observations.split_past_future to be True")
        if model_config.network.prediction_only and model_config.network.reconstruction_only:
            raise ValueError("`prediction_only` and `reconstruction_only` are mutually exclusive")
        self.encoder = get_individual_encoder(model_config)
        self.decoder = get_decoder(model_config)
        z_dim = self.encoder.zi_latent_dim

        # Aggregator for study-level representation
        aggregator_type = model_config.network.aggregator_type
        if aggregator_type == "mean":
            self.aggregator = MeanStudyAggregator()
        elif aggregator_type == "attention":
            self.aggregator = AttentionStudyAggregator(
                z_dim, model_config.network.aggregator_num_heads
            )
        else:
            raise ValueError(f"Unknown aggregator_type '{aggregator_type}'.")

        # Layers for variational study latent
        self.mu_s_layer = nn.Linear(z_dim, z_dim)
        self.logvar_s_layer = nn.Linear(z_dim, z_dim)

        # Layers for variational individual latent (target encoding)
        self.mu_i_layer = nn.Linear(z_dim, z_dim)
        self.logvar_i_layer = nn.Linear(z_dim, z_dim)

        # Layers for variational initial condition
        self.mu_init_layer = nn.Linear(z_dim, 1)
        self.logvar_init_layer = nn.Linear(z_dim, 1)

        # add extra KL head for individual latent
        # (recon, KL_s, KL_i, KL_init, KL_zs_zsN, KL_zi_ziN, invariance, init_recon, pred_loss)
        self.loss_multihead = MultiHeadLoss(mode="learnable", number_of_losses=9)
        super().__init__(model_config)

    def _sample_initial_condition(
        self,
        z_s: TensorType["B", "Z"],
        z_i: TensorType["B", "I", "Z"],
    ) -> Tuple[TensorType["B", "I", 1, 1], TensorType["B", "I", 1], TensorType["B", "I", 1]]:
        """Sample initial conditions conditioned on combined latents."""

        z_comb = self.decoder.combine_latents(z_s, z_i)  # [B, I, Z]

        mu_init = self.mu_init_layer(z_comb)  # [B, I, 1]
        logvar_init = self.logvar_init_layer(z_comb)  # [B, I, 1]

        eps = torch.randn_like(mu_init)
        init_s = mu_init + eps * torch.exp(0.5 * logvar_init)  # [B, I, 1]
        init_s = init_s.unsqueeze(-1)  # [B, I, 1, 1]

        return init_s, mu_init, logvar_init

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

    def _study_latent(
        self,
        db: AICMECompartmentsDataBatch,
        use_target: bool = False,
    ) -> Tuple[
        TensorType["B", "Z"],
        TensorType["B", "Z"],
        TensorType["B", "Z"],
    ]:
        """Encode study observations into stochastic latent z_s."""
        if use_target:
            X_raw, T_raw, M = self._include_target(db)  # [B, Ic+It, T, 1]
            if db.mask_individuals is not None:
                B = db.mask_individuals.size(0)
                I_t = db.target_obs.shape[1]
                mask_ind = torch.cat(
                    [
                        db.mask_individuals,
                        torch.ones(
                            B,
                            I_t,
                            dtype=torch.bool,
                            device=db.mask_individuals.device,
                        ),
                    ],
                    dim=1,
                )
            else:
                mask_ind = None
        else:
            X_raw, T_raw, M = (
                db.context_obs,
                db.context_obs_time,
                db.context_obs_mask,
            )  # [B, Ic, T, 1]
            mask_ind = db.mask_individuals

        # scaling statistics from **context only**
        stats = self.scaler.stats(
            db.context_obs, db.context_obs_time, db.context_obs_mask
        )  # [B, 1]
        X_s, T_s = self.scaler.forward(X_raw, T_raw, stats)  # [B, I*, T, 1]
        if use_target:
            dose = torch.cat([db.context_dosing_amounts, db.target_dosing_amounts], dim=1)
            route = torch.cat([db.context_dosing_route_types, db.target_dosing_route_types], dim=1)
        else:
            dose = db.context_dosing_amounts
            route = db.context_dosing_route_types

        z_ci = self.encoder(X_s, T_s, M, dose, route, mask_ind)  # [B, I*, Z]
        z_s_agg = self.aggregator(z_ci, mask_ind)  # [B, Z]

        mu_s = self.mu_s_layer(z_s_agg)
        logvar_s = self.logvar_s_layer(z_s_agg)
        if self.model_config.network.study_latent_deterministic:
            # Deterministic study representation [B, Z]
            z_s = mu_s
        else:
            eps = torch.randn_like(mu_s)
            z_s = mu_s + eps * torch.exp(0.5 * logvar_s)
        return z_s, mu_s, logvar_s

    def _individual_latent(
        self,
        X_t: TensorType["B", "I", "T", 1],
        T_t: TensorType["B", "I", "T", 1],
        M_t: TensorType["B", "I", "T"],
        dosing_amounts: TensorType["B", "I"],
        dosing_route_types: TensorType["B", "I"],
    ) -> Tuple[
        TensorType["B", "I", "Z"],
        TensorType["B", "I", "Z"],
        TensorType["B", "I", "Z"],
    ]:
        """Encode target observations into latent ``z_i``."""

        z_ti = self.encoder(X_t, T_t, M_t, dosing_amounts, dosing_route_types)  # [B, I, Z]
        mu_i = self.mu_i_layer(z_ti)  # [B, I, Z]
        logvar_i = self.logvar_i_layer(z_ti)  # [B, I, Z]
        if self.model_config.network.prediction_latent_deterministic:
            # Deterministic individual representation [B, I, Z]
            z_i = mu_i
        else:
            eps = torch.randn_like(mu_i)
            z_i = mu_i + eps * torch.exp(0.5 * logvar_i)
        return z_i, mu_i, logvar_i

    def _forward_reconstruction(
        self,
        db: AICMECompartmentsDataBatch,
    ) -> AICMEForwardOutputs:
        """Reconstruction loss for a single batch.

        Returns the scaling statistics so they can be reused for prediction.
        """
        Xc_raw, Tc_raw, Mc = (
            db.context_obs,
            db.context_obs_time,
            db.context_obs_mask,
        )  # [B, Ic, Tc, 1] etc.
        Xt_raw, Tt_raw, Mt = (
            db.target_obs,
            db.target_obs_time,
            db.target_obs_mask,
        )  # [B, I, Tt, 1]
        B = Xc_raw.shape[0]

        # statistics from **context only**
        stats = self.scaler.stats(Xc_raw, Tc_raw, Mc)  # [B, 1]

        outputs = AICMEForwardOutputs(
            loss_multihead=self.loss_multihead,
            use_invariance_loss=self.model_config.network.use_invariance_loss,
            device=Xt_raw.device,
        )
        outputs.stats = stats

        if self.model_config.network.prediction_only:
            # When training only on prediction, skip reconstruction computations
            _, _, _ = self._study_latent(db, use_target=False)
            z_sN, _, _ = self._study_latent(db, use_target=True)
            outputs.z_sN = z_sN
            dummy = torch.zeros_like(Xt_raw)
            outputs.update_head(
                "reconstruction",
                {
                    "mean": dummy,
                    "logvar": dummy,
                    "target": Xt_raw,
                    "mask": Mt,
                },
            )
            zero = torch.zeros((), device=dummy.device)
            outputs.update_losses(
                "reconstruction",
                {key: zero.clone() for key in AICMEForwardOutputs.LOSS_SCHEMAS["reconstruction"]},
            )
            outputs.update_losses("regularization", {"invariance": zero.clone()})
            return outputs

        Xc_s, Tc_s = self.scaler.forward(Xc_raw, Tc_raw, stats)  # [B, Ic, Tc, 1]
        Xt_s, Tt_s = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, I, Tt, 1]

        z_s, mu_s, logvar_s = self._study_latent(db, use_target=False)  # [B, Z]
        z_sN, mu_sN, logvar_sN = self._study_latent(db, use_target=True)  # [B, Z]
        outputs.z_sN = z_sN

        z_i, mu_i, logvar_i = self._individual_latent(
            Xt_s,
            Tt_s,
            Mt,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )  # [B, I, Z]

        init_true, init_mask, first_t_s = self.get_first_valid_observation(
            Xt_s, Mt, Tt_s
        )  # [B, I, 1, 1], [B,I,1], [B,I,1,1]
        init_s, mu_init, logvar_init = self._sample_initial_condition(z_s, z_i)  # [B, I, 1, 1]

        B, I, T, _ = Xt_raw.shape
        BI = B * I
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B,I,1,1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)

        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI,Tt,1]
        init_state = init_s.view(BI, 1, -1)
        first_t_feat = first_t_s.view(BI, 1, -1)
        dose_feat = dose.view(BI, 1, -1)
        route_feat = route.view(BI, 1, -1)
        mean_pred, logvar_pred, H_s, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=first_t_feat,
        )  # [BI, Tt, 1]
        mean_pred = mean_pred.view(B, I, -1, 1)  # [B, I, Tt, 1]
        logvar_pred = logvar_pred.view(B, I, -1, 1)  # [B, I, Tt, 1]
        mean_raw, _ = self.scaler.inverse(mean_pred, Tt_s, stats)

        target = Xt_raw
        mask_t = Mt

        logvar_pred = torch.clamp(logvar_pred, min=-100.0, max=100.0)
        recon_loss_dict = self.compute_loss(mean_pred, logvar_pred, Xt_s, mask_t)
        recon_loss = recon_loss_dict["loss"]
        rmse_loss = recon_loss_dict["rmse"]
        log_rmse_loss = recon_loss_dict["log_rmse"]
        r2_score = recon_loss_dict["r2"]

        init_recon = self.masked_rmse_loss(
            mu_init.unsqueeze(-1),
            init_true,
            init_mask,
        )
        init_rmse = init_recon["rmse"]

        kl_s = 0.5 * (mu_s.pow(2) + logvar_s.exp() - logvar_s - 1).mean()
        if not self.model_config.network.use_kl_s:
            kl_s = torch.tensor(0.0, device=kl_s.device)
        kl_i = 0.5 * (mu_i.pow(2) + logvar_i.exp() - logvar_i - 1).mean()
        if not self.model_config.network.use_kl_i:
            kl_i = torch.tensor(0.0, device=kl_i.device)
        kl_init = 0.5 * (mu_init.pow(2) + logvar_init.exp() - logvar_init - 1).mean()
        if not self.model_config.network.use_kl_init:
            kl_init = torch.tensor(0.0, device=kl_init.device)
        kl_zs_zsN = kl_divergence_gaussians(mu_s, logvar_s, mu_sN, logvar_sN).mean()

        outputs.update_head(
            "reconstruction",
            {
                "mean": mean_raw,
                "logvar": logvar_pred,
                "target": target,
                "mask": mask_t,
            },
        )

        recon_metrics = {
            "recon_loss": recon_loss,
            "kl_s": kl_s,
            "kl_i": kl_i,
            "kl_init": kl_init,
            "kl_zs_zsN": kl_zs_zsN,
            "rmse_norm": rmse_loss,
            "log_rmse_norm": log_rmse_loss,
            "r2_norm": r2_score,
            "log_r2_norm": recon_loss_dict["log_r2"],
            "rmse": self.masked_rmse_loss(mean_raw, target, mask_t)["rmse"],
            "log_rmse": self.masked_log_rmse_loss(mean_raw, target, mask_t)["rmse"],
            "r2": self.masked_r2_score(mean_raw, target, mask_t),
            "log_r2": self.masked_log_r2_score(mean_raw, target, mask_t),
            "init_rmse": init_rmse,
        }
        outputs.update_losses("reconstruction", recon_metrics)
        outputs.update_losses(
            "regularization", {"invariance": torch.zeros((), device=recon_loss.device)}
        )
        return outputs

    def _forward_prediction(
        self,
        db: AICMECompartmentsDataBatch,
        z_sN: torch.Tensor | None = None,
        stats: torch.Tensor | None = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Predict future remainder given past observations.

        Args:
            db: Databatch containing context and target tensors.
            z_sN: Optional precomputed study latent ``[B, Z]``.
            stats: Optional scaling statistics ``[B, 1]`` from context.

        Returns:
            Dictionary with ``loss`` and ``rmse`` scalars.
        """
        if self.model_config.network.reconstruction_only:
            dummy = torch.zeros_like(db.target_rem_sim)
            zero = torch.tensor(0.0, device=dummy.device)
            prediction = {
                "mean": dummy,
                "logvar": dummy,
                "target": dummy,
                "mask": db.target_rem_sim_mask,
            }
            pred_losses = {
                "pred_loss": zero,
                "kl_zi_ziN": zero,
                "pred_rmse": zero,
                "pred_log_rmse": zero,
                "pred_r2": zero,
                "pred_log_r2": zero,
            }
            return prediction, pred_losses

        Xc_raw = db.context_obs  # [B, Ic, Tc, 1]
        Tc_raw = db.context_obs_time  # [B, Ic, Tc, 1]
        Mc = db.context_obs_mask  # [B, Ic, Tc]
        Xt_raw = db.target_obs  # [B, It, Tt, 1]
        Tt_raw = db.target_obs_time  # [B, It, Tt, 1]
        Mt = db.target_obs_mask  # [B, It, Tt]

        if stats is None:
            stats = self.scaler.stats(Xc_raw, Tc_raw, Mc)  # [B, 1]

        Xt_s, Tt_s = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, Tt, 1]
        Xrt_s, Trt_s = self.scaler.forward(
            db.target_rem_sim, db.target_rem_sim_time, stats
        )  # [B, It, Tr, 1]

        if z_sN is None:
            z_s, _, _ = self._study_latent(db, use_target=False)  # [B, Z]
        else:
            z_s = z_sN

        z_i, mu_i, logvar_i = self._individual_latent(
            Xt_s,
            Tt_s,
            Mt,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )  # [B, It, Z]

        Xt_full = torch.cat([Xt_s, Xrt_s], dim=2)  # [B, It, Tt+Tr, 1]
        Tt_full = torch.cat([Tt_s, Trt_s], dim=2)  # [B, It, Tt+Tr, 1]
        Mt_full = torch.cat([Mt, db.target_rem_sim_mask], dim=2)  # [B, It, Tt+Tr]
        z_iN, mu_iN, logvar_iN = self._individual_latent(
            Xt_full,
            Tt_full,
            Mt_full,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )  # [B, It, Z]
        kl_zi_ziN = kl_divergence_gaussians(mu_i, logvar_i, mu_iN, logvar_iN).mean()
        if not self.model_config.network.use_kl_i_np:
            kl_zi_ziN = torch.tensor(0.0, device=kl_zi_ziN.device)

        init_raw, last_t_raw = self.get_last_valid_observation(Xt_raw, Mt, Tt_raw)  # [B, It, 1, 1]
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B, It, 1, 1]

        B = Xt_raw.size(0)
        I = init_s.size(1)
        BI = B * I
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B,I,1,1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)

        Trem_s = Trt_s  # [B, It, Tr, 1]
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)  # [BI, Tr, 1]
        init_state = init_s.view(BI, 1, -1)
        last_t_feat = last_t_s.view(BI, 1, -1)
        dose_feat = dose.view(BI, 1, -1)
        route_feat = route.view(BI, 1, -1)

        mean_pred, logvar_pred, _, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=last_t_feat,
        )  # [BI, Tr, 1]

        mean_pred = mean_pred.view(B, I, -1, 1)  # [B, It, Tr, 1]
        logvar_pred = logvar_pred.view(B, I, -1, 1)  # [B, It, Tr, 1]
        Xrem_s = Xrt_s  # [B, It, Tr, 1]
        Mrem = db.target_rem_sim_mask  # [B, It, Tr]

        logvar_pred = torch.clamp(logvar_pred, min=-100.0, max=100.0)

        loss_dict = self.compute_loss(mean_pred, logvar_pred, Xrem_s, Mrem)

        mean_raw, _ = self.scaler.inverse(mean_pred, Trem_s, stats)  # [B, It, Tr, 1]
        rem_raw = db.target_rem_sim  # [B, It, Tr, 1]
        rmse = self.masked_rmse_loss(mean_raw, rem_raw, Mrem)["rmse"]
        log_rmse = self.masked_log_rmse_loss(mean_raw, rem_raw, Mrem)["rmse"]
        r2 = self.masked_r2_score(mean_raw, rem_raw, Mrem)
        log_r2 = self.masked_log_r2_score(mean_raw, rem_raw, Mrem)

        prediction = {
            "mean": mean_raw,
            "logvar": logvar_pred,
            "target": rem_raw,
            "mask": Mrem,
        }
        pred_losses = {
            "pred_loss": loss_dict["loss"],
            "kl_zi_ziN": kl_zi_ziN,
            "pred_rmse": rmse,
            "pred_log_rmse": log_rmse,
            "pred_r2": r2,
            "pred_log_r2": log_r2,
        }

        return prediction, pred_losses

    def forward(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        return_forward_report: bool = False,
    ):
        """Forward pass over a list of ``AICMECompartmentsDataBatch`` objects.

        Parameters
        ----------
        databatch_list:
            Sequence of length ``P`` with one :class:`AICMECompartmentsDataBatch`
            per permutation. Each batch contains tensors with shapes like
            ``target_obs`` ``[B, It, Tt, 1]`` and ``target_rem_sim``
            ``[B, It, Tr, 1]``.
        """

        outputs_collector = AICMEForwardOutputs(
            loss_multihead=self.loss_multihead,
            use_invariance_loss=self.model_config.network.use_invariance_loss,
        )

        for batch in databatch_list:
            recon_outputs = self._forward_reconstruction(batch.to_reconstruct_type())
            pred_outputs, pred_losses = self._forward_prediction(
                batch,
                recon_outputs.z_sN,
                recon_outputs.stats,
            )
            recon_outputs.attach_prediction(pred_outputs, pred_losses)
            outputs_collector.add(recon_outputs)

        aggregated_outputs = outputs_collector.reduce()
        if return_forward_report:
            self.forward_report(databatch_list, aggregated_outputs)

        return aggregated_outputs

    def forward_report(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch],
        outputs_collector: AbstractForwardOutputs,
    ) -> None:
        """Compute evaluation metrics using aggregated forward outputs."""

        outputs_collector.per_substance = self.forward_report_per_substance(
            databatch_list,
            outputs_collector,
        )

    def select_unseen_dosing_from_databatch(
        self,
        db: AICMECompartmentsDataBatch,
        generator: torch.Generator | None = None,
    ) -> Tuple[
        TensorType["B"],
        TensorType["B"],
    ]:
        """Select dosing information for a new individual.

        The routine first attempts to reuse the dosing configuration from the
        available target individuals. When no targets are provided it randomly
        samples one of the context dosing configurations.

        Parameters
        ----------
        db : AICMECompartmentsDataBatch
            Batch containing context and target individuals.
        generator : torch.Generator, optional
            Random generator to control the stochastic selection among
            context individuals.

        Returns
        -------
        amount : TensorType["B"]
            Dosing amount for each new individual.
        route : TensorType["B"]
            Dosing route type for each new individual.
        """

        target_mask = db.mask_target_individuals
        context_mask = db.mask_context_individuals

        device = db.context_dosing_amounts.device
        route_device = db.context_dosing_route_types.device
        B = db.context_dosing_amounts.size(0)
        dose = torch.zeros(
            B,
            dtype=db.context_dosing_amounts.dtype,
            device=device,
        )
        route = torch.zeros(
            B,
            dtype=db.context_dosing_route_types.dtype,
            device=route_device,
        )

        for b in range(B):
            valid_targets = (
                torch.nonzero(target_mask[b], as_tuple=False).view(-1)
                if target_mask is not None
                else torch.empty(0, dtype=torch.long, device=route.device)
            )
            if valid_targets.numel() > 0:
                idx = int(valid_targets[0].item())
                dose[b] = db.target_dosing_amounts[b, idx]
                route[b] = db.target_dosing_route_types[b, idx]
                continue

            valid_context = torch.nonzero(context_mask[b], as_tuple=False).view(-1)
            if valid_context.numel() == 0:
                continue

            if generator is not None:
                choice = torch.randperm(
                    valid_context.numel(),
                    generator=generator,
                    device=device,
                )[0]
            else:
                choice = torch.randperm(valid_context.numel(), device=device)[0]
            idx = int(valid_context[int(choice)].item())
            dose[b] = db.context_dosing_amounts[b, idx]
            route[b] = db.context_dosing_route_types[b, idx]

        return dose.to(device), route.to(route_device)

    @torch.inference_mode()
    def sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ) -> Tuple[
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["B", "It", "Tr", 1],
    ]:
        """Sample predictions conditioned on a known individual.

        The initial condition for each target individual is the last
        valid observation rather than a sampled value.

            Tt: obs time of target individuals (past)
            Tr: rem time of target individuals (future)

            Ic: number of individuals in context
            It: number of individual in target   TYPICALLY 1

        Args:
            databatch: Batch containing both context and target data.
            sample_size: Number of stochastic samples ``S``.

        Returns:

            sampled_trajectories [S, B, It, Tr, 1]
            sampled_trajectories time [S, B, It, Tr, 1]
            real_trajectories [B, It, Tr, 1]
            mask_real_trajectories [B, It, Tr]
        """
        db = databatch

        stats = self.scaler.stats(
            db.context_obs, db.context_obs_time, db.context_obs_mask
        )  # [B, 1]
        Xc_s, Tc_s = self.scaler.forward(
            db.context_obs, db.context_obs_time, stats
        )  # [B, Ic, Tc, 1]
        z_ci = self.encoder(
            Xc_s,
            Tc_s,
            db.context_obs_mask,
            db.context_dosing_amounts,
            db.context_dosing_route_types,
            db.mask_individuals,
        )  # [B, Ic, Z]
        z_s = self.aggregator(z_ci, db.mask_individuals)  # [B, Z]
        mu_s = self.mu_s_layer(z_s)  # [B, Z]
        logvar_s = self.logvar_s_layer(z_s)  # [B, Z]
        if self.model_config.network.study_latent_deterministic:
            z_s = mu_s  # [B, Z]
        else:
            z_s = mu_s + torch.randn_like(mu_s) * torch.exp(0.5 * logvar_s)  # [B, Z]

        Xt_s, Tt_s = self.scaler.forward(db.target_obs, db.target_obs_time, stats)  # [B, It, Tt, 1]
        z_enc = self.encoder(
            Xt_s,
            Tt_s,
            db.target_obs_mask,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )
        z_i_mu = self.mu_i_layer(z_enc)
        z_i_logvar = self.logvar_i_layer(z_enc)
        init_raw, last_t_raw = self.get_last_valid_observation(
            db.target_obs, db.target_obs_mask, db.target_obs_time
        )  # [B, It, 1, 1]
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B, It, 1, 1]
        _, Trem_s = self.scaler.forward(
            db.target_rem_sim, db.target_rem_sim_time, stats
        )  # [B, It, Tr, 1]

        B, I, _ = z_i_mu.shape
        BI = B * I
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B,I,1,1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)
        decode_t = (Trem_s - last_t_s).view(BI, -1, 1)  # [BIt, Tr, 1]
        init_state = init_s.view(BI, 1, -1)
        last_t_feat = last_t_s.view(BI, 1, -1)
        dose_feat = dose.view(BI, 1, -1)
        route_feat = route.view(BI, 1, -1)

        samples = []
        for _ in range(sample_size):
            if self.model_config.network.prediction_latent_deterministic:
                z_i = z_i_mu  # [B, It, Z]
            else:
                eps_i = torch.randn_like(z_i_mu)
                z_i = z_i_mu + eps_i * torch.exp(0.5 * z_i_logvar)  # [B, It, Z]
            mean_s, _, _, _, _ = self.decoder(
                init_state,
                decode_t,
                z_s,
                z_i,
                dose=dose_feat,
                route=route_feat,
                first_t_s=last_t_feat,
            )  # [BIt, Tr, 1]
            mean_s = mean_s.view(B, I, -1, 1)
            mean_raw, time_raw = self.scaler.inverse(mean_s, Trem_s, stats)
            samples.append(mean_raw)

        return (
            torch.stack(samples, dim=0),
            time_raw.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1),
            db.target_rem_sim,
            db.target_rem_sim_mask,
        )

    @torch.inference_mode()
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct_max", 1],
            TensorType["B", "Tdistinct_max"],
        ]
        | None = None,
        dosing: Tuple[
            TensorType["B"],
            TensorType["B"],
        ]
        | None = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max"],
    ]:
        """
        Sample a new individual conditioned only on study context.

        Args
        ----
        db : AICMECompartmentsDataBatch
            Input batch providing context.
        sample_size : int
            Number of stochastic samples `S`.
        decode_times : (times, mask), optional
            - times: [B, Tdistinct_max, 1]
            - mask : [B, Tdistinct_max]
            If None, times are inferred automatically with
            `gather_distinct_times_per_substance`.
        dosing : (amount, route), optional
            - amount: [B]
            - route : [B]
            If None, dosing information is automatically selected with
            :meth:`select_unseen_dosing_from_databatch`.

        Returns
        -------
        samples : TensorType["S", "B", "Tdistinct_max", 1]
            Simulated individuals (one per substance).
        time_grid : TensorType["B", "Tdistinct_max", 1]
            Time grid aligned with samples.
        mask : TensorType["B", "Tdistinct_max"]
            Boolean mask indicating which decode time entries are valid
            for each batch element (substance).
        """
        # --- study context encoding ---
        stats = self.scaler.stats(
            db.context_obs, db.context_obs_time, db.context_obs_mask
        )  # [B, 1]
        Xc_s, Tc_s = self.scaler.forward(db.context_obs, db.context_obs_time, stats)
        z_ci = self.encoder(
            Xc_s,
            Tc_s,
            db.context_obs_mask,
            db.context_dosing_amounts,
            db.context_dosing_route_types,
            db.mask_individuals,
        )
        z_s = self.aggregator(z_ci, db.mask_individuals)
        mu_s = self.mu_s_layer(z_s)
        logvar_s = self.logvar_s_layer(z_s)

        B = db.context_obs.shape[0]
        I = 1  # always generate exactly one new individual per substance
        BI = B * I

        if dosing is None:
            dose_values, route_values = self.select_unseen_dosing_from_databatch(db)
        else:
            dose_values, route_values = dosing

        dose = (
            dose_values.unsqueeze(-1)
            .unsqueeze(-1)
            .to(
                device=mu_s.device,
                dtype=db.context_dosing_amounts.dtype,
            )
        )
        route = (
            route_values.unsqueeze(-1)
            .unsqueeze(-1)
            .to(
                device=mu_s.device,
                dtype=db.context_dosing_route_types.dtype,
            )
            .float()
        )

        # --- decode times (custom grid) ---
        if decode_times is None:
            times, mask = gather_distinct_times_per_substance(db)
        else:
            times, mask = decode_times  # [B,Tdistinct_max,1], [B,Tdistinct_max]

        # Broadcast raw times to [B, I, T, 1]
        Tt_raw = times.unsqueeze(1).expand(B, I, -1, -1)  # [B,I,T,1]

        # Reuse scaler for time transform
        zeros_like_vals = torch.zeros_like(Tt_raw)  # [B,I,T,1]
        _, Tt_s = self.scaler.forward(zeros_like_vals, Tt_raw, stats)  # [B,I,T,1] (scaled)

        # Relative decode times in *scaled* space
        first_t_s = Tt_s[:, :, 0:1, :]  # [B,I,1,1]
        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI,T,1]
        Tdec = decode_t.shape[1]

        # --- sampling loop ---
        samples = []
        for _ in range(sample_size):
            # Sample study latent
            z_s_samp = mu_s + torch.randn_like(mu_s) * torch.exp(0.5 * logvar_s)  # [B,Z]

            # Sample individual latent (with I=1 dim)
            z_i = torch.randn(B, I, self.encoder.zi_latent_dim, device=mu_s.device)  # [B,1,Z]

            # Initial condition: keep I=1 for compatibility
            init_s, _, _ = self._sample_initial_condition(z_s_samp, z_i)  # [B,1,1,1]

            # Decode trajectory
            mean_s, _, _, _, _ = self.decoder(
                init_s.view(BI, 1, 1),  # [B,1,1]
                decode_t,  # [B,T,1]
                z_s_samp,
                z_i,
                dose=dose.view(BI, 1, -1),
                route=route.view(BI, 1, -1),
                first_t_s=first_t_s.view(BI, 1, -1),
            )  # [B,T,1]
            mean_s = mean_s.view(B, I, -1, 1)  # [B,1,T,1]

            # Inverse-scale back to raw units
            mean_raw, _ = self.scaler.inverse(mean_s, Tt_raw, stats)
            samples.append(mean_raw.squeeze(1))  # drop I -> [B,T,1]

        samples = torch.stack(samples, dim=0)  # [S,B,T,1]

        return samples, times, mask
