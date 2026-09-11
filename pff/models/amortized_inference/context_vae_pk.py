"""Logging-free Context-VAE PK model built on the new base stack.

All tensor shapes in this module follow the conventions:
``B`` - batch size, ``I`` - number of individuals, ``T`` - time dimension and
``Z`` - latent dimension. Comments annotate intermediate tensor shapes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torchtyping import TensorType, patch_typeguard
from tqdm import tqdm

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import (
    AbstractForwardOutputs,
    NewBasePKModel,
    NewGenerativeMixin,
)
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)
from pff.models.utils.loss_utils import MultiHeadLoss
from pff.utils.tensors_operations import gather_distinct_times_per_substance

patch_typeguard()


class ContextVAEForwardOutputs(AbstractForwardOutputs):
    """Permutation-aware outputs collected by :class:`NewContextVAEPK`.

    ``HEAD_SCHEMAS`` declares a single ``"reconstruction"`` head storing the
    decoded mean and log-variance together with their targets and masks.
    ``LOSS_SCHEMAS`` exposes the scalar losses reduced across permutations: the
    reconstruction loss, the KL divergence on the study latent, the KL on the
    initial condition and the optional RMSE of the initial-condition
    reconstruction.
    """

    HEAD_SCHEMAS = {"reconstruction": ["mean", "logvar", "target", "mask"]}
    LOSS_SCHEMAS = {"reconstruction": ["recon_loss", "kl_s", "kl_init", "init_rmse", "rmse"]}

    def __init__(
        self,
        *,
        loss_multihead: Optional[Callable[[List[torch.Tensor]], Tuple[torch.Tensor, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the container with the optional multihead loss reducer."""

        super().__init__(loss_multihead=loss_multihead, device=device)
        self.stats: Optional[TensorType["B", 1]] = None

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
                _get("kl_init"),
                _get("init_rmse"),
            ]
        )
        return total_loss


class ContextVAEPK(NewBasePKModel, NewGenerativeMixin):
    """Context Variational Autoencoder for PK data (logging-free).

    This model reconstructs target observations using context data only. It
    mirrors the legacy ``ContextVAEPK`` logic but relies on the logging-free
    base stack and callbacks for visualization and empirical reporting.
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        z_dim = self.encoder.zi_latent_dim
        aggregator_type = model_config.network.aggregator_type
        self.ignore_logvar = getattr(model_config.network, "ignore_logvar", True)

        if aggregator_type == "mean":
            self.aggregator = MeanStudyAggregator()
        elif aggregator_type == "attention":
            self.aggregator = AttentionStudyAggregator(
                z_dim, model_config.network.aggregator_num_heads
            )
        else:
            raise ValueError(f"Unknown aggregator_type '{aggregator_type}'.")

        self.mu_s_layer = nn.Linear(z_dim, z_dim)
        self.logvar_s_layer = nn.Linear(z_dim, z_dim)

        self.mu_i_layer = nn.Linear(z_dim, z_dim)
        self.logvar_i_layer = nn.Linear(z_dim, z_dim)

        self.mu_init_layer = nn.Linear(z_dim, 1)
        self.logvar_init_layer = nn.Linear(z_dim, 1)

        # (recon, KL_s, KL_init, init_rmse)
        self.loss_multihead = MultiHeadLoss(mode="learnable", number_of_losses=4)

    def build_visualization_callback(self):
        """Return callbacks that handle visualization and empirical evaluation."""
        return super().build_visualization_callback()

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------
    def forward(
        self,
        databatch_list: Sequence[AICMECompartmentsDataBatch] | AICMECompartmentsDataBatch,
    ) -> ContextVAEForwardOutputs:
        """Run reconstruction-only forward passes over every permutation."""

        if isinstance(databatch_list, AICMECompartmentsDataBatch):
            databatch_list = [databatch_list]

        outputs_collector = ContextVAEForwardOutputs(loss_multihead=self.loss_multihead)
        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> ContextVAEForwardOutputs:
        """Reconstruct target observations from context-only latents."""

        Xc_raw, Tc_raw, Mc = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        # Xc_raw/Tc_raw: [B, Ic, Tc, 1], Mc: [B, Ic, Tc]
        # Xt_raw/Tt_raw: [B, It, Tt, 1], Mt: [B, It, Tt]

        outputs = ContextVAEForwardOutputs(
            loss_multihead=self.loss_multihead,
            device=Xt_raw.device,
        )

        # Scaling statistics from context only
        stats = self.scaler.stats(Xc_raw, Tc_raw, Mc)  # [B, 1]
        outputs.stats = stats

        # Scale both blocks with the same stats
        Xc_s, Tc_s = self.scaler.forward(Xc_raw, Tc_raw, stats)  # [B, Ic, Tc, 1]
        Xt_s, Tt_s = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, Tt, 1]

        # Study latent from context (deterministic, no KL on z_s)
        z_ci = self.encoder(
            Xc_s,
            Tc_s,
            Mc,
            db.context_dosing_amounts,
            db.context_dosing_route_types,
            db.mask_individuals,
        )  # [B, Ic, Z]
        z_s_agg = self.aggregator(z_ci, db.mask_individuals)  # [B, Z]
        z_s = self.mu_s_layer(z_s_agg)  # [B, Z]
        kl_s = torch.zeros((), device=z_s.device, dtype=z_s.dtype)

        # Individual latent from target (variational)
        z_ti = self.encoder(
            Xt_s,
            Tt_s,
            Mt,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )  # [B, It, Z]
        mu_i = self.mu_i_layer(z_ti)  # [B, It, Z]
        logvar_i = self.logvar_i_layer(z_ti).clamp(min=-15.0, max=15.0)  # [B, It, Z]

        if self.training and not self.ignore_logvar:
            eps = torch.randn_like(mu_i)
            z_i = mu_i + eps * torch.exp(0.5 * logvar_i)  # [B, It, Z]
        else:
            z_i = mu_i  # [B, It, Z]

        kl_i = 0.5 * (mu_i.pow(2) + logvar_i.exp() - logvar_i - 1).mean()
        if not getattr(self.model_config.network, "use_kl_i", True):
            kl_i = torch.zeros((), device=kl_i.device, dtype=kl_i.dtype)

        # Initial condition from the first valid target observation
        init_true, init_mask, first_t_s = self.get_first_valid_observation(Xt_s, Mt, Tt_s)
        # init_true: [B, It, 1, 1], init_mask: [B, It, 1], first_t_s: [B, It, 1, 1]
        init_s, mu_init, logvar_init = self._sample_initial_condition(z_s, z_i)
        # init_s: [B, It, 1, 1], mu_init/logvar_init: [B, It, 1]

        B, I, T, _ = Xt_s.shape
        BI = B * I
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B, It, 1, 1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)  # [B, It, 1, 1]

        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI, Tt, 1]
        init_state = init_s.view(BI, 1, -1)  # [BI, 1, 1]
        first_t_feat = first_t_s.view(BI, 1, -1)  # [BI, 1, 1]
        dose_feat = dose.view(BI, 1, -1)  # [BI, 1, 1]
        route_feat = route.view(BI, 1, -1)  # [BI, 1, 1]

        mean_s, logvar_s_pred, H, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=first_t_feat,
        )  # [BI, Tt, 1], [BI, Tt, 1], H: [BI, Tt, p] or None
        mean_s = mean_s.view(B, I, T, 1)  # [B, It, Tt, 1]
        logvar_s_pred = logvar_s_pred.view(B, I, T, 1)  # [B, It, Tt, 1]
        mean_raw, _ = self.scaler.inverse(mean_s, Tt_s, stats)  # [B, It, Tt, 1]

        logvar_s_pred = torch.clamp(logvar_s_pred, min=-100.0, max=100.0)
        recon_loss_dict = self.compute_loss(mean_s, logvar_s_pred, Xt_s, Mt, H)

        kl_init = 0.5 * (mu_init.pow(2) + logvar_init.exp() - logvar_init - 1).mean()
        if not getattr(self.model_config.network, "use_kl_init", True):
            kl_init = torch.zeros((), device=kl_init.device, dtype=kl_init.dtype)

        init_rmse = self.masked_rmse_loss(mu_init.unsqueeze(-1), init_true, init_mask)["rmse"]

        outputs.update_head(
            "reconstruction",
            {
                "mean": mean_raw,  # raw space for reporting
                "logvar": logvar_s_pred,
                "target": Xt_raw,
                "mask": Mt,
            },
        )

        # Only KL(z_i) is used; expose it through the "kl_s" slot for compatibility.
        outputs.update_losses(
            "reconstruction",
            {
                "recon_loss": recon_loss_dict["loss"],
                "kl_s": kl_i,
                "kl_init": kl_init,
                "init_rmse": init_rmse,
                "rmse": recon_loss_dict["rmse"],
            },
        )
        return outputs

    def _sample_initial_condition(
        self,
        z_s: TensorType["B", "Z"],
        z_i: TensorType["B", "I", "Z"],
    ) -> Tuple[
        TensorType["B", "I", 1, 1],
        TensorType["B", "I", 1],
        TensorType["B", "I", 1],
    ]:
        """Sample initial conditions conditioned on study and individual latents."""

        z_comb = self.decoder.combine_latents(z_s, z_i)  # [B, I, Z]
        mu_init = self.mu_init_layer(z_comb)  # [B, I, 1]
        logvar_init = self.logvar_init_layer(z_comb)  # [B, I, 1]
        eps_init = torch.randn_like(mu_init)
        init_s = mu_init + eps_init * torch.exp(0.5 * logvar_init)  # [B, I, 1]
        return init_s.unsqueeze(-1), mu_init, logvar_init

    # ------------------------------------------------------------------
    # NewGenerativeMixin implementation
    # ------------------------------------------------------------------
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
        ignore_logvar: bool | None = None,
        num_steps: int | None = None,
        dosing: Tuple[TensorType["B"], TensorType["B"]] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max", 1],
        TensorType["B", "Tdistinct_max"],
    ]:
        """Sample trajectories for new individuals conditioned on context."""
        _ = num_steps
        _ = include_rem
        if resolve_sampling_from_target:
            raise ValueError("ContextVAEPK does not implement target-resolved sampling.")

        if ignore_logvar is None:
            ignore_logvar = self.ignore_logvar

        # Context encoding on scaled data
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
        z_s_context = self.aggregator(z_ci, db.mask_individuals)  # [B, Z]
        z_s = self.mu_s_layer(z_s_context)  # [B, Z]

        B = db.context_obs.shape[0]
        I = 1  # always generate exactly one new individual per substance
        BI = B * I

        # Resolve decode grid (raw) and scale it like in training
        if decode_times is None:
            times, mask = gather_distinct_times_per_substance(db)  # [B, T, 1], [B, T]
        else:
            times, mask = decode_times

        device = z_s.device
        times = times.to(device)
        mask = mask.to(device)

        Tt_raw = times.unsqueeze(1).expand(B, I, -1, -1)  # [B, I, T, 1]
        zeros_like_vals = torch.zeros_like(Tt_raw)  # [B, I, T, 1]
        _, Tt_s = self.scaler.forward(zeros_like_vals, Tt_raw, stats)  # [B, I, T, 1]

        first_t_s = Tt_s[:, :, 0:1, :]  # [B, I, 1, 1]
        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI, T, 1]
        Tdec = decode_t.shape[1]

        if dosing is None:
            dose_values, route_values = self._select_context_dosing(db)
        else:
            dose_values, route_values = dosing
        dose = dose_values.to(device).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        route = route_values.to(device).float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        use_mv_covariance = bool(getattr(self.decoder, "use_covariance", False))

        samples = []
        for _ in tqdm(range(sample_size), desc="Sampling new individuals", ncols=80):
            z_i = torch.randn(B, I, self.encoder.zi_latent_dim, device=device)  # [B, 1, Z]
            init_s, _, _ = self._sample_initial_condition(z_s, z_i)  # [B, 1, 1, 1]

            mean_s, logvar_pred, H, _, _ = self.decoder(
                init_s.view(BI, 1, 1),  # [BI, 1, 1]
                decode_t,  # [BI, T, 1]
                z_s,
                z_i,
                dose=dose.view(BI, 1, -1),
                route=route.view(BI, 1, -1),
                first_t_s=first_t_s.view(BI, 1, -1),
            )  # mean_s/logvar_pred: [BI, T, 1]

            if ignore_logvar:
                sample_s = mean_s  # [BI, T, 1]
            else:
                if use_mv_covariance and (H is not None):
                    cov = H @ H.transpose(-1, -2)  # [BI, T, T]
                    L = torch.tril(cov)  # [BI, T, T]
                    xi = torch.randn(BI, Tdec, 1, device=device)  # [BI, T, 1]
                    sample_s = mean_s + (L @ xi)  # [BI, T, 1]
                else:
                    std = torch.exp(0.5 * logvar_pred)  # [BI, T, 1]
                    eps = torch.randn_like(mean_s)
                    sample_s = mean_s + eps * std  # [BI, T, 1]

            sample_s = sample_s.view(B, I, -1, 1)  # [B, 1, T, 1]
            sample_raw, _ = self.scaler.inverse(sample_s, Tt_s, stats)  # [B, 1, T, 1]
            samples.append(sample_raw.squeeze(1))  # [B, T, 1]

        stacked = torch.stack(samples, dim=0)  # [S, B, T, 1]
        return stacked, times, mask

    def _select_context_dosing(
        self, db: AICMECompartmentsDataBatch
    ) -> Tuple[TensorType["B"], TensorType["B"]]:
        """Return representative dosing (amount and route) from context individuals."""

        amount = db.context_dosing_amounts  # [B, Ic]
        route = db.context_dosing_route_types  # [B, Ic]
        mask = getattr(db, "mask_context_individuals", None)  # [B, Ic] or None

        B, I = amount.shape
        dose_out = torch.zeros(B, device=amount.device, dtype=amount.dtype)  # [B]
        route_out = torch.zeros(B, device=route.device, dtype=route.dtype)  # [B]

        for b in range(B):
            if mask is not None:
                valid = torch.nonzero(mask[b], as_tuple=False).view(-1)
            else:
                valid = torch.arange(I, device=amount.device)
            if valid.numel() == 0:
                continue
            idx = int(valid[0].item())
            dose_out[b] = amount[b, idx]
            route_out[b] = route[b, idx]
        return dose_out, route_out


__all__ = ["ContextVAEForwardOutputs", "ContextVAEPK"]
