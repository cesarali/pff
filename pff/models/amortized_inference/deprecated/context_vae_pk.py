"""Context-VAE pharmacokinetic model and forward-output helpers.

The :class:`ContextVAEPK` model specialises the generic utilities defined in
:mod:`pff.models.amortized_inference.generative_pk`.  It reuses the
encoder/decoder/scaler trio provided by :class:`BasePKModel` and relies on the
permutation-aware :class:`AbstractForwardOutputs` base class to accumulate
losses across the multiple context/target permutations emitted by the data
module.  Each permutation corresponds to a distinct split of the same study; the
model runs a reconstruction-only forward pass for every split and aggregates the
results with :meth:`AbstractForwardOutputs.reduce`.

Usage example
-------------

>>> model = ContextVAEPK(config)
>>> outputs = model.forward(databatch_list)
>>> losses = outputs.to_dict()

where ``databatch_list`` is the permutation list returned by the
``AICMECompartmentsDataModule``.  ``losses`` contains the averaged reconstruction
loss and KL terms across permutations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torchtyping import TensorType
from tqdm import tqdm

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.deprecated.generative_pk import (
    BasePKModel,
    GenerativeMixin,
)
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures.aggregators import (
    AttentionStudyAggregator,
    MeanStudyAggregator,
)
from pff.models.utils.loss_utils import MultiHeadLoss
from pff.utils.tensors_operations import gather_distinct_times_per_substance


class ContextVAEForwardOutputs(AbstractForwardOutputs):
    """Permutation-aware outputs collected by :class:`ContextVAEPK`.

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
        """Initialise the container with the optional multihead loss reducer."""

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


class ContextVAEPK(BasePKModel, GenerativeMixin):
    """Context Variational Autoencoder for Pharmacokinetics (Context-VAE-PK).

    This model reconstructs pharmacokinetic time-concentration profiles directly
    from context data, without prediction or target supervision.  It uses the
    standard :class:`BasePKModel` infrastructure (encoder, decoder and scaler)
    and relies on :class:`GenerativeMixin` for sampling utilities.

    Losses include the reconstruction objective, the KL divergence for the study
    latent (:math:`KL_S`), the KL divergence for the initial condition
    (:math:`KL_INIT`) and an optional RMSE regulariser on the initial condition.

    As with :class:`AICMEPK`, multiple permutations of the same study are
    supported through :class:`AbstractForwardOutputs`.  Each permutation produces
    an intermediate :class:`ContextVAEForwardOutputs` that is aggregated via
    :meth:`AbstractForwardOutputs.reduce` to yield averaged heads and losses.
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

    # ------------------------------------------------------------------
    # Forward computation
    # ------------------------------------------------------------------
    def forward(
        self, databatch_list: Sequence[AICMECompartmentsDataBatch]
    ) -> ContextVAEForwardOutputs:
        """Run reconstruction-only forward passes over every permutation."""

        outputs_collector = ContextVAEForwardOutputs(
            loss_multihead=self.loss_multihead,
        )
        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> ContextVAEForwardOutputs:
        """Reconstruct **target** observations.

        - z_s: deterministic study latent from aggregated context (no KL on z_s).
        - z_i: variational individual latent from targets (KL comes from here).
        - Decode on the target time grid; losses computed against target.
        """

        # ---- raw tensors
        Xc_raw, Tc_raw, Mc = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask

        outputs = ContextVAEForwardOutputs(
            loss_multihead=self.loss_multihead,
            device=Xt_raw.device,
        )

        # ---- scaling stats from context (as during training)
        stats = self.scaler.stats(Xc_raw, Tc_raw, Mc)
        outputs.stats = stats

        # Scale both blocks with the SAME stats
        Xc_s, Tc_s = self.scaler.forward(Xc_raw, Tc_raw, stats)
        Xt_s, Tt_s = self.scaler.forward(Xt_raw, Tt_raw, stats)

        # ---- study latent from context (DETERMINISTIC; no KL)
        z_ci = self.encoder(
            Xc_s,
            Tc_s,
            Mc,
            db.context_dosing_amounts,
            db.context_dosing_route_types,
            db.mask_individuals,
        )  # [B, Ic, Z]
        z_s_agg = self.aggregator(z_ci, db.mask_individuals)  # [B, Z]
        z_s = self.mu_s_layer(z_s_agg)  # [B, Z] projection only
        kl_s = torch.zeros((), device=z_s.device, dtype=z_s.dtype)

        # ---- individual latent from target (VARIATIONAL; this is the only KL)
        z_ti = self.encoder(
            Xt_s,
            Tt_s,
            Mt,
            db.target_dosing_amounts,
            db.target_dosing_route_types,
        )  # [B, It, Z]

        mu_i = self.mu_i_layer(z_ti)  # [B, It, Z]
        logvar_i = self.logvar_i_layer(z_ti)  # [B, It, Z]
        logvar_i = torch.clamp(logvar_i, min=-15.0, max=15.0)

        if self.training and not self.ignore_logvar:
            eps = torch.randn_like(mu_i)
            z_i = mu_i + eps * torch.exp(0.5 * logvar_i)  # reparam sample
        else:
            z_i = mu_i  # mean at eval or if ignoring logvar

        # KL for z_i ~ N(mu_i, diag(exp(logvar_i))) vs N(0, I)
        kl_i = 0.5 * (mu_i.pow(2) + logvar_i.exp() - logvar_i - 1).mean()

        # ---- initial condition from target first valid obs (unchanged)
        init_true, init_mask, first_t_s = self.get_first_valid_observation(Xt_s, Mt, Tt_s)
        init_s, mu_init, logvar_init = self._sample_initial_condition(z_s, z_i)

        # ---- decode on target grid (relative to first target time)
        B, I, T, _ = Xt_s.shape
        BI = B * I

        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B,It,1,1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)

        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI,Tt,1]
        init_state = init_s.view(BI, 1, -1)  # [BI,1,1]
        first_t_feat = first_t_s.view(BI, 1, -1)
        dose_feat = dose.view(BI, 1, -1)
        route_feat = route.view(BI, 1, -1)

        mean_s, logvar_s_pred, H, _, _ = self.decoder(
            init_state,
            decode_t,
            z_s,
            z_i,
            dose=dose_feat,
            route=route_feat,
            first_t_s=first_t_feat,
        )  # [BI,Tt,1], [BI,Tt,1], ...
        mean_s = mean_s.view(B, I, T, 1)
        logvar_s_pred = logvar_s_pred.view(B, I, T, 1)
        mean_raw, _ = self.scaler.inverse(mean_s, Tt_s, stats)

        # ---- losses against target
        logvar_s_pred = torch.clamp(logvar_s_pred, min=-100.0, max=100.0)
        recon_loss_dict = self.compute_loss(mean_s, logvar_s_pred, Xt_s, Mt, H)

        kl_init = 0.5 * (mu_init.pow(2) + logvar_init.exp() - logvar_init - 1).mean()
        if not getattr(self.model_config.network, "use_kl_init", True):
            kl_init = torch.zeros_like(kl_init)

        init_rmse = self.masked_rmse_loss(mu_init.unsqueeze(-1), init_true, init_mask)["rmse"]

        # ---- heads + scalars
        outputs.update_head(
            "reconstruction",
            {
                "mean": mean_raw,  # RAW space for reporting
                "logvar": logvar_s_pred,
                "target": Xt_raw,
                "mask": Mt,
            },
        )

        # IMPORTANT: keep the 4-head structure (recon, KL_s, KL_init, init_rmse)
        # We place KL(z_i) into the 'kl_s' slot, since z_s has no KL.
        outputs.update_losses(
            "reconstruction",
            {
                "recon_loss": recon_loss_dict["loss"],
                "kl_s": kl_i,  # <- ONLY KL term for latents
                "kl_init": kl_init,
                "init_rmse": init_rmse,
                "rmse": recon_loss_dict["rmse"],
            },
        )
        return outputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_initial_condition(
        self,
        z_s: TensorType["B", "Z"],
        z_i: TensorType["B", "I", "Z"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample initial conditions conditioned on study and individual latents."""

        z_comb = self.decoder.combine_latents(z_s, z_i)
        mu_init = self.mu_init_layer(z_comb)
        logvar_init = self.logvar_init_layer(z_comb)
        eps_init = torch.randn_like(mu_init)
        init_s = mu_init + eps_init * torch.exp(0.5 * logvar_init)
        return init_s.unsqueeze(-1), mu_init, logvar_init

    # ------------------------------------------------------------------
    # GenerativeMixin implementation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
        num_steps: int = None,
        *,
        ignore_logvar: Optional[bool] = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        """Sample trajectories for new individuals conditioned on context.

        Options
        -------
        ignore_logvar:
            If True, ignore predictive uncertainty and sample deterministically
            using the decoder mean.
            If None it fallbacks to the config, meaning the input to the function overrides
            config

        use_mv_covariance:
            If None, inferred from `self.decoder.use_covariance` (if present).
            If True and decoder returns `H` (shape [BI,T,p]), build
            L = lower(H Hᵀ) by zeroing the upper triangle, and sample with
            sample = mean + L @ ξ,  ξ ~ N(0, I_T).
            If False or H is None, fall back to diagonal sampling with logvar.
        """
        if ignore_logvar is None:
            ignore_logvar = self.ignore_logvar

        # ---- (A) Context encoding on *scaled* data (same as training) ----
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)
        Xc_s, Tc_s = self.scaler.forward(db.context_obs, db.context_obs_time, stats)

        z_ci = self.encoder(
            Xc_s,
            Tc_s,
            db.context_obs_mask,
            db.context_dosing_amounts,
            db.context_dosing_route_types,
            db.mask_individuals,
        )
        z_s_context = self.aggregator(z_ci, db.mask_individuals)
        z_s = self.mu_s_layer(z_s_context)

        B = db.context_obs.shape[0]
        I = 1
        BI = B * I

        # ---- (B) Resolve decode grid (RAW), then scale it like in training ----
        if decode_times is None:
            times, mask = gather_distinct_times_per_substance(db)  # times: [B,T,1], mask: [B,T]
        else:
            times, mask = decode_times

        device = z_s.device
        times = times.to(device)
        mask = mask.to(device)

        # Broadcast raw times to [B, I, T, 1]
        Tt_raw = times.unsqueeze(1).expand(B, I, -1, -1)  # [B,I,T,1]

        # Reuse scaler for time transform
        zeros_like_vals = torch.zeros_like(Tt_raw)  # [B,I,T,1]
        _, Tt_s = self.scaler.forward(zeros_like_vals, Tt_raw, stats)  # [B,I,T,1] (scaled)

        # Relative decode times in *scaled* space
        first_t_s = Tt_s[:, :, 0:1, :]  # [B,I,1,1]
        decode_t = (Tt_s - first_t_s).view(BI, -1, 1)  # [BI,T,1]
        Tdec = decode_t.shape[1]

        # ---- (C) Dosing / route tensors ----
        dose_values, route_values = self._select_context_dosing(db)
        dose = dose_values.to(device).unsqueeze(-1).unsqueeze(-1)  # [B,I,1,1]
        route = route_values.to(device).float().unsqueeze(-1).unsqueeze(-1)  # [B,I,1,1]

        # ---- (D) Sampling loop ----
        use_mv_covariance = bool(getattr(self.decoder, "use_covariance", False))

        samples = []
        for _ in tqdm(range(sample_size), desc="Sampling new individuals", ncols=80):
            # individual-level latent
            z_i = torch.randn(B, I, self.encoder.zi_latent_dim, device=device)

            # initial condition
            init_s, _, _ = self._sample_initial_condition(z_s, z_i)

            # decode in scaled-time space
            mean_s, logvar_pred, H, _, _ = self.decoder(
                init_s.view(BI, 1, 1),
                decode_t,
                z_s,
                z_i,
                dose=dose.view(BI, 1, -1),
                route=route.view(BI, 1, -1),
                first_t_s=first_t_s.view(BI, 1, -1),
            )  # mean_s/logvar_pred: [BI,T,1]; H: [BI,T,p] or None

            if ignore_logvar:
                sample_s = mean_s  # deterministic path
            else:
                if use_mv_covariance and (H is not None):
                    # ---  GP-style sampling with L = lower(H Hᵀ) ---
                    # Build covariance surrogate and take its strict lower-triangular part
                    cov = H @ H.transpose(-1, -2)  # [BI,T,T]
                    L = torch.tril(cov)  # lower(H Hᵀ)  (zero upper triangle)
                    xi = torch.randn(BI, Tdec, 1, device=device)  # [BI,T,1]
                    sample_s = mean_s + (L @ xi)  # [BI,T,1]
                else:
                    # Diagonal sampling using log-variance
                    std = torch.exp(0.5 * logvar_pred)
                    eps = torch.randn_like(mean_s)
                    sample_s = mean_s + eps * std

            # reshape back to [B,I,T,1] and inverse-scale to RAW space
            sample_s = sample_s.view(B, I, -1, 1)
            sample_raw, _ = self.scaler.inverse(sample_s, Tt_s, stats)  # [B,I,T,1]
            samples.append(sample_raw.squeeze(1))  # [B,T,1]

        stacked = torch.stack(samples, dim=0)  # [S,B,T,1]
        return stacked, times, mask

    def _select_context_dosing(
        self, db: AICMECompartmentsDataBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return representative dosing (amount and route) from context individuals."""

        amount = db.context_dosing_amounts
        route = db.context_dosing_route_types
        mask = getattr(db, "mask_context_individuals", None)
        B, I = amount.shape
        dose_out = torch.zeros(B, device=amount.device, dtype=amount.dtype)
        route_out = torch.zeros(B, device=route.device, dtype=route.dtype)
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
