"""
Diffusion-based PK models (discrete and continuous time).

This module mirrors the high-level API of :class:`FlowPK` but replaces the
flow-matching objective with diffusion-based objectives.  The encoder,
scaler, latent aggregation and decoder are *identical* to those used in
``flows_pk.py`` so that model configurations can be reused with minimal
changes.  The main differences are:

* The reconstruction objective is formulated as denoising diffusion
  (discrete or continuous time).
* The decoder is interpreted as a *denoiser* predicting Gaussian noise
  on scaled PK trajectories rather than a vector field on an
  interpolation path.
* The shapes of all tensors follow exactly the same conventions as in
  :class:`FlowPK`:

    - ``Xc``: [B, Ic, T, 1]  context observations
    - ``Xt``: [B, It, T, 1]  target observations
    - ``Tc``, ``Tt``: [B, I, T, 1]  times
    - masks: [B, I, T]
    - latent study embedding ``z_s``: [B, Z]
    - latent individual embeddings ``z_ci``: [B, Ic, Z]
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torchtyping import TensorType

from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.deprecated.generative_pk import (
    BasePKModel,
    GenerativeMixin,
)
from pff.models.amortized_inference.generative_pk import AbstractForwardOutputs
from pff.models.architectures.aggregators import MeanStudyAggregator
from pff.models.diffusion.continuous_diffusion import ContinuousDiffusion
from pff.models.diffusion.discrete_diffusion import GPDiffusion
from pff.models.utils.loss_utils import MultiHeadLoss

# ---------------------------------------------------------------------------
# Forward-output container
# ---------------------------------------------------------------------------


class DiffusionForwardOutputs(AbstractForwardOutputs):
    """
    Forward outputs for diffusion-based PK models.

    This is intentionally almost identical to :class:`FlowForwardOutputs`
    used by :class:`FlowPK` in ``flows_pk.py``:

    * A single ``"reconstruction"`` head with fields:

        - ``prediction``: model outputs (here, predicted noise)
        - ``target``: diffusion targets (true noise)
        - ``mask``: per-individual mask

    * A single ``"rmse"`` loss in the ``"reconstruction"`` scope.
    """

    HEAD_SCHEMAS = {"reconstruction": ["prediction", "target", "mask"]}
    LOSS_SCHEMAS = {"reconstruction": ["rmse"]}  # Single scalar loss

    def __init__(
        self,
        *,
        loss_multihead: Optional[Callable[[List[torch.Tensor]], Tuple[torch.Tensor, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(loss_multihead=loss_multihead, device=device)
        # Optionally cache scaling statistics alongside permutation outputs.
        self.stats: Optional[TensorType["B", 1]] = None

    def _compute_total_loss(self, flat_losses: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Compute total loss from flattened losses.

        For diffusion PK models we default to a single reconstruction loss
        (``"rmse"``) and optionally route it through a
        :class:`MultiHeadLoss` wrapper to keep the same interface as other
        PK models.
        """
        if not flat_losses:
            return None
        if self.loss_multihead is None:
            return super()._compute_total_loss(flat_losses)

        device = self._infer_device()
        zero = torch.zeros((), device=device)
        mse_loss = flat_losses.get("rmse", zero)

        # MultiHeadLoss expects a list of losses.
        total_loss, _ = self.loss_multihead([mse_loss])
        return total_loss


# ---------------------------------------------------------------------------
# Beta schedule utilities
# ---------------------------------------------------------------------------


class LinearBetaSchedule:
    """
    Simple linear beta schedule usable for both discrete and continuous diffusion.

    The schedule is defined on the normalised interval :math:`t ∈ [0, 1]`:

        β(t) = β_min + (β_max - β_min) * t

    For continuous diffusion we additionally expose the primitive

        ∫₀ᵗ β(s) ds = β_min * t + 0.5 * (β_max - β_min) * t²

    which is what :class:`ContinuousDiffusion` expects via
    ``beta_fn.integral``.
    """

    def __init__(self, beta_min: float = 1e-4, beta_max: float = 2e-2) -> None:
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate β(t) element-wise on tensor ``t``.

        Parameters
        ----------
        t:
            Diffusion time(s) in [0, 1].  Shape is arbitrary.

        Returns
        -------
        Tensor
            Same shape as ``t``.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the time integral ∫₀ᵗ β(s) ds.

        Parameters
        ----------
        t:
            Diffusion time(s) in [0, 1].  Shape is arbitrary.

        Returns
        -------
        Tensor
            Same shape as ``t``.
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2


# ---------------------------------------------------------------------------
# Shared base class for diffusion PK models
# ---------------------------------------------------------------------------


class _BaseDiffusionPK(BasePKModel, GenerativeMixin):
    """
    Shared utilities for diffusion-based PK models.

    This class encapsulates the encoder / decoder, latent aggregation and
    scaling logic common to both discrete and continuous diffusion
    variants.  Subclasses only implement:

    * construction of the diffusion object (discrete vs continuous), and
    * the specific diffusion training and sampling logic.
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        # Latent dimensionality of individual encoder.
        self.z_dim = self.encoder.zi_latent_dim  # type: ignore[attr-defined]
        self.aggregator = MeanStudyAggregator()
        self.loss_multihead = MultiHeadLoss(mode="fixed", number_of_losses=1)

        # ------------------------------------------------------------------
        # Diffusion hyper-parameters from configuration (with safe defaults)
        # ------------------------------------------------------------------
        net_cfg = getattr(model_config, "network", None)

        def _get(name: str, default: Any) -> Any:
            return getattr(net_cfg, name, default) if net_cfg is not None else default

        # Number of diffusion steps (discrete case) / discretisation points.
        self.num_diffusion_steps: int = int(_get("diffusion_num_steps", 100))

        # Final diffusion time for continuous diffusion (t1 in [0, 1]).
        self.diffusion_t1: float = float(_get("diffusion_t1", 1.0))

        # Beta schedule parameters.
        self.beta_min: float = float(_get("diffusion_beta_min", 1e-4))
        self.beta_max: float = float(_get("diffusion_beta_max", 2e-2))

        # Whether the model predicts unit Gaussian noise or correlated noise.
        self.predict_gaussian_noise: bool = bool(_get("diffusion_predict_gaussian_noise", True))

        # Marginal GP variance scale used in time-series diffusion.
        self.diffusion_sigma: float = float(_get("diffusion_sigma", 0.1))

        # Shared beta schedule object usable by discrete and continuous diffusion.
        self.beta_schedule = LinearBetaSchedule(self.beta_min, self.beta_max)

    # ------------------------------------------------------------------
    # High-level forward API (shared)
    # ------------------------------------------------------------------

    def forward(
        self, databatch_list: Sequence[AICMECompartmentsDataBatch]
    ) -> DiffusionForwardOutputs:
        """
        Run diffusion-based reconstruction over all permutations in a study.

        Parameters
        ----------
        databatch_list:
            Sequence of :class:`AICMECompartmentsDataBatch` objects representing
            different context/target permutations for a single study.

        Returns
        -------
        DiffusionForwardOutputs
            Aggregated losses and (optionally) reconstruction heads.
        """
        outputs_collector = DiffusionForwardOutputs(loss_multihead=self.loss_multihead)

        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    # ------------------------------------------------------------------
    # Latent encoding and scaling utilities (shared)
    # ------------------------------------------------------------------

    def _study_latent(
        self, db: AICMECompartmentsDataBatch, use_target: bool = False
    ) -> Tuple[
        TensorType["B", "Z"],  # z_s: study-level latent
        TensorType["B", "Ic", "Z"],  # z_ci: context individual latents
        TensorType["B", 1],  # stats placeholder from scaler
    ]:
        """
        Encode a study into a shared latent representation ``z_s``.

        Parameters
        ----------
        db:
            Single permutation minibatch.
        use_target:
            Unused but kept for API compatibility with FlowPK.

        Returns
        -------
        z_s:
            Study-level latent embedding with shape ``[B, Z]``.
        z_ci:
            Context individual latent embeddings with shape ``[B, Ic, Z]``.
        stats:
            Scaling statistics object produced by :class:`PKScaler`.
        """
        del use_target  # unused, kept for signature compatibility

        Xc_raw, Tc_raw, M = db.context_obs, db.context_obs_time, db.context_obs_mask
        mask_ind = db.mask_individuals

        # Scaling statistics from context only (identical to FlowPK).
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)
        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)
        # Xc: [B, Ic, T, 1]
        # Tc: [B, Ic, T, 1]

        dose = db.context_dosing_amounts  # [B, Ic, 1]
        route = db.context_dosing_route_types  # [B, Ic, 1]

        # Encoder produces individual latents per context individual.
        z_ci = self.encoder(Xc, Tc, M, dose, route, mask_ind)  # [B, Ic, Z]
        z_s = self.aggregator(z_ci, mask_ind)  # [B, Z]

        return z_s, z_ci, stats

    # ------------------------------------------------------------------
    # Decoder wrapper shared by discrete and continuous diffusion
    # ------------------------------------------------------------------

    def _make_denoiser(
        self,
        *,
        Xt: TensorType["B", "It", "T", 1],
        Tt: TensorType["B", "It", "T", 1],
        db: AICMECompartmentsDataBatch,
        z_s: TensorType["B", "Z"],
        stats: TensorType["B", 1],
    ) -> Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]:
        """
        Build a closure that wraps the PK decoder as a diffusion *denoiser*.

        The returned callable has signature::

            denoiser(x_noisy, i, t) -> pred_noise

        where

        * ``x_noisy``: [B, It, T, 1]  noisy trajectory at diffusion time i
        * ``i``      : [B, It, T, 1]  diffusion time index / scalar per batch
        * ``t``      : [B, It, T, 1]  scaled observation times

        and returns

        * ``pred_noise``: [B, It, T, 1] predicted (Gaussian) noise.

        Internally this wrapper reuses the same features and shapes as the
        flow-matching decoder in :class:`FlowPK`:

        * first valid observation per individual as initial state,
        * study and individual latents (currently using the study latent for
          target individuals, exactly like FlowPK),
        * dosing and route features,
        * flow_time = normalised diffusion step in [0, 1],
        * flow_path = current path ``x_noisy`` with individual dimension
          squeezed, i.e. [B, T, 1].
        """
        # Shapes
        # -------
        # Xt, Tt: [B, It, T, 1]
        B, It, T, _ = Xt.shape
        BI = B * It
        device = Xt.device

        # First (scaled) observation per target individual and first time.
        init_true, init_mask, first_t_s = self.get_first_valid_observation(
            Xt, db.target_obs_mask, Tt
        )
        # init_true: [B, It, 1, 1]
        # first_t_s: [B, It, 1, 1]

        # Study-level latent is currently broadcast to target individuals,
        # just like in FlowPK.
        z_i = z_s.unsqueeze(1).repeat(1, It, 1)  # [B, It, Z]

        # Target dosing and route, expanded to match decoder expectations.
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B, It, 1, 1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)
        # route: [B, It, 1, 1]

        # Flatten individual dimension for features that are per-individual.
        init_state = init_true.view(BI, 1, -1)  # [B*It, 1, 1]
        first_t_feat = first_t_s.view(BI, 1, -1)  # [B*It, 1, 1]
        dose_feat = dose.view(BI, 1, -1)  # [B*It, 1, 1]
        route_feat = route.view(BI, 1, -1)  # [B*It, 1, 1]

        # Decoder will see time grid per substance: [B, T, 1]
        decode_times = Tt.squeeze(1)  # [B, T, 1] (It is usually == 1 in AICME)

        def denoiser(
            x_noisy: torch.Tensor,  # [B, It, T, 1]
            i: torch.Tensor,  # [B, It, T, 1]
            t: torch.Tensor,  # [B, It, T, 1] (scaled times, unused here)
        ) -> torch.Tensor:
            # Ensure shapes match expected batch size.
            assert x_noisy.shape[0] == B and x_noisy.shape[1] == It and x_noisy.shape[2] == T, (
                "Denoiser called with inconsistent shapes."
            )

            # Diffusion step index is constant across individuals and time for a
            # given batch element (by construction in diffusion code).  We take
            # one representative per batch and normalise to [0, 1].
            #   i_flat: [B]  scalar step per batch item
            i_flat = i[:, 0, 0, 0].float()
            denom = max(self.num_diffusion_steps - 1, 1)
            tau = i_flat / denom  # [B]
            flow_time = tau.view(B, 1, 1).to(device)  # [B, 1, 1]

            # Decoder expects path without individual dimension, [B, T, 1].
            flow_path = x_noisy.squeeze(1)  # [B, T, 1]; It is assumed 1.

            # Call the shared PK decoder.  This matches the call pattern in
            # FlowPK where:
            #
            #   init_state: [B*It, 1, 1]
            #   decode_times: [B, T, 1]
            #   z_s: [B, Z]
            #   z_i: [B, It, Z]
            #   dose_feat, route_feat, first_t_feat: [B*It, 1, 1]
            #   flow_time: [B, 1, 1]
            #   flow_path: [B, T, 1]
            vtau = self.decoder(
                init_state,
                decode_times,
                z_s,
                z_i,
                dose=dose_feat,
                route=route_feat,
                first_t_s=first_t_feat,
                flow_time=flow_time,
                flow_path=flow_path,
            )  # Expected shape: [B, 1, T, 1]

            # If there are multiple target individuals (It > 1), broadcast the
            # decoder output across them.  In AICME we typically have It == 1.
            if vtau.shape[1] == 1 and It > 1:
                vtau = vtau.expand(-1, It, -1, -1)  # [B, It, T, 1]

            return vtau

        return denoiser

    # ------------------------------------------------------------------
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------

    def _forward_reconstruction(
        self, db: AICMECompartmentsDataBatch
    ) -> DiffusionForwardOutputs:  # pragma: no cover - implemented in subclasses
        raise NotImplementedError

    @torch.inference_mode()
    def sample_new_individual(  # pragma: no cover - implemented in subclasses
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        num_steps: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Discrete-time diffusion PK model
# ---------------------------------------------------------------------------


class DiscreteDiffusionPK(_BaseDiffusionPK):
    """
    Discrete denoising diffusion PK model with GP noise.

    This class uses :class:`GPDiffusion` (discrete-time diffusion with
    Gaussian-process noise) applied to the scaled target trajectories
    ``Xt`` (shape [B, It, T, 1]).  The encoder, decoder and latent
    structure are identical to :class:`FlowPK`; only the training
    objective and sampling are changed to diffusion.

    The decoder is trained to predict *Gaussian* noise as in standard DDPM
    setups (``predict_gaussian_noise=True``).
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        # Discrete GP diffusion over the last dimension dim=1, with temporal
        # correlation across T via a Gaussian process.
        self.diffusion = GPDiffusion(
            dim=1,
            num_steps=self.num_diffusion_steps,
            beta_fn=self.beta_schedule,
            predict_gaussian_noise=self.predict_gaussian_noise,
            sigma=self.diffusion_sigma,
        )

    # ------------------------------------------------------------------
    # Training / reconstruction
    # ------------------------------------------------------------------

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> DiffusionForwardOutputs:
        """
        Single-permutation forward pass with discrete diffusion reconstruction.

        The main steps are:

        1. Encode context into study latent ``z_s``.
        2. Scale context and target trajectories.
        3. Form a noisy sample ``x_τ`` via discrete diffusion q(x_τ | x₀).
        4. Use the decoder as denoiser to predict Gaussian noise.
        5. Compute a masked MSE over the noise field.
        """
        # Encode study-level latent from context.
        z_s, z_ci, stats = self._study_latent(db, use_target=False)  # [B, Z], [B, Ic, Z]
        Xc_raw, Tc_raw, Mc = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        Mc_individuals = db.mask_context_individuals  # [B, Ic]
        Mt_individuals = db.mask_target_individuals  # [B, It]

        # Scale context and target using context-only statistics.
        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)  # [B, Ic, T, 1], [B, Ic, T, 1]
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1], [B, It, T, 1]

        # Shapes
        B, It, T, _ = Xt.shape
        device = Xt.device

        outputs = DiffusionForwardOutputs(loss_multihead=self.loss_multihead, device=Xt_raw.device)
        outputs.stats = stats

        # Build decoder wrapper that behaves like a diffusion denoiser.
        denoiser = self._make_denoiser(Xt=Xt, Tt=Tt, db=db, z_s=z_s, stats=stats)

        # ------------------------------------------------------------------
        # Discrete diffusion step
        # ------------------------------------------------------------------
        x_clean = Xt  # [B, It, T, 1]

        # Sample discrete diffusion step index i ~ Uniform{0, ..., N-1}
        # one scalar per batch element and broadcast to [B, It, T, 1].
        i_scalar = torch.randint(
            low=0,
            high=self.num_diffusion_steps,
            size=(B,),
            device=device,
        )  # [B]
        i = i_scalar.view(B, 1, 1, 1).expand_as(x_clean[..., :1])  # [B, It, T, 1]

        # Forward diffusion: x_noisy, noise ~ GP-correlated but we predict
        # *Gaussian* noise returned by GPDiffusion (because
        # predict_gaussian_noise=True).
        x_noisy, noise = self.diffusion.forward(x_clean, i, t=Tt)  # both [B, It, T, 1]

        # Predict Gaussian noise with the decoder.
        pred_noise = denoiser(x_noisy, i=i, t=Tt)  # [B, It, T, 1]

        # ------------------------------------------------------------------
        # Masked noise MSE loss
        # ------------------------------------------------------------------
        # For discrete diffusion we use an unweighted MSE on the noise field.
        # To reuse BasePKModel.masked_mse_loss we optionally multiply both
        # prediction and target by sqrt(weight).  Here weight=1.
        weight = torch.ones_like(pred_noise)  # [B, It, T, 1]
        scaled_pred = weight * pred_noise
        scaled_target = weight * noise

        mse_dict = self.masked_mse_loss(
            scaled_pred,
            scaled_target,
            Mt,  # [B, It, T]
            mask_individuals=Mt_individuals,  # [B, It]
        )

        # Store outputs for potential inspection (noise prediction task).
        outputs.update_head(
            "reconstruction",
            {
                "prediction": pred_noise,  # [B, It, T, 1]
                "target": noise,  # [B, It, T, 1]
                "mask": Mt_individuals,  # [B, It]
            },
        )

        outputs.update_losses(
            "reconstruction",
            {
                "rmse": mse_dict["mse"],  # Single MSE-like loss (scalar).
            },
        )

        return outputs

    # ------------------------------------------------------------------
    # Sampling new individuals
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        num_steps: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        """
        Sample trajectories for new individuals conditioned on context via
        discrete diffusion.

        Parameters
        ----------
        db:
            Single permutation minibatch.
        sample_size:
            Number of independent synthetic trajectories to draw.
        num_steps:
            Unused for discrete diffusion (diffusion steps are governed by
            ``self.num_diffusion_steps``) but kept for API compatibility
            with :class:`FlowPK`.
        decode_times:
            Optional pre-computed decode times; currently unused and kept for
            API compatibility.

        Returns
        -------
        samples:
            Tensor of shape [S, B, Tdistinct, 1] with sampled (unscaled)
            trajectories.
        times:
            Tensor of shape [B, Tdistinct, 1] with decode times; currently
            this is the scaled target time grid.
        mask:
            Boolean mask [B, Tdistinct] indicating valid entries.
        """
        del num_steps, decode_times  # Not used, kept for signature compatibility.

        # Encode context and scale.
        z_s, _, stats = self._study_latent(db, use_target=False)  # [B, Z], [B, Ic, Z]
        Xc_raw, Tc_raw, _ = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask

        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1]

        B, It, T, _ = Xt.shape
        device = self.device

        # Build denoiser closure.
        denoiser = self._make_denoiser(Xt=Xt, Tt=Tt, db=db, z_s=z_s, stats=stats)

        samples: List[torch.Tensor] = []

        for _ in range(sample_size):
            # DiscreteDiffusion.sample expects a callable model(x, i=..., **kwargs).
            # We pass times via kwargs so the GP covariance is well-defined.
            x_sample = self.diffusion.sample(
                model=lambda x, i, t: denoiser(x, i=i, t=t),  # type: ignore[arg-type]
                num_samples=(B, It, T, 1),
                device=device,
                t=Tt,
            )  # [B, It, T, 1]

            # Inverse scaling back to original PK units.
            X_raw, Tt_out = self.scaler.inverse(x_sample, Tt, stats)
            # X_raw: [B, It, T, 1], Tt_out: [B, It, T, 1]

            # For now we assume a single target individual per batch (It == 1)
            # as in AICME.  Squeeze individual dimension.
            samples.append(X_raw.squeeze(1))  # [B, T, 1]

        stacked_samples = torch.stack(samples, dim=0)  # [S, B, T, 1]

        return stacked_samples, Tt_out.squeeze(1), Mt.squeeze(1)


# ---------------------------------------------------------------------------
# Continuous-time diffusion PK model
# ---------------------------------------------------------------------------


class ContinuousDiffusionPK(_BaseDiffusionPK):
    """
    Continuous-time diffusion PK model based on SDEs.

    This class uses :class:`ContinuousDiffusion` with GP noise, applied to
    the scaled target trajectories.  The decoder is treated as a score
    approximator by predicting Gaussian noise at arbitrary diffusion
    times ``i ∈ [0, t1]``.
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        # Continuous diffusion with GP noise on time series.
        self.diffusion = ContinuousDiffusion(
            dim=1,
            beta_fn=self.beta_schedule,
            t1=self.diffusion_t1,
            noise_fn=None,  # Will default to Normal unless overridden below.
            is_time_series=True,
            predict_gaussian_noise=self.predict_gaussian_noise,
        )

        # Replace default noise with GP noise consistent with discrete case.
        # We keep this as a separate attribute because ContinuousDiffusion
        # only stores the callable.
        from pff.models.diffusion.noise import GaussianProcess

        self.diffusion.noise = GaussianProcess(dim=1, sigma=self.diffusion_sigma)

    # ------------------------------------------------------------------
    # Training / reconstruction
    # ------------------------------------------------------------------

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> DiffusionForwardOutputs:
        """
        Single-permutation forward pass with continuous-time diffusion.

        The structure mirrors :meth:`DiscreteDiffusionPK._forward_reconstruction`
        but uses a continuous-time diffusion index ``i ∈ [0, t1]`` and the
        loss is weighted by ``loss_weighting(i)`` as implemented in
        :class:`ContinuousDiffusion`.
        """
        # Encode study-level latent from context.
        z_s, z_ci, stats = self._study_latent(db, use_target=False)
        Xc_raw, Tc_raw, Mc = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        Mc_individuals = db.mask_context_individuals
        Mt_individuals = db.mask_target_individuals

        # Scale context and target.
        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1]

        B, It, T, _ = Xt.shape
        device = Xt.device

        outputs = DiffusionForwardOutputs(loss_multihead=self.loss_multihead, device=Xt_raw.device)
        outputs.stats = stats

        # Decoder as denoiser.
        denoiser = self._make_denoiser(Xt=Xt, Tt=Tt, db=db, z_s=z_s, stats=stats)

        # ------------------------------------------------------------------
        # Continuous diffusion step
        # ------------------------------------------------------------------
        x_clean = Xt  # [B, It, T, 1]

        # Sample continuous diffusion time i ∈ [0, t1] as in ContinuousDiffusion.get_loss.
        i = torch.rand(
            B,
            *(1,) * (x_clean.ndim - 1),
            device=device,
        ).expand_as(x_clean[..., :1])
        i = i * self.diffusion.t1  # [B, It, T, 1]

        # Forward SDE marginal: obtain x_noisy and Gaussian noise.
        x_noisy, noise = self.diffusion.forward(x_clean, i, t=Tt)  # [B, It, T, 1]

        # Predict Gaussian noise.
        pred_noise = denoiser(x_noisy, i=i, t=Tt)  # [B, It, T, 1]

        # Loss weighting as defined in ContinuousDiffusion (typically
        # a function of beta and i).
        weights = self.diffusion.loss_weighting(i)  # [B, It, T, 1] or broadcastable
        if not torch.is_tensor(weights):
            weights = torch.as_tensor(
                weights,
                device=pred_noise.device,
                dtype=pred_noise.dtype,
            )

        # Ensure broadcast to prediction shape.
        if weights.shape != pred_noise.shape:
            weights = weights.expand_as(pred_noise)

        # Implement weighted MSE via scaling both prediction and target by
        # sqrt(weight), so we can reuse masked_mse_loss.
        scale = torch.sqrt(weights.clamp_min(1e-12))
        scaled_pred = scale * pred_noise
        scaled_target = scale * noise

        mse_dict = self.masked_mse_loss(
            scaled_pred,
            scaled_target,
            Mt,  # [B, It, T]
            mask_individuals=Mt_individuals,
        )

        outputs.update_head(
            "reconstruction",
            {
                "prediction": pred_noise,
                "target": noise,
                "mask": Mt_individuals,
            },
        )

        outputs.update_losses(
            "reconstruction",
            {
                "rmse": mse_dict["mse"],
            },
        )

        return outputs

    # ------------------------------------------------------------------
    # Sampling new individuals
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        num_steps: int = 10,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        """
        Sample trajectories for new individuals via continuous-time diffusion.

        The sampler delegates to :meth:`ContinuousDiffusion.sample`, which
        can internally use either ODE or SDE sampling (we default to ODE).
        Context conditioning is injected via the decoder-based denoiser.
        """
        del num_steps, decode_times  # Not used here, kept for compatibility.

        # Encode context and scale.
        z_s, _, stats = self._study_latent(db, use_target=False)
        Xc_raw, Tc_raw, _ = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask

        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1]

        B, It, T, _ = Xt.shape
        device = self.device

        denoiser = self._make_denoiser(Xt=Xt, Tt=Tt, db=db, z_s=z_s, stats=stats)

        samples: List[torch.Tensor] = []

        for _ in range(sample_size):
            # Use ODE-based sampler by default for deterministic trajectories.
            x_sample = self.diffusion.sample(
                model=lambda x, i, t: denoiser(x, i=i, t=t),  # type: ignore[arg-type]
                num_samples=(B, It, T, 1),
                device=device,
                use_ode=True,
                t=Tt,
            )  # [B, It, T, 1]

            X_raw, Tt_out = self.scaler.inverse(x_sample, Tt, stats)
            samples.append(X_raw.squeeze(1))  # [B, T, 1]

        stacked_samples = torch.stack(samples, dim=0)  # [S, B, T, 1]

        return stacked_samples, Tt_out.squeeze(1), Mt.squeeze(1)


__all__ = [
    "DiffusionForwardOutputs",
    "DiscreteDiffusionPK",
    "ContinuousDiffusionPK",
]

# Deprecated import path compatibility
# ------------------------------------
# The actively maintained implementation lives in
# ``pff.models.amortized_inference.diffusion_pk``. Re-export those
# classes from this legacy module so older imports receive the updated
# generative-only, FlowPK-vector-field diffusion behavior without duplicating
# the implementation here.
_current_diffusion_pk = import_module("pff.models.amortized_inference.diffusion_pk")
DiffusionForwardOutputs = _current_diffusion_pk.DiffusionForwardOutputs  # noqa: F811
DiscreteDiffusionPK = _current_diffusion_pk.DiscreteDiffusionPK  # noqa: F811
ContinuousDiffusionPK = _current_diffusion_pk.ContinuousDiffusionPK  # noqa: F811
del _current_diffusion_pk
