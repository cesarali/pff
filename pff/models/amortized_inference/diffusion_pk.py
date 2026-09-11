"""
Logging-free diffusion-based PK models (discrete and continuous time).

This module mirrors the generative API of :class:`FlowPK` but replaces the
flow-matching objective with diffusion-based objectives. New configurations use
the same FlowPK point-cloud ``TransformerVectorField`` architecture, scaler, and
source-process configuration so that model capacity stays aligned while the
training objective changes. Legacy encoder-decoder configs remain supported as a
private compatibility path.
The main differences are:

* The reconstruction objective is formulated as denoising diffusion
  (discrete or continuous time).
* The decoder is interpreted as a *denoiser* predicting Gaussian noise
  on scaled PK trajectories rather than a vector field on an
  interpolation path.
* DiffusionPK is generative-only: it exposes ``sample_new_individual`` and
  target-schedule generative sampling, but not ``sample_individual_prediction``.
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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torchtyping import TensorType

from pff.config_classes.diffusion_pk_config import DiffusionPKExperimentConfig
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.config_classes.source_process_config import SourceProcessConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import (
    AbstractForwardOutputs,
    NewBasePKModel,
    NewGenerativeMixin,
)
from pff.models.architectures.aggregators import MeanStudyAggregator
from pff.models.architectures.vector_fields_pk import TransformerVectorField
from pff.models.diffusion.continuous_diffusion import ContinuousDiffusion
from pff.models.diffusion.discrete_diffusion import DiscreteDiffusion
from pff.models.diffusion.source_process import build_source_process
from pff.models.utils.loss_utils import MultiHeadLoss
from pff.models.utils.target_axis_ops import (
    flatten_many_target_axis_3d,
    flatten_many_target_axis_4d,
    unflatten_target_axis_4d,
)
from pff.utils.tensors_operations import gather_distinct_times_per_substance


def _require_target_observation_schedule(db: AICMECompartmentsDataBatch) -> None:
    """Require at least one valid target observation time per batch element."""

    target_obs_valid = db.target_obs_mask.bool() & db.mask_target_individuals.unsqueeze(-1)
    missing = torch.nonzero(~target_obs_valid.any(dim=(1, 2)), as_tuple=False).view(-1)
    if missing.numel() > 0:
        missing_str = ", ".join(str(int(idx)) for idx in missing.tolist())
        raise ValueError(
            "resolve_sampling_from_target=True requires at least one valid target "
            f"observation per batch element. Missing batch indices: {missing_str}."
        )

# ---------------------------------------------------------------------------
# Forward-output container
# ---------------------------------------------------------------------------


class DiffusionForwardOutputs(AbstractForwardOutputs):
    """
    Forward outputs for diffusion-based PK models.

    This is intentionally almost identical to :class:`FlowForwardOutputs`
    used by :class:`NewFlowPK` in ``new_flows_pk.py``:

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


class _BaseDiffusionPK(NewBasePKModel):
    """
    Shared utilities for diffusion-based PK models.

    This class encapsulates FlowPK-style context preprocessing, decoder
    wrapping, optional legacy latent aggregation, and scaling logic common to
    both discrete and continuous diffusion variants. Subclasses only implement:

    * construction of the diffusion object (discrete vs continuous), and
    * the specific diffusion training and sampling logic.
    """

    def __init__(
        self, model_config: Union[NodePKExperimentConfig, DiffusionPKExperimentConfig]
    ) -> None:
        super().__init__(model_config)

        vector_field_cfg = getattr(model_config, "vector_field", None)
        if vector_field_cfg is not None:
            self.z_dim = vector_field_cfg.zi_latent_dim
        elif self.encoder is not None:
            self.z_dim = self.encoder.zi_latent_dim  # type: ignore[attr-defined]
        else:
            raise ValueError(
                "DiffusionPK requires either a FlowPK-style `vector_field` section "
                "or a legacy `network` section with an encoder."
            )
        self.aggregator = MeanStudyAggregator()
        self.loss_multihead = MultiHeadLoss(mode="fixed", number_of_losses=1)
        # Vector-field decoder uses a different call signature (point clouds + masks).
        self._use_vector_field_decoder = isinstance(self.decoder, TransformerVectorField)

        # ------------------------------------------------------------------
        # Diffusion hyper-parameters from configuration (with safe defaults)
        # ------------------------------------------------------------------
        source_cfg = getattr(model_config, "source_process", None)

        def _get(name: str, default: Any) -> Any:
            if hasattr(model_config, name):
                return getattr(model_config, name)
            net_cfg = getattr(model_config, "network", None)
            if net_cfg is not None and hasattr(net_cfg, name):
                return getattr(net_cfg, name)
            return default

        # Number of diffusion steps (discrete case) / discretisation points.
        self.num_diffusion_steps: int = int(_get("diffusion_num_steps", 100))

        # Final diffusion time for continuous diffusion (t1 in [0, 1]).
        self.diffusion_t1: float = float(_get("diffusion_t1", 1.0))

        # Beta schedule parameters.
        self.beta_min: float = float(_get("diffusion_beta_min", 1e-4))
        self.beta_max: float = float(_get("diffusion_beta_max", 2e-2))

        # Whether the model predicts unit Gaussian noise or correlated noise.
        self.predict_gaussian_noise: bool = bool(
            getattr(model_config, "predict_gaussian_noise", _get("diffusion_predict_gaussian_noise", True))
        )

        # Shared beta schedule object usable by discrete and continuous diffusion.
        self.beta_schedule = LinearBetaSchedule(self.beta_min, self.beta_max)

        # Source process shared by discrete and continuous diffusion.
        if source_cfg is None:
            source_cfg = SourceProcessConfig()

        self.source_process, self.source_is_time_series = build_source_process(
            source_cfg, dim=1
        )

    def build_visualization_callback(self):
        """Return callbacks that handle visualization and empirical evaluation."""
        return super().build_visualization_callback()

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
            Unused but kept for API compatibility with NewFlowPK.

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
        if self.encoder is None:
            raise RuntimeError(
                "Latent study encoding is only available for legacy network-based "
                "DiffusionPK configs. FlowPK-style vector-field diffusion uses "
                "`preprocess_study` instead."
            )

        Xc_raw, Tc_raw, M = db.context_obs, db.context_obs_time, db.context_obs_mask
        mask_ind = db.mask_individuals

        # Scaling statistics from context only (identical to NewFlowPK).
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
        z_s: TensorType["B", "Z"] | None = None,
        stats: TensorType["B", 1],
        study_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        init_state: TensorType["BI", 1, 1] | None = None,
        first_t_s: TensorType["BI", 1, 1] | None = None,
        z_i: TensorType["B", "It", "Z"] | None = None,
        decode_times: TensorType["B", "It", "T", 1] | None = None,
        target_mask: TensorType["B", "It", "T"] | None = None,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
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

        For FlowPK-style vector-field configs this wrapper reuses the same
        point-cloud features, masks, dose features, and shape conventions as
        :class:`FlowPK`:

        * context point cloud ``[B, Ic*T, 2]`` with ``[time, value]`` channels,
        * target point cloud ``[B, It*T, 2]`` with ``[time, noisy_value]``,
        * dose and route features at context and target time points,
        * flow_time = normalised diffusion step in [0, 1],
        * valid-observation and block attention masks.

        Decoder interface notes
        -----------------------
        If the configured decoder is a TransformerVectorField, this wrapper
        switches to the FlowPK-style interface and uses point clouds plus
        attention masks. In that case, ``z_s``, ``z_i``, ``dose``, ``route``,
        ``init_state``, and ``first_t_s`` are ignored.
        """
        # Shapes
        # -------
        # Xt, Tt: [B, It, T, 1]
        B, It, T, _ = Xt.shape
        BI = B * It
        device = Xt.device

        if self._use_vector_field_decoder:
            if target_mask is None:
                target_mask = db.target_obs_mask
            if study_ctx is None:
                study_ctx, _ = self.preprocess_study(db, stats=stats)
            dose_values, route_values = self._resolve_target_dosing_for_shape(
                db,
                dosing=dosing,
                batch_size=B,
                num_targets=It,
            )
            dose = self._target_dose_features(dose_values, route_values, T)  # [B, It, T, 2]

            def vector_field_denoiser(
                x_noisy: torch.Tensor,  # [B, It, T, 1]
                i: torch.Tensor,  # [B, It, T, 1]
                t: torch.Tensor,  # [B, It, T, 1]
            ) -> torch.Tensor:
                """Denoise with the same point-cloud decoder interface used by FlowPK."""

                if x_noisy.shape[:3] != (B, It, T):
                    raise ValueError(
                        "Denoiser called with inconsistent shapes: expected "
                        f"{(B, It, T)}, got {tuple(x_noisy.shape[:3])}."
                    )

                tau = self._normalize_diffusion_time(i).view(B, 1).to(device)  # [B, 1]

                x_flat, t_flat = flatten_many_target_axis_4d(
                    x_noisy, t
                )  # [B*It, 1, T, 1], [B*It, 1, T, 1]
                mask_flat = flatten_many_target_axis_3d(target_mask)[0]  # [B*It, 1, T]
                dose_flat = self._flatten_target_dose(dose)  # [B*It, T, 2]
                study_ctx_flat = self._repeat_study_context_for_targets(study_ctx, It)
                (
                    x_ctx_flat,
                    t_ctx_flat,
                    ctx_mask_flat,
                    ctx_ind_mask_flat,
                    dose_ctx_flat,
                ) = study_ctx_flat

                pc_ctx, mask_pc_ctx, mask_attn_ctx = self.pointcloud_from_obs(
                    x_ctx_flat,
                    t_ctx_flat,
                    ctx_mask_flat,
                    mask_individuals=ctx_ind_mask_flat,
                )  # [B*It, Nc, 2], [B*It, Nc], [B*It, Nc, Nc]
                pc_x, mask_pc_x = self.pointcloud_from_obs(
                    x_flat,
                    t_flat,
                    mask_flat,
                    return_attn_mask=False,
                )  # [B*It, T, 2], [B*It, T]

                pred_flat = self.decoder(
                    x=pc_x,  # [B*It, T, 2]
                    ctx=pc_ctx,  # [B*It, Nc, 2]
                    flow_t=tau.repeat_interleave(It, dim=0),  # [B*It, 1]
                    mask_pad_x=mask_pc_x,  # [B*It, T]
                    mask_pad_ctx=mask_pc_ctx,  # [B*It, Nc]
                    mask_attn_ctx=mask_attn_ctx,  # [B*It, Nc, Nc]
                    dose=dose_flat,  # [B*It, T, 2]
                    dose_ctx=dose_ctx_flat,  # [B*It, Nc, 2]
                )  # [B*It, 1, T, 1]
                return unflatten_target_axis_4d(pred_flat, B, It)  # [B, It, T, 1]

            return vector_field_denoiser

        if z_s is None:
            raise RuntimeError("Legacy diffusion denoiser requires a study latent `z_s`.")

        # First (scaled) observation per target individual and first time
        # unless overridden by the caller (e.g., for prediction).
        if init_state is None or first_t_s is None:
            if target_mask is None:
                target_mask = db.target_obs_mask
            init_true, _, first_t_s_local = self.get_first_valid_observation(
                Xt, target_mask, Tt
            )
            # init_true: [B, It, 1, 1]
            # first_t_s: [B, It, 1, 1]
            if init_state is None:
                init_state = init_true.view(BI, 1, -1)  # [B*It, 1, 1]
            if first_t_s is None:
                first_t_s = first_t_s_local.view(BI, 1, -1)  # [B*It, 1, 1]

        # Study-level latent is currently broadcast to target individuals,
        # just like in NewFlowPK, unless overridden.
        if z_i is None:
            z_i = z_s.unsqueeze(1).repeat(1, It, 1)  # [B, It, Z]

        # Target dosing and route, expanded to match decoder expectations.
        if dosing is None:
            dose_values = db.target_dosing_amounts
            route_values = db.target_dosing_route_types
        else:
            dose_values, route_values = dosing
            if dose_values.ndim == 1:
                dose_values = dose_values.unsqueeze(-1)
            if route_values.ndim == 1:
                route_values = route_values.unsqueeze(-1)
        dose = dose_values.unsqueeze(-1).unsqueeze(-1)  # [B, It, 1, 1]
        route = route_values.float().unsqueeze(-1).unsqueeze(-1)
        # route: [B, It, 1, 1]

        # Flatten individual dimension for features that are per-individual.
        init_state = init_state.view(BI, 1, -1)  # [B*It, 1, 1]
        first_t_feat = first_t_s.view(BI, 1, -1)  # [B*It, 1, 1]
        dose_feat = dose.view(BI, 1, -1)  # [B*It, 1, 1]
        route_feat = route.view(BI, 1, -1)  # [B*It, 1, 1]

        # Decoder will see time grid per substance: [B, T, 1]
        if decode_times is None:
            decode_times = Tt
        decode_times = decode_times.squeeze(1)  # [B, T, 1] (It is usually == 1 in AICME)

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
            # NewFlowPK where:
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

    def preprocess_study(
        self,
        db: AICMECompartmentsDataBatch,
        stats: TensorType["B", 1] | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        TensorType["B", 1],
    ]:
        """Scale the context study and build FlowPK-style context dose features."""

        B, num_individuals, C, _ = db.context_obs.shape
        N = num_individuals * C
        if stats is None:
            stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        x_ctx, t_ctx = self.scaler.forward(
            db.context_obs, db.context_obs_time, stats
        )  # [B, I, C, 1], [B, I, C, 1]
        x_ctx = x_ctx * db.context_obs_mask.unsqueeze(-1).float()  # [B, I, C, 1]

        dose_amount = db.context_dosing_amounts.unsqueeze(-1).expand(-1, -1, C).reshape(B, N)
        dose_route = db.context_dosing_route_types.unsqueeze(-1).expand(-1, -1, C).reshape(B, N)
        dose_ctx = torch.stack([dose_amount, dose_route], dim=-1)  # [B, N, 2]

        study_ctx = (
            x_ctx,
            t_ctx,
            db.context_obs_mask,
            db.mask_context_individuals,
            dose_ctx,
        )
        return study_ctx, stats

    def pointcloud_from_obs(
        self,
        x: TensorType["B", "I", "T", 1],
        time: TensorType["B", "I", "T", 1],
        obs_mask: TensorType["B", "I", "T"],
        return_attn_mask: bool = True,
        mask_individuals: torch.Tensor | None = None,
    ):
        """Convert observations to FlowPK point-cloud tensors.

        Args:
            x: Observation values with shape ``[B, I, T, 1]``.
            time: Observation times with shape ``[B, I, T, 1]``.
            obs_mask: Observation validity mask with shape ``[B, I, T]``.
            return_attn_mask: Whether to return a block-structured attention mask.
            mask_individuals: Optional individual validity mask ``[B, I]``.

        Returns:
            ``pc`` with shape ``[B, I*T, 2]`` storing ``[time, value]`` pairs,
            ``padding_mask`` with shape ``[B, I*T]``, and optionally
            ``attn_mask`` with shape ``[B, I*T, I*T]``.
        """

        B, num_individuals, T, _ = x.shape
        N = num_individuals * T
        pc = torch.concat([time, x], dim=-1).reshape(B, N, 2)  # [B, N, 2]
        padding_mask = obs_mask.reshape(B, N).bool()  # [B, N]

        if mask_individuals is not None:
            ind_mask = mask_individuals.unsqueeze(-1).expand(B, num_individuals, T).reshape(B, N)
            padding_mask = padding_mask & ind_mask.bool()

        if not return_attn_mask:
            return pc, padding_mask

        block_ids = (
            torch.arange(num_individuals, device=x.device)
            .unsqueeze(1)
            .expand(num_individuals, T)
            .reshape(N)
        )
        block_attn = block_ids.unsqueeze(0) == block_ids.unsqueeze(1)  # [N, N]
        attn_mask = block_attn.unsqueeze(0) & padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)
        return pc, padding_mask, attn_mask

    def _pointcloud_from_obs(self, x, time, return_attn_mask: bool = True):
        """Backward-compatible point-cloud wrapper for legacy denoisers."""

        obs_mask = time.squeeze(-1) > 0.0  # [B, I, T]
        return self.pointcloud_from_obs(x, time, obs_mask, return_attn_mask=return_attn_mask)

    def _normalize_diffusion_time(self, i: torch.Tensor) -> torch.Tensor:
        """Map discrete or continuous diffusion time tensors to ``[0, 1]``."""

        i_flat = i[:, 0, 0, 0].float()  # [B]
        if getattr(self, "diffusion_time_kind", "discrete") == "continuous":
            denom = max(float(self.diffusion_t1), 1.0e-12)
        else:
            denom = max(int(self.num_diffusion_steps) - 1, 1)
        return (i_flat / denom).clamp(0.0, 1.0)  # [B]

    @staticmethod
    def _flatten_target_dose(dose: torch.Tensor) -> torch.Tensor:
        """Flatten target-axis dose tensor from ``[B, It, T, 2]`` to ``[B*It, T, 2]``."""

        B, It, T, D = dose.shape
        return dose.reshape(B * It, T, D)

    @staticmethod
    def _repeat_study_context_for_targets(
        study_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        num_targets: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repeat a study context once for each target individual."""

        x_ctx, t_ctx, context_obs_mask, mask_context_individuals, dose_ctx = study_ctx
        return (
            x_ctx.repeat_interleave(num_targets, dim=0),
            t_ctx.repeat_interleave(num_targets, dim=0),
            context_obs_mask.repeat_interleave(num_targets, dim=0),
            mask_context_individuals.repeat_interleave(num_targets, dim=0),
            dose_ctx.repeat_interleave(num_targets, dim=0),
        )

    @staticmethod
    def _repeat_stats(
        stats_dict: dict[str, torch.Tensor], repeats: int, base_batch: int
    ) -> dict[str, torch.Tensor]:
        """Repeat per-batch scaler statistics for target-resolved diffusion sampling."""

        stats_out: dict[str, torch.Tensor] = {}
        for key, value in stats_dict.items():
            if torch.is_tensor(value) and value.shape[0] == base_batch:
                stats_out[key] = value.repeat_interleave(repeats, dim=0)
            else:
                stats_out[key] = value
        return stats_out

    def _resolve_target_dosing_for_shape(
        self,
        db: AICMECompartmentsDataBatch,
        *,
        dosing: tuple[torch.Tensor, torch.Tensor] | None,
        batch_size: int,
        num_targets: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve dose and route tensors with shape ``[B, It]``."""

        if dosing is None:
            dose_values = db.target_dosing_amounts
            route_values = db.target_dosing_route_types
        else:
            dose_values, route_values = dosing

        if dose_values.ndim == 1:
            dose_values = dose_values.unsqueeze(-1)  # [B, 1]
        if route_values.ndim == 1:
            route_values = route_values.unsqueeze(-1)  # [B, 1]
        if dose_values.shape == (batch_size, 1) and num_targets > 1:
            dose_values = dose_values.expand(-1, num_targets)  # [B, It]
        if route_values.shape == (batch_size, 1) and num_targets > 1:
            route_values = route_values.expand(-1, num_targets)  # [B, It]

        dose_values = dose_values.to(db.target_dosing_amounts.device)
        route_values = route_values.to(db.target_dosing_route_types.device)

        expected = (batch_size, num_targets)
        if tuple(dose_values.shape) != expected:
            raise ValueError(
                f"Target dosing amounts must have shape {expected}, got {tuple(dose_values.shape)}."
            )
        if tuple(route_values.shape) != expected:
            raise ValueError(
                f"Target dosing routes must have shape {expected}, got {tuple(route_values.shape)}."
            )
        return dose_values, route_values.float()

    @staticmethod
    def _target_dose_features(
        dose_values: torch.Tensor,
        route_values: torch.Tensor,
        time_steps: int,
    ) -> torch.Tensor:
        """Expand per-target dose and route values to ``[B, It, T, 2]``."""

        dose_amount = dose_values.unsqueeze(-1).expand(-1, -1, time_steps)  # [B, It, T]
        dose_route = route_values.unsqueeze(-1).expand(-1, -1, time_steps)  # [B, It, T]
        return torch.stack([dose_amount, dose_route], dim=-1)  # [B, It, T, 2]

    def select_unseen_dosing_from_databatch(
        self,
        db: AICMECompartmentsDataBatch,
        generator: torch.Generator | None = None,
    ) -> tuple[TensorType["B", 1], TensorType["B", 1]]:
        """Select one target dosing when available, otherwise sample a context dosing."""

        target_mask = db.mask_target_individuals  # [B, It]
        context_mask = db.mask_context_individuals  # [B, Ic]
        device = db.context_dosing_amounts.device
        route_device = db.context_dosing_route_types.device
        B = db.context_dosing_amounts.size(0)
        dose = torch.zeros(B, dtype=db.context_dosing_amounts.dtype, device=device)
        route = torch.zeros(B, dtype=db.context_dosing_route_types.dtype, device=route_device)

        for b in range(B):
            valid_targets = torch.nonzero(target_mask[b], as_tuple=False).view(-1)
            if valid_targets.numel() > 0:
                idx = int(valid_targets[0].item())
                dose[b] = db.target_dosing_amounts[b, idx]
                route[b] = db.target_dosing_route_types[b, idx]
                continue

            valid_context = torch.nonzero(context_mask[b], as_tuple=False).view(-1)
            if valid_context.numel() == 0:
                continue
            if generator is not None:
                choice = torch.randperm(valid_context.numel(), generator=generator, device=device)[0]
            else:
                choice = torch.randperm(valid_context.numel(), device=device)[0]
            idx = int(valid_context[int(choice)].item())
            dose[b] = db.context_dosing_amounts[b, idx]
            route[b] = db.context_dosing_route_types[b, idx]

        return dose.unsqueeze(-1), route.unsqueeze(-1)

    def _prepare_sample_grid(
        self,
        db: AICMECompartmentsDataBatch,
        stats: TensorType["B", 1],
        *,
        decode_times: tuple[torch.Tensor, torch.Tensor] | None,
        resolve_sampling_from_target: bool,
        include_rem: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare scaled zero-valued decode paths and masks for generative sampling."""

        if decode_times is None:
            if resolve_sampling_from_target:
                target_sources = (
                    ("target_obs_time", "target_rem_sim_time")
                    if include_rem
                    else ("target_obs_time",)
                )
                raw_times, raw_mask = gather_distinct_times_per_substance(
                    db,
                    time_sources=target_sources,
                )
            else:
                raw_times, raw_mask = gather_distinct_times_per_substance(db)
        else:
            raw_times, raw_mask = decode_times  # [B, T, 1], [B, T]

        if resolve_sampling_from_target:
            It = db.target_obs.shape[1]
            Tt_raw = raw_times.unsqueeze(1).expand(-1, It, -1, -1)  # [B, It, T, 1]
            Mt = raw_mask.unsqueeze(1).expand(-1, It, -1).bool()  # [B, It, T]
        else:
            Tt_raw = raw_times.unsqueeze(1)  # [B, 1, T, 1]
            Mt = raw_mask.unsqueeze(1).bool()  # [B, 1, T]

        Xt_raw = torch.zeros_like(Tt_raw)  # [B, It/1, T, 1]
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It/1, T, 1]
        return Xt, Tt, Mt, raw_times, raw_mask.bool()

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
        ignore_logvar: bool = True,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Discrete-time diffusion PK model
# ---------------------------------------------------------------------------


class DiscreteDiffusionPK(_BaseDiffusionPK, NewGenerativeMixin):
    """
    Discrete denoising diffusion PK model with GP noise.

    This class applies discrete-time diffusion to scaled target trajectories
    ``Xt`` (shape [B, It, T, 1]). The default architecture is the same
    FlowPK point-cloud vector field; only the training objective and sampler
    are changed to diffusion.

    The decoder is trained to predict *Gaussian* noise as in standard DDPM
    setups (``predict_gaussian_noise=True``).
    """

    def __init__(
        self, model_config: Union[NodePKExperimentConfig, DiffusionPKExperimentConfig]
    ) -> None:
        super().__init__(model_config)

        diffusion_type = getattr(model_config, "diffusion_type", None)
        if diffusion_type is not None and str(diffusion_type).lower() != "discrete":
            raise ValueError(
                f"DiscreteDiffusionPK requires diffusion_type 'discrete', got {diffusion_type!r}."
            )

        self.diffusion_time_kind = "discrete"
        self.diffusion = DiscreteDiffusion(
            dim=1,
            num_steps=self.num_diffusion_steps,
            beta_fn=self.beta_schedule,
            noise_fn=self.source_process,
            is_time_series=self.source_is_time_series,
            predict_gaussian_noise=self.predict_gaussian_noise,
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
        if self._use_vector_field_decoder:
            study_ctx, stats = self.preprocess_study(db)
            z_s = None
        else:
            study_ctx = None
            z_s, _, stats = self._study_latent(db, use_target=False)  # [B, Z], [B, Ic, Z]
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        Mt_individuals = db.mask_target_individuals  # [B, It]

        # Scale context and target using context-only statistics.
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1], [B, It, T, 1]

        # Shapes
        B, It, T, _ = Xt.shape
        device = Xt.device

        outputs = DiffusionForwardOutputs(loss_multihead=self.loss_multihead, device=Xt_raw.device)
        outputs.stats = stats

        # Build decoder wrapper that behaves like a diffusion denoiser.
        denoiser = self._make_denoiser(
            Xt=Xt,
            Tt=Tt,
            db=db,
            z_s=z_s,
            stats=stats,
            study_ctx=study_ctx,
            target_mask=Mt,
        )

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
        # To reuse NewBasePKModel.masked_mse_loss we optionally multiply both
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
        ignore_logvar: bool = True,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
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
            with :class:`NewFlowPK`.
        decode_times:
            Optional pre-computed decode times for true generative sampling.

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
        del num_steps
        _ = ignore_logvar  # Not used, kept for signature compatibility.
        if resolve_sampling_from_target:
            _require_target_observation_schedule(db)

        if self._use_vector_field_decoder:
            study_ctx, stats = self.preprocess_study(db)
            z_s = None
        else:
            study_ctx = None
            z_s, _, stats = self._study_latent(db, use_target=False)  # [B, Z], [B, Ic, Z]

        Xt, Tt, Mt, raw_times, raw_mask = self._prepare_sample_grid(
            db,
            stats,
            decode_times=decode_times,
            resolve_sampling_from_target=resolve_sampling_from_target,
            include_rem=include_rem,
        )  # [B, It/1, T, 1], [B, It/1, T, 1], [B, It/1, T]

        B, It, T, _ = Xt.shape
        device = self.device
        if dosing is None and not resolve_sampling_from_target:
            dosing = self.select_unseen_dosing_from_databatch(db)  # [B, 1], [B, 1]

        # Build denoiser closure.
        denoiser = self._make_denoiser(
            Xt=Xt,
            Tt=Tt,
            db=db,
            z_s=z_s,
            stats=stats,
            study_ctx=study_ctx,
            decode_times=Tt,
            target_mask=Mt,
            dosing=dosing,
        )

        samples: List[torch.Tensor] = []
        Tt_out = Tt

        num_draws = 1 if resolve_sampling_from_target else int(sample_size)
        for _ in range(num_draws):
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
            if resolve_sampling_from_target:
                samples_out = X_raw.permute(1, 0, 2, 3).contiguous()  # [It, B, T, 1]
                return samples_out, Tt_out[:, 0], raw_mask

            # For now we assume a single target individual per batch (It == 1)
            # as in AICME.  Squeeze individual dimension.
            samples.append(X_raw.squeeze(1))  # [B, T, 1]

        stacked_samples = torch.stack(samples, dim=0)  # [S, B, T, 1]

        return stacked_samples, Tt_out.squeeze(1), raw_mask

    # ------------------------------------------------------------------
    # Predictive sampling (known individuals)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _legacy_sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ) -> Tuple[
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["B", "It", "Tr", 1],
        TensorType["B", "It", "Tr"],
    ]:
        """Legacy predictive sampler retained for reference; not part of public DiffusionPK."""
        db = databatch

        X_obs_raw, T_obs_raw, M_obs = db.target_obs, db.target_obs_time, db.target_obs_mask
        # X_obs_raw/T_obs_raw: [B,It,To,1], M_obs: [B,It,To]
        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim,
            db.target_rem_sim_time,
            db.target_rem_sim_mask,
        )  # [B,It,Tr,1], [B,It,Tr,1], [B,It,Tr]

        # Study-level latent and context scaling stats.
        z_s, _, stats = self._study_latent(db, use_target=False)  # [B,Z]

        # Scale remainder block using context stats.
        Xrem_s, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)  # [B,It,Tr,1]

        # Initial state from the last observed point.
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B,It,1,1]

        B, It, Tr, _ = Xrem_s.shape
        BI = B * It
        init_state = init_s.view(BI, 1, -1)  # [B*It,1,1]
        last_t_feat = last_t_s.view(BI, 1, -1)  # [B*It,1,1]

        # Build denoiser with a prediction-specific initial state.
        denoiser = self._make_denoiser(
            Xt=Xrem_s,
            Tt=Trem_s,
            db=db,
            z_s=z_s,
            stats=stats,
            init_state=init_state,
            first_t_s=last_t_feat,
        )

        samples: List[torch.Tensor] = []
        for _ in range(sample_size):
            x_sample = self.diffusion.sample(
                model=lambda x, i, t: denoiser(x, i=i, t=t),  # type: ignore[arg-type]
                num_samples=(B, It, Tr, 1),
                device=Xrem_s.device,
                t=Trem_s,
            )  # [B,It,Tr,1]

            X_raw, T_raw = self.scaler.inverse(x_sample, Trem_s, stats)
            samples.append(X_raw)  # [B,It,Tr,1]

        stacked_samples = torch.stack(samples, dim=0)  # [S,B,It,Tr,1]
        times = T_raw.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)  # [S,B,It,Tr,1]

        return stacked_samples, times, Xrem_raw, Mrem


# ---------------------------------------------------------------------------
# Continuous-time diffusion PK model
# ---------------------------------------------------------------------------


class ContinuousDiffusionPK(_BaseDiffusionPK, NewGenerativeMixin):
    """
    Continuous-time diffusion PK model based on SDEs.

    This class uses :class:`ContinuousDiffusion` with GP noise, applied to
    the scaled target trajectories.  The decoder is treated as a score
    approximator by predicting Gaussian noise at arbitrary diffusion
    times ``i ∈ [0, t1]``.
    """

    def __init__(
        self, model_config: Union[NodePKExperimentConfig, DiffusionPKExperimentConfig]
    ) -> None:
        super().__init__(model_config)

        diffusion_type = getattr(model_config, "diffusion_type", None)
        if diffusion_type is not None and str(diffusion_type).lower() != "continuous":
            raise ValueError(
                f"ContinuousDiffusionPK requires diffusion_type 'continuous', got {diffusion_type!r}."
            )

        self.diffusion_time_kind = "continuous"
        self.diffusion = ContinuousDiffusion(
            dim=1,
            beta_fn=self.beta_schedule,
            t1=self.diffusion_t1,
            noise_fn=self.source_process,
            is_time_series=self.source_is_time_series,
            predict_gaussian_noise=self.predict_gaussian_noise,
        )

    # ------------------------------------------------------------------
    # Training / reconstruction
    # ------------------------------------------------------------------

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> DiffusionForwardOutputs:
        """
        Single-permutation forward pass with continuous-time diffusion.

        The structure mirrors :meth:`NewDiscreteDiffusionPK._forward_reconstruction`
        but uses a continuous-time diffusion index ``i ∈ [0, t1]`` and the
        loss is weighted by ``loss_weighting(i)`` as implemented in
        :class:`ContinuousDiffusion`.
        """
        if self._use_vector_field_decoder:
            study_ctx, stats = self.preprocess_study(db)
            z_s = None
        else:
            study_ctx = None
            z_s, _, stats = self._study_latent(db, use_target=False)
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        Mt_individuals = db.mask_target_individuals

        # Scale context and target.
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B, It, T, 1]

        B, It, T, _ = Xt.shape
        device = Xt.device

        outputs = DiffusionForwardOutputs(loss_multihead=self.loss_multihead, device=Xt_raw.device)
        outputs.stats = stats

        # Decoder as denoiser.
        denoiser = self._make_denoiser(
            Xt=Xt,
            Tt=Tt,
            db=db,
            z_s=z_s,
            stats=stats,
            study_ctx=study_ctx,
            target_mask=Mt,
        )

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
        ignore_logvar: bool = True,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
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
        del num_steps
        _ = ignore_logvar  # Not used here, kept for compatibility.
        if resolve_sampling_from_target:
            _require_target_observation_schedule(db)

        if self._use_vector_field_decoder:
            study_ctx, stats = self.preprocess_study(db)
            z_s = None
        else:
            study_ctx = None
            z_s, _, stats = self._study_latent(db, use_target=False)

        Xt, Tt, Mt, raw_times, raw_mask = self._prepare_sample_grid(
            db,
            stats,
            decode_times=decode_times,
            resolve_sampling_from_target=resolve_sampling_from_target,
            include_rem=include_rem,
        )  # [B, It/1, T, 1], [B, It/1, T, 1], [B, It/1, T]

        B, It, T, _ = Xt.shape
        device = self.device
        if dosing is None and not resolve_sampling_from_target:
            dosing = self.select_unseen_dosing_from_databatch(db)  # [B, 1], [B, 1]

        denoiser = self._make_denoiser(
            Xt=Xt,
            Tt=Tt,
            db=db,
            z_s=z_s,
            stats=stats,
            study_ctx=study_ctx,
            decode_times=Tt,
            target_mask=Mt,
            dosing=dosing,
        )

        samples: List[torch.Tensor] = []
        Tt_out = Tt

        num_draws = 1 if resolve_sampling_from_target else int(sample_size)
        for _ in range(num_draws):
            # Use ODE-based sampler by default for deterministic trajectories.
            x_sample = self.diffusion.sample(
                model=lambda x, i, t: denoiser(x, i=i, t=t),  # type: ignore[arg-type]
                num_samples=(B, It, T, 1),
                device=device,
                use_ode=True,
                t=Tt,
            )  # [B, It, T, 1]

            X_raw, Tt_out = self.scaler.inverse(x_sample, Tt, stats)
            if resolve_sampling_from_target:
                samples_out = X_raw.permute(1, 0, 2, 3).contiguous()  # [It, B, T, 1]
                return samples_out, Tt_out[:, 0], raw_mask
            samples.append(X_raw.squeeze(1))  # [B, T, 1]

        stacked_samples = torch.stack(samples, dim=0)  # [S, B, T, 1]

        return stacked_samples, Tt_out.squeeze(1), raw_mask

    # ------------------------------------------------------------------
    # Predictive sampling (known individuals)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _legacy_sample_individual_prediction(
        self,
        databatch: AICMECompartmentsDataBatch,
        sample_size: int = 1,
    ) -> Tuple[
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["S", "B", "It", "Tr", 1],
        TensorType["B", "It", "Tr", 1],
        TensorType["B", "It", "Tr"],
    ]:
        """Legacy predictive sampler retained for reference; not part of public DiffusionPK."""
        db = databatch

        X_obs_raw, T_obs_raw, M_obs = db.target_obs, db.target_obs_time, db.target_obs_mask
        # X_obs_raw/T_obs_raw: [B,It,To,1], M_obs: [B,It,To]
        Xrem_raw, Trem_raw, Mrem = (
            db.target_rem_sim,
            db.target_rem_sim_time,
            db.target_rem_sim_mask,
        )  # [B,It,Tr,1], [B,It,Tr,1], [B,It,Tr]

        # Study-level latent and context scaling stats.
        z_s, _, stats = self._study_latent(db, use_target=False)  # [B,Z]

        # Scale remainder block using context stats.
        Xrem_s, Trem_s = self.scaler.forward(Xrem_raw, Trem_raw, stats)  # [B,It,Tr,1]

        # Initial state from the last observed point.
        init_raw, last_t_raw = self.get_last_valid_observation(X_obs_raw, M_obs, T_obs_raw)
        init_s, last_t_s = self.scaler.forward(init_raw, last_t_raw, stats)  # [B,It,1,1]

        B, It, Tr, _ = Xrem_s.shape
        BI = B * It
        init_state = init_s.view(BI, 1, -1)  # [B*It,1,1]
        last_t_feat = last_t_s.view(BI, 1, -1)  # [B*It,1,1]

        # Build denoiser with a prediction-specific initial state.
        denoiser = self._make_denoiser(
            Xt=Xrem_s,
            Tt=Trem_s,
            db=db,
            z_s=z_s,
            stats=stats,
            init_state=init_state,
            first_t_s=last_t_feat,
        )

        samples: List[torch.Tensor] = []
        for _ in range(sample_size):
            x_sample = self.diffusion.sample(
                model=lambda x, i, t: denoiser(x, i=i, t=t),  # type: ignore[arg-type]
                num_samples=(B, It, Tr, 1),
                device=Xrem_s.device,
                use_ode=True,
                t=Trem_s,
            )  # [B,It,Tr,1]

            X_raw, T_raw = self.scaler.inverse(x_sample, Trem_s, stats)
            samples.append(X_raw)  # [B,It,Tr,1]

        stacked_samples = torch.stack(samples, dim=0)  # [S,B,It,Tr,1]
        times = T_raw.unsqueeze(0).repeat(sample_size, 1, 1, 1, 1)  # [S,B,It,Tr,1]

        return stacked_samples, times, Xrem_raw, Mrem


__all__ = [
    "DiffusionForwardOutputs",
    "DiscreteDiffusionPK",
    "ContinuousDiffusionPK",
]
