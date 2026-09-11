"""Flow matching PK models built on the logging-free PK base stack."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchtyping import TensorType
from tqdm import tqdm

try:  # pragma: no cover - dependency availability is environment-specific
    import ot as pot
except ModuleNotFoundError:  # pragma: no cover - only required for OT coupling
    pot = None

from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.config_classes.source_process_config import SourceProcessConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.generative_pk import (
    AbstractForwardOutputs,
    NewBasePKModel,
    NewGenerativeMixin,
    NewPredictiveMixin,
)
from pff.models.diffusion.noise import GaussianProcessRegression, WhiteNoiseProcess
from pff.models.diffusion.source_process import normalize_source_type
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


class FlowForwardOutputs(AbstractForwardOutputs):
    """Simplified forward outputs collector for Flow Matching PK models.

    Contains minimal schema with single reconstruction head and single MSE loss.
    """

    HEAD_SCHEMAS = {"reconstruction": ["prediction", "target", "mask"]}
    LOSS_SCHEMAS = {"reconstruction": ["mse"]}  # Single MSE loss

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
        """Compute total loss - simplified to single MSE loss."""
        if not flat_losses:
            return None
        if self.loss_multihead is None:
            return super()._compute_total_loss(flat_losses)

        # Single loss case - just the MSE
        device = self._infer_device()
        zero = torch.zeros((), device=device)
        mse_loss = flat_losses.get("mse", zero)

        # For compatibility with MultiHeadLoss expecting a list
        total_loss, _ = self.loss_multihead([mse_loss])
        return total_loss

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Flatten losses and add derived RMSE for logging/checkpointing."""
        flat = super().to_dict()
        mse = flat.get("mse")
        if mse is not None:
            flat["rmse"] = torch.sqrt(torch.clamp(mse, min=0.0))
        return flat


class FlowPK(NewBasePKModel, NewPredictiveMixin, NewGenerativeMixin):
    """Flow Matching PK model with single MSE loss."""

    def _resolve_flow_source_config(self):
        """Return a source-process config object with safe defaults for FlowPK."""
        source_cfg = getattr(self.model_config, "source_process", None)
        if source_cfg is None:
            # Legacy experiment configs can omit the section entirely.
            return SourceProcessConfig()
        if isinstance(source_cfg, dict):
            return SourceProcessConfig(**source_cfg)
        return source_cfg

    def _ensure_source_process_initialized(self) -> None:
        """Initialize FlowPK source process for fresh and legacy checkpoints."""
        if hasattr(self, "source_process") and self.source_process is not None:
            return

        source_cfg = self._resolve_flow_source_config()
        source_type = normalize_source_type(
            str(getattr(source_cfg, "source_type", "gaussian_process"))
        )

        # FlowPK requires a source process that supports conditional queries
        # `(t_query, t_cond, x_cond, mask_query, mask_cond)` in sampling/forward.
        if source_type in (
            "gaussian_process",
            "gp",
            "gaussian_process_regression",
            "gp_regression",
        ):
            self.source_process = GaussianProcessRegression(
                variance=float(getattr(source_cfg, "gp_variance", 1.0)),
                length_scale=float(getattr(source_cfg, "gp_length_scale", 0.1)),
                epsilon=float(getattr(source_cfg, "gp_eps", 1e-8)),
                transform=str(getattr(source_cfg, "gp_transform", "softplus")),
            )
            self.source_is_time_series = True
            return

        if source_type in ("normal", "gaussian", "white_noise"):
            self.source_process = WhiteNoiseProcess(
                epsilon=float(getattr(source_cfg, "gp_eps", 1e-8)),
                transform=str(getattr(source_cfg, "gp_transform", "softplus")),
            )
            self.source_is_time_series = False
            return

        raise ValueError(
            "FlowPK only supports conditional source types "
            "['gaussian_process', 'gp_regression', 'white_noise'] in `source_process.source_type`, "
            f"got '{source_type}'."
        )

    def __init__(self, experiment_config: FlowPKExperimentConfig) -> None:
        super().__init__(experiment_config)
        vector_field_cfg = getattr(experiment_config, "vector_field", None)
        source_cfg = self._resolve_flow_source_config()
        if vector_field_cfg is not None:
            self.z_dim = vector_field_cfg.zi_latent_dim
        else:
            raise ValueError("FlowPK requires a vector_field or network configuration.")
        self.loss_multihead = MultiHeadLoss(mode="fixed", number_of_losses=1)
        self.sigma = float(getattr(source_cfg, "flow_sigma", 1e-4)) if source_cfg else 1e-4
        self.use_OT_coupling = getattr(source_cfg, "use_OT_coupling", False)
        self.flow_num_steps = int(getattr(experiment_config, "flow_num_steps", 100))
        mix_cfg = getattr(experiment_config, "mix_data", None)
        self.sample_size = int(
            getattr(
                mix_cfg,
                "sample_size_for_generative_evaluation_val",
                getattr(experiment_config, "sample_size_for_generative_evaluation", 10),
            )
        )
        self._ensure_source_process_initialized()

        self.sample_chunk_size = 50  # chunk size for sample batches in Euler loop
        self.attn_query_chunk_size = (
            50  # chunk size over query tokens in attention (limits N×M peak memory)
        )

    def forward(self, databatch_list: Sequence[AICMECompartmentsDataBatch]) -> FlowForwardOutputs:
        """Run forward passes over every permutation and aggregate results."""

        outputs_collector = FlowForwardOutputs(loss_multihead=self.loss_multihead)

        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    # === flow matching components ===

    def sample_path(self, x0, x1, mask_cond):
        """Sample a path from the source process."""
        B = x0.shape[0]
        m = mask_cond.unsqueeze(-1).float()
        assert torch.equal(x0 * (1.0 - m), x1 * (1.0 - m)), (
            "Cond OT triangular map failed: past values of xt do not match x1"
        )
        eps = torch.randn_like(x1, device=x1.device)
        t = torch.rand((B, 1, 1, 1), device=x1.device)  # flow time =/= obs time!
        xt = t * x1 + (1.0 - t) * x0 + self.sigma * eps  # [B,1,T,1]
        xt = m * xt + (1.0 - m) * x0  # ensure past values of xt match x1 for conditioning
        return xt, t.view(B, 1)

    def coupling_with_conditioning(self, x0, x1, obs_times, mask_obs, dose, study_ctx):
        if self.use_OT_coupling:
            ot = OTSampler(batch_size=x0.shape[0], replace=False)
            return ot.sample_plan_with_conditioning(
                x0=x0,
                x1=x1,
                obs_times=obs_times,
                mask_obs=mask_obs,
                dose=dose,
                study_ctx=study_ctx,
            )
        return x0, x1, obs_times, mask_obs, dose, study_ctx

    def conditional_velocity_field(self, x0, x1, mask_cond):
        """Compute the conditional velocity field between x0 and x1."""
        m = mask_cond.unsqueeze(-1).float()
        ut = m * x1 - m * x0  # [B,1,T,1]
        return ut

    def neural_velocity_field(self, t, target, context, mask_cond):
        """
        - t: flow ode time [B, 1]
        - target: tuple of (xt, obs_times, mask_obs, dose) for the target individuals
        - context: tuple of (x_ctx, obs_times_ctx, context_obs_mask, mask_context_individuals, dose_ctx) for the study context
        - mask_cond: future mask for conditioning the velocity field to be zero for past values
        """
        xt, obs_times, mask_obs, dose = target
        if dose.ndim == 3:
            # Backward compatibility for single-target generation path.
            dose = dose.unsqueeze(1)  # [B, 1, T, 2]
        B, It, _, _ = xt.shape

        # Flatten target-individual axis so each target path is decoded independently.
        xt_flat, obs_times_flat = flatten_many_target_axis_4d(
            xt, obs_times
        )  # [B*It,1,T,1], [B*It,1,T,1]
        mask_obs_flat, mask_cond_flat = flatten_many_target_axis_3d(
            mask_obs, mask_cond
        )  # [B*It,1,T], [B*It,1,T]
        dose_flat = self._flatten_target_dose(dose)  # [B*It, T, 2]
        study_ctx_flat = self._repeat_study_context_for_targets(context, It)
        (
            x_ctx_flat,
            obs_times_ctx_flat,
            context_obs_mask_flat,
            mask_context_individuals_flat,
            dose_ctx_flat,
        ) = study_ctx_flat
        t_flat = t.repeat_interleave(It, dim=0)  # [B*It, 1]
        m = mask_cond_flat.unsqueeze(-1).float()  # [B*It, 1, T, 1]

        # study point cloud for conditioning
        pc_c, mask_pc_c, mask_attn_c = self.pointcloud_from_obs(
            x_ctx_flat,
            obs_times_ctx_flat,
            context_obs_mask_flat,
            mask_individuals=mask_context_individuals_flat,
        )  # [B*It, N, 2], [B*It, N], [B*It, N, N]

        # target point cloud for prediction
        pc_xt, mask_pc = self.pointcloud_from_obs(
            xt_flat,
            obs_times_flat,
            mask_obs_flat,
            return_attn_mask=False,
        )  # [B*It, T, 2], [B*It, T]

        vt = self.decoder(
            x=pc_xt,  # [B*It, T, 2] target point cloud
            ctx=pc_c,  # [B*It, N, 2] study point cloud
            flow_t=t_flat,  # [B*It, 1]
            mask_pad_x=mask_pc,  # [B*It, T]
            mask_pad_ctx=mask_pc_c,  # [B*It, N]
            mask_attn_ctx=mask_attn_c,  # [B*It, N, N]
            dose=dose_flat,  # [B*It, T, 2]
            dose_ctx=dose_ctx_flat,  # [B*It, N, 2]
        )
        vt = vt * m  # [B*It, 1, T, 1] zero out predictions for past
        return unflatten_target_axis_4d(vt, B, It)  # [B, It, T, 1]

    # === forward pass and sampling ===

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> FlowForwardOutputs:
        """Flow matching reconstruction with MSE loss."""
        self._ensure_source_process_initialized()

        # ... context study
        study_ctx, stats = self.preprocess_study(db)

        # ...target individual x1 with past and future concatenated

        x_past, t_past = self.scaler.forward(db.target_obs, db.target_obs_time, stats)
        x_future, t_future = self.scaler.forward(db.target_rem_sim, db.target_rem_sim_time, stats)
        x_past = x_past * db.target_obs_mask.unsqueeze(-1).float()
        x_future = x_future * db.target_rem_sim_mask.unsqueeze(-1).float()

        obs_times, x1, mask_obs, is_future = self.concat_past_and_future(
            t_past, x_past, db.target_obs_mask, t_future, x_future, db.target_rem_sim_mask
        )  # [B,It,T,1], [B,It,T,1], [B,It,T], [B,It,T]

        B, It, T, _ = obs_times.shape
        dose_amount = db.target_dosing_amounts.unsqueeze(-1).expand(-1, It, T)  # [B, It, T]
        dose_route = db.target_dosing_route_types.unsqueeze(-1).expand(-1, It, T)  # [B, It, T]
        dose = torch.stack([dose_amount, dose_route], dim=-1)  # [B, It, T, 2]

        # ...source GP regression
        t_future_flat, t_past_flat, x_past_flat = flatten_many_target_axis_4d(
            t_future, t_past, x_past
        )  # [B*It,1,T_future,1], [B*It,1,T_past,1], [B*It,1,T_past,1]
        future_mask_flat, past_mask_flat = flatten_many_target_axis_3d(
            db.target_rem_sim_mask, db.target_obs_mask
        )  # [B*It,1,T_future], [B*It,1,T_past]

        x0_future = self.source_process(
            t_query=t_future_flat,
            t_cond=t_past_flat,
            x_cond=x_past_flat,
            mask_query=future_mask_flat,
            mask_cond=past_mask_flat,
        )
        x0_future = unflatten_target_axis_4d(x0_future, B, It)  # [B, It, T_future, 1]
        _, x0, _, _ = self.concat_past_and_future(
            t_past, x_past, db.target_obs_mask, t_future, x0_future, db.target_rem_sim_mask
        )

        # ...sample from coupling and path interpolation
        x0, x1, obs_times, mask_obs, dose, study_ctx = self.coupling_with_conditioning(
            x0, x1, obs_times, mask_obs, dose, study_ctx
        )
        xt, t = self.sample_path(x0, x1, is_future)  # [B,It,T,1], [B,1]
        target = (xt, obs_times, mask_obs, dose)

        # ...loss
        ut = self.conditional_velocity_field(x0, x1, is_future)
        vt = self.neural_velocity_field(t, target, context=study_ctx, mask_cond=is_future)

        mse_dict = self.masked_mse_loss(vt, ut, is_future)

        # Store outputs
        outputs = FlowForwardOutputs(loss_multihead=self.loss_multihead, device=x1.device)
        outputs.stats = stats  # type: ignore

        outputs.update_head(
            "reconstruction",
            {
                "prediction": vt,
                "target": ut,
                "mask": is_future,
            },
        )
        outputs.update_losses(
            "reconstruction",
            {
                "mse": mse_dict["mse"],  # Single MSE loss
            },
        )
        return outputs

    def _prepare_new_individual_eval_times(
        self,
        db: AICMECompartmentsDataBatch,
        stats: TensorType["B", 1],
        *,
        resolve_sampling_from_target: bool,
        include_rem: bool = False,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare scaled evaluation times and masks for new-individual sampling."""

        if decode_times is None:
            if resolve_sampling_from_target:
                target_time_sources = (
                    ("target_obs_time", "target_rem_sim_time")
                    if include_rem
                    else ("target_obs_time",)
                )
                eval_times, mask_obs = gather_distinct_times_per_substance(
                    db,
                    time_sources=target_time_sources,
                )
            else:
                eval_times, mask_obs = gather_distinct_times_per_substance(db)
        else:
            eval_times, mask_obs = decode_times  # [B, Tdistinct_max, 1], [B, Tdistinct_max]

        mask_obs_base = mask_obs  # [B, T]
        if resolve_sampling_from_target:
            It = db.target_obs.shape[1]
            eval_times = eval_times.unsqueeze(1).expand(-1, It, -1, -1)  # [B, It, T, 1]
            mask_obs = mask_obs.unsqueeze(1).expand(-1, It, -1)  # [B, It, T]
        else:
            eval_times = eval_times.unsqueeze(1)  # [B, 1, T, 1]
            mask_obs = mask_obs.unsqueeze(1)  # [B, 1, T]

        dummy_vals = torch.zeros_like(eval_times)  # [B, It/1, T, 1]
        _, eval_times = self.scaler.forward(dummy_vals, eval_times, stats)  # [B, It/1, T, 1]
        is_future = mask_obs  # [B, It/1, T]
        return eval_times, mask_obs, mask_obs_base, is_future

    def _prepare_new_individual_dose_routes(
        self,
        db: AICMECompartmentsDataBatch,
        *,
        T: int,
        resolve_sampling_from_target: bool,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Prepare per-time dose and route features for new-individual sampling."""

        if resolve_sampling_from_target:
            dose_values, route_values = self.resolve_target_dosing_from_databatch(db, dosing)
            dose_amount = dose_values.unsqueeze(-1).expand(-1, -1, T)  # [B, It, T]
            dose_route = route_values.unsqueeze(-1).expand(-1, -1, T)  # [B, It, T]
            return torch.stack([dose_amount, dose_route], dim=-1)  # [B, It, T, 2]

        if dosing is None:
            dose_values, route_values = self.select_unseen_dosing_from_databatch(
                db
            )  # [B, 1], [B, 1]
        else:
            dose_values, route_values = dosing

        if dose_values.ndim == 1:
            dose_values = dose_values.unsqueeze(-1)  # [B, 1]
        if route_values.ndim == 1:
            route_values = route_values.unsqueeze(-1)  # [B, 1]
        dose_amount = dose_values.expand(-1, T)  # [B, T]
        dose_route = route_values.expand(-1, T)  # [B, T]
        return torch.stack([dose_amount, dose_route], dim=-1)  # [B, T, 2]

    @torch.inference_mode()
    def sample_new_individual(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int = 10,
        num_steps: int | None = None,
        decode_times: Tuple[
            TensorType["B", "Tdistinct", 1],
            TensorType["B", "Tdistinct"],
        ]
        | None = None,
        ignore_logvar: bool = True,
        dosing: tuple[TensorType["B", 1], TensorType["B", 1]] | None = None,
        resolve_sampling_from_target: bool = False,
        include_rem: bool = False,
    ) -> Tuple[
        TensorType["S", "B", "T", 1],
        TensorType["B", "T", 1],
        TensorType["B", "T"],
    ]:
        """Sample new individuals from context or resolve one sample per target.

        When ``resolve_sampling_from_target`` is ``False`` this method preserves
        the historical sample-first layout ``[S, B, T, 1]``. When it is
        ``True`` the target axis becomes the leading output axis and the method
        returns ``[It, B, T, 1]`` using target-derived times and dosing.
        Setting ``include_rem=True`` extends the inferred target-resolved time
        grid with ``target_rem_sim_time`` when ``decode_times`` is not given.
        """
        self._ensure_source_process_initialized()
        if num_steps is None:
            num_steps = self.flow_num_steps
        if num_steps <= 0:
            raise ValueError(f"`num_steps` must be a positive integer, got {num_steps}.")

        device = self.device
        B = db.context_obs.shape[0]
        if resolve_sampling_from_target:
            _require_target_observation_schedule(db)

        # ...context study
        study_ctx, stats = self.preprocess_study(db)

        eval_times, mask_obs, mask_obs_base, is_future = self._prepare_new_individual_eval_times(
            db,
            stats,
            resolve_sampling_from_target=resolve_sampling_from_target,
            include_rem=include_rem,
            decode_times=decode_times,
        )
        T = mask_obs_base.shape[1]
        dose = self._prepare_new_individual_dose_routes(
            db,
            T=T,
            resolve_sampling_from_target=resolve_sampling_from_target,
            dosing=dosing,
        )

        # ...chunked vectorized Euler integration
        delta_t = 1.0 / num_steps
        t_vals = torch.linspace(delta_t, 1.0, steps=num_steps, device=device)

        self._set_attn_query_chunk_size(self.attn_query_chunk_size)
        try:
            if resolve_sampling_from_target:
                It = db.target_obs.shape[1]
                BIt = B * It

                study_ctx_k = tuple(self._repeat_batch(v, It) for v in study_ctx)
                stats_k = self._repeat_stats(stats, It, B)
                ev = flatten_many_target_axis_4d(eval_times)[0]  # [BIt, 1, T, 1]
                mo, isf = flatten_many_target_axis_3d(mask_obs, is_future)  # [BIt, 1, T]
                d = dose.reshape(BIt, T, 2)  # [BIt, T, 2]

                x = self.source_process(t_query=ev, mask_query=mo)  # [BIt, 1, T, 1]
                for step in tqdm(
                    t_vals,
                    desc="Sampling trajectories (target-resolved new individual)",
                    ncols=80,
                ):
                    t = step.view(1, 1).expand(BIt, 1)
                    vt = self.neural_velocity_field(
                        t, (x, ev, mo, d), context=study_ctx_k, mask_cond=isf
                    )
                    x += vt * delta_t

                x1, times = self.scaler.inverse(x, ev, stats_k)
                samples_out = (
                    x1.reshape(B, It, *x1.shape[1:]).permute(1, 0, 2, 3, 4).contiguous().squeeze(2)
                )  # [It, B, T, 1]
                times_out = times.reshape(B, It, *times.shape[1:])[:, 0].squeeze(1)
            else:
                S = int(sample_size)
                K = min(self.sample_chunk_size, S) if self.sample_chunk_size else S
                chunk_results: list[torch.Tensor] = []
                times_out: torch.Tensor | None = None

                for chunk_start in range(0, S, K):
                    chunk = min(K, S - chunk_start)
                    BK = B * chunk

                    study_ctx_k = tuple(self._repeat_batch(v, chunk) for v in study_ctx)
                    stats_k = self._repeat_stats(stats, chunk, B)
                    rep = self._repeat_many(
                        chunk,
                        eval_times=eval_times,
                        mask_obs=mask_obs,
                        is_future=is_future,
                        dose=dose,
                    )
                    ev = rep["eval_times"]  # [BK, 1, T, 1]
                    mo = rep["mask_obs"]  # [BK, 1, T]
                    isf = rep["is_future"]  # [BK, 1, T]
                    d = rep["dose"]  # [BK, T, 2]

                    x = self.source_process(t_query=ev, mask_query=mo)  # [BK, 1, T, 1]
                    for step in tqdm(
                        t_vals, desc="Sampling trajectories (new individual)", ncols=80
                    ):
                        t = step.view(1, 1).expand(BK, 1)
                        vt = self.neural_velocity_field(
                            t, (x, ev, mo, d), context=study_ctx_k, mask_cond=isf
                        )
                        x += vt * delta_t

                    x1, times = self.scaler.inverse(x, ev, stats_k)
                    chunk_results.append(
                        x1.reshape(B, chunk, *x1.shape[1:])
                        .permute(1, 0, 2, 3, 4)
                        .contiguous()
                        .squeeze(2)
                    )
                    if times_out is None:
                        times_out = (
                            times.reshape(B, chunk, *times.shape[1:])
                            .permute(1, 0, 2, 3, 4)
                            .contiguous()[0]
                            .squeeze(1)
                        )
                samples_out = torch.cat(chunk_results, dim=0)  # [S, B, T, 1]
        finally:
            self._set_attn_query_chunk_size(None)

        return samples_out, times_out, mask_obs_base

    @torch.inference_mode()
    def sample_individual_prediction(
        self,
        db: AICMECompartmentsDataBatch,
        sample_size: int | None = None,
        num_steps: int | None = None,
        dosing: tuple[TensorType["B", 1], TensorType["B", 1]] | None = None,
    ):
        self._ensure_source_process_initialized()
        db = self._select_single_target_individual(db)
        device = self.device
        B = db.context_obs.shape[0]
        if sample_size is None:
            S = int(self.sample_size)
        else:
            S = int(sample_size)
        if S <= 0:
            raise ValueError(f"`sample_size` must be a positive integer, got {S}.")

        if num_steps is None:
            num_steps = int(self.flow_num_steps)
        if num_steps <= 0:
            raise ValueError(f"`num_steps` must be a positive integer, got {num_steps}.")
        BS = B * S

        # ...context study
        study_ctx, stats = self.preprocess_study(db)

        # ...eval time and dosing context for generation
        x_past, t_past = self.scaler.forward(db.target_obs, db.target_obs_time, stats)
        x_future, t_future = self.scaler.forward(db.target_rem_sim, db.target_rem_sim_time, stats)
        x_past = x_past * db.target_obs_mask.unsqueeze(-1).float()
        x_future = x_future * db.target_rem_sim_mask.unsqueeze(-1).float()

        eval_times, _, mask_obs, is_future = self.concat_past_and_future(
            t_past, x_past, db.target_obs_mask, t_future, x_future, db.target_rem_sim_mask
        )  # [B,It,T,1], [B,It,T,1], [B,It,T], [B,It,T]

        _, It, T, _ = eval_times.shape
        dose_amount = db.target_dosing_amounts.unsqueeze(-1).expand(-1, It, T)  # [B, It, T]
        dose_route = db.target_dosing_route_types.unsqueeze(-1).expand(-1, It, T)  # [B, It, T]
        dose = torch.stack([dose_amount, dose_route], dim=-1)  # [B, It, T, 2]

        # ...chunked vectorized Euler integration
        K = min(self.sample_chunk_size, S) if self.sample_chunk_size else S
        delta_t = 1.0 / num_steps
        t_vals = torch.linspace(delta_t, 1.0, steps=num_steps, device=device)
        n_future = db.target_rem_sim.shape[-2]

        chunk_x1: list[torch.Tensor] = []
        chunk_t: list[torch.Tensor] = []

        self._set_attn_query_chunk_size(self.attn_query_chunk_size)
        try:
            for chunk_start in range(0, S, K):
                chunk = min(K, S - chunk_start)
                BK = B * chunk

                study_ctx_k = tuple(self._repeat_batch(v, chunk) for v in study_ctx)
                stats_k = self._repeat_stats(stats, chunk, B)
                rep = self._repeat_many(
                    chunk,
                    t_past=t_past,
                    x_past=x_past,
                    t_future=t_future,
                    eval_times=eval_times,
                    mask_obs=mask_obs,
                    is_future=is_future,
                    dose=dose,
                    past_mask=db.target_obs_mask,
                    future_mask=db.target_rem_sim_mask,
                )
                tp = rep["t_past"]  # [BK,It,T_past,1]
                xp = rep["x_past"]  # [BK,It,T_past,1]
                tf = rep["t_future"]  # [BK,It,T_future,1]
                ev = rep["eval_times"]  # [BK,It,T,1]
                mo = rep["mask_obs"]  # [BK,It,T]
                isf = rep["is_future"]  # [BK,It,T]
                d = rep["dose"]  # [BK,It,T,2]
                pm = rep["past_mask"]  # [BK,It,T_past]
                fm = rep["future_mask"]  # [BK,It,T_future]

                BK = tp.shape[0]
                It = tp.shape[1]
                tf_flat, tp_flat, xp_flat = flatten_many_target_axis_4d(
                    tf, tp, xp
                )  # [BK*It,1,T_future,1], [BK*It,1,T_past,1], [BK*It,1,T_past,1]
                fm_flat, pm_flat = flatten_many_target_axis_3d(
                    fm, pm
                )  # [BK*It,1,T_future], [BK*It,1,T_past]

                x0_future = self.source_process(
                    t_query=tf_flat,
                    t_cond=tp_flat,
                    x_cond=xp_flat,
                    mask_query=fm_flat,
                    mask_cond=pm_flat,
                )  # [BK*It,1,T_future,1]
                x0_future = unflatten_target_axis_4d(x0_future, BK, It)  # [BK,It,T_future,1]

                _, x, _, _ = self.concat_past_and_future(
                    tp, xp, pm, tf, x0_future, fm
                )  # [BK,It,T,1]

                for step in tqdm(
                    t_vals, desc="Sampling trajectories (individual prediction)", ncols=80
                ):
                    t = step.view(1, 1).expand(BK, 1)
                    vt = self.neural_velocity_field(
                        t, (x, ev, mo, d), context=study_ctx_k, mask_cond=isf
                    )
                    x += vt * delta_t

                x1, ev_out = self.scaler.inverse(x, ev, stats_k)
                # x1 = x1.exp() * mo.unsqueeze(-1)
                tf_out, x1f = self.split_past_and_future(ev_out, x1, mo, isf, n_future=n_future)

                chunk_x1.append(
                    x1f.reshape(B, chunk, *x1f.shape[1:]).permute(1, 0, 2, 3, 4).contiguous()
                )
                chunk_t.append(
                    tf_out.reshape(B, chunk, *tf_out.shape[1:]).permute(1, 0, 2, 3, 4).contiguous()
                )
        finally:
            self._set_attn_query_chunk_size(None)

        x1_future = torch.cat(chunk_x1, dim=0)  # [S, B, It, T_rem, 1]
        times_future = torch.cat(chunk_t, dim=0)  # [S, B, It, T_rem, 1]
        return x1_future, times_future, db.target_rem_sim, db.target_rem_sim_mask

    # === helper methods ===

    @staticmethod
    def _flatten_target_dose(dose: torch.Tensor) -> torch.Tensor:
        """Flatten target axis in dose tensor: [B, It, T, 2] -> [B*It, T, 2]."""
        B, It, T, D = dose.shape
        return dose.reshape(B * It, T, D)

    @staticmethod
    def _repeat_study_context_for_targets(
        study_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        num_targets: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Repeat study-level context along batch axis once per target individual."""
        x_ctx, obs_times_ctx, context_obs_mask, mask_context_individuals, dose_ctx = study_ctx
        return (
            x_ctx.repeat_interleave(num_targets, dim=0),
            obs_times_ctx.repeat_interleave(num_targets, dim=0),
            context_obs_mask.repeat_interleave(num_targets, dim=0),
            mask_context_individuals.repeat_interleave(num_targets, dim=0),
            dose_ctx.repeat_interleave(num_targets, dim=0),
        )

    @staticmethod
    def _first_valid_target_indices(mask_target_individuals: torch.Tensor) -> torch.Tensor:
        """Return one target index per batch item (first valid if available, else zero)."""
        has_valid = mask_target_individuals.any(dim=1)  # [B]
        first_valid = torch.argmax(mask_target_individuals.long(), dim=1)  # [B]
        fallback = torch.zeros_like(first_valid)
        return torch.where(has_valid, first_valid, fallback)

    @classmethod
    def _take_target_axis(cls, x: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
        """Gather one target individual per batch: [B, It, ...] -> [B, 1, ...]."""
        b_idx = torch.arange(x.shape[0], device=x.device)
        return x[b_idx, target_indices].unsqueeze(1)

    @classmethod
    def _select_single_target_individual(
        cls, db: AICMECompartmentsDataBatch
    ) -> AICMECompartmentsDataBatch:
        """Select one target individual per batch item for sampling-only APIs."""
        target_indices = cls._first_valid_target_indices(db.mask_target_individuals)  # [B]

        selected_names: list[list[str]] = []
        idx_cpu = target_indices.detach().cpu().tolist()
        for b, idx in enumerate(idx_cpu):
            names_b = db.target_subject_name[b] if b < len(db.target_subject_name) else []
            if names_b and 0 <= idx < len(names_b):
                selected_names.append([names_b[idx]])
            elif names_b:
                selected_names.append([names_b[0]])
            else:
                selected_names.append([f"target_{b}_0"])

        return db._replace(
            target_obs=cls._take_target_axis(db.target_obs, target_indices),  # [B,1,T_obs,1]
            target_obs_time=cls._take_target_axis(
                db.target_obs_time, target_indices
            ),  # [B,1,T_obs,1]
            target_obs_mask=cls._take_target_axis(
                db.target_obs_mask, target_indices
            ),  # [B,1,T_obs]
            target_rem_sim=cls._take_target_axis(
                db.target_rem_sim, target_indices
            ),  # [B,1,T_rem,1]
            target_rem_sim_time=cls._take_target_axis(
                db.target_rem_sim_time, target_indices
            ),  # [B,1,T_rem,1]
            target_rem_sim_mask=cls._take_target_axis(
                db.target_rem_sim_mask, target_indices
            ),  # [B,1,T_rem]
            target_dosing_amounts=cls._take_target_axis(
                db.target_dosing_amounts, target_indices
            ),  # [B,1]
            target_dosing_route_types=cls._take_target_axis(
                db.target_dosing_route_types, target_indices
            ),  # [B,1]
            mask_target_individuals=torch.ones(
                db.mask_target_individuals.shape[0],
                1,
                dtype=db.mask_target_individuals.dtype,
                device=db.mask_target_individuals.device,
            ),  # [B,1]
            target_subject_name=selected_names,
        )

    def preprocess_study(self, db, stats=None):
        B, I, C, _ = db.context_obs.shape
        N = I * C  # total number of observations in study

        if stats is None:
            stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)

        x_ctx, obs_times_ctx = self.scaler.forward(
            db.context_obs, db.context_obs_time, stats
        )  # [B,I,C,1], [B,I,C,1]

        x_ctx = x_ctx * db.context_obs_mask.unsqueeze(-1).float()

        # ...get dosing information for context study, expand to observation level
        dose_amount = db.context_dosing_amounts  # [B, I]
        dose_amount = dose_amount.unsqueeze(-1).expand(-1, -1, C).reshape(B, N)  # [B, N]
        dose_route = db.context_dosing_route_types  # [B, I]
        dose_route = dose_route.unsqueeze(-1).expand(-1, -1, C).reshape(B, N)  # [B, N]
        dose = torch.stack([dose_amount, dose_route], dim=-1)  # [B, N, 2]

        study = (x_ctx, obs_times_ctx, db.context_obs_mask, db.mask_context_individuals, dose)

        return study, stats

    def pointcloud_from_obs(
        self,
        x,
        time,
        obs_mask,
        return_attn_mask: bool = True,
        mask_individuals: torch.Tensor = None,
    ):
        """Convert observation tensor to point cloud format.
        Args:
            x: TensorType["B", "I", "T", 1] - observations values
            time: TensorType["B", "I", "T", 1] - observation times
            obs_mask: TensorType["B", "I", "T"] - observation-level mask
            mask_individuals: TensorType["B", "I"] - optional mask for valid individuals
        Returns:
            pc: TensorType["B", "N", 2] - 2D point cloud [time, value] with N = I*T points
            mask: TensorType["B", "N"] - zero-padding mask for valid points
            attn_mask: TensorType["B", "N", "N"] -  Block structure respecting individual exchangeability
                                                    and observation exchangeability within individuals.

        """
        B, I, T, _ = x.shape
        N = I * T
        pc = torch.concat([time, x], dim=-1).reshape(B, N, 2)
        padding_mask = obs_mask.reshape(B, N)  # [B, N]

        if mask_individuals is not None:
            ind_mask_expanded = (
                mask_individuals.unsqueeze(-1).expand(B, I, T).reshape(B, N)
            )  # [B, N]
            padding_mask = padding_mask & ind_mask_expanded

        # Block structure for attention mask
        if not return_attn_mask:
            return pc, padding_mask

        block_ids = torch.arange(I, device=x.device).unsqueeze(1).expand(I, T).reshape(N)
        block_attn = block_ids.unsqueeze(0) == block_ids.unsqueeze(1)  # [N, N]
        attn_mask = block_attn.unsqueeze(0) & padding_mask.unsqueeze(2) & padding_mask.unsqueeze(1)

        return pc, padding_mask, attn_mask

    def select_unseen_dosing_from_databatch(
        self,
        db: AICMECompartmentsDataBatch,
        generator: torch.Generator | None = None,
    ) -> Tuple[
        TensorType["B", 1],
        TensorType["B", 1],
    ]:
        """Select dosing information for a new individual."""

        target_mask = db.mask_target_individuals  # [B, It]
        context_mask = db.mask_context_individuals  # [B, Ic]

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

        dose = dose.unsqueeze(-1)  # [B, 1]
        route = route.unsqueeze(-1)  # [B, 1]
        return dose.to(device), route.to(route_device)

    def resolve_target_dosing_from_databatch(
        self,
        db: AICMECompartmentsDataBatch,
        dosing: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve per-target dosing tensors shaped ``[B, It]``."""

        B, It = db.target_dosing_amounts.shape
        if dosing is None:
            dose_values = db.target_dosing_amounts
            route_values = db.target_dosing_route_types
        else:
            dose_values, route_values = dosing

        if dose_values.ndim == 1:
            dose_values = dose_values.unsqueeze(-1).expand(-1, It)
        if route_values.ndim == 1:
            route_values = route_values.unsqueeze(-1).expand(-1, It)

        if dose_values.shape != (B, It):
            raise ValueError(
                "Target-resolved dosing amounts must have shape [B, It], got "
                f"{tuple(dose_values.shape)} for batch {(B, It)}."
            )
        if route_values.shape != (B, It):
            raise ValueError(
                "Target-resolved dosing routes must have shape [B, It], got "
                f"{tuple(route_values.shape)} for batch {(B, It)}."
            )

        return dose_values, route_values

    def concat_past_and_future(
        self, past_time, past_values, past_mask, future_time, future_values, future_mask
    ):
        """
        Concatenate past and future observations for a single batch, moving padding to the end.
        Args:
            past_time: (B, I, T_past, 1)
            past_values: (B, I, T_past, 1)
            past_mask: (B, I, T_past)
            future_time: (B, I, T_future, 1)
            future_values: (B, I, T_future, 1)
            future_mask: (B, I, T_future)
        Returns:
            T: Concatenated times with padding at end. Shape: (B, I, T_past + T_future, 1)
            X: Concatenated values with padding at end. Shape: (B, I, T_past + T_future, 1)
            M: Combined validity mask. Shape: (B, I, T_past + T_future)
            is_future: Mask indicating future obs (True=future, False=past/padding). Shape: (B, I, T_past + T_future)
        """

        T_naive = torch.cat([past_time, future_time], dim=-2)
        X_naive = torch.cat([past_values, future_values], dim=-2)
        M_naive = torch.cat([past_mask, future_mask], dim=-1)

        n_past = past_time.shape[-2]
        n_future = future_time.shape[-2]
        device = past_time.device

        base_priority = torch.cat(
            [torch.zeros(n_past, device=device), torch.ones(n_future, device=device)]
        )
        base_priority = base_priority.expand_as(M_naive)
        priority = torch.where(M_naive, base_priority, base_priority + 2)
        sort_indices = torch.argsort(priority, dim=-1, stable=True)
        sort_indices_expanded = sort_indices.unsqueeze(-1)
        T = torch.gather(T_naive, dim=-2, index=sort_indices_expanded)
        X = torch.gather(X_naive, dim=-2, index=sort_indices_expanded.expand_as(X_naive))
        M = torch.gather(M_naive, dim=-1, index=sort_indices)
        original_is_future = torch.cat(
            [
                torch.zeros(n_past, dtype=torch.bool, device=device),
                torch.ones(n_future, dtype=torch.bool, device=device),
            ]
        ).expand_as(M_naive)
        is_future = torch.gather(original_is_future, dim=-1, index=sort_indices)
        is_future = is_future & M
        return T, X, M, is_future

    def split_past_and_future(
        self,
        T: torch.Tensor,
        X: torch.Tensor,
        M: torch.Tensor,
        is_future: torch.Tensor,
        n_future: int,
    ):
        """Inverse of concat_past_future: extract future observations from concatenated tensor.

        Args:
            T:         (B, I, n_past + n_future, 1)  - concatenated times
            X:         (B, I, n_past + n_future, 1)  - concatenated values
            M:         (B, I, n_past + n_future)     - combined validity mask
            is_future: (B, I, n_past + n_future)     - True where position is a valid future obs
            n_future:  int                         - original number of future time slots
        Returns:
            future_time:   (B, I, n_future, 1)
            future_values: (B, I, n_future, 1)
            future_mask:   (B, I, n_future)        - True where slot has a valid future obs
        """
        B, I, N, _ = X.shape
        device = X.device

        future_order = torch.argsort(~is_future, dim=-1, stable=True)  # (B, I, N): future idx first
        future_idx = future_order[..., :n_future]  # (B, I, n_future)

        future_mask = torch.gather(is_future, dim=-1, index=future_idx)  # (B, I, n_future)

        idx_expanded = future_idx.unsqueeze(-1)  # (B, I, n_future, 1)
        future_time = torch.gather(T, dim=-2, index=idx_expanded)  # (B, I, n_future, 1)
        future_time = future_time * future_mask.unsqueeze(-1)  # zero out invalid future times
        future_values = torch.gather(X, dim=-2, index=idx_expanded.expand(B, I, n_future, 1))
        future_values = future_values * future_mask.unsqueeze(-1)  # zero out invalid future values
        return future_time, future_values

    def _set_attn_query_chunk_size(self, size):
        """Propagate query_chunk_size to every attention module that supports it.
        Set size=None to restore the default full-matrix path.
        """
        for m in self.modules():
            if hasattr(m, "query_chunk_size"):
                m.query_chunk_size = size

    @staticmethod
    def _repeat_batch(x: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
        """Repeat batch dimension `repeats` times for tensorized sampling."""
        if x is None:
            return None
        return x.repeat_interleave(repeats, dim=0)

    @staticmethod
    def _repeat_stats(
        stats_dict: dict[str, torch.Tensor], repeats: int, base_batch: int
    ) -> dict[str, torch.Tensor]:
        """Repeat per-batch statistics for tensorized sampling."""
        stats_out: dict[str, torch.Tensor] = {}
        for k, v in stats_dict.items():
            if torch.is_tensor(v) and v.shape[0] == base_batch:
                stats_out[k] = v.repeat_interleave(repeats, dim=0)
            else:
                stats_out[k] = v
        return stats_out

    @classmethod
    def _repeat_many(
        cls, repeats: int, **tensors: torch.Tensor | None
    ) -> dict[str, torch.Tensor | None]:
        """Repeat batch dimension for multiple tensors in one call."""
        return {k: cls._repeat_batch(v, repeats) for k, v in tensors.items()}


class OTSampler:
    """Optimal Transport Sampler for coupling source and target distributions.

    Uses the Earth Mover's Distance (EMD) to compute an optimal transport plan
    between batches of samples, enabling OT-based coupling in flow matching.

    Args:
        batch_size: Number of samples to draw from the transport plan.
        replace: Whether to sample with replacement from the transport plan.
    """

    def __init__(self, batch_size, replace=False):
        if pot is None:
            raise ModuleNotFoundError(
                "Optimal-transport coupling requires POT. Install the 'pot' package "
                "or set source_process.use_OT_coupling=false."
            )
        self.ot_fn = partial(pot.emd, numThreads=1)
        self.batch_size = batch_size
        self.replace = replace

    def get_map(self, x0, x1):
        """Compute the optimal transport map between x0 and x1.

        Args:
            x0: Source samples with shape [B, 1, T, 1] or [B, ...].
            x1: Target samples with shape [B, 1, T, 1] or [B, ...].

        Returns:
            Transport plan matrix of shape [B, B].
        """
        # Flatten tensors to [B, -1] for distance computation
        # x0, x1 typically have shape [B, 1, T, 1], need [B, T] for cdist
        x0_flat = x0.reshape(x0.shape[0], -1)  # [B, T]
        x1_flat = x1.reshape(x1.shape[0], -1)  # [B, T]

        a = pot.unif(x0_flat.shape[0])  # uniform distribution over B source samples
        b = pot.unif(x1_flat.shape[0])  # uniform distribution over B target samples
        M = torch.cdist(x0_flat, x1_flat) ** 2  # [B, B] squared Euclidean distance
        p = self.ot_fn(a, b, M.detach().cpu().numpy())

        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            print("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi):
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=self.batch_size, replace=self.replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan_with_conditioning(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        obs_times: torch.Tensor,
        mask_obs: torch.Tensor,
        dose: torch.Tensor,
        study_ctx: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ):
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi)
        x_ctx, obs_times_ctx, context_obs_mask, mask_context_individuals, dose_ctx = study_ctx
        mapped_study_ctx = (
            x_ctx[j],
            obs_times_ctx[j],
            context_obs_mask[j],
            mask_context_individuals[j],
            dose_ctx[j],
        )
        return x0[i], x1[j], obs_times[j], mask_obs[j], dose[j], mapped_study_ctx


__all__ = ["FlowForwardOutputs", "FlowPK", "OTSampler"]
