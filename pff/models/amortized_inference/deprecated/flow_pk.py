from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch
from pff.models.amortized_inference.flows_pk import FlowForwardOutputs
from pff.models.amortized_inference.generative_pk import (
    NewBasePKModel,
    NewGenerativeMixin,
)
from pff.models.architectures.aggregators import MeanStudyAggregator
from pff.models.diffusion.noise import GaussianProcess
from pff.models.utils.loss_utils import MultiHeadLoss


import torch
from torchtyping import TensorType
from tqdm import tqdm


from typing import Sequence, Tuple


class FlowPK___depracated(NewBasePKModel, NewGenerativeMixin):
    """Flow Matching PK model with single RMSE loss.

    Simplified PK model for flow matching with:
    - Basic encoder-decoder architecture from NewBasePKModel
    - Study-level aggregation for context information
    - Single RMSE loss computation
    - Flow matching interpolation path
    """

    def __init__(self, model_config: NodePKExperimentConfig) -> None:
        super().__init__(model_config)

        vector_field_cfg = getattr(model_config, "vector_field", None)
        if vector_field_cfg is not None:
            self.z_dim = vector_field_cfg.zi_latent_dim
        elif self.encoder is not None:
            self.z_dim = self.encoder.zi_latent_dim
        else:
            raise ValueError("FlowPK requires a vector_field or network configuration.")
        self.aggregator = MeanStudyAggregator()
        self.loss_multihead = MultiHeadLoss(mode="fixed", number_of_losses=1)
        self.sigma = 0.1
        self.source_process = GaussianProcess(dim=1)

    def build_visualization_callback(self):
        """Return callbacks that handle visualization and empirical evaluation."""
        train_cfg = getattr(getattr(self, "model_config", None), "train", None)
        callbacks_scheduler = getattr(train_cfg, "callbacks_scheduler", None)
        if not callbacks_scheduler:
            return []

        from pff.training.callbacks.scheduler import BaseSchedulerCallback
        from pff.training.callbacks.task_registry import TASK_REGISTRY

        return [BaseSchedulerCallback.from_config(cfg=callbacks_scheduler, registry=TASK_REGISTRY)]

    def forward(self, databatch_list: Sequence[AICMECompartmentsDataBatch]) -> FlowForwardOutputs:
        """Run forward passes over every permutation and aggregate results."""

        outputs_collector = FlowForwardOutputs(loss_multihead=self.loss_multihead)

        for batch in databatch_list:
            outputs = self._forward_reconstruction(batch)
            outputs_collector.add(outputs)

        aggregated_outputs = outputs_collector.reduce()
        return aggregated_outputs

    def _reshape_time_like(self, t, state):
        if isinstance(t, (float, int)):
            return t
        return t.reshape(-1, *([1] * (state.ndim - 1)))

    def _study_latent(
        self, db: AICMECompartmentsDataBatch, use_target: bool = False
    ) -> Tuple[TensorType["B", "Z"], TensorType["B", 1]]:
        """Encode study observations into latent z_s."""

        Xc_raw, Tc_raw, M = db.context_obs, db.context_obs_time, db.context_obs_mask
        # Xc_raw/Tc_raw: [B,Ic,Tc,1], M: [B,Ic,Tc]
        mask_ind = db.mask_individuals  # [B,Ic]

        # Scaling statistics from context only
        stats = self.scaler.stats(db.context_obs, db.context_obs_time, db.context_obs_mask)
        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)  # [B,Ic,Tc,1], [B,Ic,Tc,1]

        dose = db.context_dosing_amounts  # [B,Ic]
        route = db.context_dosing_route_types  # [B,Ic]

        z_ci = self.encoder(Xc, Tc, M, dose, route, mask_ind)  # [B,Ic,Z]
        z_s = self.aggregator(z_ci, mask_ind)  # [B,Z]

        return z_s, z_ci, stats

    def _forward_reconstruction(self, db: AICMECompartmentsDataBatch) -> FlowForwardOutputs:
        """Flow matching reconstruction with RMSE loss."""

        z_s, z_i, stats = self._study_latent(db, use_target=False)  # [B,Z], [B,Ic,Z]

        Xc_raw, Tc_raw, Mc = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        # Xc_raw: [B,Ic,Tc,1], Xt_raw: [B,It,Tt,1]
        Mc_individuals = db.mask_context_individuals  # [B,Ic]
        Mt_individuals = db.mask_target_individuals  # [B,It]

        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)  # [B,Ic,T,1], [B,Ic,T,1]
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B,It,T,1], [B,It,T,1]

        # shapes
        B, It, T, _ = Xt.shape
        BI = B * It

        outputs = FlowForwardOutputs(loss_multihead=self.loss_multihead, device=Xt_raw.device)
        outputs.stats = stats

        # Flow matching: sample interpolation path
        eps = torch.randn_like(Xt, device=Xt.device)  # [B,It,T,1]
        Xt_0 = self.source_process(t=Tt, device=Xt.device)  # [B,It,T,1]
        tau = torch.rand((B, 1, 1, 1), device=Xt.device)  # [B,1,1,1]
        Xtau = tau * Xt + (1 - tau) * Xt_0  # [B,It,T,1]
        Xtau += self.sigma * eps  # [B,It,T,1]

        # first observation
        init_state, init_mask, first_t_s = self.get_first_valid_observation(Xt, Mt, Tt)
        # z_i = z_s.unsqueeze(1).repeat(1, It, 1)  # [B,It,Z] TODO: encode target individuals
        dose = db.target_dosing_amounts  # [B,It,1,1]
        route = db.target_dosing_route_types.float()  # [B,It,1,1]

        # vector fields
        utau = Xt - Xt_0  # [B,It,T,1] conditional vector field

        print(
            ">>>> ",
            Xtau.shape,
            Tt.shape,
            z_s.shape,
            z_i.shape,
            tau.shape,
            init_state.shape,
            first_t_s.shape,
            dose.shape,
            route.shape,
        )

        vtau = self.decoder(
            x=Xtau,  # [B,It,T,1]
            decode_time=Tt,  # [B,It,T,1]
            z_s=z_s,  # [B,Z]
            z_i=z_i,  # [B,Ic,Z]
            flow_time=tau.view(B, 1),  # [B,1]
            init_state=init_state,  # [B,It,1,1]
            first_t_s=first_t_s,  # [B,It,1,1]
            dose=dose,  # [B,It]
            route=route,  # [B,It]
        )

        mse_dict = self.masked_mse_loss(vtau, utau, Mt, mask_individuals=Mt_individuals)

        # Store outputs
        outputs.update_head(
            "reconstruction",
            {
                "prediction": vtau,
                "target": utau,
                "mask": Mt_individuals,
            },
        )

        outputs.update_losses(
            "reconstruction",
            {
                "mse": mse_dict["mse"],  # Single RMSE loss
            },
        )

        return outputs

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
    ) -> Tuple[
        TensorType["S", "B", "Tdistinct", 1],
        TensorType["B", "Tdistinct", 1],
        TensorType["B", "Tdistinct"],
    ]:
        """Sample trajectories for new individuals conditioned on context."""

        _ = decode_times
        _ = ignore_logvar

        z_s, z_i, stats = self._study_latent(db, use_target=False)  # [B,Z]
        Xc_raw, Tc_raw, _ = db.context_obs, db.context_obs_time, db.context_obs_mask
        Xt_raw, Tt_raw, Mt = db.target_obs, db.target_obs_time, db.target_obs_mask
        # Xc_raw: [B,Ic,Tc,1], Xt_raw: [B,It,Tt,1]
        Xc, Tc = self.scaler.forward(Xc_raw, Tc_raw, stats)
        Xt, Tt = self.scaler.forward(Xt_raw, Tt_raw, stats)  # [B,It,T,1], [B,It,T,1]

        # shapes
        B, It, T, _ = Xt.shape
        BI = B * It
        device = self.device

        # setup context
        init_true, init_mask, first_t_s = self.get_first_valid_observation(Xt, Mt, Tt)
        init_state = init_true  # [B,It,1,1]
        # z_i = z_s.unsqueeze(1).repeat(1, It, 1)  # [B,It,Z]
        dose = db.target_dosing_amounts.unsqueeze(-1).unsqueeze(-1)  # [B,It,1,1]
        route = db.target_dosing_route_types.float().unsqueeze(-1).unsqueeze(-1)  # [B,It,1,1]
        dose_feat = dose.view(BI, 1, -1)  # [BI,1,1]
        route_feat = route.view(BI, 1, -1)  # [BI,1,1]

        # flow matching sampling
        tau_steps = torch.linspace(0.0, 1.0, num_steps, device=device)
        delta_tau = (tau_steps[-1] - tau_steps[0]) / (len(tau_steps) - 1)

        samples: list[torch.Tensor] = []

        for _ in tqdm(range(sample_size), desc="Sampling new individuals", ncols=80):
            X0 = self.source_process(t=Tt, device=device)  # [B,It,T,1]
            X = X0.clone().to(device)  # [B,It,T,1]

            for step in tau_steps:  # Euler integration steps
                tau = torch.full((len(X),), step.item(), device=device)  # [B,]
                vtau = self.decoder(
                    x=X.squeeze(1),  # [B,T,1]
                    decode_time=Tt.squeeze(1),
                    z_s=z_s,
                    z_i=z_i,
                    flow_time=tau.view(-1, 1, 1),  # [B,1,1]
                    init_state=init_state.view(BI, 1, -1),
                    first_t_s=first_t_s.view(BI, 1, -1),
                    dose=dose_feat,
                    route=route_feat,
                )  # [B,It,T,1]

                X += vtau * delta_tau
            X_raw, Tt_raw = self.scaler.inverse(X, Tt, stats)
            samples.append(X_raw.squeeze(1))  # [B,T,1]

        stacked_samples = torch.stack(samples, dim=0)  # [S,B,T,1]

        return stacked_samples, Tt_raw.squeeze(1), Mt.squeeze(1)
