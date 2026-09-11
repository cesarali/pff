import os
import random
import tempfile
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from pff import data_dir
from pff.config_classes.data_config import ObservationsConfig
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.data.data_empirical.builder import (
    EmpiricalBatchConfig,
    JSON2AICMEBuilder,
)
from pff.data.data_empirical.json_schema import StudyJSON
from pff.data.data_generation.compartment_models_management import (
    _build_synthetic_vpc_data_list,
    build_sample_experiment_studies,
    prepare_full_simulation,
    prepare_full_simulation_with_repeated_targets,
)
from pff.data.data_generation.observations_classes import (
    ObservationStrategyFactory,
)
from pff.data.datasets.aicme_batch import (
    AICMECompartmentsDataBatch,
)
from pff.utils.tensors_operations import ensure_mask_or_empty, ensure_tensor_or_empty


def ensure_min_valid(mask, min_length):
    """
    Ensures that each row of the last dimension in the mask has at least `min_length` valid (1s) entries.
    """
    valid_counts = mask.sum(dim=-1, keepdim=True)  # Count valid entries along time dimension
    needs_fixing = valid_counts < min_length  # Identify sequences needing more valid entries

    if needs_fixing.any():
        # Find the top `min_length` indices in each row (sorted for deterministic filling)
        _, topk_indices = torch.topk(
            mask + torch.rand_like(mask) * 0.01, k=min_length, dim=-1, sorted=True
        )

        # Create an empty mask and scatter `1`s at selected indices
        fixed_mask = torch.zeros_like(mask)
        fixed_mask.scatter_(-1, topk_indices, 1.0)

        # Combine the original and fixed masks
        mask = torch.where(needs_fixing, fixed_mask, mask)

    return mask


def is_valid_simulation(sim: torch.Tensor) -> bool:
    """Returns True if the simulation is numerically valid and all values are < 10."""
    return torch.isfinite(sim).all() and (sim >= 0).all() and (sim < 10).all()


def _stack_one_perm(
    batches: Sequence["AICMECompartmentsDataBatch"],
) -> "AICMECompartmentsDataBatch":
    """Collate one permutation worth of AICME databatches.

    Two tensor layouts are supported:

    1. Dataset-style items without a leading batch axis, for example
       ``target_obs.shape == [I, T, 1]``. These should be stacked to
       ``[B, I, T, 1]`` via :func:`default_collate`.
    2. Builder-style items that already carry ``B=1``, for example
       ``target_obs.shape == [1, I, T, 1]``. These should be concatenated
       along the existing batch axis to avoid introducing an extra singleton
       dimension such as ``[B, 1, I, T, 1]``.
    """

    if not batches:
        raise ValueError("Cannot collate an empty sequence of AICME databatches.")

    first_batch = batches[0]
    has_leading_batch_dim = (
        isinstance(first_batch.target_obs, torch.Tensor) and first_batch.target_obs.dim() >= 4
    )

    result = []
    for f in AICMECompartmentsDataBatch._fields:
        items = [getattr(b, f) for b in batches]

        if f in {"study_name", "substance_name"}:
            merged = []
            for it in items:
                if isinstance(it, (list, tuple)):
                    merged.extend(map(str, it))
                elif isinstance(it, str):
                    merged.append(it)
                else:
                    raise TypeError(f"Unexpected type for {f}: {type(it)}")
            result.append(merged)
            continue

        if f in {"context_subject_name", "target_subject_name"}:
            merged_lls = []
            for it in items:
                if isinstance(it, (list, tuple)):
                    merged_lls.extend([list(inner) for inner in it])
                else:
                    raise TypeError(f"Unexpected type for {f}: {type(it)}")
            result.append(merged_lls)
            continue

        if torch.is_tensor(items[0]) and has_leading_batch_dim:
            result.append(torch.cat(items, dim=0))
            continue

        result.append(default_collate(items))

    return AICMECompartmentsDataBatch(*result)


def _collate_aicme_batches(batch_list):
    """
    Handles:
      - [B] of AICMECompartmentsDataBatch → returns one collated batch
      - [B][P] of AICMECompartmentsDataBatch → returns list of P collated batches
    """
    if not batch_list:
        return batch_list

    first = batch_list[0]

    # Case 1: flat list of AICME batches
    if hasattr(first, "_fields"):  # NamedTuple-like
        return _stack_one_perm(batch_list)

    # Case 2: nested [B][P]
    if isinstance(first, (list, tuple)) and hasattr(first[0], "_fields"):
        # transpose [B][P] -> [P][B]
        transposed = list(zip(*batch_list))
        return [_stack_one_perm(list(group)) for group in transposed]

    # If we reach here and elements are Tensors, do NOT recurse further.
    if torch.is_tensor(first):
        raise TypeError(
            "Got a list of tensors instead of AICMECompartmentsDataBatch. "
            "Check that your Dataset returns AICMECompartmentsDataBatch, not raw tensors."
        )

    raise TypeError(
        f"Unexpected element type in batch_list: {type(first)}. "
        "Expected AICMECompartmentsDataBatch or list thereof."
    )


def split_individuals_tensor_batch(
    full_tensor_a: torch.Tensor,
    full_tensor_b: torch.Tensor,
    full_tensor_c: Optional[torch.Tensor],
    n_of_target_individuals: int,
    seed: Optional[int] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    num_individuals = full_tensor_a.shape[0]
    if seed is not None:
        random.seed(seed)

    if n_of_target_individuals == 0:
        return full_tensor_a, full_tensor_b, full_tensor_c, None, None, None

    all_indices = list(range(num_individuals))
    target_indices = random.sample(all_indices, n_of_target_individuals)
    context_indices = [i for i in all_indices if i not in target_indices]

    context_a = full_tensor_a[context_indices]
    context_b = full_tensor_b[context_indices]
    context_c = full_tensor_c[context_indices] if full_tensor_c is not None else None

    target_a = full_tensor_a[target_indices]
    target_b = full_tensor_b[target_indices]
    target_c = full_tensor_c[target_indices] if full_tensor_c is not None else None

    return context_a, context_b, context_c, target_a, target_b, target_c


def list_of_databath_to_device(
    batch_list: List[AICMECompartmentsDataBatch],
    device: torch.device | str,
) -> List[AICMECompartmentsDataBatch]:
    """Move a list of batches to ``device``.

    Parameters
    ----------
    batch_list:
        List of :class:`AICMECompartmentsDataBatch` objects.
    device:
        Target device.
    """
    return [b.to_device(device) for b in batch_list]


def build_reconstruction_db(
    db: AICMECompartmentsDataBatch,
) -> AICMECompartmentsDataBatch:
    """
    Reconstruct the target trajectories by concatenating observed and remainder
    segments, then right-padding so that the target has the same time dimension
    as the context. The context is left untouched.

    Returns a new AICMECompartmentsDataBatch.
    """
    B, Ic, Tc, _ = db.context_obs.shape  # context shape is the reference
    _, It, _, _ = db.target_obs.shape

    # reference length for padding (use context time dim)
    T_max = Tc

    # allocate new target tensors
    Xt_full = torch.zeros(B, It, T_max, 1, dtype=db.target_obs.dtype, device=db.target_obs.device)
    Tt_full = torch.zeros(
        B, It, T_max, 1, dtype=db.target_obs_time.dtype, device=db.target_obs_time.device
    )
    Mt_full = torch.zeros(B, It, T_max, dtype=torch.bool, device=db.target_obs_mask.device)

    # fill reconstructed target
    for b in range(B):
        for i in range(It):
            o_len = int(db.target_obs_mask[b, i].sum().item())
            r_len = int(db.target_rem_sim_mask[b, i].sum().item())
            total = o_len + r_len
            if total == 0:
                continue
            Xt_full[b, i, :o_len] = db.target_obs[b, i, :o_len]
            Xt_full[b, i, o_len:total] = db.target_rem_sim[b, i, :r_len]
            Tt_full[b, i, :o_len] = db.target_obs_time[b, i, :o_len]
            Tt_full[b, i, o_len:total] = db.target_rem_sim_time[b, i, :r_len]
            Mt_full[b, i, :total] = True

    # replace only the target fields
    return db._replace(
        target_obs=Xt_full,
        target_obs_time=Tt_full,
        target_obs_mask=Mt_full,
    )


class AICMECompartmentsDataset(Dataset):
    """Dataset generating synthetic PK batches for AICME models.

    Target observation strategies should already divide past and future
    observations (``split_past_future=True``).
    """

    def __init__(
        self,
        model_config: FlowPKExperimentConfig,
        ctx_fn,
        tgt_fn,
        number_of_process=1000,
        *,
        store_in_tempfile: bool = False,
        keep_tempfile: bool = False,
        recreate_tempfile: bool = False,
        tempfile_path: str | None = None,
        show_progress: bool = True,
        split: str = "",
        use_shared_target_dosing: bool = False,
        shared_target_n_targets: int = 100,
    ):
        self.mix_data_config = model_config.mix_data
        self.meta_study_config = model_config.meta_study
        self.meta_dosing_config = model_config.dosing
        self.number_of_process = number_of_process
        # ``n_of_permutations`` specifies how many shuffled versions of the
        # context/target split are generated for a single simulation.
        # ``n_of_databatches`` is a deprecated alias kept for backward
        # compatibility and mirrors ``n_of_permutations``.
        self.n_of_permutations = model_config.mix_data.n_of_permutations
        self.n_of_databatches = self.n_of_permutations  # deprecated alias
        self.n_of_target_individuals = int(model_config.mix_data.n_of_target_individuals)
        if self.n_of_target_individuals < 0:
            raise ValueError("n_of_target_individuals must be >= 0")

        # `num_individuals_range` controls context individuals only.
        self.min_context_individuals = int(self.meta_study_config.num_individuals_range[0])
        self.max_context_individuals = int(self.meta_study_config.num_individuals_range[-1])
        if self.min_context_individuals < 0:
            raise ValueError("meta_study.num_individuals_range minimum must be >= 0")
        if self.max_context_individuals < self.min_context_individuals:
            raise ValueError("meta_study.num_individuals_range must satisfy max >= min")

        # Fixed total capacity used by downstream consumers.
        self.max_individuals = self.max_context_individuals + self.n_of_target_individuals

        self.context_fn = ctx_fn
        self.target_fn = tgt_fn
        self.store_in_tempfile = store_in_tempfile
        self.keep_tempfile = keep_tempfile
        self.recreate_tempfile = recreate_tempfile
        self.show_progress = True
        self._tmpfile_path: List[str] | None = None
        self._loaded_data = None
        self.run_id = getattr(model_config, "run_index", 0)
        self.model_name = model_config.name_str

        if self.store_in_tempfile:
            self._prepare_tempfile_data(tempfile_path=tempfile_path, split=split)

        self.use_shared_target_dosing = use_shared_target_dosing
        self.shared_target_n_targets = shared_target_n_targets

    def __del__(self):
        if (
            self.store_in_tempfile
            and not self.keep_tempfile
            and self._tmpfile_path
            and os.path.exists(self._tmpfile_path)
        ):
            os.remove(self._tmpfile_path)

    def __len__(self):
        return self.number_of_process  # Arbitrary large number to simulate infinite data

    def _prepare_tempfile_data(self, *, tempfile_path: str | None, split: str) -> None:
        """Handle creation and (re)generation of the temporary data file."""
        if tempfile_path is None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
            self._tmpfile_path = tmp.name
            tmp.close()
        else:
            # Allow both Tuple paths from YAML and plain strings
            if isinstance(tempfile_path, (tuple, list)):
                base_path = os.path.join(data_dir, *tempfile_path)
            else:
                base_path = tempfile_path

            dirname = os.path.dirname(base_path)
            basename = os.path.basename(base_path)

            suffix = f"_{self.model_name}_{split}"
            if self.run_id is not None:
                suffix += f"_run{self.run_id}"
            new_basename = basename + suffix + ".tr"

            self._tmpfile_path = os.path.join(dirname, new_basename)

        if self.recreate_tempfile or not os.path.exists(self._tmpfile_path):
            print("RECREATING DATASET!")
            iterator = range(self.number_of_process)
            if self.show_progress:
                from tqdm.auto import tqdm

                iterator = tqdm(iterator, desc="Generating AICME data")
            data = [self._generate_item(i) for i in iterator]
            torch.save(data, self._tmpfile_path)

    def split_simulations(
        self, full_simulation, full_simulation_times
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        list[int],
        list[int],
    ]:
        """
        From the full simulation, randomly select `n_of_target_individuals` as targets and keep the rest as context.
        If `n_of_target_individuals == 0`, returns None for the target fields.
        """
        n_of_target_individuals = self.n_of_target_individuals
        num_individuals = full_simulation.shape[0]

        if n_of_target_individuals == 0:
            context_simulation = full_simulation
            context_simulation_times = full_simulation_times
            return (
                context_simulation,
                context_simulation_times,
                None,
                None,
                list(range(num_individuals)),
                [],
            )

        if num_individuals < n_of_target_individuals:
            raise ValueError(
                "Simulation contains fewer individuals than requested targets: "
                f"num_individuals={num_individuals}, "
                f"n_of_target_individuals={n_of_target_individuals}."
            )

        # Randomly select indices for target individuals
        target_indices = random.sample(range(num_individuals), n_of_target_individuals)
        context_indices = [i for i in range(num_individuals) if i not in target_indices]

        # Split the simulations, times, and masks
        target_simulation = full_simulation[target_indices]
        target_simulation_times = full_simulation_times[target_indices]
        context_simulation = full_simulation[context_indices]
        context_simulation_times = full_simulation_times[context_indices]

        return (
            context_simulation,
            context_simulation_times,
            target_simulation,
            target_simulation_times,
            context_indices,
            target_indices,
        )

    def _build_generation_meta_study_config(self):
        """Return a meta-study config where totals include fixed target individuals.

        The user-facing ``meta_study.num_individuals_range`` represents context
        individuals only. For raw simulation generation, we therefore sample
        ``context + n_of_target_individuals`` total individuals.
        """
        total_min = self.min_context_individuals + self.n_of_target_individuals
        total_max = self.max_context_individuals + self.n_of_target_individuals

        if getattr(self.meta_study_config, "simple_mode", False):
            total_individuals = random.randint(total_min, total_max)
            return replace(
                self.meta_study_config,
                num_individuals=total_individuals,
                num_individuals_range=(total_individuals, total_individuals),
            )

        return replace(
            self.meta_study_config,
            num_individuals_range=(total_min, total_max),
        )

    def __getitem__(self, idx):
        if self.store_in_tempfile:
            if self._loaded_data is None:
                self._loaded_data = torch.load(self._tmpfile_path, weights_only=False)
            # If in distributed mode, adjust the index based on process rank/world size
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                total_len = len(self._loaded_data)
                # Compute adjusted indices for this rank
                adjusted_idx = idx * world_size + rank
                if adjusted_idx >= total_len:
                    # If we would go out of bounds, wrap around to get a valid index
                    adjusted_idx = adjusted_idx % total_len
                return self._loaded_data[adjusted_idx]
            return self._loaded_data[idx]

        if self.use_shared_target_dosing:
            return self._generate_item_sample_target_dosing(
                idx, n_targets=self.shared_target_n_targets
            )
        return self._generate_item(idx)

    def _generate_item(self, idx) -> List[AICMECompartmentsDataBatch]:
        """Generate a list of ``AICMECompartmentsDataBatch`` objects.

        Each element corresponds to one permutation of the context/target split.
        Target observations are generated using ``target_fn``, which is expected
        to divide past and future observations.
        """
        (
            full_simulation,
            full_simulation_times,
            dosing_amounts,
            dosing_routes,
            time_points,
            time_scales,
        ) = prepare_full_simulation(
            self._build_generation_meta_study_config(),
            self.meta_dosing_config,
        )

        list_of_databatches: List[AICMECompartmentsDataBatch] = []
        for _ in range(self.n_of_permutations):
            # Split into context and target
            (
                context_simulation,
                context_simulation_times,
                target_simulation,
                target_simulation_times,
                context_indices,
                target_indices,
            ) = self.split_simulations(full_simulation, full_simulation_times)

            context_observations = self._safe_generate(
                self.context_fn,
                context_simulation,
                context_simulation_times,
                time_scales=time_scales,
            )

            target_observations = self._safe_generate(
                self.target_fn,
                target_simulation,
                target_simulation_times,
                time_scales=time_scales,
            )

            (
                context_obs,  # [c_ind, num_obs_c, 1]
                context_obs_time,  # [c_ind, num_obs_c, 1]
                context_obs_mask,  # [c_ind, num_obs_c]
                context_rem_sim,  # [c_ind, rem_obs_c, 1]
                context_rem_sim_time,  # [c_ind, rem_obs_c, 1]
                context_rem_sim_mask,  # [c_ind, rem_obs_c]
                context_time_scales,
            ) = context_observations

            (
                target_obs,  # [t_ind, num_obs_t, 1]
                target_obs_time,  # [t_ind, num_obs_t, 1]
                target_obs_mask,  # [t_ind, num_obs_t]
                target_rem_sim,  # [t_ind, rem_obs_t, 1]
                target_rem_sim_time,  # [t_ind, rem_obs_t, 1]
                target_rem_sim_mask,  # [t_ind, rem_obs_t]
                target_time_scales,
            ) = target_observations

            # Use provided time scales or fall back to simulation defaults
            ts = (
                context_time_scales
                if context_time_scales is not None
                else target_time_scales
                if target_time_scales is not None
                else time_scales
            )

            batch = self._build_padded_batch(
                context_obs,
                context_obs_time,
                context_obs_mask,
                context_rem_sim,
                context_rem_sim_time,
                context_rem_sim_mask,
                dosing_amounts[context_indices],
                dosing_routes[context_indices],
                target_obs,
                target_obs_time,
                target_obs_mask,
                target_rem_sim,
                target_rem_sim_time,
                target_rem_sim_mask,
                dosing_amounts[target_indices] if len(target_indices) > 0 else None,
                dosing_routes[target_indices] if len(target_indices) > 0 else None,
                ts,
            )
            list_of_databatches.append(batch)

        return list_of_databatches

    def _generate_item_sample_target_dosing(
        self,
        idx: int,
        n_targets: int = 100,
        different_dosing: bool = False,
    ):
        (
            context_sim,
            context_times,
            target_sim,
            target_times,
            dosing_amounts_ctx,
            dosing_routes_ctx,
            dosing_amounts_tgt,
            dosing_routes_tgt,
            time_points,
            time_scales,
        ) = prepare_full_simulation_with_repeated_targets(
            self.meta_study_config,
            self.meta_dosing_config,
            n_targets,
            different_dosing=different_dosing,
            idx=idx,
        )

        # Observations
        context_obs_pack = self._safe_generate(
            self.context_fn, context_sim, context_times, time_scales=time_scales
        )
        target_obs_pack = self._safe_generate(
            self.target_fn, target_sim, target_times, time_scales=time_scales
        )

        (
            context_obs,
            context_obs_time,
            context_obs_mask,
            context_rem_sim,
            context_rem_sim_time,
            context_rem_sim_mask,
            context_time_scales,
        ) = context_obs_pack

        (
            target_obs,
            target_obs_time,
            target_obs_mask,
            target_rem_sim,
            target_rem_sim_time,
            target_rem_sim_mask,
            target_time_scales,
        ) = target_obs_pack

        ts = (
            context_time_scales
            if context_time_scales is not None
            else (target_time_scales or time_scales)
        )

        # Build batch
        batch = self._build_padded_batch(
            # context
            context_obs,
            context_obs_time,
            context_obs_mask,
            context_rem_sim,
            context_rem_sim_time,
            context_rem_sim_mask,
            dosing_amounts_ctx,
            dosing_routes_ctx,
            # target
            target_obs,
            target_obs_time,
            target_obs_mask,
            target_rem_sim,
            target_rem_sim_time,
            target_rem_sim_mask,
            dosing_amounts_tgt,
            dosing_routes_tgt,
            # time scales
            ts=ts,
            target_capacity=n_targets,
        )

        return [batch]

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #

    def _build_padded_batch(
        self,
        ctx_obs: Tensor,  # [c_ind, num_obs_c]
        ctx_time: Tensor,  # [c_ind, num_obs_c]
        ctx_mask: Tensor,  # [c_ind, num_obs_c]
        ctx_rem: Optional[Tensor],  # [c_ind, rem_obs_c] | None
        ctx_rem_time: Optional[Tensor],  # [c_ind, rem_obs_c] | None
        ctx_rem_mask: Optional[Tensor],  # [c_ind, rem_obs_c] | None
        ctx_dose: Tensor,  # [c_ind]
        ctx_route: Tensor,  # [c_ind]
        tgt_obs: Optional[Tensor],  # [t_ind, num_obs_t] | None
        tgt_time: Optional[Tensor],  # [t_ind, num_obs_t] | None
        tgt_mask: Optional[Tensor],  # [t_ind, num_obs_t] | None
        tgt_rem: Optional[Tensor],  # [t_ind, rem_obs_t] | None
        tgt_rem_time: Optional[Tensor],  # [t_ind, rem_obs_t] | None
        tgt_rem_mask: Optional[Tensor],  # [t_ind, rem_obs_t] | None
        tgt_dose: Optional[Tensor],  # [t_ind] | None
        tgt_route: Optional[Tensor],  # [t_ind] | None
        ts: Tensor,  # [B(=1), 2]
        *,
        target_capacity: Optional[
            int
        ] = None,  # ← NEW (optional). If None, use self.n_of_target_individuals
    ) -> AICMECompartmentsDataBatch:
        """Pad context and target tensors then pack them into a batch."""

        max_c = self.max_context_individuals  # (unchanged)
        max_t = (
            target_capacity if target_capacity is not None else self.n_of_target_individuals
        )  # ← ONLY CHANGE

        # ── target padding (unchanged) ─────────────────────────────────────────
        t_obs_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                tgt_obs.unsqueeze(-1) if tgt_obs is not None else None, (1, 1, 1)
            ),  # to [t_ind, Tt, 1]
            max_t,
        )
        t_time_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                tgt_time.unsqueeze(-1) if tgt_time is not None else None, (1, 1, 1)
            ),  # to [t_ind, Tt, 1]
            max_t,
        )
        t_mask_p = self._pad_first_dim(
            ensure_mask_or_empty(
                tgt_mask if tgt_mask is not None else None, (1, 1)
            ),  # to [t_ind, Tt]
            max_t,
        )
        t_rem_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                tgt_rem.unsqueeze(-1) if tgt_rem is not None else None, (t_obs_p.size(0), 1, 1)
            ),  # [t_ind, Rt,1]
            max_t,
        )
        t_rem_time_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                tgt_rem_time.unsqueeze(-1) if tgt_rem_time is not None else None,
                (t_obs_p.size(0), 1, 1),
            ),
            max_t,
        )
        t_rem_mask_p = self._pad_first_dim(
            ensure_mask_or_empty(
                tgt_rem_mask if tgt_rem_mask is not None else None, (t_obs_p.size(0), 1)
            ),
            max_t,
        )
        t_dose_p = self._pad_first_dim(
            ensure_tensor_or_empty(tgt_dose if tgt_dose is not None else None, (1,)),  # [t_ind]
            max_t,
        )
        t_route_p = self._pad_first_dim(
            ensure_tensor_or_empty(tgt_route if tgt_route is not None else None, (1,)),  # [t_ind]
            max_t,
        ).long()

        # ── context padding (unchanged) ────────────────────────────────────────
        c_obs_p = self._pad_first_dim(ctx_obs, max_c).unsqueeze(-1)  # [c_ind, Tc, 1]
        c_time_p = self._pad_first_dim(ctx_time, max_c).unsqueeze(-1)  # [c_ind, Tc, 1]
        c_mask_p = self._pad_first_dim(ctx_mask, max_c)  # [c_ind, Tc]
        c_rem_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                ctx_rem.unsqueeze(-1) if ctx_rem is not None else None, (ctx_obs.size(0), 1, 1)
            ),
            max_c,
        )
        c_rem_time_p = self._pad_first_dim(
            ensure_tensor_or_empty(
                ctx_rem_time.unsqueeze(-1) if ctx_rem_time is not None else None,
                (ctx_obs.size(0), 1, 1),
            ),
            max_c,
        )
        c_rem_mask_p = self._pad_first_dim(
            ensure_mask_or_empty(
                ctx_rem_mask if ctx_rem_mask is not None else None, (ctx_obs.size(0), 1)
            ),
            max_c,
        )
        c_dose_p = self._pad_first_dim(ctx_dose, max_c)  # [c_ind]
        c_route_p = self._pad_first_dim(ctx_route, max_c).long()  # [c_ind]

        total_c = ctx_obs.size(0)
        mask_c_inds = torch.zeros(self.max_context_individuals, dtype=torch.bool)
        mask_c_inds[:total_c] = True

        total_t = tgt_obs.size(0) if tgt_obs is not None else 0
        mask_t_inds = torch.zeros(
            max_t, dtype=torch.bool
        )  # ← use max_t here (unchanged logic, just variable)
        mask_t_inds[:total_t] = True

        return AICMECompartmentsDataBatch(
            target_obs=t_obs_p,
            target_obs_time=t_time_p,
            target_obs_mask=t_mask_p,
            target_rem_sim=t_rem_p,
            target_rem_sim_time=t_rem_time_p,
            target_rem_sim_mask=t_rem_mask_p,
            target_dosing_amounts=t_dose_p,
            target_dosing_route_types=t_route_p,
            context_obs=c_obs_p,
            context_obs_time=c_time_p,
            context_obs_mask=c_mask_p,
            context_rem_sim=c_rem_p,
            context_rem_sim_time=c_rem_time_p,
            context_rem_sim_mask=c_rem_mask_p,
            context_dosing_amounts=c_dose_p,
            context_dosing_route_types=c_route_p,
            mask_context_individuals=mask_c_inds,
            mask_target_individuals=mask_t_inds,
            study_name=[""],
            context_subject_name=[[""] * max_c],
            target_subject_name=[[""] * max_t],  # ← still uses max_t
            substance_name=[""],
            time_scales=ts,
            is_empirical=False,
        )

    @staticmethod
    def _safe_generate(strategy, sim, times, **kw):
        """
        Call ObservationStrategy.generate() only when `sim` is not None.
        Returns a 7-tuple of Nones otherwise.
        """
        if sim is None:
            return (None, None, None, None, None, None, None)

        for _ in range(10):  # retries, like old manager
            out = strategy.generate(sim, times, **kw)
            if out[0] is not None:  # got a non-empty slice
                return out
        raise RuntimeError(
            "Unable to generate non-empty observations "
            "after 10 attempts – check strategy parameters."
        )

    @staticmethod
    def _pad_first_dim(t: torch.Tensor, size: int) -> torch.Tensor:
        """Pad tensor along the first dimension up to ``size``.

        Parameters
        ----------
        t : TensorType["I", *Ts]
            Input tensor where ``I`` may be smaller than ``size``.
        size : int
            Desired first-dimension size after padding.

        Returns
        -------
        TensorType["size", *Ts]
            Tensor padded with zeros (or ``False`` for bool tensors) so that the
            first dimension equals ``size``. If ``t`` already has ``size`` or
            more elements along the first dimension, it is truncated.
        """

        current = t.size(0)
        if current >= size:
            return t[:size]

        pad_shape = (size - current, *t.shape[1:])
        pad_value = False if t.dtype == torch.bool else 0.0
        padding = torch.full(pad_shape, pad_value, dtype=t.dtype, device=t.device)
        return torch.cat([t, padding], dim=0)


class AICMESyntheticExperimentDataset(Dataset):
    """Dataset generating sample-experiment batches from synthetic ``StudyJSON`` records.

    Each dataset item is one independent synthetic sample experiment. The item
    structure mirrors the AICME synthetic dataset convention and returns a
    list of ``P=n_dosings`` :class:`AICMECompartmentsDataBatch` objects.
    """

    def __init__(
        self,
        datamodule: "AICMECompartmentsDataModule",
        *,
        n_targets: int,
        n_dosings: int,
        dosing_mode: str,
        dataset_size: int,
        dosing_list_generation: str = "dosing_from_samples",
        logdose_range: Optional[Tuple[float, float]] = None,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")

        self.datamodule = datamodule
        self.n_targets = int(n_targets)
        self.n_dosings = int(n_dosings)
        self.dosing_mode = str(dosing_mode)
        self.dataset_size = int(dataset_size)
        self.dosing_list_generation = str(dosing_list_generation)
        self.logdose_range = logdose_range
        self.synthetic_target_observation_config = synthetic_target_observation_config

    def __len__(self) -> int:
        """Return how many independent synthetic experiments can be drawn."""

        return self.dataset_size

    def __getitem__(self, idx: int) -> List[AICMECompartmentsDataBatch]:
        """Generate one fresh synthetic experiment as a list of databatches."""

        studies = self.datamodule.generate_synthetic_study_experiment_list(
            n_targets=self.n_targets,
            n_dosings=self.n_dosings,
            dosing_mode=self.dosing_mode,
            dosing_list_generation=self.dosing_list_generation,
            logdose_range=self.logdose_range,
            synthetic_target_observation_config=self.synthetic_target_observation_config,
        )
        return self.datamodule._build_synthetic_experiment_batch_list(
            studies,
            n_targets=self.n_targets,
        )


class SimpleSyntheticExperimentDataset(Dataset):
    """Dataset generating simple-mode synthetic experiment batches.

    Each dataset item returns a list of ``n_dosings`` independently sampled
    :class:`AICMECompartmentsDataBatch` objects. Within each list element,
    context and target tensors are produced from the same underlying simple
    simulation draw while target observations use the deterministic fixed-grid
    strategy required by synthetic experiment consumers.
    """

    def __init__(
        self,
        datamodule: "AICMECompartmentsDataModule",
        *,
        n_targets: int,
        n_dosings: int,
        dosing_mode: str,
        dataset_size: int,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
    ) -> None:
        if dataset_size < 0:
            raise ValueError("dataset_size must be non-negative")
        if not getattr(datamodule.meta_config, "simple_mode", False):
            raise ValueError(
                "SimpleSyntheticExperimentDataset requires meta_study.simple_mode=True"
            )
        if str(dosing_mode) != "diverse_dosing":
            raise ValueError(
                "Simple synthetic experiments currently support only dosing_mode='diverse_dosing'."
            )

        self.datamodule = datamodule
        self.n_targets = int(n_targets)
        self.n_dosings = int(n_dosings)
        self.dataset_size = int(dataset_size)
        self.synthetic_target_observation_config = synthetic_target_observation_config

        generator_model_config = replace(
            datamodule.model_config,
            mix_data=replace(
                datamodule.model_config.mix_data,
                n_of_target_individuals=self.n_targets,
                n_of_permutations=1,
            ),
        )
        self.generator_dataset = AICMECompartmentsDataset(
            generator_model_config,
            ctx_fn=datamodule.context_strategy,
            tgt_fn=datamodule._build_synthetic_experiment_target_strategy(
                synthetic_target_observation_config
            ),
            number_of_process=self.dataset_size,
        )

    def __len__(self) -> int:
        """Return how many simple synthetic experiment items can be drawn."""

        return self.dataset_size

    def __getitem__(self, idx: int) -> List[AICMECompartmentsDataBatch]:
        """Generate one simple synthetic experiment as ``n_dosings`` batches."""

        batch_list: List[AICMECompartmentsDataBatch] = []
        for _ in range(self.n_dosings):
            generated = self.generator_dataset._generate_item(idx)
            if len(generated) != 1:
                raise RuntimeError(
                    "SimpleSyntheticExperimentDataset expects exactly one permutation per draw."
                )
            batch_list.append(generated[0])
        return batch_list


class AICMECompartmentsDataModule(pl.LightningDataModule):
    """LightningDataModule for synthetic PK simulation data."""

    # Empirical target batches always use the legacy PK observation strategy
    # with a fixed capacity profile, independent from synthetic target config.
    _EMPIRICAL_TARGET_MAX_NUM_OBS = 15
    _EMPIRICAL_TARGET_MIN_PAST = 0
    _EMPIRICAL_TARGET_MAX_PAST = 5

    def __init__(
        self,
        model_config: FlowPKExperimentConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.context_config = model_config.context_observations
        self.target_config = model_config.target_observations
        self.meta_config = model_config.meta_study
        self.data_config = model_config.mix_data
        self.study_config = model_config.meta_study
        self.num_workers = model_config.train.num_workers
        self.persistent_workers = model_config.train.persistent_workers
        self.shuffle_val = getattr(model_config.train, "shuffle_val", True)
        self.train_size = self.data_config.train_size
        self.val_size = self.data_config.val_size
        self.test_size = self.data_config.test_size
        self.batch_size = model_config.train.batch_size
        self._prepared = False
        # Cached shape parameters for empirical batch builders
        self.max_individuals: int | None = None
        self.max_observations: int | None = None
        self.max_remaining: int | None = None
        self.empirical_target_config = None
        self.empirical_target_strategy = None
        self.empirical_test_batches: Dict[str, List["AICMECompartmentsDataBatch"]] = {}
        self.empirical_test_batches_no_heldout: Dict[str, List["AICMECompartmentsDataBatch"]] = {}

    def prepare_data(self):
        # Use this method to download or prepare data if needed.
        # This is called only once and on a single GPU.
        # Here the Observation Manager Also Handles Empirical Data
        tempfile_path = getattr(self.data_config, "tempfile_path", None)
        if tempfile_path:
            temp_dir = Path(data_dir).joinpath(*tempfile_path)
        else:
            temp_dir = Path(data_dir) / "preprocessed"
        temp_dir.mkdir(parents=True, exist_ok=True)

        self.context_strategy = ObservationStrategyFactory.from_config(
            self.context_config,
            self.meta_config,
        )
        self.target_strategy = ObservationStrategyFactory.from_config(
            self.target_config,
            self.meta_config,
        )
        # Empirical target path: enforce legacy PK strategy and fixed capacities.
        # This is intentionally decoupled from synthetic target strategy settings.
        self.empirical_target_config = replace(
            self.target_config,
            type=None,
            split_past_future=True,
            max_num_obs=self._EMPIRICAL_TARGET_MAX_NUM_OBS,
            min_past=self._EMPIRICAL_TARGET_MIN_PAST,
            max_past=self._EMPIRICAL_TARGET_MAX_PAST,
        )
        self.empirical_target_strategy = ObservationStrategyFactory.from_config(
            self.empirical_target_config,
            self.meta_config,
        )
        self.train_dataset = AICMECompartmentsDataset(
            self.model_config,
            ctx_fn=self.context_strategy,
            tgt_fn=self.target_strategy,
            number_of_process=self.train_size,
            store_in_tempfile=self.data_config.store_in_tempfile,
            keep_tempfile=self.data_config.keep_tempfile,
            recreate_tempfile=self.data_config.recreate_tempfile,
            tempfile_path=self.data_config.tempfile_path,
            show_progress=self.data_config.tqdm_progress,
            split="train",
        )
        self.val_dataset = AICMECompartmentsDataset(
            self.model_config,
            ctx_fn=self.context_strategy,
            tgt_fn=self.target_strategy,
            number_of_process=self.val_size,
            store_in_tempfile=self.data_config.store_in_tempfile,
            keep_tempfile=self.data_config.keep_tempfile,
            recreate_tempfile=self.data_config.recreate_tempfile,
            tempfile_path=self.data_config.tempfile_path,
            show_progress=self.data_config.tqdm_progress,
            split="val",
        )
        self.test_dataset = AICMECompartmentsDataset(
            self.model_config,
            ctx_fn=self.context_strategy,
            tgt_fn=self.target_strategy,
            number_of_process=self.test_size,
            store_in_tempfile=self.data_config.store_in_tempfile,
            keep_tempfile=self.data_config.keep_tempfile,
            recreate_tempfile=self.data_config.recreate_tempfile,
            tempfile_path=self.data_config.tempfile_path,
            show_progress=self.data_config.tqdm_progress,
            split="test",
        )
        # Record shapes for empirical builders
        ctx_obs, ctx_rem = self.context_strategy.get_shapes()
        tgt_obs, tgt_rem = self.target_strategy.get_shapes()
        self.max_observations = max(ctx_obs, tgt_obs)
        self.max_remaining = max(ctx_rem, tgt_rem)
        self.max_individuals = max(
            self.train_dataset.max_context_individuals,
            self.train_dataset.n_of_target_individuals,
        )
        self._prepared = True
        self._empirical_loaded = False

        # Preload empirical datasets during prepare_data so they are available
        # before training callbacks query them.
        # In DDP, keep network/download activity on rank 0 only.
        if self._is_global_zero_process():
            self._load_empirical_test_batches()
            self._empirical_loaded = True

    def setup(self, stage=None):
        # Use this method to split data into train, validation, and test sets.
        # This is called on every GPU.
        if not self._prepared:
            self.prepare_data()

    def train_dataloader(self):
        # Returns the training dataloader.
        num_workers, persistent_workers = self._resolve_dataloader_workers()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=_collate_aicme_batches,
        )

    def val_dataloader(self):
        # Returns the validation dataloader.
        num_workers, persistent_workers = self._resolve_dataloader_workers()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=_collate_aicme_batches,
        )

    def test_dataloader(self):
        # Optional: Returns the test dataloader.
        # If you don't have a test set, you can omit this method.
        num_workers, persistent_workers = self._resolve_dataloader_workers()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=_collate_aicme_batches,
        )

    def obtain_shapes(self) -> Tuple[int, int, int]:
        """Expose dataset shape parameters for empirical batching.

        Returns
        -------
        Tuple[int, int, int]
            ``(max_individuals, max_observations, max_remaining)`` as used by
            :class:`AICMECompartmentsDataset`.
        """

        if not self._prepared:
            self.prepare_data()

        assert self.max_individuals is not None
        assert self.max_observations is not None
        assert self.max_remaining is not None
        return (
            self.max_individuals,
            self.max_observations,
            self.max_remaining,
        )

    def _resolve_dataloader_workers(self) -> Tuple[int, bool]:
        """Return DataLoader worker settings that are safe for single-process runs."""
        num_workers = max(0, int(self.num_workers))
        persistent_workers = self.persistent_workers and num_workers > 0
        return num_workers, persistent_workers

    def _build_synthetic_experiment_target_config(
        self,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
    ) -> ObservationsConfig:
        """Return the synthetic target observation config used only for sample experiments.

        When no explicit synthetic target configuration is provided, the
        synthetic path starts from the context observation settings but forces
        ``fixed_grid_start_index=3`` so the deterministic target grid begins
        near zero rather than exactly at ``t=0``.
        """

        base_config = (
            synthetic_target_observation_config
            if synthetic_target_observation_config is not None
            else replace(self.context_config, fixed_grid_start_index=3)
        )
        return replace(
            base_config,
            type="fixed_regular_grid",
            add_rem=False,
            split_past_future=False,
            min_past=None,
            max_past=None,
        )

    def _build_synthetic_experiment_target_strategy(
        self,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
    ):
        """Instantiate the synthetic target observation strategy on demand."""

        target_config = self._build_synthetic_experiment_target_config(
            synthetic_target_observation_config
        )
        return ObservationStrategyFactory.from_config(target_config, self.meta_config)

    @staticmethod
    def _infer_study_block_capacity(
        studies: Sequence[StudyJSON],
        block_name: str,
    ) -> Tuple[int, int, int]:
        """Infer ``(I, T_obs, T_rem)`` capacities from one StudyJSON block.

        The returned tuple corresponds to the maximum number of individuals,
        observed time points, and remainder time points across the supplied
        studies for ``block_name``.
        """

        max_individuals = 0
        max_observations = 0
        max_remaining = 0

        for study in studies:
            individuals = list(study.get(block_name, []))
            max_individuals = max(max_individuals, len(individuals))
            for individual in individuals:
                max_observations = max(
                    max_observations,
                    len(individual.get("observations", [])),
                )
                max_remaining = max(
                    max_remaining,
                    len(individual.get("remaining_times", [])),
                    len(individual.get("remaining", [])),
                )

        return max_individuals, max_observations, max_remaining

    def _build_synthetic_experiment_builder(
        self,
        studies: Sequence[StudyJSON],
        *,
        n_targets: int,
    ) -> JSON2AICMEBuilder:
        """Create a builder sized for the current synthetic experiment request.

        Synthetic sample experiments are already serialized with explicit
        ``context`` and ``target`` blocks. This helper therefore only derives
        padding capacities and does not apply any additional split logic.
        """

        if not self._prepared:
            self.prepare_data()

        max_inds, max_obs, max_rem = self.obtain_shapes()
        ctx_obs_cap, ctx_rem_cap = self.context_strategy.get_shapes()
        target_strategy = self.empirical_target_strategy or self.target_strategy
        tgt_obs_cap, tgt_rem_cap = target_strategy.get_shapes()

        ctx_inds_seen, ctx_obs_seen, ctx_rem_seen = self._infer_study_block_capacity(
            studies,
            "context",
        )
        tgt_inds_seen, tgt_obs_seen, tgt_rem_seen = self._infer_study_block_capacity(
            studies,
            "target",
        )

        train_dataset = getattr(self, "train_dataset", None)
        default_ctx_cap = getattr(train_dataset, "max_context_individuals", max_inds)
        default_tgt_cap = getattr(train_dataset, "n_of_target_individuals", max_inds)

        cfg = EmpiricalBatchConfig(
            max_databatch_size=1,
            max_individuals=max(max_inds, ctx_inds_seen, tgt_inds_seen, n_targets),
            max_observations=max(max_obs, ctx_obs_seen, tgt_obs_seen),
            max_remaining=max(max_rem, ctx_rem_seen, tgt_rem_seen),
            max_context_individuals=max(default_ctx_cap, ctx_inds_seen),
            max_target_individuals=max(default_tgt_cap, tgt_inds_seen, n_targets),
            max_context_observations=max(ctx_obs_cap, ctx_obs_seen),
            # Sample experiments use one fixed regular target grid; builder
            # padding must accommodate the serialized target capacity.
            max_target_observations=max(tgt_obs_cap, ctx_obs_cap, tgt_obs_seen),
            max_context_remaining=max(ctx_rem_cap, ctx_rem_seen),
            max_target_remaining=max(tgt_rem_cap, ctx_rem_cap, tgt_rem_seen),
        )
        return JSON2AICMEBuilder(cfg)

    def _build_synthetic_experiment_batch_list(
        self,
        studies: Sequence[StudyJSON],
        *,
        n_targets: int,
    ) -> List["AICMECompartmentsDataBatch"]:
        """Convert one synthetic sample experiment into a list of databatches.

        Each ``StudyJSON`` is converted independently so that every returned
        :class:`AICMECompartmentsDataBatch` keeps a leading batch dimension
        ``B=1``. The dataloader later collates these per-item lists into the
        standard nested ``List[AICMECompartmentsDataBatch]`` structure.
        """

        if not studies:
            return []

        builder = self._build_synthetic_experiment_builder(
            studies,
            n_targets=n_targets,
        )
        return [
            builder.build_one_aicmebatch([study], self.model_config.dosing) for study in studies
        ]

    def _build_synthetic_vpc_evaluation_batch(
        self,
        study: StudyJSON,
    ) -> "AICMECompartmentsDataBatch":
        """Convert one synthetic VPC study into a single evaluation databatch.

        The synthetic truth-vs-model VPC task operates on native ``StudyJSON``
        records returned by :meth:`generate_synthetic_vpc_data_list`, while the
        model-side VPC sampler consumes :class:`AICMECompartmentsDataBatch`.
        This helper keeps that conversion private and reuses the existing
        synthetic-study builder path so the scheduler task does not need to
        create a second public abstraction for one-off VPC evaluation batches.
        """

        builder = self._build_synthetic_experiment_builder(
            [study],
            n_targets=len(study.get("target", [])),
        )
        return builder.build_one_aicmebatch([study], self.model_config.dosing)

    def get_synthetic_experiment_dataloader(
        self,
        *,
        n_targets: int,
        n_dosings: int,
        dosing_mode: str,
        dataset_size: int,
        dosing_list_generation: str = "dosing_from_samples",
        logdose_range: Optional[Tuple[float, float]] = None,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Build a dataloader of independent synthetic sample experiments.

        Parameters
        ----------
        n_targets:
            Number of target individuals in each generated study.
        n_dosings:
            Number of dosing realizations generated per synthetic experiment.
            Each item therefore returns a list of length ``n_dosings``.
        dosing_mode:
            Sample-experiment dosing mode forwarded to
            :meth:`generate_synthetic_study_experiment_list`.
        dataset_size:
            Number of independent synthetic experiments in the dataset.
        dosing_list_generation:
            Dosing-list generation mode used only when
            ``dosing_mode="dosing_list"``.
        logdose_range:
            Optional log-dose range forwarded to the public sample-experiment
            generator.
        synthetic_target_observation_config:
            Optional synthetic-only target observation configuration. When
            omitted, the sample-experiment path defaults to a fixed regular
            grid with ``fixed_grid_start_index=3``.
        shuffle:
            Whether to shuffle experiment items before batching.

        Notes
        -----
        Invalid sampled PK simulations are handled through internal
        replacement resampling in the generation layer, so loader items keep
        returning complete sample experiments instead of leaking transient
        rejection-sampling failures to callers.

        When ``meta_study.simple_mode=True``, this method bypasses the
        ``StudyJSON`` sample-experiment generator and instead reuses the
        regular synthetic dataset generation flow. In that mode only
        ``dosing_mode='diverse_dosing'`` is supported, and target individuals
        are serialized with the deterministic fixed regular grid strategy.
        The synthetic VPC-specific ``vpc_context`` mode is not available in
        simple mode.
        """

        if n_dosings <= 0:
            raise ValueError("n_dosings must be positive when building a synthetic loader")

        if not self._prepared:
            self.prepare_data()

        if getattr(self.meta_config, "simple_mode", False):
            if dosing_mode != "diverse_dosing":
                raise ValueError(
                    "Simple synthetic experiment loaders support only dosing_mode='diverse_dosing'."
                )
            dataset = SimpleSyntheticExperimentDataset(
                self,
                n_targets=n_targets,
                n_dosings=n_dosings,
                dosing_mode=dosing_mode,
                dataset_size=dataset_size,
                synthetic_target_observation_config=synthetic_target_observation_config,
            )
        else:
            dataset = AICMESyntheticExperimentDataset(
                self,
                n_targets=n_targets,
                n_dosings=n_dosings,
                dosing_mode=dosing_mode,
                dataset_size=dataset_size,
                dosing_list_generation=dosing_list_generation,
                logdose_range=logdose_range,
                synthetic_target_observation_config=synthetic_target_observation_config,
            )
        num_workers, persistent_workers = self._resolve_dataloader_workers()
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            collate_fn=_collate_aicme_batches,
        )

    @staticmethod
    def _is_global_zero_process() -> bool:
        """Return True for rank 0 (or single-process execution)."""

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
        return True

    def _load_empirical_test_batches(self) -> None:
        """Download and cache empirical Hugging Face datasets for evaluation."""

        from pff.data.data_empirical import load_empirical_hf_batches_as_dm

        datasets = getattr(self.data_config, "test_empirical_datasets", [])
        self.empirical_test_batches = {}
        self.empirical_test_batches_no_heldout = {}
        if not datasets:
            return

        for repo_id in datasets:
            try:
                batches = load_empirical_hf_batches_as_dm(
                    repo_id,
                    meta_dosing=self.model_config.dosing,
                    datamodule=self,
                    held_out=True,
                )
            except Exception as exc:  # noqa: BLE001 - surface download issues
                warnings.warn(
                    f"Failed to load empirical dataset '{repo_id}': {exc}",
                    stacklevel=2,
                )
                continue

            if not batches:
                warnings.warn(
                    f"No empirical batches returned for dataset '{repo_id}'",
                    stacklevel=2,
                )
                continue

            self.empirical_test_batches[repo_id] = batches
            try:
                no_heldout_batches = load_empirical_hf_batches_as_dm(
                    repo_id,
                    meta_dosing=self.model_config.dosing,
                    datamodule=self,
                    held_out=False,
                )
            except Exception as exc:  # noqa: BLE001 - surface download issues
                warnings.warn(
                    f"Failed to load no-heldout empirical dataset '{repo_id}': {exc}",
                    stacklevel=2,
                )
                continue

            if not no_heldout_batches:
                warnings.warn(
                    f"No no-heldout empirical batches returned for dataset '{repo_id}'",
                    stacklevel=2,
                )
                continue

            self.empirical_test_batches_no_heldout[repo_id] = no_heldout_batches

    def get_empirical_test_batches(
        self,
        *,
        no_heldout: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> Dict[str, List["AICMECompartmentsDataBatch"]]:
        """Return cached empirical batches keyed by Hugging Face dataset id.

        Parameters
        ----------
        no_heldout:
            If ``True``, return batches where all empirical individuals remain
            in context (no held-out target). If ``False`` (default), return the
            leave-one-out batches.
        device:
            Optional device where returned batches should live. When provided,
            returned batches are moved to ``device`` without mutating the
            internal cache.
        """

        # Safety fallback for direct/manual datamodule usage.
        if not getattr(self, "_empirical_loaded", False):
            if not self._prepared:
                self.prepare_data()
            elif self._is_global_zero_process():
                self._load_empirical_test_batches()
            self._empirical_loaded = True

        batch_map = (
            self.empirical_test_batches_no_heldout if no_heldout else self.empirical_test_batches
        )

        if device is None:
            return batch_map

        return {
            repo_id: list_of_databath_to_device(batch_list, device)
            for repo_id, batch_list in batch_map.items()
        }

    def get_empirical_batches(
        self,
        *,
        split: str,
        empirical_name: Optional[str],
        device: Optional[torch.device | str] = None,
    ) -> List["AICMECompartmentsDataBatch"]:
        """Return one empirical batch list using scheduler-oriented split aliases.

        Supported split aliases:
        - ``empirical_heldout``: leave-one-out empirical targets
        - ``empirical_no_heldout``: all empirical individuals remain in context
        """

        normalized_split = str(split).strip().lower()
        if normalized_split == "empirical_heldout":
            batch_map = self.get_empirical_test_batches(no_heldout=False, device=device)
        elif normalized_split == "empirical_no_heldout":
            batch_map = self.get_empirical_test_batches(no_heldout=True, device=device)
        else:
            raise ValueError(
                f"Unsupported empirical split alias '{split}'. "
                "Expected 'empirical_heldout' or 'empirical_no_heldout'."
            )

        if empirical_name is None:
            raise ValueError("`empirical_name` must be provided for empirical scheduler tasks.")
        try:
            return batch_map[str(empirical_name)]
        except KeyError as exc:
            raise ValueError(
                f"No empirical batches found for split='{split}' and empirical_name='{empirical_name}'."
            ) from exc

    @staticmethod
    def _normalize_substance_name(name: object) -> str:
        """Normalize substance names for robust matching."""

        return "".join(ch.lower() for ch in str(name) if ch.isalnum())

    def select_empirical_batch_list(
        self,
        dataset_key: Optional[str] = None,
        *,
        no_heldout: bool = False,
    ) -> Tuple[Optional[str], List["AICMECompartmentsDataBatch"]]:
        """Select one empirical dataset batch list for plotting/evaluation.

        Parameters
        ----------
        dataset_key:
            Explicit dataset key to use. If missing or unknown, the first
            non-empty dataset in cache is selected.
        no_heldout:
            Whether to read from the no-heldout cache.

        Returns
        -------
        Tuple[Optional[str], List[AICMECompartmentsDataBatch]]
            Selected dataset key (or ``None`` if unavailable) and batch list.
        """

        empirical_batches = self.get_empirical_test_batches(no_heldout=no_heldout)
        if dataset_key is not None and dataset_key in empirical_batches:
            selected_key = dataset_key
            batch_list = empirical_batches[dataset_key]
        else:
            selected_key = None
            batch_list = None
            for repo_id, batches in empirical_batches.items():
                if batches:
                    selected_key = repo_id
                    batch_list = batches
                    break

        if not batch_list:
            label = "no-heldout" if no_heldout else "heldout"
            raise RuntimeError(f"No empirical {label} batches available for predictive plotting.")

        return selected_key, batch_list

    def describe_empirical_test_batches(
        self,
        empirical_batches: Optional[Dict[str, List["AICMECompartmentsDataBatch"]]] = None,
        *,
        no_heldout: bool = False,
        batch_index: int = 0,
        print_available: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Describe empirical test batches and return available studies/drugs.

        This helper is designed to be called after
        :meth:`get_empirical_test_batches` in notebook/script workflows.

        Parameters
        ----------
        empirical_batches:
            Optional pre-fetched empirical batches (typically from
            :meth:`get_empirical_test_batches`). If ``None``, batches are
            fetched internally.
        no_heldout:
            Whether to describe no-heldout batches.
        batch_index:
            Batch index to inspect within each dataset. Default is ``0``.
        print_available:
            If ``True``, print available datasets/studies/drugs.

        Returns
        -------
        Tuple[List[str], List[str]]
            Unique available study names and drug names from the selected
            ``batch_index`` across datasets.
        """

        batch_map = empirical_batches
        if batch_map is None:
            batch_map = self.get_empirical_test_batches(no_heldout=no_heldout)

        if batch_index < 0:
            raise ValueError("batch_index must be non-negative")

        available_studies: List[str] = []
        available_drugs: List[str] = []
        seen_studies: set[str] = set()
        seen_drugs: set[str] = set()

        if print_available:
            label = "no_heldout=True" if no_heldout else "heldout"
            print(f"Available empirical datasets ({label}):", list(batch_map.keys()))

        for repo_id, batch_list in batch_map.items():
            if print_available:
                print(f"Dataset '{repo_id}' contains {len(batch_list)} empirical batch(es).")
            if batch_index >= len(batch_list):
                if print_available:
                    print(
                        f"  Skipping dataset '{repo_id}': batch_index={batch_index} "
                        f"is out of range."
                    )
                continue

            batch = batch_list[batch_index]
            studies, drugs = self.describe_empirical_batch(batch, print_available=False)
            for study in studies:
                if study not in seen_studies:
                    seen_studies.add(study)
                    available_studies.append(study)
            for drug in drugs:
                if drug not in seen_drugs:
                    seen_drugs.add(drug)
                    available_drugs.append(drug)

            if print_available:
                print(f"  Batch {batch_index} studies:", studies)
                print(f"  Batch {batch_index} drugs:", drugs)

        if print_available:
            print("Available studies:", available_studies)
            print("Available drugs:", available_drugs)

        return available_studies, available_drugs

    @staticmethod
    def describe_empirical_batch(
        batch: "AICMECompartmentsDataBatch",
        *,
        print_available: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Return display-ready study and substance names for a batch.

        Parameters
        ----------
        batch:
            Empirical batch to inspect.
        print_available:
            If ``True``, print available studies and drugs to stdout.
        """

        studies = [str(name) if name else f"study_{i}" for i, name in enumerate(batch.study_name)]
        drugs = [
            str(name) if name else f"substance_{i}" for i, name in enumerate(batch.substance_name)
        ]

        if print_available:
            print("Available studies in selected batch:", studies)
            print("Available drugs in selected batch:", drugs)

        return studies, drugs

    @staticmethod
    def slice_single_substance_batch(
        batch: "AICMECompartmentsDataBatch",
        b_idx: int,
    ) -> "AICMECompartmentsDataBatch":
        """Extract one substance entry from a multi-substance batch.

        Parameters
        ----------
        batch:
            Batch with leading batch dimension ``B``.
        b_idx:
            Substance index along ``B``.

        Returns
        -------
        AICMECompartmentsDataBatch
            Single-substance batch with tensors sliced to ``B=1``.
        """

        if b_idx < 0 or b_idx >= len(batch.substance_name):
            raise IndexError(
                f"Substance index {b_idx} is out of range for batch size "
                f"{len(batch.substance_name)}."
            )

        values = []
        for field_name in batch._fields:
            value = getattr(batch, field_name)
            if isinstance(value, torch.Tensor):
                # Keep tensor rank stable by preserving a singleton leading B axis.
                values.append(value[b_idx : b_idx + 1])
            elif field_name in {
                "study_name",
                "substance_name",
                "context_subject_name",
                "target_subject_name",
            }:
                values.append([value[b_idx]])
            else:
                values.append(value)
        return batch.__class__(*values)

    @classmethod
    def slice_single_substance_batch_by_name(
        cls,
        batch: "AICMECompartmentsDataBatch",
        substance_name: str,
    ) -> "AICMECompartmentsDataBatch":
        """Extract one substance entry by matching drug name."""

        _, available_drugs = cls.describe_empirical_batch(batch, print_available=False)
        norm_target = cls._normalize_substance_name(substance_name)
        matches = [
            i
            for i, name in enumerate(available_drugs)
            if cls._normalize_substance_name(name) == norm_target
        ]
        if not matches:
            raise ValueError(
                f"Selected drug '{substance_name}' not found in heldout batch. "
                f"Choose from: {available_drugs}"
            )
        return cls.slice_single_substance_batch(batch, matches[0])

    def select_empirical_drug_batch(
        self,
        empirical_batches: Dict[str, List["AICMECompartmentsDataBatch"]],
        selected_drug: str,
        *,
        permutation_indexes: Optional[int | Sequence[int]] = None,
        print_selection: bool = True,
    ) -> Tuple[
        "AICMECompartmentsDataBatch | List[AICMECompartmentsDataBatch]",
        str,
        str,
    ]:
        """Select one drug from empirical batches, optionally across permutations.

        Parameters
        ----------
        empirical_batches:
            Mapping returned by :meth:`get_empirical_test_batches`.
        selected_drug:
            Drug name to match across all empirical batches.
        permutation_indexes:
            Optional permutation index or list of permutation indices within the
            selected empirical dataset's batch list. When ``None`` (default),
            the method preserves legacy behaviour and returns the first matching
            single-substance batch. When a list/tuple is provided, returns a
            list of single-substance batches in the requested permutation order.
        print_selection:
            If ``True``, print where the match was found.
        """

        norm_target = self._normalize_substance_name(selected_drug)
        requested_permutations: Optional[List[int]]
        return_many = isinstance(permutation_indexes, (list, tuple))
        if permutation_indexes is None:
            requested_permutations = None
        elif return_many:
            if len(permutation_indexes) == 0:
                raise ValueError("'permutation_indexes' must not be empty.")
            requested_permutations = [int(idx) for idx in permutation_indexes]
            if len(set(requested_permutations)) != len(requested_permutations):
                raise ValueError("'permutation_indexes' must contain unique indices.")
        else:
            requested_permutations = [int(permutation_indexes)]

        all_available_drugs: List[str] = []
        seen_drugs: set[str] = set()

        for repo_id, batch_list in empirical_batches.items():
            for batch_index, batch in enumerate(batch_list):
                _, available_drugs = self.describe_empirical_batch(batch, print_available=False)
                for drug in available_drugs:
                    if drug not in seen_drugs:
                        seen_drugs.add(drug)
                        all_available_drugs.append(drug)

                matches = [
                    i
                    for i, name in enumerate(available_drugs)
                    if self._normalize_substance_name(name) == norm_target
                ]
                if matches:
                    if requested_permutations is None:
                        selected_batches: List[AICMECompartmentsDataBatch] = [
                            self.slice_single_substance_batch(batch, matches[0])
                        ]
                        chosen_permutations = [batch_index]
                    else:
                        selected_batches = []
                        chosen_permutations = requested_permutations
                        for permutation_index in requested_permutations:
                            if permutation_index < 0 or permutation_index >= len(batch_list):
                                raise IndexError(
                                    f"Permutation index {permutation_index} is out of range for "
                                    f"dataset '{repo_id}' with {len(batch_list)} permutations."
                                )

                            perm_batch = batch_list[permutation_index]
                            _, perm_drugs = self.describe_empirical_batch(
                                perm_batch, print_available=False
                            )
                            perm_matches = [
                                i
                                for i, name in enumerate(perm_drugs)
                                if self._normalize_substance_name(name) == norm_target
                            ]
                            if not perm_matches:
                                raise ValueError(
                                    f"Selected drug '{selected_drug}' was not found in dataset "
                                    f"'{repo_id}' at permutation index {permutation_index}."
                                )
                            selected_batches.append(
                                self.slice_single_substance_batch(perm_batch, perm_matches[0])
                            )

                    studies, drugs = self.describe_empirical_batch(
                        selected_batches[0], print_available=False
                    )
                    selected_study = studies[0]
                    selected_name = drugs[0]
                    if print_selection:
                        print("Selected empirical dataset key:", repo_id)
                        if len(chosen_permutations) == 1:
                            print("Selected empirical batch index:", chosen_permutations[0])
                        else:
                            print("Selected empirical batch indexes:", chosen_permutations)
                        print("Selected study:", selected_study)
                        print("Selected drug:", selected_name)
                    if return_many:
                        return selected_batches, selected_study, selected_name
                    return selected_batches[0], selected_study, selected_name

        raise ValueError(
            f"Selected drug '{selected_drug}' was not found in empirical batches. "
            f"Choose from: {all_available_drugs}"
        )

    def _select_strategy(self, who: str):
        """Return the observation strategy requested via ``who``.

        Parameters
        ----------
        who:
            Either ``"target"`` or ``"context"``.

        Returns
        -------
        ObservationStrategy
            The strategy matching the requested role.
        """

        if who == "target":
            return self.target_strategy
        if who == "context":
            return self.context_strategy
        raise ValueError("'who' must be either 'target' or 'context'.")

    def _select_strategies(self, who: str) -> List[object]:
        """Return strategy list for the requested role.

        For ``who='target'`` this includes both synthetic and empirical target
        strategies so past-selection overrides remain consistent when empirical
        batches are generated from the datamodule.
        """

        if who == "context":
            return [self.context_strategy]
        if who == "target":
            strategies: List[object] = [self.target_strategy]
            empirical_target_strategy = getattr(self, "empirical_target_strategy", None)
            if empirical_target_strategy is not None:
                strategies.append(empirical_target_strategy)
            # Keep order stable while avoiding duplicate objects.
            deduped: List[object] = []
            seen_ids: set[int] = set()
            for strategy in strategies:
                strategy_id = id(strategy)
                if strategy_id in seen_ids:
                    continue
                seen_ids.add(strategy_id)
                deduped.append(strategy)
            return deduped
        raise ValueError("'who' must be either 'target' or 'context'.")

    def fix_past_selection(self, fix_past_value: int, *, who: str = "target") -> None:
        """Force a fixed number of past observations for the selected strategy.

        The override is only applied for strategies with ``split_past_future``
        enabled; for others the call is ignored.
        """

        if not self._prepared:
            self.prepare_data()

        for strategy in self._select_strategies(who):
            if hasattr(strategy, "fix_past_selection"):
                strategy.fix_past_selection(fix_past_value)
        # Reset lazy-load flag so empirical data is reloaded with new strategy settings
        self._empirical_loaded = False

    def release_past_selection(self, *, who: str = "target") -> None:
        """Restore the default past sampling behaviour for the given strategy."""

        if not self._prepared:
            self.prepare_data()

        for strategy in self._select_strategies(who):
            if hasattr(strategy, "release_past_selection"):
                strategy.release_past_selection()
        # Reset lazy-load flag so empirical data is reloaded with restored strategy settings
        self._empirical_loaded = False

    def set_shared_target_dosing(self, enable: bool = True, n_targets: int = 100) -> None:
        """Enable/disable shared target dosing across all datasets.

        Parameters
        ----------
        enable : bool
            Whether to enable shared-target dosing.
        n_targets : int
            Number of target individuals to sample when enabled.
        """
        self.use_shared_target_dosing = enable
        self.shared_target_n_targets = n_targets

        for ds in (
            getattr(self, "train_dataset", None),
            getattr(self, "val_dataset", None),
            getattr(self, "test_dataset", None),
        ):
            if ds is not None:
                ds.use_shared_target_dosing = enable
                ds.shared_target_n_targets = n_targets

    def unset_shared_target_dosing(self) -> None:
        """Disable shared target dosing and restore default behaviour."""
        self.set_shared_target_dosing(False)

    @staticmethod
    def _add_batch_dim_to_synthetic_batch(
        batch: AICMECompartmentsDataBatch,
    ) -> AICMECompartmentsDataBatch:
        """Add a leading ``B=1`` axis to tensor fields missing batch dimension."""

        values: list = []
        for name, value in zip(batch._fields, batch):
            if not isinstance(value, torch.Tensor):
                values.append(value)
                continue

            if name == "time_scales":
                # ``time_scales`` is often already [B, 2] while other fields are
                # emitted as [I, ...] by dataset-level generation.
                values.append(value.unsqueeze(0) if value.dim() == 1 else value)
                continue

            values.append(value.unsqueeze(0))

        return AICMECompartmentsDataBatch(*values)

    def generate_synthetic_study_experiment_list(
        self,
        n_targets: int,
        n_dosings: int,
        dosing_mode: str,
        dosing_list_generation: str = "dosing_from_samples",
        logdose_range: Optional[Tuple[float, float]] = None,
        synthetic_target_observation_config: Optional[ObservationsConfig] = None,
    ) -> list[StudyJSON]:
        """Generate synthetic sample-experiment studies as ``StudyJSON`` records.

        This API preserves ``context_observations`` for context serialization,
        while synthetic targets are serialized through a synthetic-only target
        observation configuration provided at call time. When no synthetic
        target config is supplied, a deterministic ``fixed_regular_grid`` with
        ``fixed_grid_start_index=3`` is used. ``n_dosings`` controls how many
        study JSONs are returned for every dosing mode. ``dosing_list_generation``
        and ``logdose_range`` are only used when ``dosing_mode="dosing_list"``.

        Supported dosing modes are:
        - ``repeated_dosing``: ``n_dosings`` studies where each study uses one
          repeated target dosing setup shared by all target individuals.
        - ``dosing_list``: ``n_dosings`` studies sharing context while target dosing changes.
          Use ``dosing_list_generation="dosing_from_samples"`` for repeated-target
          stochastic draws from one shared study-level dosing distribution,
          or ``dosing_list_generation="dosing_from_range"`` together with
          ``logdose_range=(min_logdose, max_logdose)`` for a deterministic log-dose grid.
        - ``diverse_dosing``: ``n_dosings`` studies where target individuals
          receive independently sampled dosing within each study.
        - ``vpc_context``: ``n_dosings`` context-only studies where ``n_targets``
          observed synthetic individuals are serialized under ``context`` and
          ``target`` is always empty.

        Invalid sampled context or target simulations are handled with
        internal replacement resampling, so this method either returns a
        complete list of ``n_dosings`` studies or raises only after exhausting
        the bounded retry budget in the generation layer.
        """

        if n_targets < 0:
            raise ValueError("n_targets must be non-negative")
        if n_dosings < 0:
            raise ValueError("n_dosings must be non-negative")
        if dosing_mode not in {
            "repeated_dosing",
            "dosing_list",
            "diverse_dosing",
            "vpc_context",
        }:
            raise ValueError(
                "dosing_mode must be one of {'repeated_dosing', 'dosing_list', "
                "'diverse_dosing', 'vpc_context'}"
            )
        if dosing_mode == "dosing_list" and dosing_list_generation not in {
            "dosing_from_samples",
            "dosing_from_range",
        }:
            raise ValueError(
                "dosing_list_generation must be one of {'dosing_from_samples', 'dosing_from_range'}"
            )

        if not self._prepared:
            self.prepare_data()

        target_observation_config = self._build_synthetic_experiment_target_config(
            synthetic_target_observation_config
        )

        return build_sample_experiment_studies(
            meta_study_config=self.meta_config,
            meta_dosing_config=self.model_config.dosing,
            context_observation_config=self.context_config,
            target_observation_config=target_observation_config,
            n_targets=n_targets,
            n_dosings=n_dosings,
            dosing_mode=dosing_mode,
            dosing_list_generation=dosing_list_generation,
            logdose_range=logdose_range,
        )

    def generate_synthetic_vpc_data_list(
        self,
        n_cases: int,
        n_observed_individuals: int,
        sample_size: int,
    ) -> list[tuple[StudyJSON, list[StudyJSON]]]:
        """Build native synthetic VPC cases as ``(observed_study, replicates)`` tuples.

        Each returned case samples one study-level state once, builds one
        context-only observed ``StudyJSON`` with ``target=[]``, then generates
        ``sample_size`` context-only replicate studies. The replicate studies
        preserve the observed study's realized dosing layout and explicit VPC
        schedule while resampling only the individual latent configurations.
        Observation sampling for these context-only studies uses the same
        context observation configuration as the datamodule's normal training
        data path.
        """

        if getattr(self.meta_config, "simple_mode", False):
            raise ValueError(
                "Synthetic VPC case generation is not supported when meta_study.simple_mode=True."
            )

        if not self._prepared:
            self.prepare_data()

        return _build_synthetic_vpc_data_list(
            meta_study_config=self.meta_config,
            meta_dosing_config=self.model_config.dosing,
            observation_config=self.context_config,
            n_cases=n_cases,
            n_observed_individuals=n_observed_individuals,
            sample_size=sample_size,
        )
