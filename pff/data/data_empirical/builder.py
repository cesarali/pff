from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from datasets import load_dataset
from torchtyping import TensorType as TT

from pff.config_classes.data_config import (
    MetaDosingConfig,
)
from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch

if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule

from .json_schema import IndividualJSON, StudyJSON, canonicalize_study
from .json_stats import EmpiricalJSONStats, compute_json_stats


@dataclass
class EmpiricalBatchConfig:
    """Configuration for empirical batch construction.

    Attributes
    ----------
    pad_value_time:
        Value used to pad time tensors.
    pad_value_obs:
        Value used to pad observation tensors.
    max_databatch_size:
        Maximum number of studies that can be stacked into a single batch.
    max_individuals:
        Maximum number of individuals per context or target block.
    max_observations:
        Maximum number of observation time points per individual.
    max_remaining:
        Maximum number of remaining time points per individual.
    max_context_individuals / max_target_individuals:
        Optional overrides specifying separate capacities for context and
        target individual counts.
    max_context_observations / max_target_observations:
        Optional overrides specifying per-block observation capacities.
    max_context_remaining / max_target_remaining:
        Optional overrides specifying per-block remaining simulation
        capacities.
    """

    pad_value_time: float = 0.0
    pad_value_obs: float = 0.0
    max_databatch_size: int = 8
    max_individuals: int = 1
    max_observations: int = 0
    max_remaining: int = 0
    max_context_individuals: Optional[int] = None
    max_target_individuals: Optional[int] = None
    max_context_observations: Optional[int] = None
    max_target_observations: Optional[int] = None
    max_context_remaining: Optional[int] = None
    max_target_remaining: Optional[int] = None


class JSON2AICMEBuilder:
    """Convert empirical study JSON to :class:`AICMECompartmentsDataBatch`.

    The builder pads context and target individuals to fixed sizes and
    assembles the :class:`AICMECompartmentsDataBatch` expected by the models.
    """

    def __init__(self, cfg: EmpiricalBatchConfig) -> None:
        self.cfg = cfg

    def _ctx_cap(self) -> int:
        return (
            self.cfg.max_context_individuals
            if self.cfg.max_context_individuals is not None
            else self.cfg.max_individuals
        )

    def _tgt_cap(self) -> int:
        return (
            self.cfg.max_target_individuals
            if self.cfg.max_target_individuals is not None
            else self.cfg.max_individuals
        )

    def _ctx_obs_cap(self) -> int:
        return (
            self.cfg.max_context_observations
            if self.cfg.max_context_observations is not None
            else self.cfg.max_observations
        )

    def _tgt_obs_cap(self) -> int:
        return (
            self.cfg.max_target_observations
            if self.cfg.max_target_observations is not None
            else self.cfg.max_observations
        )

    def _ctx_rem_cap(self) -> int:
        return (
            self.cfg.max_context_remaining
            if self.cfg.max_context_remaining is not None
            else self.cfg.max_remaining
        )

    def _tgt_rem_cap(self) -> int:
        return (
            self.cfg.max_target_remaining
            if self.cfg.max_target_remaining is not None
            else self.cfg.max_remaining
        )

    def _block_from_inds(
        self,
        inds: List[IndividualJSON],
        *,
        max_individuals: int,
        obs_cap: int,
        rem_cap: int,
    ) -> Dict[str, TT]:
        """Assemble tensors for a list of individuals.

        Padding is applied so that each block has the same number of
        individuals (``max_individuals``) and time steps
        (``max_observations``/``max_remaining``).
        """

        I_max = max(0, max_individuals)
        ET = max(0, obs_cap)
        R = max(0, rem_cap)

        obs_tensor = torch.full((I_max, ET), self.cfg.pad_value_obs)  # [I, ET]
        time_tensor = torch.full((I_max, ET), self.cfg.pad_value_time)  # [I, ET]
        mask_tensor = torch.zeros((I_max, ET), dtype=torch.bool)  # [I, ET]

        rem_tensor = (
            torch.full((I_max, R), self.cfg.pad_value_obs) if R else torch.zeros(I_max, 0)
        )  # [I, R]
        rem_time_tensor = (
            torch.full((I_max, R), self.cfg.pad_value_time) if R else torch.zeros(I_max, 0)
        )  # [I, R]
        rem_mask_tensor = (
            torch.zeros((I_max, R), dtype=torch.bool)
            if R
            else torch.zeros(I_max, 0, dtype=torch.bool)
        )  # [I, R]

        for i, ind in enumerate(inds[:I_max]):
            obs = torch.tensor(ind.get("observations", []), dtype=torch.float32)  # [ET?]
            time = torch.tensor(ind.get("observation_times", []), dtype=torch.float32)  # [ET?]
            L = min(obs.shape[0], ET)
            obs_tensor[i, :L] = obs[:L]
            time_tensor[i, :L] = time[:L]
            mask_tensor[i, :L] = True

            rem_t = torch.tensor(ind.get("remaining_times", []), dtype=torch.float32)  # [R?]
            if "remaining" in ind:
                rem = torch.tensor(ind.get("remaining", []), dtype=torch.float32)  # [R?]
            else:
                rem = torch.zeros(rem_t.shape[0], dtype=torch.float32)  # [R?]
            Lr = min(rem_t.shape[0], R)
            if R:
                rem_tensor[i, :Lr] = rem[:Lr]
                rem_time_tensor[i, :Lr] = rem_t[:Lr]
                rem_mask_tensor[i, :Lr] = True

        return {
            "obs": obs_tensor,
            "time": time_tensor,
            "mask": mask_tensor,
            "rem": rem_tensor,
            "rem_time": rem_time_tensor,
            "rem_mask": rem_mask_tensor,
        }

    def build_study_batch(
        self, study: StudyJSON, meta_dosing: MetaDosingConfig
    ) -> AICMECompartmentsDataBatch:
        """Build a batch for a single study.
        DOES NOT USES OBSERVATIONS STRATEGIESM,
        takes the observation structure as given by the JSON data

        Parameters
        ----------
        study:
            Canonicalised representation of one study.
        meta_dosing:
            Global dosing configuration.

        Returns
        -------
        AICMECompartmentsDataBatch
            Batch with ``B=1``.
        """

        study = canonicalize_study(study)
        ctx_cap = self._ctx_cap()
        tgt_cap = self._tgt_cap()

        ctx_block = self._block_from_inds(
            study["context"],
            max_individuals=ctx_cap,
            obs_cap=self._ctx_obs_cap(),
            rem_cap=self._ctx_rem_cap(),
        )
        tgt_block = self._block_from_inds(
            study["target"],
            max_individuals=tgt_cap,
            obs_cap=self._tgt_obs_cap(),
            rem_cap=self._tgt_rem_cap(),
        )

        route_vocab = {r: i for i, r in enumerate(meta_dosing.route_options)}

        def _dose_route(inds: List[IndividualJSON], I_max: int):
            amounts = torch.zeros(1, I_max, dtype=torch.float32)  # [1, I]
            routes = torch.zeros(1, I_max, dtype=torch.long)  # [1, I]
            for i, ind in enumerate(inds[:I_max]):
                if ind.get("dosing"):
                    amounts[0, i] = ind["dosing"][0]
                    routes[0, i] = route_vocab.get(ind["dosing_type"][0], 0)
            return amounts, routes

        c_dose, c_route = _dose_route(study["context"], ctx_cap)
        t_dose, t_route = _dose_route(study["target"], tgt_cap)

        def _unsqueeze(block):
            obs = block["obs"].unsqueeze(0).unsqueeze(-1)  # [1, I, ET, 1]
            time = block["time"].unsqueeze(0).unsqueeze(-1)  # [1, I, ET, 1]
            mask = block["mask"].unsqueeze(0)  # [1, I, ET]
            rem = block["rem"].unsqueeze(0).unsqueeze(-1)  # [1, I, R, 1]
            rem_time = block["rem_time"].unsqueeze(0).unsqueeze(-1)  # [1, I, R, 1]
            rem_mask = block["rem_mask"].unsqueeze(0)  # [1, I, R]
            return obs, time, mask, rem, rem_time, rem_mask

        t_obs, t_time, t_mask, t_rem, t_rem_time, t_rem_mask = _unsqueeze(tgt_block)
        c_obs, c_time, c_mask, c_rem, c_rem_time, c_rem_mask = _unsqueeze(ctx_block)

        mask_ctx_inds = torch.zeros(1, ctx_cap, dtype=torch.bool)  # [1, I]
        mask_ctx_inds[0, : min(len(study["context"]), ctx_cap)] = True
        mask_tgt_inds = torch.zeros(1, tgt_cap, dtype=torch.bool)  # [1, I]
        mask_tgt_inds[0, : min(len(study["target"]), tgt_cap)] = True

        study_name = [study["meta_data"]["study_name"]]
        substance_name = [study["meta_data"].get("substance_name", "")]

        context_subject_name = [
            [
                study["context"][i].get("name_id", "") if i < len(study["context"]) else ""
                for i in range(ctx_cap)
            ]
        ]
        target_subject_name = [
            [
                study["target"][i].get("name_id", "") if i < len(study["target"]) else ""
                for i in range(tgt_cap)
            ]
        ]

        batch = AICMECompartmentsDataBatch(
            target_obs=t_obs,
            target_obs_time=t_time,
            target_obs_mask=t_mask,
            target_rem_sim=t_rem,
            target_rem_sim_time=t_rem_time,
            target_rem_sim_mask=t_rem_mask,
            context_obs=c_obs,
            context_obs_time=c_time,
            context_obs_mask=c_mask,
            context_rem_sim=c_rem,
            context_rem_sim_time=c_rem_time,
            context_rem_sim_mask=c_rem_mask,
            target_dosing_amounts=t_dose,
            target_dosing_route_types=t_route,
            context_dosing_amounts=c_dose,
            context_dosing_route_types=c_route,
            mask_context_individuals=mask_ctx_inds,
            mask_target_individuals=mask_tgt_inds,
            study_name=study_name,
            context_subject_name=context_subject_name,
            target_subject_name=target_subject_name,
            substance_name=substance_name,
            time_scales=torch.tensor([[0.0, 0.0]]),
            is_empirical=True,
        )
        return batch

    @staticmethod
    def _stack_B(
        batches: List[AICMECompartmentsDataBatch],
    ) -> AICMECompartmentsDataBatch:
        """Concatenate ``batches`` along the batch dimension ``B``.

        Each input batch must have ``B=1``; the returned batch will have
        ``B=len(batches)`` with index order preserved.
        """

        if not batches:
            raise ValueError("batches must not be empty")

        stacked_fields = []
        for values in zip(*batches):
            first = values[0]
            if isinstance(first, torch.Tensor):
                stacked_fields.append(torch.cat(values, dim=0))  # [B, ...]
            elif isinstance(first, list):
                if first and isinstance(first[0], list):
                    merged_nested: List[List[str]] = []
                    for v in values:
                        merged_nested.extend(v)
                    stacked_fields.append(merged_nested)
                else:
                    merged: List[str] = []
                    for v in values:
                        merged.extend(v)
                    stacked_fields.append(merged)
            else:
                stacked_fields.append(first)
        return AICMECompartmentsDataBatch(*stacked_fields)

    def build_one_aicmebatch(
        self, studies: List[StudyJSON], meta_dosing: MetaDosingConfig
    ) -> AICMECompartmentsDataBatch:
        """Build a single batch from multiple studies.

        Parameters
        ----------
        studies:
            List of studies to combine. The resulting batch will have
            ``B=len(studies)``.
        meta_dosing:
            Global dosing configuration shared across studies.

        Returns
        -------
        AICMECompartmentsDataBatch
            Combined batch with batch dimension indexing the supplied
            studies in order.
        """

        per_study = [self.build_study_batch(s, meta_dosing) for s in studies]
        return self._stack_B(per_study)

    def build_one_aicmebatch_as_dataset(
        self,
        studies: List[StudyJSON],
        context_strategy,
        target_strategy,
        meta_dosing: MetaDosingConfig,
        *,
        return_studies: bool = False,  # ← debugging flag (default = True)
    ) -> List[Union[AICMECompartmentsDataBatch, List[StudyJSON]]]:
        """Create batches mirroring ``AICMECompartmentsDataset`` processing.

        For each study we generate leave-one-out permutations using
        :func:`held_out_ind_json`. The provided ``context_strategy`` and
        ``target_strategy`` are then used to apply the same empirical splitting
        between observed and remaining measurements as performed in
        :class:`AICMECompartmentsDataset`. Each permutation across all studies is
        stacked along the batch dimension ``B``.

        Parameters
        ----------
        studies:
            List of empirical studies. Each study is expected to contain only a
            context block; target individuals are produced via leave-one-out
            permutations.
        context_strategy / target_strategy:
            Observation strategies matching those used by
            :class:`AICMECompartmentsDataset` for shaping context and target
            data respectively.
        meta_dosing:
            Global dosing configuration.
        return_studies:
            If True (default), return the intermediate permuted study dicts
            instead of building full ``AICMECompartmentsDataBatch`` objects.
            Useful for debugging.

        Returns
        -------
        List[Union[AICMECompartmentsDataBatch, List[StudyJSON]]]
            If `return_studies` is True → list of permuted study dicts.
            If `return_studies` is False → list of ``AICMECompartmentsDataBatch``.
        """
        canon_studies = [canonicalize_study(s, drop_tgt_too_few=False) for s in studies]
        max_perm = max(len(s["context"]) for s in canon_studies)
        per_study_perms = [held_out_ind_json(s, max_perm) for s in canon_studies]

        batches = []
        for perm_idx in range(max_perm):
            permuted_studies = [
                self._process_one_study_perm(
                    study_perms[perm_idx], context_strategy, target_strategy
                )
                for study_perms in per_study_perms
            ]
            if return_studies:
                batches.append(permuted_studies)  # debugging: raw dicts
            else:
                batches.append(self.build_one_aicmebatch(permuted_studies, meta_dosing))
        return batches

    def build_one_aicmebatch_as_dataset_no_heldout(
        self,
        studies: List[StudyJSON],
        context_strategy,
        target_strategy,
        meta_dosing: MetaDosingConfig,
        *,
        return_studies: bool = False,
    ) -> List[Union[AICMECompartmentsDataBatch, List[StudyJSON]]]:
        """Create a single empirical batch without leave-one-out targets.

        This method mirrors :meth:`build_one_aicmebatch_as_dataset` preprocessing
        but does not move any individual from context to target. All individuals
        remain in context and the returned list has a single element.

        Parameters
        ----------
        studies:
            List of empirical studies.
        context_strategy / target_strategy:
            Observation strategies matching those used by
            :class:`AICMECompartmentsDataset`.
        meta_dosing:
            Global dosing configuration.
        return_studies:
            If ``True``, return the processed ``StudyJSON`` records instead of a
            fully built :class:`AICMECompartmentsDataBatch`.

        Returns
        -------
        List[Union[AICMECompartmentsDataBatch, List[StudyJSON]]]
            A list with length one containing either processed studies or one
            ``AICMECompartmentsDataBatch``.
        """
        canon_studies = [canonicalize_study(s, drop_tgt_too_few=False) for s in studies]
        context_only_studies: List[StudyJSON] = []
        for study in canon_studies:
            all_inds = list(study.get("context", [])) + list(study.get("target", []))
            context_only_studies.append(
                {
                    "context": all_inds,
                    "target": [],
                    "meta_data": dict(study.get("meta_data", {})),
                }
            )
        processed_studies = [
            self._process_one_study_perm(study, context_strategy, target_strategy)
            for study in context_only_studies
        ]

        if return_studies:
            return [processed_studies]
        return [self.build_one_aicmebatch(processed_studies, meta_dosing)]

    def _process_one_study_perm(
        self,
        study: StudyJSON,
        context_strategy,
        target_strategy,
    ) -> StudyJSON:
        """Turn one permuted study into tensors and apply strategies."""
        processed = {"context": [], "target": [], "meta_data": study["meta_data"]}

        for block, inds, strat in (
            ("context", study["context"], context_strategy),
            ("target", study["target"], target_strategy),
        ):
            processed[block] = self._process_block(inds, strat)

        return processed

    def _process_block(
        self,
        inds: List[IndividualJSON],
        strat,
    ) -> List[IndividualJSON]:
        """Convert a list of individuals into padded tensors, then apply strategy."""
        if not inds:
            return []

        obs, times, mask = self._pack_individuals(inds)
        obs_o, time_o, mask_o, rem_o, rem_t, rem_m = strat.generate_empirical(obs, times, mask)

        return self._rebuild_individuals(inds, obs_o, time_o, mask_o, rem_o, rem_t, rem_m)

    def _pack_individuals(
        self,
        inds: List[IndividualJSON],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad individuals into (obs, times, mask)."""
        I = len(inds)
        ET = max(len(ind["observations"]) for ind in inds)
        obs = torch.full((I, ET), self.cfg.pad_value_obs)
        times = torch.full((I, ET), self.cfg.pad_value_time)
        mask = torch.zeros((I, ET), dtype=torch.bool)

        for i, ind in enumerate(inds):
            o = torch.tensor(ind["observations"], dtype=torch.float32)
            t = torch.tensor(ind["observation_times"], dtype=torch.float32)
            L = o.shape[0]
            obs[i, :L], times[i, :L], mask[i, :L] = o, t, True
        return obs, times, mask

    def _rebuild_individuals(
        self,
        inds: List[IndividualJSON],
        obs_o: torch.Tensor,
        time_o: torch.Tensor,
        mask_o: torch.Tensor,
        rem_o: Optional[torch.Tensor],
        rem_t: Optional[torch.Tensor],
        rem_m: Optional[torch.Tensor],
    ) -> List[IndividualJSON]:
        """Convert tensors back to JSON-like dicts for each individual."""
        block_inds = []
        for i in range(obs_o.shape[0]):
            ind_dict: IndividualJSON = {
                "observations": obs_o[i][mask_o[i]].tolist(),
                "observation_times": time_o[i][mask_o[i]].tolist(),
            }
            name_id = inds[i].get("name_id") if i < len(inds) else None
            if name_id:
                ind_dict["name_id"] = name_id
            if rem_o is not None and rem_m is not None:
                ind_dict["remaining"] = rem_o[i][rem_m[i]].tolist()
                ind_dict["remaining_times"] = rem_t[i][rem_m[i]].tolist()
            block_inds.append(ind_dict)
        return block_inds


def databatch_to_study_jsons(
    batch: AICMECompartmentsDataBatch,
    meta_dosing: MetaDosingConfig,
) -> list[StudyJSON]:
    """Convert an ``AICMECompartmentsDataBatch`` back to ``StudyJSON`` records.

    Parameters
    ----------
    batch:
        Batch carrying tensors with a leading study dimension ``B``.
    meta_dosing:
        Dosing configuration used to decode route type indices.

    Returns
    -------
    List[StudyJSON]
        One study per element along the batch dimension ``B``. Missing
        ``study_name`` or ``substance_name`` entries are replaced by
        fallback placeholders ``study_{b}`` and ``substance_{b}``.
    """
    route_options = meta_dosing.route_options
    studies: list[StudyJSON] = []
    B = batch.context_obs.shape[0]

    def _block(
        obs: TT["B", "I", "T", 1],
        time: TT["B", "I", "T", 1],
        mask: TT["B", "I", "T"],
        rem: TT["B", "I", "R", 1],
        rem_time: TT["B", "I", "R", 1],
        rem_mask: TT["B", "I", "R"],
        doses: TT["B", "I"],
        routes: TT["B", "I"],
        ind_mask: TT["B", "I"],
        names: list[list[str]],
    ) -> list[IndividualJSON]:
        inds: list[IndividualJSON] = []
        for i in range(obs.shape[1]):
            if not ind_mask[b, i]:
                continue
            name_list = names[b] if b < len(names) else []
            ind: IndividualJSON = {}
            if i < len(name_list) and name_list[i]:
                ind["name_id"] = name_list[i]
            obs_i = obs[b, i, :, 0]  # [T]
            time_i = time[b, i, :, 0]  # [T]
            mask_i = mask[b, i]  # [T]
            ind["observations"] = obs_i[mask_i].tolist()
            ind["observation_times"] = time_i[mask_i].tolist()
            rem_i = rem[b, i, :, 0]  # [R]
            rem_time_i = rem_time[b, i, :, 0]  # [R]
            rem_mask_i = rem_mask[b, i]  # [R]
            rem_vals = rem_i[rem_mask_i].tolist()
            rem_times = rem_time_i[rem_mask_i].tolist()
            if rem_vals:
                ind["remaining"] = rem_vals
                ind["remaining_times"] = rem_times
            dose = float(doses[b, i].item())
            route_idx = int(routes[b, i].item())
            if dose or route_idx:
                route = (
                    route_options[route_idx] if route_idx < len(route_options) else str(route_idx)
                )
                ind["dosing"] = [dose]
                ind["dosing_type"] = [route]
                ind["dosing_times"] = [meta_dosing.time]
                ind["dosing_name"] = [route]
            inds.append(ind)
        return inds

    for b in range(B):
        study_name = (
            batch.study_name[b]
            if b < len(batch.study_name) and batch.study_name[b]
            else f"study_{b}"
        )
        substance_name = (
            batch.substance_name[b]
            if b < len(batch.substance_name) and batch.substance_name[b]
            else f"substance_{b}"
        )
        meta = {"study_name": study_name, "substance_name": substance_name}
        ctx = _block(
            batch.context_obs,
            batch.context_obs_time,
            batch.context_obs_mask,
            batch.context_rem_sim,
            batch.context_rem_sim_time,
            batch.context_rem_sim_mask,
            batch.context_dosing_amounts,
            batch.context_dosing_route_types,
            batch.mask_context_individuals,
            batch.context_subject_name,
        )
        tgt = _block(
            batch.target_obs,
            batch.target_obs_time,
            batch.target_obs_mask,
            batch.target_rem_sim,
            batch.target_rem_sim_time,
            batch.target_rem_sim_mask,
            batch.target_dosing_amounts,
            batch.target_dosing_route_types,
            batch.mask_target_individuals,
            batch.target_subject_name,
        )
        studies.append({"context": ctx, "target": tgt, "meta_data": meta})
    return studies


def prediction_to_study_jsons(
    prediction_sample: TT["S", "B", "It", "Tr", 1],
    prediction_time: TT["S", "B", "It", "Tr", 1],
    batch: AICMECompartmentsDataBatch,
    meta_dosing: MetaDosingConfig,
) -> list[StudyJSON]:
    """Attach prediction samples to study records.

    Parameters
    ----------
    prediction_sample:
        Predicted trajectories with a leading sample dimension ``S``.
    prediction_time:
        Time points corresponding to ``prediction_sample``.
    batch:
        Original :class:`AICMECompartmentsDataBatch` used to generate the
        predictions.
    meta_dosing:
        Dosing configuration for route decoding.

    Returns
    -------
    list[StudyJSON]
        Studies with ``prediction_samples`` and ``prediction_times`` fields in
        each predicted target individual.

    Notes
    -----
    Some predictive samplers (for example FlowPK individual prediction) may
    return predictions for only a subset of target individuals compared with
    the original batch. In that case this function keeps only the first ``It``
    target entries (where ``It`` is inferred from ``prediction_sample``) so
    JSON plots and exported records stay aligned with the predicted tensors.
    """

    studies = databatch_to_study_jsons(batch, meta_dosing)
    _, B, It, _, _ = prediction_sample.shape  # [S, B, It, Tr, 1]
    for b in range(B):
        # Keep studies aligned with the number of predicted target individuals.
        studies[b]["target"] = studies[b]["target"][:It]
        for i in range(min(It, len(studies[b]["target"]))):
            samples = prediction_sample[:, b, i, :, 0]  # [S, Tr]
            times = prediction_time[0, b, i, :, 0]  # [Tr]
            studies[b]["target"][i]["prediction_samples"] = samples.tolist()
            studies[b]["target"][i]["prediction_times"] = times.tolist()
    return studies


def simulation_obs_to_study_json(
    obs_out: torch.Tensor,
    time_out: torch.Tensor,
    mask_out: torch.Tensor,
    rem_sim: Optional[torch.Tensor],
    rem_time: Optional[torch.Tensor],
    rem_mask: Optional[torch.Tensor],
    dosing_config_array: list,
    dosing_amounts: torch.Tensor,
    study_config,
    idx: int,
) -> StudyJSON:
    """Convert processed simulation tensors into a :class:`StudyJSON` entry.

    Parameters
    ----------
    obs_out, time_out, mask_out:
        Tensors describing the observed concentrations and time points for the
        simulated individuals.  ``mask_out`` identifies valid entries in the
        padded tensors.
    rem_sim, rem_time, rem_mask:
        Optional tensors describing the remaining (unobserved) simulation
        trajectory.  When provided, the tensors must have the same leading
        dimensions as ``obs_out`` and ``time_out`` with ``rem_mask`` marking
        valid entries.
    dosing_config_array:
        Sequence with dosing configuration objects for each individual.
    dosing_amounts:
        Tensor containing the dosing amount per individual.
    study_config:
        Configuration object describing the simulated study. Only the
        ``drug_id`` attribute is accessed, if present.
    idx:
        Index used to label the generated study name.

    Returns
    -------
    StudyJSON
        JSON-compatible dictionary describing the context block of the
        simulation.
    """

    context: list[IndividualJSON] = []
    num_individuals = obs_out.shape[0]

    for ind_idx in range(num_individuals):
        mask = mask_out[ind_idx].to(torch.bool)
        observations = obs_out[ind_idx][mask].tolist()
        observation_times = time_out[ind_idx][mask].tolist()

        individual: IndividualJSON = {
            "name_id": f"context_{ind_idx}",
            "observations": observations,
            "observation_times": observation_times,
        }

        if rem_sim is not None and rem_time is not None and rem_mask is not None:
            rem_mask_row = rem_mask[ind_idx].to(torch.bool)
            if rem_mask_row.any():
                individual["remaining"] = rem_sim[ind_idx][rem_mask_row].tolist()
                individual["remaining_times"] = rem_time[ind_idx][rem_mask_row].tolist()

        dosing_cfg = dosing_config_array[ind_idx]
        dose = float(dosing_amounts[ind_idx].item())
        route = getattr(dosing_cfg, "route", "")
        dosing_time = float(getattr(dosing_cfg, "time", 0.0))

        if dose or route:
            individual["dosing"] = [dose]
            individual["dosing_type"] = [route]
            individual["dosing_times"] = [dosing_time]
            individual["dosing_name"] = [route]

        context.append(individual)

    study_json: StudyJSON = {
        "context": context,
        "target": [],
        "meta_data": {
            "study_name": f"simulated_study_{idx}",
            "substance_name": getattr(study_config, "drug_id", "simulated_substance"),
        },
    }

    return study_json


def held_out_ind_json(study: StudyJSON, max_held_out_individuals: int) -> List[StudyJSON]:
    """Create study permutations with one individual moved to target.

    Parameters
    ----------
    study:
        Study JSON containing only context individuals (``target`` must be empty).
    max_held_out_individuals:
        Maximum number of permutations to generate.

    Returns
    -------
    List[StudyJSON]
        List with ``max_held_out_individuals`` studies where each of the first
        ``len(context)`` entries corresponds to one context individual being
        moved to the target block. Remaining entries repeat the original study
        with an empty target.
    """
    context = list(study.get("context", []))
    meta = dict(study.get("meta_data", {}))
    out: List[StudyJSON] = []
    n_ctx = len(context)
    limit = min(max_held_out_individuals, n_ctx)
    for idx in range(limit):
        target = [context[idx]]
        ctx = context[:idx] + context[idx + 1 :]
        out.append({"context": ctx, "target": target, "meta_data": meta})
    base = {"context": context, "target": [], "meta_data": meta}
    while len(out) < max_held_out_individuals:
        out.append(base)
    return out


def held_out_list_json(
    builder: JSON2AICMEBuilder,
    studies: List[StudyJSON],
    meta_dosing: MetaDosingConfig,
    max_held_out_individuals: int,
) -> List[AICMECompartmentsDataBatch]:
    """
    Generate batches for leave-one-out permutations across studies.

    Parameters
    ----------
    builder:
        Instance used to convert studies to :class:`AICMECompartmentsDataBatch`.
    studies:
        Studies where only the context block is populated.
    meta_dosing:
        Global dosing configuration.
    max_held_out_individuals:
        Maximum number of held-out permutations per study.

    Returns
    -------
    List[AICMECompartmentsDataBatch]
        ``max_held_out_individuals`` batches. The ``i``-th batch contains the
        ``i``-th permutation from each study stacked along the batch
        dimension.
    """
    per_study = [held_out_ind_json(s, max_held_out_individuals) for s in studies]
    batches: List[AICMECompartmentsDataBatch] = []
    for i in range(max_held_out_individuals):
        perm = [per_study[j][i] for j in range(len(studies))]
        batches.append(builder.build_one_aicmebatch(perm, meta_dosing))
    return batches


def load_empirical_json_batches(
    json_path: Path,
    meta_dosing: Optional[MetaDosingConfig] = None,
    stats: Optional[EmpiricalJSONStats] = None,
    datamodule: Optional[AICMECompartmentsDataModule] = None,
) -> List[AICMECompartmentsDataBatch]:
    """
    Load an empirical study JSON file and build leave-one-out batches.

    We place all the individuals in the context

    Parameters
    ----------
    json_path:
        Path to a JSON file containing a list of :class:`StudyJSON` records.
    meta_dosing:
        Global dosing configuration. If ``None`` a default
        :class:`MetaDosingConfig` is used.
    stats:
        Pre-computed statistics describing the dataset. When ``None`` the
        statistics are calculated from ``json_path`` via
        :func:`compute_json_stats`.
    datamodule:
        Optional synthetic data module providing shape information via
        :meth:`AICMECompartmentsDataModule.obtain_shapes`. When given, these
        shapes override those inferred from ``stats``.

    Returns
    -------
    List[AICMECompartmentsDataBatch]
        Leave-one-out batches constructed from the studies in ``json_path``.

    Notes
    -----
    The function canonicalises all studies and either uses the provided
    ``stats`` or computes them from the JSON file to determine the number of
    leave-one-out permutations. When ``datamodule`` is supplied the padding
    shapes ``(max_individuals, max_observations, max_remaining)`` are taken
    from :meth:`AICMECompartmentsDataModule.obtain_shapes`.
    """

    # read file SHOULD BE A LIST OF STUDY JSON
    with json_path.open() as f:
        raw_studies = json.load(f)

    if not isinstance(raw_studies, list):
        raise ValueError("Expected JSON file to contain a list of StudyJSON records")

    # ensure data quality
    canon_studies: List[StudyJSON] = [
        canonicalize_study(s, drop_tgt_too_few=False) for s in raw_studies
    ]

    # we set all the individuals as context
    studies: List[StudyJSON] = []
    for study in canon_studies:
        all_individuals = list(study.get("context", [])) + list(study.get("target", []))
        studies.append(
            {"context": all_individuals, "target": [], "meta_data": study.get("meta_data", {})}
        )

    # define shapes
    if not studies:
        raise ValueError("No studies found in JSON file")
    if datamodule is not None:
        max_inds, max_obs, max_rem = datamodule.obtain_shapes()  # (I, T, R)
        ctx_cap = getattr(datamodule.train_dataset, "max_context_individuals", max_inds)
        tgt_cap = getattr(datamodule.train_dataset, "n_of_target_individuals", max_inds)
    else:
        # compute statitics of the whole dataset
        stats = compute_json_stats(canon_studies)
        max_inds, max_obs, max_rem = (
            stats.max_total_individuals,
            stats.max_observations,
            stats.max_remaining,
        )
        ctx_cap = max_inds
        tgt_cap = max_inds

    # the maximum batch is so that we have all the empirical at once
    cfg = EmpiricalBatchConfig(
        max_databatch_size=len(studies),
        max_individuals=max_inds,
        max_observations=max_obs,
        max_remaining=max_rem,
        max_context_individuals=ctx_cap,
        max_target_individuals=tgt_cap,
    )
    builder = JSON2AICMEBuilder(cfg)
    meta = meta_dosing or MetaDosingConfig()

    return held_out_list_json(
        builder, studies, meta, max_held_out_individuals=stats.max_total_individuals
    )


def load_empirical_json_batches_as_dm(
    json_path: Optional[Path] = None,
    meta_dosing: Optional[MetaDosingConfig] = None,
    stats: Optional[EmpiricalJSONStats] = None,
    datamodule: Optional[AICMECompartmentsDataModule] = None,
    raw_studies: Optional[List[StudyJSON]] = None,
    *,
    held_out: bool = True,
) -> List[AICMECompartmentsDataBatch]:
    """Load an empirical study JSON file and build leave-one-out batches.

    This variant mirrors the empirical preprocessing performed by
    :class:`AICMECompartmentsDataset` by relying on the observation strategies
    of a provided :class:`AICMECompartmentsDataModule` and using
    :meth:`JSON2AICMEBuilder.build_one_aicmebatch_as_dataset`.

    Parameters
    ----------
    json_path:
        Path to a JSON file containing a list of :class:`StudyJSON` records.
    meta_dosing:
        Global dosing configuration. If ``None`` a default
        :class:`MetaDosingConfig` is used.
    stats:
        Pre-computed statistics describing the dataset. When ``None`` the
        statistics are calculated from ``json_path`` via
        :func:`compute_json_stats`.
    datamodule:
        Synthetic data module providing observation strategies and shape
        information via :meth:`AICMECompartmentsDataModule.obtain_shapes`.
        The module must be provided; its shapes override those inferred from
        ``stats``.
    held_out:
        If ``True`` (default), build leave-one-out permutations (one empirical
        individual in target). If ``False``, keep all empirical individuals in
        context and return a single batch.

    Returns
    -------
    List[AICMECompartmentsDataBatch]
        Leave-one-out batches constructed from the studies in ``json_path``
        using the datamodule's strategies.
    """

    if datamodule is None:
        raise ValueError("datamodule must be provided to supply observation strategies")

    if raw_studies is None:
        with json_path.open() as f:
            raw_studies = json.load(f)

    if not isinstance(raw_studies, list):
        raise ValueError("Expected JSON file to contain a list of StudyJSON records")

    canon_studies: List[StudyJSON] = [
        canonicalize_study(s, drop_tgt_too_few=False) for s in raw_studies
    ]

    if stats is None:
        stats = compute_json_stats(canon_studies)

    studies: List[StudyJSON] = []
    for study in canon_studies:
        all_inds = list(study.get("context", [])) + list(study.get("target", []))
        studies.append({"context": all_inds, "target": [], "meta_data": study.get("meta_data", {})})

    if not studies:
        raise ValueError("No studies found in JSON file")

    max_inds, max_obs, max_rem = datamodule.obtain_shapes()  # (I, T, R)
    ctx_cap = getattr(datamodule.train_dataset, "max_context_individuals", max_inds)
    tgt_cap = getattr(datamodule.train_dataset, "n_of_target_individuals", max_inds)
    context_strategy = getattr(datamodule, "context_strategy", None)
    # For empirical targets we prefer the dedicated datamodule override
    # (legacy PK behavior + fixed capacities), falling back to target_strategy.
    target_strategy = getattr(datamodule, "empirical_target_strategy", None)
    if target_strategy is None:
        target_strategy = getattr(datamodule, "target_strategy", None)
    if context_strategy is None or target_strategy is None:
        raise ValueError("datamodule is missing context or target strategies")

    ctx_obs_cap, ctx_rem_cap = context_strategy.get_shapes()
    tgt_obs_cap, tgt_rem_cap = target_strategy.get_shapes()

    cfg = EmpiricalBatchConfig(
        max_databatch_size=len(studies),
        max_individuals=max_inds,
        max_observations=max_obs,
        max_remaining=max_rem,
        max_context_individuals=ctx_cap,
        max_target_individuals=tgt_cap,
        max_context_observations=ctx_obs_cap,
        max_target_observations=tgt_obs_cap,
        max_context_remaining=ctx_rem_cap,
        max_target_remaining=tgt_rem_cap,
    )
    builder = JSON2AICMEBuilder(cfg)
    meta = meta_dosing or MetaDosingConfig()

    if held_out:
        return builder.build_one_aicmebatch_as_dataset(
            studies, context_strategy, target_strategy, meta
        )
    return builder.build_one_aicmebatch_as_dataset_no_heldout(
        studies, context_strategy, target_strategy, meta
    )


def load_empirical_hf_batches_as_dm(
    repo_id: str,
    split: str = "train",
    meta_dosing: Optional[MetaDosingConfig] = None,
    stats: Optional[EmpiricalJSONStats] = None,
    datamodule: Optional[AICMECompartmentsDataModule] = None,
    *,
    held_out: bool = True,
) -> List[AICMECompartmentsDataBatch]:
    """Load a StudyJSON dataset from Hugging Face Hub.

    Parameters
    ----------
    repo_id:
        Hugging Face dataset id.
    split:
        Dataset split to load.
    meta_dosing:
        Dosing configuration.
    stats:
        Optional precomputed dataset statistics.
    datamodule:
        Datamodule providing empirical shape and strategy information.
    held_out:
        If ``True`` (default), build leave-one-out permutations. If ``False``,
        keep all empirical individuals in context and return a single batch.
    """

    if datamodule is None:
        raise ValueError("datamodule must be provided to supply observation strategies")

    # Load from HF Hub
    ds = load_dataset(repo_id, split=split)
    raw_studies = [dict(study) for study in ds]  # Hugging Face rows are dict-like

    # reuse your old code
    canon_studies: List[StudyJSON] = [
        canonicalize_study(s, drop_tgt_too_few=False) for s in raw_studies
    ]

    if stats is None:
        stats = compute_json_stats(canon_studies)

    studies: List[StudyJSON] = []
    for study in canon_studies:
        all_inds = list(study.get("context", [])) + list(study.get("target", []))
        studies.append({"context": all_inds, "target": [], "meta_data": study.get("meta_data", {})})

    if not studies:
        raise ValueError("No studies found in HF dataset")

    max_inds, max_obs, max_rem = datamodule.obtain_shapes()
    ctx_cap = getattr(datamodule.train_dataset, "max_context_individuals", max_inds)
    tgt_cap = getattr(datamodule.train_dataset, "n_of_target_individuals", max_inds)
    context_strategy = getattr(datamodule, "context_strategy", None)
    # For empirical targets we prefer the dedicated datamodule override
    # (legacy PK behavior + fixed capacities), falling back to target_strategy.
    target_strategy = getattr(datamodule, "empirical_target_strategy", None)
    if target_strategy is None:
        target_strategy = getattr(datamodule, "target_strategy", None)
    if context_strategy is None or target_strategy is None:
        raise ValueError("datamodule is missing context or target strategies")

    ctx_obs_cap, ctx_rem_cap = context_strategy.get_shapes()
    tgt_obs_cap, tgt_rem_cap = target_strategy.get_shapes()

    cfg = EmpiricalBatchConfig(
        max_databatch_size=len(studies),
        max_individuals=max_inds,
        max_observations=max_obs,
        max_remaining=max_rem,
        max_context_individuals=ctx_cap,
        max_target_individuals=tgt_cap,
        max_context_observations=ctx_obs_cap,
        max_target_observations=tgt_obs_cap,
        max_context_remaining=ctx_rem_cap,
        max_target_remaining=tgt_rem_cap,
    )
    builder = JSON2AICMEBuilder(cfg)
    meta = meta_dosing or MetaDosingConfig()

    if held_out:
        return builder.build_one_aicmebatch_as_dataset(
            studies, context_strategy, target_strategy, meta
        )
    return builder.build_one_aicmebatch_as_dataset_no_heldout(
        studies, context_strategy, target_strategy, meta
    )
