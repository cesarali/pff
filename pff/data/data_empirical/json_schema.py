"""TypedDict schemas for empirical pharmacokinetic JSON inputs."""

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, TypedDict

try:  # pragma: no cover - optional torch dependency
    import torch
    from torchtyping import TensorType as TT
except ModuleNotFoundError:  # pragma: no cover - allow missing torch
    torch = None  # type: ignore
    TT = object  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pff.data.datasets.aicme_batch import AICMECompartmentsDataBatch


class IndividualJSON(TypedDict, total=False):
    """Schema for a single individual's PK data.

    Optional ``prediction_samples`` and ``prediction_times`` fields allow
    storing model forecasts for the individual's future trajectory.
    Each element in ``prediction_samples`` corresponds to a full simulated
    trajectory for the times listed in ``prediction_times``.

    For inference requests, ``remaining_times`` may be provided without
    ``remaining``. In that case the time grid is treated as the requested
    decode schedule and placeholder values can be materialized later by the
    empirical builder.
    """

    name_id: str
    observations: List[float]
    observation_times: List[float]
    remaining: List[float]
    remaining_times: List[float]
    dosing: List[float]
    dosing_type: List[str]
    dosing_times: List[float]
    dosing_name: List[str]
    prediction_samples: List[List[float]]
    prediction_times: List[float]
    covariates: Dict[str, object]


class StudyJSON(TypedDict):
    """Schema for a full study consisting of context and target individuals."""

    context: List[IndividualJSON]
    target: List[IndividualJSON]
    meta_data: Dict[str, str]


MIN_OBS_DEFAULT = 0


class ValidationError(Exception):
    """Raised when data do not conform to :class:`StudyJSON`."""

    pass


def canonicalize_individual(
    ind: IndividualJSON,
    *,
    min_obs: int = MIN_OBS_DEFAULT,
    drop_if_too_few: bool = True,
) -> Optional[IndividualJSON]:
    """Return a canonical version of ``ind``.

    Parameters
    ----------
    ind:
        Individual JSON record to canonicalize. The input dictionary is **not**
        mutated.
    min_obs:
        Minimum required number of observations. Defaults to
        :data:`MIN_OBS_DEFAULT`.
    drop_if_too_few:
        If ``True`` and the individual has fewer than ``min_obs`` observations
        after sorting/de-duplication, ``None`` is returned.

    Returns
    -------
    Optional[IndividualJSON]
        Canonicalized record or ``None`` when dropped.

    Notes
    -----
    The function performs the following steps:

    - Validate the presence and equal length of ``observations`` and
      ``observation_times``.
    - Sort observations by ascending time and remove duplicate time entries
      keeping the first occurrence.
    - Optionally drop the individual if the number of observations is below
      ``min_obs``.
    - Ensure ``remaining``/``remaining_times`` are disjoint from
      ``observation_times`` and of equal length.
    - If any dosing related fields are provided, require that all dosing fields
      are present and have equal lengths.
    """

    # --- observations & times ---
    if "observations" not in ind or "observation_times" not in ind:
        raise ValidationError("observations and observation_times are required")

    obs = list(ind["observations"])
    times = list(ind["observation_times"])
    if len(obs) != len(times):
        raise ValidationError("observations and observation_times must match in length")

    # sort and de-duplicate by time (stable sort keeps first occurrence)
    pairs = sorted(zip(times, obs), key=lambda x: x[0])
    seen = set()
    obs_sorted: List[float] = []
    times_sorted: List[float] = []
    for t, o in pairs:
        if t in seen:
            continue
        seen.add(t)
        times_sorted.append(t)
        obs_sorted.append(o)

    if len(obs_sorted) < min_obs and drop_if_too_few:
        return None

    new_ind: IndividualJSON = {}
    if "name_id" in ind:
        new_ind["name_id"] = ind["name_id"]
    new_ind["observations"] = obs_sorted
    new_ind["observation_times"] = times_sorted

    # --- remaining ---
    has_rem_values = "remaining" in ind
    has_rem_times = "remaining_times" in ind
    if has_rem_values or has_rem_times:
        if has_rem_values and not has_rem_times:
            raise ValidationError("remaining_times is required when remaining is provided")

        rem_t = list(ind.get("remaining_times", []))
        rem = list(ind.get("remaining", []))
        if has_rem_values and len(rem) != len(rem_t):
            raise ValidationError("remaining and remaining_times must match in length")

        obs_time_set = set(times_sorted)
        rem_filtered: List[float] = []
        rem_t_filtered: List[float] = []

        if has_rem_values:
            for t, r in zip(rem_t, rem):
                if t in obs_time_set:
                    continue
                rem_t_filtered.append(t)
                rem_filtered.append(r)
            new_ind["remaining"] = rem_filtered
            new_ind["remaining_times"] = rem_t_filtered
        else:
            for t in rem_t:
                if t in obs_time_set:
                    continue
                rem_t_filtered.append(t)
            new_ind["remaining_times"] = rem_t_filtered

    # --- dosing ---
    dosing_keys = ["dosing", "dosing_type", "dosing_times", "dosing_name"]
    present_dosing = [k for k in dosing_keys if k in ind]
    if present_dosing:
        if len(present_dosing) != len(dosing_keys):
            raise ValidationError("all dosing fields must be present when dosing is provided")
        lengths = [len(ind[k]) for k in dosing_keys]  # type: ignore[index]
        if len(set(lengths)) != 1:
            raise ValidationError("dosing fields must have equal lengths")
        for k in dosing_keys:
            new_ind[k] = list(ind[k])  # type: ignore[index]

    # --- covariates ---
    if "covariates" in ind:
        new_ind["covariates"] = dict(ind["covariates"])

    # --- prediction samples ---
    if "prediction_samples" in ind:
        new_ind["prediction_samples"] = [list(s) for s in ind["prediction_samples"]]
    if "prediction_times" in ind:
        new_ind["prediction_times"] = list(ind["prediction_times"])
    if "prediction_mean" in ind:
        new_ind["prediction_mean"] = list(ind["prediction_mean"])
    if "prediction_std" in ind:
        new_ind["prediction_std"] = list(ind["prediction_std"])

    return new_ind


def canonicalize_study(
    study: StudyJSON,
    *,
    min_obs_ctx: int = MIN_OBS_DEFAULT,
    min_obs_tgt: int = MIN_OBS_DEFAULT,
    drop_tgt_too_few: bool = True,
) -> StudyJSON:
    """Canonicalize all individuals in ``study`` and validate meta data."""

    meta = study.get("meta_data", {})
    if not meta.get("study_name") or not meta.get("substance_name"):
        raise ValidationError("meta_data must include non-empty study_name and substance_name")

    context_canon: List[IndividualJSON] = []
    for ind in study.get("context", []):
        canon = canonicalize_individual(ind, min_obs=min_obs_ctx, drop_if_too_few=False)
        if canon is not None:
            context_canon.append(canon)

    target_canon: List[IndividualJSON] = []
    for ind in study.get("target", []):
        canon = canonicalize_individual(ind, min_obs=min_obs_tgt, drop_if_too_few=drop_tgt_too_few)
        if canon is not None:
            target_canon.append(canon)

    new_study: StudyJSON = {
        "context": context_canon,
        "target": target_canon,
        "meta_data": dict(meta),
    }
    return new_study


def studies_from_sampled_targets(
    *,
    db: "AICMECompartmentsDataBatch",
    samples: "TT['S', 'B', 'T', 1]",
    times: "TT['B', 'T', 1]",
    mask: "TT['B', 'T']",
    route_options: Sequence[str],
    dosing_time: float,
    name_prefix: str = "new_individual",
    resolve_sampling_from_target: bool = False,
) -> List[StudyJSON]:
    """Convert sampled trajectories into :class:`StudyJSON` records.

    Parameters
    ----------
    db:
        Batch containing the conditioning study information. Only the fields
        accessed in this function are required, allowing reuse with compatible
        NamedTuple implementations used throughout the project.
    samples, times, mask:
        Output tensors from ``sample_new_individual`` where ``samples`` carries
        the simulated trajectories, ``times`` their corresponding decode times
        and ``mask`` selects valid entries along the temporal dimension.
    route_options:
        Lookup table translating dosing route indices into human readable
        labels. Indices outside the provided range are returned as their string
        representation.
    dosing_time:
        Absolute time at which the dosing event occurred. Used for both context
        and newly sampled target individuals when dosing information is
        present.
    name_prefix:
        Prefix for generated target individual identifiers. Defaults to
        ``"new_individual"``.
    resolve_sampling_from_target:
        Interpret the leading axis of ``samples`` as target individuals rather
        than Monte Carlo samples. In this mode one target record is emitted per
        target axis entry using the corresponding target dosing metadata.

    Returns
    -------
    list[StudyJSON]
        One ``StudyJSON`` per batch element in ``db``.
    """

    if torch is None:
        raise ValidationError("torch is required to build StudyJSON records from tensors")

    leading_count, B, _, _ = samples.shape
    studies: List[StudyJSON] = []

    for b in range(B):
        study_name = (
            db.study_name[b] if b < len(db.study_name) and db.study_name[b] else f"study_{b}"
        )
        substance_name = (
            db.substance_name[b]
            if b < len(db.substance_name) and db.substance_name[b]
            else f"substance_{b}"
        )

        context_list: List[IndividualJSON] = []
        I = db.context_obs.shape[1]
        for i in range(I):
            if not db.mask_context_individuals[b, i]:
                continue

            ind: IndividualJSON = {}
            if b < len(db.context_subject_name) and i < len(db.context_subject_name[b]):
                name = db.context_subject_name[b][i]
                if name:
                    ind["name_id"] = name

            obs_i = db.context_obs[b, i, :, 0]
            time_i = db.context_obs_time[b, i, :, 0]
            mask_i = db.context_obs_mask[b, i]
            ind["observations"] = obs_i[mask_i].tolist()
            ind["observation_times"] = time_i[mask_i].tolist()

            if db.context_rem_sim.shape[2] > 0:
                rem_i = db.context_rem_sim[b, i, :, 0]
                rem_t = db.context_rem_sim_time[b, i, :, 0]
                rem_m = db.context_rem_sim_mask[b, i]
                rem_vals = rem_i[rem_m].tolist()
                rem_times = rem_t[rem_m].tolist()
                if rem_vals:
                    ind["remaining"] = rem_vals
                    ind["remaining_times"] = rem_times

            dose = float(db.context_dosing_amounts[b, i].item())
            route_idx = int(db.context_dosing_route_types[b, i].item())
            if dose or route_idx:
                route = (
                    route_options[route_idx] if route_idx < len(route_options) else str(route_idx)
                )
                ind["dosing"] = [dose]
                ind["dosing_type"] = [route]
                ind["dosing_times"] = [dosing_time]
                ind["dosing_name"] = [route]

            context_list.append(ind)

        target_list: List[IndividualJSON] = []
        valid_mask = mask[b]
        valid_times = times[b, valid_mask, 0].tolist()
        for sample_idx in range(leading_count):
            traj = samples[sample_idx, b, valid_mask, 0].tolist()
            if resolve_sampling_from_target:
                target_idx = sample_idx
                if (
                    b < len(db.target_subject_name)
                    and target_idx < len(db.target_subject_name[b])
                    and db.target_subject_name[b][target_idx]
                ):
                    name_id = db.target_subject_name[b][target_idx]
                else:
                    name_id = f"{name_prefix}_{target_idx}"
                dose = float(db.target_dosing_amounts[b, target_idx].item())
                route_idx = int(db.target_dosing_route_types[b, target_idx].item())
            else:
                name_id = f"{name_prefix}_{sample_idx}"
                dose = float(db.target_dosing_amounts[b, 0].item())
                route_idx = int(db.target_dosing_route_types[b, 0].item())

            ind: IndividualJSON = {
                "name_id": name_id,
                "observations": traj,
                "observation_times": valid_times,
            }
            if dose or route_idx:
                route = (
                    route_options[route_idx] if route_idx < len(route_options) else str(route_idx)
                )
                ind["dosing"] = [dose]
                ind["dosing_type"] = [route]
                ind["dosing_times"] = [dosing_time]
                ind["dosing_name"] = [route]
            target_list.append(ind)

        studies.append(
            {
                "context": context_list,
                "target": target_list,
                "meta_data": {
                    "study_name": study_name,
                    "substance_name": substance_name,
                },
            }
        )

    return studies


def prediction_stats(study: StudyJSON) -> StudyJSON:
    """Compute prediction mean and std for target individuals.

    For each target individual with ``prediction_samples`` the function
    calculates the mean and standard deviation across the sample dimension and
    stores the results in ``prediction_mean`` and ``prediction_std`` fields.

    Parameters
    ----------
    study:
        ``StudyJSON`` record containing prediction samples.

    Returns
    -------
    StudyJSON
        The input study where target individuals now also carry ``prediction_mean``
        and ``prediction_std`` fields. The input mapping is mutated for
        convenience.
    """

    for ind in study.get("target", []):
        samples = ind.get("prediction_samples")
        if samples:
            if torch is None:
                raise ValidationError("torch is required to compute prediction summaries")
            samples_t: TT["S", "Tr"] = torch.tensor(samples)
            ind["prediction_mean"] = samples_t.mean(dim=0).tolist()
            ind["prediction_std"] = samples_t.std(dim=0, unbiased=False).tolist()
    return study
