"""This is only used for checking the shapes of the empirical data that are passed to the Dataloader"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set

from .json_schema import StudyJSON


@dataclass
class EmpiricalJSONStats:
    """Basic statistics collected from empirical study JSON files.

    Attributes
    ----------
    min_context_individuals, max_context_individuals:
        Range of context individuals across studies.
    min_target_individuals, max_target_individuals:
        Range of target individuals across studies.
    min_observation, max_observation:
        Extremal observed values across all individuals.
    substances:
        Sorted list of distinct substance names.
    max_total_individuals:
        Maximum combined number of context and target individuals in a study.
    max_observations:
        Maximum number of observation time points for any individual.
    max_remaining:
        Maximum number of remaining time points for any individual.
    substance_summaries:
        Nested mapping keyed by substance name containing per-substance
        statistics. Each inner dictionary exposes the total number of
        individuals, the minimum and maximum number of observation time points
        per individual, and the sorted list of unique time steps observed
        across all individuals (including observation and remaining times) for
        the substance.
    studies_by_substance:
        Mapping from substance name to the list of studies associated with it.
    """

    min_context_individuals: int
    max_context_individuals: int
    min_target_individuals: int
    max_target_individuals: int
    min_observation: float
    max_observation: float
    substances: List[str]
    max_total_individuals: int
    max_observations: int
    max_remaining: int
    substance_summaries: Dict[str, Dict[str, object]]
    studies_by_substance: Dict[str, List[StudyJSON]]

    def studies_for_substance(self, substance: str) -> List[StudyJSON]:
        """Return all studies that reference ``substance``.

        Parameters
        ----------
        substance:
            Name of the substance whose studies should be returned.

        Returns
        -------
        List[StudyJSON]
            Study dictionaries associated with ``substance``. An empty list is
            returned when the substance was not observed.
        """

        return list(self.studies_by_substance.get(substance, []))

    def get_substance_summary(self, substance: str) -> Dict[str, object]:
        """Return the per-substance statistics for ``substance``.

        Parameters
        ----------
        substance:
            Name of the substance whose statistics should be retrieved.

        Returns
        -------
        Dict[str, object]
            Dictionary containing the ``individual_count``,
            ``min_observations``, ``max_observations`` and
            ``observation_time_steps`` entries. An empty dictionary is returned
            if the substance is unknown.
        """

        summary = self.substance_summaries.get(substance)
        if summary is None:
            return {}
        return dict(summary)


def compute_json_stats(studies: Sequence[StudyJSON]) -> EmpiricalJSONStats:
    """Compute statistics across empirical pharmacokinetic studies.

    Parameters
    ----------
    studies:
        Sequence of :class:`StudyJSON` objects to aggregate.

    Returns
    -------
    EmpiricalJSONStats
        Aggregated statistics across all provided studies.
    """

    min_ctx = float("inf")
    max_ctx = 0
    min_tgt = float("inf")
    max_tgt = 0
    min_obs = float("inf")
    max_obs = float("-inf")
    substances = set()
    max_total_inds = 0
    max_obs_len = 0
    max_rem_len = 0

    substance_counts: Dict[str, int] = defaultdict(int)
    substance_min_obs: Dict[str, int] = {}
    substance_max_obs: Dict[str, int] = {}
    substance_times: Dict[str, Set[float]] = defaultdict(set)
    studies_by_substance: Dict[str, List[StudyJSON]] = defaultdict(list)

    for study in studies:
        c_len = len(study.get("context", []))
        t_len = len(study.get("target", []))
        total_len = c_len + t_len
        min_ctx = min(min_ctx, c_len)
        max_ctx = max(max_ctx, c_len)
        min_tgt = min(min_tgt, t_len)
        max_tgt = max(max_tgt, t_len)
        max_total_inds = max(max_total_inds, total_len)
        meta = study.get("meta_data", {})
        substance = meta.get("substance_name")
        if substance:
            substances.add(substance)
            studies_by_substance[substance].append(study)
        for ind in study.get("context", []) + study.get("target", []):
            obs = ind.get("observations", [])
            obs_len = len(obs)
            rem = ind.get("remaining", [])
            times = ind.get("observation_times", [])
            rem_times = ind.get("remaining_times", [])

            max_obs_len = max(max_obs_len, len(obs))
            max_rem_len = max(max_rem_len, len(rem))
            if obs:
                min_obs = min(min_obs, min(obs))
                max_obs = max(max_obs, max(obs))

            if substance:
                substance_counts[substance] += 1
                current_min = substance_min_obs.get(substance)
                if current_min is None:
                    substance_min_obs[substance] = obs_len
                else:
                    substance_min_obs[substance] = min(current_min, obs_len)
                current_max = substance_max_obs.get(substance)
                if current_max is None:
                    substance_max_obs[substance] = obs_len
                else:
                    substance_max_obs[substance] = max(current_max, obs_len)
                substance_times[substance].update(times)
                substance_times[substance].update(rem_times)

    if min_ctx == float("inf"):
        min_ctx = 0
    if min_tgt == float("inf"):
        min_tgt = 0
    if min_obs == float("inf"):
        min_obs = float("nan")
    if max_obs == float("-inf"):
        max_obs = float("nan")

    substance_summaries = {
        substance: {
            "individual_count": substance_counts.get(substance, 0),
            "min_observations": substance_min_obs.get(substance, 0),
            "max_observations": substance_max_obs.get(substance, 0),
            "observation_time_steps": sorted(substance_times.get(substance, set())),
        }
        for substance in sorted(substances)
    }

    return EmpiricalJSONStats(
        min_context_individuals=int(min_ctx),
        max_context_individuals=int(max_ctx),
        min_target_individuals=int(min_tgt),
        max_target_individuals=int(max_tgt),
        min_observation=float(min_obs),
        max_observation=float(max_obs),
        substances=sorted(substances),
        max_total_individuals=int(max_total_inds),
        max_observations=int(max_obs_len),
        max_remaining=int(max_rem_len),
        substance_summaries=substance_summaries,
        studies_by_substance={k: list(v) for k, v in studies_by_substance.items()},
    )
