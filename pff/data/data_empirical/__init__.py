"""Utilities for working with empirical JSON study data."""

try:  # pragma: no cover - optional torch dependency
    from .builder import (
        JSON2AICMEBuilder,
        EmpiricalBatchConfig,
        held_out_ind_json,
        held_out_list_json,
        load_empirical_json_batches,
        load_empirical_json_batches_as_dm,
        load_empirical_hf_batches_as_dm,
        databatch_to_study_jsons,
        prediction_to_study_jsons,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - allow missing torch
    if exc.name != "torch":
        raise
    JSON2AICMEBuilder = EmpiricalBatchConfig = None  # type: ignore
    held_out_ind_json = held_out_list_json = None  # type: ignore
    load_empirical_json_batches = load_empirical_json_batches_as_dm = None  # type: ignore
    databatch_to_study_jsons = prediction_to_study_jsons = None  # type: ignore

__all__ = [
    "json_schema",
    "JSON2AICMEBuilder",
    "EmpiricalBatchConfig",
    "held_out_ind_json",
    "held_out_list_json",
    "load_empirical_json_batches",
    "load_empirical_json_batches_as_dm",
    "load_empirical_hf_batches_as_dm",
    "databatch_to_study_jsons",
    "prediction_to_study_jsons",
    "json_stats",
]
