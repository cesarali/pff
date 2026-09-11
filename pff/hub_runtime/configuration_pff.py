"""Hugging Face configuration for self-contained PK runtime bundles."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import PretrainedConfig

from pff.hub_runtime.runtime_contract import STUDY_JSON_IO_VERSION


class PKHubConfig(PretrainedConfig):
    """Public Hub config describing a consumer-facing PK runtime bundle."""

    model_type = "pff"

    def __init__(
        self,
        architecture_name: Optional[str] = None,
        experiment_type: str = "nodepk",
        experiment_config: Optional[Dict[str, Any]] = None,
        builder_config: Optional[Dict[str, Any]] = None,
        supported_tasks: Optional[List[str]] = None,
        default_task: Optional[str] = None,
        io_schema_version: str = STUDY_JSON_IO_VERSION,
        original_repo_id: Optional[str] = None,
        runtime_repo_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.architecture_name = architecture_name
        self.experiment_type = experiment_type
        self.experiment_config = dict(experiment_config or {})
        self.builder_config = dict(builder_config or {})
        self.supported_tasks = list(supported_tasks or [])
        self.default_task = default_task or (self.supported_tasks[0] if self.supported_tasks else None)
        self.io_schema_version = io_schema_version
        self.original_repo_id = original_repo_id
        self.runtime_repo_id = runtime_repo_id


__all__ = ["PKHubConfig"]
