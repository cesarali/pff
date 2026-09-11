"""Hugging Face AutoModel wrapper for consumer-facing PK runtime bundles."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import torch
from transformers import PreTrainedModel

from pff.data.data_empirical.json_schema import StudyJSON
from pff.hub_runtime.configuration_pff import PKHubConfig
from pff.hub_runtime.runtime_contract import (
    RuntimeBuilderConfig,
    build_batch_from_studies,
    infer_supported_tasks,
    instantiate_backbone_from_hub_config,
    normalize_studies_input,
    split_runtime_samples,
    validate_studies_for_task,
)
from pff.models.amortized_inference.generative_pk import (
    NewGenerativeMixin,
    NewPredictiveMixin,
)


class PKHubModel(PreTrainedModel):
    """Thin wrapper exposing a stable StudyJSON runtime API on top of PK models."""

    config_class = PKHubConfig
    base_model_prefix = "backbone"

    def __init__(self, config: PKHubConfig, backbone: Optional[torch.nn.Module] = None) -> None:
        super().__init__(config)
        self.backbone = backbone if backbone is not None else instantiate_backbone_from_hub_config(config)
        self.backbone.eval()

    def forward(self, *args, **kwargs):
        """Delegate raw forward calls to the wrapped PK backbone."""

        return self.backbone(*args, **kwargs)

    @property
    def supported_tasks(self) -> Sequence[str]:
        """Tasks supported by this runtime model."""

        return tuple(getattr(self.config, "supported_tasks", []) or infer_supported_tasks(self.backbone))

    @torch.inference_mode()
    def run_task(
        self,
        *,
        task: str,
        studies: Union[StudyJSON, Sequence[StudyJSON]],
        num_samples: int = 1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run the public StudyJSON inference contract for the requested task."""

        supported_tasks = list(self.supported_tasks)
        if task not in supported_tasks:
            raise ValueError(
                f"Unsupported task {task!r}. Supported tasks: {supported_tasks or 'none'}."
            )
        if int(num_samples) < 1:
            raise ValueError("num_samples must be >= 1.")

        canonical_studies = normalize_studies_input(studies)
        builder_config = RuntimeBuilderConfig.from_dict(self.config.builder_config)
        validate_studies_for_task(canonical_studies, task=task, builder_config=builder_config)

        experiment_config_payload = getattr(self.config, "experiment_config", {})
        meta_dosing_payload = experiment_config_payload.get("dosing", {})
        batch = build_batch_from_studies(
            canonical_studies,
            builder_config=builder_config,
            meta_dosing=self.backbone.meta_dosing.__class__(**meta_dosing_payload)
            if meta_dosing_payload
            else self.backbone.meta_dosing,
        )
        batch = batch.to(self.device)

        if task == "generate":
            if not isinstance(self.backbone, NewGenerativeMixin):
                raise ValueError(f"Backbone {type(self.backbone).__name__} does not support generate.")
            output_studies = self.backbone.sample_new_individuals_to_studyjson(
                batch,
                sample_size=int(num_samples),
                num_steps=kwargs.get("num_steps"),
            )
        elif task == "predict":
            if not isinstance(self.backbone, NewPredictiveMixin):
                raise ValueError(f"Backbone {type(self.backbone).__name__} does not support predict.")
            output_studies = self.backbone.sample_individual_prediction_from_batch_list_to_studyjson(
                [batch],
                sample_size=int(num_samples),
            )[0]
        else:
            raise ValueError(f"Unsupported task {task!r}.")

        results = [
            {
                "input_index": index,
                "samples": split_runtime_samples(task, study),
            }
            for index, study in enumerate(output_studies)
        ]

        return {
            "task": task,
            "io_schema_version": self.config.io_schema_version,
            "model_info": {
                "architecture_name": self.config.architecture_name,
                "experiment_type": self.config.experiment_type,
                "supported_tasks": supported_tasks,
                "runtime_repo_id": self.config.runtime_repo_id,
                "original_repo_id": self.config.original_repo_id,
            },
            "results": results,
        }


__all__ = ["PKHubModel"]
