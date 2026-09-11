"""Shared runtime-contract helpers for consumer-facing Hub bundles.

This module is imported both by the local exporter and by the copied package
inside the generated Hugging Face runtime bundle. Keep dependencies limited to
modules that are already required for model inference.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, get_args, get_origin

import torch
from transformers import PretrainedConfig

from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    MixDataConfig,
    ObservationsConfig,
    SimpleMetaStudyConfig,
)
from pff.config_classes.diffusion_pk_config import DiffusionPKExperimentConfig
from pff.config_classes.flow_pk_config import FlowPKExperimentConfig, VectorFieldPKConfig
from pff.config_classes.node_pk_config import (
    EncoderDecoderNetworkConfig,
    NodePKExperimentConfig,
)
from pff.config_classes.source_process_config import SourceProcessConfig
from pff.config_classes.training_config import TrainingConfig
from pff.data.data_empirical.builder import EmpiricalBatchConfig, JSON2AICMEBuilder
from pff.data.data_empirical.json_schema import (
    IndividualJSON,
    StudyJSON,
    canonicalize_study,
)
from pff.data.data_generation.observations_classes import ObservationStrategyFactory
from pff.models import get_model_class
from pff.models.amortized_inference.generative_pk import (
    NewGenerativeMixin,
    NewPredictiveMixin,
)

SUPPORTED_RUNTIME_ARCHITECTURES = {
    "AICMEPK",
    "ContextVAEPK",
    "FlowPK",
    "PredictionPK",
}
STUDY_JSON_IO_VERSION = "studyjson-v1"


@dataclass
class RuntimeBuilderConfig:
    """Fixed builder capacities serialized into the Hub runtime config."""

    max_context_individuals: int
    max_target_individuals: int
    max_context_observations: int
    max_target_observations: int
    max_context_remaining: int
    max_target_remaining: int

    def to_dict(self) -> Dict[str, int]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeBuilderConfig":
        """Instantiate the builder capacities from serialized config payload."""

        return cls(
            max_context_individuals=int(payload["max_context_individuals"]),
            max_target_individuals=int(payload["max_target_individuals"]),
            max_context_observations=int(payload["max_context_observations"]),
            max_target_observations=int(payload["max_target_observations"]),
            max_context_remaining=int(payload["max_context_remaining"]),
            max_target_remaining=int(payload["max_target_remaining"]),
        )

    def to_empirical_batch_config(self, *, max_databatch_size: int) -> EmpiricalBatchConfig:
        """Translate runtime capacities to the builder used by StudyJSON IO."""

        return EmpiricalBatchConfig(
            max_databatch_size=int(max_databatch_size),
            max_individuals=max(self.max_context_individuals, self.max_target_individuals),
            max_observations=max(self.max_context_observations, self.max_target_observations),
            max_remaining=max(self.max_context_remaining, self.max_target_remaining),
            max_context_individuals=self.max_context_individuals,
            max_target_individuals=self.max_target_individuals,
            max_context_observations=self.max_context_observations,
            max_target_observations=self.max_target_observations,
            max_context_remaining=self.max_context_remaining,
            max_target_remaining=self.max_target_remaining,
        )


def _coerce_annotation(annotation: Any, value: Any) -> Any:
    """Best-effort coercion of JSON-loaded values into dataclass field types."""

    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Union:
        non_none = [arg for arg in args if arg is not type(None)]
        for candidate in non_none:
            if candidate in (dict, Dict, Any, Mapping):
                continue
            try:
                return _coerce_annotation(candidate, value)
            except Exception:
                continue
        return value

    if origin in (list, List, Sequence):
        (inner_type,) = args if args else (Any,)
        return [_coerce_annotation(inner_type, item) for item in value]

    if origin in (tuple,):
        if not args:
            return tuple(value)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_annotation(args[0], item) for item in value)
        return tuple(_coerce_annotation(inner, item) for inner, item in zip(args, value))

    if origin in (dict, Dict, Mapping):
        return dict(value)

    if annotation is Any:
        return value

    if annotation is MetaStudyConfig and isinstance(value, Mapping) and value.get("simple_mode"):
        return SimpleMetaStudyConfig(**dict(value))

    if hasattr(annotation, "__dataclass_fields__") and isinstance(value, Mapping):
        kwargs = {}
        for field_name, field_def in annotation.__dataclass_fields__.items():
            if field_name in value:
                kwargs[field_name] = _coerce_annotation(field_def.type, value[field_name])
        return annotation(**kwargs)

    return value


def _rebuild_node_config(payload: Mapping[str, Any]) -> NodePKExperimentConfig:
    """Reconstruct a ``NodePKExperimentConfig`` from serialized dict content."""

    return NodePKExperimentConfig(
        experiment_type=str(payload.get("experiment_type", "nodepk")).lower(),
        name_str=str(payload.get("name_str", "NodePK")),
        comet_ai_key=payload.get("comet_ai_key"),
        experiment_name=str(payload.get("experiment_name", "node_pk_compartments")),
        hugging_face_token=payload.get("hugging_face_token"),
        upload_to_hf_hub=bool(payload.get("upload_to_hf_hub", False)),
        hf_model_name=str(payload.get("hf_model_name", "NodePK_runtime")),
        hf_model_card_path=tuple(payload.get("hf_model_card_path", ("hf_model_cards", "README.md"))),
        tags=list(payload.get("tags", [])),
        experiment_indentifier=payload.get("experiment_indentifier"),
        my_results_path=payload.get("my_results_path"),
        experiment_dir=payload.get("experiment_dir"),
        verbose=bool(payload.get("verbose", False)),
        run_index=int(payload.get("run_index", 0)),
        debug_test=bool(payload.get("debug_test", False)),
        network=_coerce_annotation(EncoderDecoderNetworkConfig, payload.get("network", {})),
        mix_data=_coerce_annotation(MixDataConfig, payload.get("mix_data", {})),
        context_observations=_coerce_annotation(
            ObservationsConfig, payload.get("context_observations", {})
        ),
        target_observations=_coerce_annotation(
            ObservationsConfig, payload.get("target_observations", {})
        ),
        meta_study=_coerce_annotation(MetaStudyConfig, payload.get("meta_study", {})),
        dosing=_coerce_annotation(MetaDosingConfig, payload.get("dosing", {})),
        train=_coerce_annotation(TrainingConfig, payload.get("train", {})),
    )


def _rebuild_flow_config(payload: Mapping[str, Any]) -> FlowPKExperimentConfig:
    """Reconstruct a ``FlowPKExperimentConfig`` from serialized dict content."""

    return FlowPKExperimentConfig(
        experiment_type=str(payload.get("experiment_type", "flowpk")).lower(),
        name_str=str(payload.get("name_str", "FlowPK")),
        comet_ai_key=payload.get("comet_ai_key"),
        experiment_name=str(payload.get("experiment_name", "flow_pk_compartments")),
        hugging_face_token=payload.get("hugging_face_token"),
        upload_to_hf_hub=bool(payload.get("upload_to_hf_hub", False)),
        hf_model_name=str(payload.get("hf_model_name", "FlowPK_runtime")),
        hf_model_card_path=tuple(payload.get("hf_model_card_path", ("hf_model_cards", "README.md"))),
        tags=list(payload.get("tags", [])),
        experiment_indentifier=payload.get("experiment_indentifier"),
        my_results_path=payload.get("my_results_path"),
        experiment_dir=payload.get("experiment_dir"),
        verbose=bool(payload.get("verbose", False)),
        run_index=int(payload.get("run_index", 0)),
        debug_test=bool(payload.get("debug_test", False)),
        flow_num_steps=int(payload.get("flow_num_steps", 50)),
        vector_field=_coerce_annotation(VectorFieldPKConfig, payload.get("vector_field", {})),
        source_process=_coerce_annotation(SourceProcessConfig, payload.get("source_process", {})),
        mix_data=_coerce_annotation(MixDataConfig, payload.get("mix_data", {})),
        context_observations=_coerce_annotation(
            ObservationsConfig, payload.get("context_observations", {})
        ),
        target_observations=_coerce_annotation(
            ObservationsConfig, payload.get("target_observations", {})
        ),
        meta_study=_coerce_annotation(MetaStudyConfig, payload.get("meta_study", {})),
        dosing=_coerce_annotation(MetaDosingConfig, payload.get("dosing", {})),
        train=_coerce_annotation(TrainingConfig, payload.get("train", {})),
    )


def _rebuild_diffusion_config(payload: Mapping[str, Any]) -> DiffusionPKExperimentConfig:
    """Reconstruct a ``DiffusionPKExperimentConfig`` from serialized dict content."""

    return DiffusionPKExperimentConfig(
        experiment_type=str(payload.get("experiment_type", "diffusionpk")).lower(),
        name_str=str(payload.get("name_str", "ContinuousDiffusionPK")),
        diffusion_type=str(payload.get("diffusion_type", "continuous")),
        comet_ai_key=payload.get("comet_ai_key"),
        experiment_name=str(payload.get("experiment_name", "diffusion_pk_compartments")),
        hugging_face_token=payload.get("hugging_face_token"),
        upload_to_hf_hub=bool(payload.get("upload_to_hf_hub", False)),
        hf_model_name=str(payload.get("hf_model_name", "DiffusionPK_runtime")),
        hf_model_card_path=tuple(payload.get("hf_model_card_path", ("hf_model_cards", "README.md"))),
        tags=list(payload.get("tags", [])),
        experiment_indentifier=payload.get("experiment_indentifier"),
        my_results_path=payload.get("my_results_path"),
        experiment_dir=payload.get("experiment_dir"),
        verbose=bool(payload.get("verbose", False)),
        run_index=int(payload.get("run_index", 0)),
        debug_test=bool(payload.get("debug_test", False)),
        predict_gaussian_noise=bool(payload.get("predict_gaussian_noise", True)),
        diffusion_num_steps=int(payload.get("diffusion_num_steps", 100)),
        diffusion_t1=float(payload.get("diffusion_t1", 1.0)),
        diffusion_beta_min=float(payload.get("diffusion_beta_min", 1e-4)),
        diffusion_beta_max=float(payload.get("diffusion_beta_max", 2e-2)),
        vector_field=(
            _coerce_annotation(VectorFieldPKConfig, payload.get("vector_field", {}))
            if payload.get("vector_field") is not None
            else None
        ),
        network=(
            _coerce_annotation(EncoderDecoderNetworkConfig, payload.get("network", {}))
            if payload.get("network") is not None
            else None
        ),
        source_process=_coerce_annotation(SourceProcessConfig, payload.get("source_process", {})),
        mix_data=_coerce_annotation(MixDataConfig, payload.get("mix_data", {})),
        context_observations=_coerce_annotation(
            ObservationsConfig, payload.get("context_observations", {})
        ),
        target_observations=_coerce_annotation(
            ObservationsConfig, payload.get("target_observations", {})
        ),
        meta_study=_coerce_annotation(MetaStudyConfig, payload.get("meta_study", {})),
        dosing=_coerce_annotation(MetaDosingConfig, payload.get("dosing", {})),
        train=_coerce_annotation(TrainingConfig, payload.get("train", {})),
    )


def rebuild_experiment_config(
    payload: Mapping[str, Any],
) -> Union[NodePKExperimentConfig, FlowPKExperimentConfig, DiffusionPKExperimentConfig]:
    """Rebuild the serialized experiment config stored in the Hub config."""

    experiment_type = str(payload.get("experiment_type", "nodepk")).lower()
    if experiment_type == "nodepk":
        return _rebuild_node_config(payload)
    if experiment_type == "flowpk":
        return _rebuild_flow_config(payload)
    if experiment_type == "diffusionpk":
        return _rebuild_diffusion_config(payload)
    raise ValueError(f"Unsupported experiment_type for runtime bundle: {experiment_type!r}.")


def compute_runtime_builder_config(
    exp_config: Union[NodePKExperimentConfig, FlowPKExperimentConfig, DiffusionPKExperimentConfig],
) -> RuntimeBuilderConfig:
    """Compute fixed empirical StudyJSON capacities from the experiment config."""

    context_strategy = ObservationStrategyFactory.from_config(
        exp_config.context_observations,
        exp_config.meta_study,
    )
    target_strategy = ObservationStrategyFactory.from_config(
        exp_config.target_observations,
        exp_config.meta_study,
    )
    ctx_obs_cap, ctx_rem_cap = context_strategy.get_shapes()
    tgt_obs_cap, tgt_rem_cap = target_strategy.get_shapes()

    max_context_individuals = int(exp_config.meta_study.num_individuals_range[-1])
    max_target_individuals = int(getattr(exp_config.mix_data, "n_of_target_individuals", 1))
    if max_target_individuals < 0:
        raise ValueError("n_of_target_individuals must be >= 0 for Hub runtime export.")

    return RuntimeBuilderConfig(
        max_context_individuals=max_context_individuals,
        max_target_individuals=max_target_individuals,
        max_context_observations=int(ctx_obs_cap),
        max_target_observations=int(tgt_obs_cap),
        max_context_remaining=int(ctx_rem_cap),
        max_target_remaining=int(tgt_rem_cap),
    )


def infer_supported_tasks(backbone: torch.nn.Module) -> List[str]:
    """Infer the public task surface supported by the wrapped model."""

    tasks: List[str] = []
    if isinstance(backbone, NewGenerativeMixin):
        tasks.append("generate")
    if isinstance(backbone, NewPredictiveMixin):
        tasks.append("predict")
    return tasks


def validate_runtime_architecture(backbone: torch.nn.Module) -> str:
    """Ensure the loaded architecture is supported by the runtime bundle v1."""

    architecture_name = backbone.__class__.__name__
    if architecture_name not in SUPPORTED_RUNTIME_ARCHITECTURES:
        raise ValueError(
            "Runtime Hub export only supports "
            f"{sorted(SUPPORTED_RUNTIME_ARCHITECTURES)}, got {architecture_name!r}."
        )
    return architecture_name


def build_runtime_config_payload(
    *,
    backbone: torch.nn.Module,
    exp_config: Union[NodePKExperimentConfig, FlowPKExperimentConfig, DiffusionPKExperimentConfig],
    original_repo_id: Optional[str],
    runtime_repo_id: Optional[str],
) -> Dict[str, Any]:
    """Build the serializable fields stored in the Hub config."""

    architecture_name = validate_runtime_architecture(backbone)
    supported_tasks = infer_supported_tasks(backbone)
    if not supported_tasks:
        raise ValueError(f"Model {architecture_name!r} does not expose runtime tasks.")

    builder_config = compute_runtime_builder_config(exp_config)
    return {
        "architecture_name": architecture_name,
        "experiment_type": str(getattr(exp_config, "experiment_type", "nodepk")).lower(),
        "experiment_config": asdict(exp_config),
        "builder_config": builder_config.to_dict(),
        "supported_tasks": supported_tasks,
        "default_task": supported_tasks[0],
        "io_schema_version": STUDY_JSON_IO_VERSION,
        "original_repo_id": original_repo_id,
        "runtime_repo_id": runtime_repo_id,
    }


def instantiate_backbone_from_hub_config(config: PretrainedConfig) -> torch.nn.Module:
    """Rebuild the internal PK model represented by the public Hub wrapper."""

    experiment_config_payload = getattr(config, "experiment_config", None)
    if not isinstance(experiment_config_payload, Mapping):
        raise ValueError("Hub config is missing the serialized experiment_config payload.")
    exp_config = rebuild_experiment_config(experiment_config_payload)
    model_cls = get_model_class(exp_config)
    backbone = model_cls(exp_config)
    backbone.eval()
    return backbone


def normalize_studies_input(
    studies: Union[StudyJSON, Sequence[StudyJSON]],
) -> List[StudyJSON]:
    """Normalize runtime input to a mutable list of canonicalized studies."""

    if isinstance(studies, Mapping):
        raw_studies = [dict(studies)]
    else:
        raw_studies = [dict(study) for study in studies]
    return [canonicalize_study(study, drop_tgt_too_few=False) for study in raw_studies]


def validate_studies_for_task(
    studies: Sequence[StudyJSON],
    *,
    task: str,
    builder_config: RuntimeBuilderConfig,
) -> None:
    """Validate task semantics and reject inputs that exceed runtime capacities."""

    for study_idx, study in enumerate(studies):
        context = list(study.get("context", []))
        target = list(study.get("target", []))

        if task == "generate":
            if not context:
                raise ValueError("`generate` requires at least one context individual per study.")
            if target:
                raise ValueError("`generate` expects target to be empty in the input StudyJSON.")
        elif task == "predict":
            if not target:
                raise ValueError("`predict` requires at least one target individual per study.")
        else:
            raise ValueError(f"Unsupported task {task!r}.")

        if len(context) > builder_config.max_context_individuals:
            raise ValueError(
                f"Study {study_idx} exceeds context individual capacity "
                f"({len(context)} > {builder_config.max_context_individuals})."
            )
        if len(target) > builder_config.max_target_individuals:
            raise ValueError(
                f"Study {study_idx} exceeds target individual capacity "
                f"({len(target)} > {builder_config.max_target_individuals})."
            )

        _validate_individual_block(
            study_idx=study_idx,
            block_name="context",
            individuals=context,
            max_observations=builder_config.max_context_observations,
            max_remaining=builder_config.max_context_remaining,
        )
        _validate_individual_block(
            study_idx=study_idx,
            block_name="target",
            individuals=target,
            max_observations=builder_config.max_target_observations,
            max_remaining=builder_config.max_target_remaining,
        )


def _validate_individual_block(
    *,
    study_idx: int,
    block_name: str,
    individuals: Sequence[IndividualJSON],
    max_observations: int,
    max_remaining: int,
) -> None:
    """Reject studies that would otherwise be truncated by the empirical builder."""

    for ind_idx, individual in enumerate(individuals):
        obs_len = len(individual.get("observations", []))
        rem_len = len(individual.get("remaining", []))
        if obs_len > max_observations:
            raise ValueError(
                f"Study {study_idx} {block_name}[{ind_idx}] exceeds observation capacity "
                f"({obs_len} > {max_observations})."
            )
        if rem_len > max_remaining:
            raise ValueError(
                f"Study {study_idx} {block_name}[{ind_idx}] exceeds remaining capacity "
                f"({rem_len} > {max_remaining})."
            )


def build_batch_from_studies(
    studies: Sequence[StudyJSON],
    *,
    builder_config: RuntimeBuilderConfig,
    meta_dosing: MetaDosingConfig,
):
    """Convert canonical studies into the internal PK databatch representation."""

    builder = JSON2AICMEBuilder(
        builder_config.to_empirical_batch_config(max_databatch_size=max(1, len(studies)))
    )
    return builder.build_one_aicmebatch(list(studies), meta_dosing)


def split_runtime_samples(task: str, study: StudyJSON) -> List[StudyJSON]:
    """Convert model-specific StudyJSON outputs into per-sample StudyJSONs."""

    if task == "generate":
        return _split_generate_samples(study)
    if task == "predict":
        return _split_predict_samples(study)
    raise ValueError(f"Unsupported task {task!r}.")


def _split_generate_samples(study: StudyJSON) -> List[StudyJSON]:
    """Split generated target individuals into one StudyJSON per sample."""

    targets = list(study.get("target", []))
    if not targets:
        return [deepcopy(study)]

    split: List[StudyJSON] = []
    for target in targets:
        split.append(
            {
                "context": deepcopy(study.get("context", [])),
                "target": [deepcopy(target)],
                "meta_data": deepcopy(study.get("meta_data", {})),
            }
        )
    return split


def _split_predict_samples(study: StudyJSON) -> List[StudyJSON]:
    """Split target prediction samples into one StudyJSON per sample index."""

    targets = list(study.get("target", []))
    if not targets:
        return [deepcopy(study)]

    sample_count = 0
    for target in targets:
        sample_count = max(sample_count, len(target.get("prediction_samples", [])))
    if sample_count == 0:
        return [deepcopy(study)]

    split: List[StudyJSON] = []
    for sample_idx in range(sample_count):
        target_block: List[IndividualJSON] = []
        for target in targets:
            target_copy: IndividualJSON = deepcopy(target)
            samples = list(target.get("prediction_samples", []))
            if samples:
                if sample_idx >= len(samples):
                    raise ValueError(
                        "All target individuals must expose the same number of prediction samples."
                    )
                target_copy["prediction_samples"] = [deepcopy(samples[sample_idx])]
            target_block.append(target_copy)

        split.append(
            {
                "context": deepcopy(study.get("context", [])),
                "target": target_block,
                "meta_data": deepcopy(study.get("meta_data", {})),
            }
        )
    return split


def runtime_readme_text(
    *,
    base_model_card: str,
    runtime_repo_id: str,
    original_repo_id: Optional[str],
    supported_tasks: Sequence[str],
    default_task: str,
) -> str:
    """Compose the README uploaded with the consumer-facing runtime bundle."""

    original_line = (
        f"- Native training/artifact repo: `{original_repo_id}`"
        if original_repo_id
        else "- Native training/artifact repo: not recorded"
    )
    tasks_literal = ", ".join(f"`{task}`" for task in supported_tasks)

    usage = f"""

## Runtime Bundle

This repository is the consumer-facing runtime bundle for this PK model.

- Runtime repo: `{runtime_repo_id}`
{original_line}
- Supported tasks: {tasks_literal}
- Default task: `{default_task}`
- Load path: `AutoModel.from_pretrained(..., trust_remote_code=True)`

### Installation

You do **not** need to install `pff` to use this runtime bundle.

`transformers` is the public loading entrypoint, but `transformers` alone is
not sufficient because this is a PyTorch model with custom runtime code. A
reliable consumer environment is:

```bash
pip install torch transformers huggingface_hub lightning datasets pandas torchtyping gpytorch pot torchdiffeq torchsde ruamel.yaml pyyaml
```

### Python Usage

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("{runtime_repo_id}", trust_remote_code=True)

studies = [
    {{
        "context": [
            {{
                "name_id": "ctx_0",
                "observations": [0.2, 0.5, 0.3],
                "observation_times": [0.5, 1.0, 2.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }}
        ],
        "target": [],
        "meta_data": {{"study_name": "demo", "substance_name": "drug_x"}},
    }}
]

outputs = model.run_task(
    task="{default_task}",
    studies=studies,
    num_samples=4,
)
print(outputs["results"][0]["samples"])
```

### Predictive Sampling

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("{runtime_repo_id}", trust_remote_code=True)

predict_studies = [
    {{
        "context": [
            {{
                "name_id": "ctx_0",
                "observations": [0.2, 0.5, 0.3],
                "observation_times": [0.5, 1.0, 2.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }}
        ],
        "target": [
            {{
                "name_id": "tgt_0",
                "observations": [0.25, 0.31],
                "observation_times": [0.5, 1.0],
                "remaining": [0.0, 0.0, 0.0],
                "remaining_times": [2.0, 4.0, 8.0],
                "dosing": [1.0],
                "dosing_type": ["oral"],
                "dosing_times": [0.0],
                "dosing_name": ["oral"],
            }}
        ],
        "meta_data": {{"study_name": "demo", "substance_name": "drug_x"}},
    }}
]

outputs = model.run_task(
    task="predict",
    studies=predict_studies,
    num_samples=4,
)
print(outputs["results"][0]["samples"][0]["target"][0]["prediction_samples"])
```

### Notes

- `trust_remote_code=True` is required because this model uses custom Hugging Face Hub runtime code.
- The consumer API is `transformers` + `run_task(...)`; the consumer does not need a local clone of this repository.
- This runtime bundle is intentionally separate from the native training export so you can evaluate both distribution paths in parallel.
"""

    return base_model_card.rstrip() + "\n" + usage.strip() + "\n"


def resolve_model_card_text(model_card_path: Path) -> str:
    """Read and validate the model card that seeds the runtime README."""

    if not model_card_path.is_file():
        raise FileNotFoundError(f"Model card not found at: {model_card_path}")
    return model_card_path.read_text(encoding="utf-8")
