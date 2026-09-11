import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml  # type: ignore

from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    MixDataConfig,
    ObservationsConfig,
    SimpleMetaStudyConfig,
)
from pff.config_classes.flow_pk_config import VectorFieldPKConfig
from pff.config_classes.node_pk_config import EncoderDecoderNetworkConfig
from pff.config_classes.source_process_config import SourceProcessConfig
from pff.config_classes.training_config import TrainingConfig
from pff.config_classes.utils import TupleSafeLoader


@dataclass
class DiffusionPKExperimentConfig:
    """Experiment configuration dedicated to diffusion PK models."""

    experiment_type: str = "diffusionpk"
    name_str: str = "ContinuousDiffusionPK"
    diffusion_type: str = "continuous"  # "continuous" or "discrete"

    comet_ai_key: str = None
    experiment_name: str = "diffusion_pk_compartments"
    hugging_face_token: str = None
    upload_to_hf_hub: bool = True
    hf_model_name: str = "DiffusionPK_test"
    hf_model_card_path: Tuple[str, str, str] = ("hf_model_card", "DIFFUSION-PK_Readme.md")

    tags: List[str] = field(default_factory=lambda: ["diffusion-pk", "B-0"])
    experiment_indentifier: str = None
    my_results_path: str = None
    experiment_dir: str = None
    verbose: bool = False
    run_index: int = 0
    debug_test: bool = False

    # Diffusion training knob: predict unit Gaussian noise or correlated noise.
    predict_gaussian_noise: bool = True
    diffusion_num_steps: int = 100
    diffusion_t1: float = 1.0
    diffusion_beta_min: float = 1e-4
    diffusion_beta_max: float = 2e-2

    # New diffusion configs share FlowPK's vector-field architecture. ``network``
    # remains optional for old checkpoints/configs that still use latent
    # encoder-decoder diffusion.
    vector_field: Optional[VectorFieldPKConfig] = field(default_factory=VectorFieldPKConfig)
    network: Optional[EncoderDecoderNetworkConfig] = None
    source_process: SourceProcessConfig = field(default_factory=SourceProcessConfig)
    mix_data: MixDataConfig = field(default_factory=MixDataConfig)

    context_observations: ObservationsConfig = field(default_factory=ObservationsConfig)
    target_observations: ObservationsConfig = field(default_factory=ObservationsConfig)

    meta_study: MetaStudyConfig = field(default_factory=MetaStudyConfig)
    dosing: MetaDosingConfig = field(default_factory=MetaDosingConfig)

    train: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_yaml(file_path: str) -> "DiffusionPKExperimentConfig":
        """Initializes the class from a YAML file."""

        with open(file_path, "r") as file:
            config_dict = yaml.load(file, Loader=TupleSafeLoader) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected experiment YAML to be a mapping.")

        exp_type = config_dict.get("experiment_type")
        if exp_type is not None and str(exp_type).lower() != "diffusionpk":
            raise ValueError(
                "Expected experiment_type 'diffusionpk' for DiffusionPKExperimentConfig, "
                f"got {exp_type!r}."
            )

        base_dir = os.path.dirname(os.path.abspath(file_path))

        data_cfg_dict = (
            DiffusionPKExperimentConfig._load_ref_yaml(config_dict.get("data_config"), base_dir)
            or {}
        )
        training_cfg_dict = (
            DiffusionPKExperimentConfig._load_ref_yaml(config_dict.get("training_config"), base_dir)
            or {}
        )
        model_cfg_dict = (
            DiffusionPKExperimentConfig._load_ref_yaml(config_dict.get("model_config"), base_dir)
            or {}
        )

        observations_section = DiffusionPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "observations_config"
        )
        if observations_section is None:
            observations_section = DiffusionPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "observations_config"
            )
        if observations_section is not None:
            context_observations_base = observations_section.get("context_observations")
            target_observations_base = observations_section.get("target_observations")
        else:
            context_observations_base = data_cfg_dict.get("context_observations")
            target_observations_base = data_cfg_dict.get("target_observations")

        mix_data_section = DiffusionPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "mix_data_config"
        )
        if mix_data_section is None:
            mix_data_section = DiffusionPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "mix_data_config"
            )
        if mix_data_section is None:
            mix_data_section = data_cfg_dict.get("mix_data")

        meta_study_section = DiffusionPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_study_config"
        )
        if meta_study_section is None:
            meta_study_section = DiffusionPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_study_config"
            )
        meta_dosing_section = DiffusionPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_dosing_config"
        )
        if meta_dosing_section is None:
            meta_dosing_section = DiffusionPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_dosing_config"
            )

        meta_study_base = DiffusionPKExperimentConfig._extract_config_mapping(
            meta_study_section, "meta_study"
        )
        if meta_study_base is None and meta_dosing_section is not None:
            meta_study_base = DiffusionPKExperimentConfig._extract_config_mapping(
                meta_dosing_section, "meta_study"
            )
        if meta_study_base is None:
            meta_study_base = data_cfg_dict.get("meta_study")

        dosing_base = DiffusionPKExperimentConfig._extract_config_mapping(
            meta_dosing_section, "dosing"
        )
        if dosing_base is None:
            dosing_base = data_cfg_dict.get("dosing")

        mix_data_cfg = DiffusionPKExperimentConfig._merge_dicts(
            mix_data_section, config_dict.get("mix_data")
        )
        context_obs_cfg = DiffusionPKExperimentConfig._merge_dicts(
            context_observations_base, config_dict.get("context_observations")
        )
        target_obs_cfg = DiffusionPKExperimentConfig._merge_dicts(
            target_observations_base, config_dict.get("target_observations")
        )
        meta_study_cfg = DiffusionPKExperimentConfig._merge_dicts(
            meta_study_base, config_dict.get("meta_study")
        )
        dosing_cfg = DiffusionPKExperimentConfig._merge_dicts(
            dosing_base, config_dict.get("dosing")
        )

        train_section = training_cfg_dict.get("train", training_cfg_dict)
        train_cfg = DiffusionPKExperimentConfig._merge_dicts(
            train_section, config_dict.get("train")
        )

        vector_field_section = model_cfg_dict.get("vector_field")
        vector_field_cfg = None
        if vector_field_section is not None or config_dict.get("vector_field") is not None:
            vector_field_cfg = DiffusionPKExperimentConfig._merge_dicts(
                vector_field_section, config_dict.get("vector_field")
            )

        network_cfg = None
        if vector_field_cfg is None:
            network_section = model_cfg_dict.get("network")
            if network_section is None and "vector_field" not in model_cfg_dict:
                network_section = model_cfg_dict
            if network_section is not None or config_dict.get("network") is not None:
                network_cfg = DiffusionPKExperimentConfig._merge_dicts(
                    network_section, config_dict.get("network")
                )

        diffusion_section = model_cfg_dict.get("diffusion", {})
        diffusion_cfg = DiffusionPKExperimentConfig._merge_dicts(
            diffusion_section, config_dict.get("diffusion")
        )

        source_section = DiffusionPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "source_config"
        )
        if source_section is None:
            source_section = DiffusionPKExperimentConfig._resolve_config_section(
                model_cfg_dict, base_dir, "source_config"
            )
        if source_section is None:
            source_section = model_cfg_dict.get("source_process") or model_cfg_dict.get("noise_model")
        source_section = DiffusionPKExperimentConfig._extract_config_mapping(
            source_section, "source_process"
        )
        if isinstance(source_section, dict) and "noise_model" in source_section:
            source_section = source_section.get("noise_model")
        source_cfg = DiffusionPKExperimentConfig._merge_dicts(
            source_section, config_dict.get("source_process")
        )

        if meta_study_cfg.get("simple_mode", False):
            meta_study_instance = SimpleMetaStudyConfig(**meta_study_cfg)
        else:
            meta_study_instance = MetaStudyConfig(**meta_study_cfg)

        train_cfg = TrainingConfig._filter_kwargs(train_cfg)

        return DiffusionPKExperimentConfig(
            experiment_type=str(config_dict.get("experiment_type", "diffusionpk")).lower(),
            name_str=config_dict.get("name_str", "ContinuousDiffusionPK"),
            diffusion_type=config_dict.get("diffusion_type", "continuous"),
            tags=config_dict.get("tags", ["diffusion-pk", "B-0"]),
            experiment_name=config_dict.get("experiment_name", "diffusion_pk_compartments"),
            experiment_indentifier=config_dict.get("experiment_indentifier", None),
            my_results_path=config_dict.get("my_results_path", None),
            experiment_dir=config_dict.get("experiment_dir", None),
            comet_ai_key=config_dict.get("comet_ai_key", None),
            hugging_face_token=config_dict.get("hugging_face_token", None),
            upload_to_hf_hub=config_dict.get("upload_to_hf_hub", True),
            hf_model_name=config_dict.get("hf_model_name", "DiffusionPK_test"),
            hf_model_card_path=tuple(
                config_dict.get(
                    "hf_model_card_path", ("hf_model_card", "DIFFUSION-PK_Readme.md")
                )
            ),
            debug_test=config_dict.get("debug_test", False),
            predict_gaussian_noise=bool(
                config_dict.get(
                    "predict_gaussian_noise",
                    diffusion_cfg.get("predict_gaussian_noise", True),
                )
            ),
            diffusion_num_steps=int(
                config_dict.get(
                    "diffusion_num_steps",
                    diffusion_cfg.get("diffusion_num_steps", diffusion_cfg.get("num_steps", 100)),
                )
            ),
            diffusion_t1=float(
                config_dict.get("diffusion_t1", diffusion_cfg.get("diffusion_t1", 1.0))
            ),
            diffusion_beta_min=float(
                config_dict.get(
                    "diffusion_beta_min",
                    diffusion_cfg.get("diffusion_beta_min", diffusion_cfg.get("beta_min", 1e-4)),
                )
            ),
            diffusion_beta_max=float(
                config_dict.get(
                    "diffusion_beta_max",
                    diffusion_cfg.get("diffusion_beta_max", diffusion_cfg.get("beta_max", 2e-2)),
                )
            ),
            vector_field=(
                VectorFieldPKConfig(**vector_field_cfg)
                if vector_field_cfg is not None
                else None
            ),
            network=(
                EncoderDecoderNetworkConfig(**network_cfg)
                if network_cfg is not None
                else None
            ),
            source_process=SourceProcessConfig(**source_cfg),
            mix_data=MixDataConfig(**mix_data_cfg),
            context_observations=ObservationsConfig(**context_obs_cfg),
            target_observations=ObservationsConfig(**target_obs_cfg),
            meta_study=meta_study_instance,
            dosing=MetaDosingConfig(**dosing_cfg),
            train=TrainingConfig(**train_cfg),
        )

    @staticmethod
    def _merge_dicts(
        base_dict: Optional[Dict[str, Any]], override_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge two optional dictionaries returning a new dictionary."""

        merged: Dict[str, Any] = {}

        if base_dict:
            if not isinstance(base_dict, dict):
                raise TypeError(
                    "Expected base_dict to be a mapping when merging configuration sections."
                )
            merged = deepcopy(base_dict)

        if override_dict:
            if not isinstance(override_dict, dict):
                raise TypeError(
                    "Expected override_dict to be a mapping when merging configuration sections."
                )
            merged.update(override_dict)

        return merged

    @staticmethod
    def _extract_config_mapping(
        section: Optional[Dict[str, Any]], nested_key: str
    ) -> Optional[Dict[str, Any]]:
        """Return a nested configuration mapping or the section itself."""

        if section is None:
            return None

        if not isinstance(section, dict):
            raise TypeError(
                "Expected configuration section to be a mapping when extracting nested"
                f" '{nested_key}' values."
            )

        if nested_key in section:
            nested_value = section[nested_key]
            if nested_value is None:
                return None
            if not isinstance(nested_value, dict):
                raise TypeError(
                    f"Expected '{nested_key}' section to be a mapping when extracting configuration values."
                )
            return nested_value

        return section

    @staticmethod
    def _load_ref_yaml(
        ref: Optional[Union[str, Dict[str, Any]]], base_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load a referenced YAML block or return inline dictionaries as-is."""

        if ref is None:
            return None

        if isinstance(ref, dict):
            return ref

        if isinstance(ref, str):
            ref_path = ref
            if not os.path.isabs(ref_path):
                ref_path = os.path.join(base_dir, ref_path)

            with open(ref_path, "r") as handle:
                return yaml.load(handle, Loader=TupleSafeLoader) or {}

        raise TypeError("Expected configuration reference to be a mapping or string path.")

    @staticmethod
    def _resolve_config_section(
        cfg_dict: Dict[str, Any], base_dir: str, key: str
    ) -> Optional[Dict[str, Any]]:
        """Resolve nested configuration references within a configuration block."""

        if key not in cfg_dict:
            return None

        section = cfg_dict[key]

        if section is None:
            return None

        if isinstance(section, dict):
            ref_value = section.get("_ref") if "_ref" in section else None
            if ref_value is not None:
                loaded = DiffusionPKExperimentConfig._load_ref_yaml(ref_value, base_dir)
                return loaded or {}
            return section

        if isinstance(section, str):
            loaded = DiffusionPKExperimentConfig._load_ref_yaml(section, base_dir)
            return loaded or {}

        raise TypeError(
            f"Expected configuration section '{key}' to be a mapping or string reference."
        )
