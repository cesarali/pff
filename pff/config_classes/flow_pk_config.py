import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

try:  # pragma: no cover - exercised indirectly via configuration loading
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml

try:  # pragma: no cover - optional dependency for HF integration
    from transformers import PretrainedConfig  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - allow configuration utilities without transformers

    class PretrainedConfig:  # type: ignore
        def __init__(self, **kwargs):
            super().__init__()

from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    MixDataConfig,
    ObservationsConfig,
    SimpleMetaStudyConfig,
)
from pff.config_classes.source_process_config import SourceProcessConfig
from pff.config_classes.training_config import TrainingConfig
from pff.config_classes.utils import TupleSafeLoader


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.inf
    # guard against NaN
    if math.isnan(v):
        return math.inf
    return v


def _raise_flowpk_network_migration() -> None:
    raise ValueError(
        "FlowPK configs no longer accept a 'network' section. "
        "Please rename 'network' to 'vector_field' and set 'experiment_type: flowpk' in your YAML."
    )


@dataclass
class VectorFieldPKConfig:
    """Configuration for the transformer vector field used by FlowPK."""

    # Transformer vector field configuration
    hidden_dim: int = 64
    fourier_modes: int = 16
    use_spectral_qkv: bool = False
    time_fourier_max_freq: int = 64
    encoder_num_heads: int = 4
    decoder_num_heads: int = 4
    encoder_attention_layers: int = 2
    decoder_attention_layers: int = 2
    dropout: float = 0.0

    # # Latent/conditioning settings required by the vector field implementation
    cov_proj_dim: int = 16  # p in the paper
    combine_latent_mode: str = "mlp"  # Options: "mlp", "sum"
    zi_latent_dim: int = 200

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "VectorFieldPKConfig":
        """Instantiate the vector field configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict) and "network" in config_dict:
            _raise_flowpk_network_migration()

        if isinstance(config_dict, dict) and "vector_field" in config_dict:
            config_dict = config_dict.get("vector_field") or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected 'vector_field' section in YAML to be a mapping.")

        return cls(**config_dict)


@dataclass
class FlowPKExperimentConfig:
    """Experiment configuration for FlowPK (vector field only)."""

    experiment_type: str = "flowpk"
    name_str: str = "FlowPK"
    comet_ai_key: str = None
    experiment_name: str = "flow_pk_compartments"
    hugging_face_token: str = None
    upload_to_hf_hub: bool = True
    hf_model_name: str = "FlowPK_test"
    hf_model_card_path: Tuple[str, str, str] = ("hf_model_card", "CVAE_Readme.md")

    tags: List[str] = field(default_factory=lambda: ["flow-pk", "B-0"])
    experiment_indentifier: str = None
    my_results_path: str = None
    experiment_dir: str = None
    verbose: bool = False
    run_index: int = 0
    debug_test: bool = False
    # Default Euler integration steps used by FlowPK sampling when callers
    # do not provide ``num_steps`` explicitly (for example VPC callbacks).
    flow_num_steps: int = 50

    vector_field: VectorFieldPKConfig = field(default_factory=VectorFieldPKConfig)
    source_process: SourceProcessConfig = field(default_factory=SourceProcessConfig)
    mix_data: MixDataConfig = field(default_factory=MixDataConfig)

    context_observations: ObservationsConfig = field(default_factory=ObservationsConfig)
    target_observations: ObservationsConfig = field(default_factory=ObservationsConfig)

    meta_study: MetaStudyConfig = field(default_factory=MetaStudyConfig)
    dosing: MetaDosingConfig = field(default_factory=MetaDosingConfig)

    train: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_yaml(file_path: str) -> "FlowPKExperimentConfig":
        """Initializes the class from a YAML file.

        Supports both monolithic experiment YAML files as well as files that
        reference dedicated data, training, and model configuration YAMLs.
        """

        with open(file_path, "r") as file:
            config_dict = yaml.load(file, Loader=TupleSafeLoader) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected experiment YAML to be a mapping.")

        exp_type = config_dict.get("experiment_type")
        if exp_type is not None and str(exp_type).lower() != "flowpk":
            raise ValueError(
                f"Expected experiment_type 'flowpk' for FlowPKExperimentConfig, got {exp_type!r}."
            )

        if "network" in config_dict:
            _raise_flowpk_network_migration()

        base_dir = os.path.dirname(os.path.abspath(file_path))

        data_cfg_dict = (
            FlowPKExperimentConfig._load_ref_yaml(config_dict.get("data_config"), base_dir) or {}
        )
        training_cfg_dict = (
            FlowPKExperimentConfig._load_ref_yaml(config_dict.get("training_config"), base_dir)
            or {}
        )
        model_cfg_dict = (
            FlowPKExperimentConfig._load_ref_yaml(config_dict.get("model_config"), base_dir) or {}
        )

        if isinstance(model_cfg_dict, dict) and "network" in model_cfg_dict:
            _raise_flowpk_network_migration()

        observations_section = FlowPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "observations_config"
        )
        if observations_section is None:
            observations_section = FlowPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "observations_config"
            )
        if observations_section is not None:
            context_observations_base = observations_section.get("context_observations")
            target_observations_base = observations_section.get("target_observations")
        else:
            context_observations_base = data_cfg_dict.get("context_observations")
            target_observations_base = data_cfg_dict.get("target_observations")

        mix_data_section = FlowPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "mix_data_config"
        )
        if mix_data_section is None:
            mix_data_section = FlowPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "mix_data_config"
            )
        if mix_data_section is None:
            mix_data_section = data_cfg_dict.get("mix_data")

        meta_study_section = FlowPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_study_config"
        )
        if meta_study_section is None:
            meta_study_section = FlowPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_study_config"
            )
        meta_dosing_section = FlowPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_dosing_config"
        )
        if meta_dosing_section is None:
            meta_dosing_section = FlowPKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_dosing_config"
            )

        meta_study_base = FlowPKExperimentConfig._extract_config_mapping(
            meta_study_section, "meta_study"
        )
        if meta_study_base is None and meta_dosing_section is not None:
            meta_study_base = FlowPKExperimentConfig._extract_config_mapping(
                meta_dosing_section, "meta_study"
            )
        if meta_study_base is None:
            meta_study_base = data_cfg_dict.get("meta_study")

        dosing_base = FlowPKExperimentConfig._extract_config_mapping(meta_dosing_section, "dosing")
        if dosing_base is None:
            dosing_base = data_cfg_dict.get("dosing")

        mix_data_inline = config_dict.get("mix_data")
        if mix_data_inline is not None and not isinstance(mix_data_inline, dict):
            raise TypeError("Expected 'mix_data' section in experiment YAML to be a mapping.")

        # Backward compatibility: allow mix-data keys at experiment top-level.
        # Nested `mix_data:` values take precedence over these legacy top-level keys.
        legacy_mix_data_inline = {
            field_meta.name: config_dict[field_meta.name]
            for field_meta in fields(MixDataConfig)
            if field_meta.name in config_dict
        }
        merged_mix_data_inline = dict(legacy_mix_data_inline)
        if isinstance(mix_data_inline, dict):
            merged_mix_data_inline.update(mix_data_inline)

        mix_data_cfg = FlowPKExperimentConfig._merge_dicts(
            mix_data_section, merged_mix_data_inline
        )
        context_obs_cfg = FlowPKExperimentConfig._merge_dicts(
            context_observations_base, config_dict.get("context_observations")
        )
        target_obs_cfg = FlowPKExperimentConfig._merge_dicts(
            target_observations_base, config_dict.get("target_observations")
        )
        meta_study_cfg = FlowPKExperimentConfig._merge_dicts(
            meta_study_base, config_dict.get("meta_study")
        )
        dosing_cfg = FlowPKExperimentConfig._merge_dicts(dosing_base, config_dict.get("dosing"))

        train_section = training_cfg_dict.get("train", training_cfg_dict)
        train_cfg = FlowPKExperimentConfig._merge_dicts(train_section, config_dict.get("train"))

        vector_field_section = model_cfg_dict.get("vector_field", model_cfg_dict)
        if isinstance(vector_field_section, dict) and "network" in vector_field_section:
            _raise_flowpk_network_migration()
        vector_field_cfg = FlowPKExperimentConfig._merge_dicts(
            vector_field_section, config_dict.get("vector_field")
        )

        source_section = FlowPKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "source_config"
        )
        if source_section is None:
            source_section = FlowPKExperimentConfig._resolve_config_section(
                model_cfg_dict, base_dir, "source_config"
            )
        if source_section is None:
            source_section = model_cfg_dict.get("source_process") or model_cfg_dict.get("noise_model")
        source_section = FlowPKExperimentConfig._extract_config_mapping(
            source_section, "source_process"
        )
        if isinstance(source_section, dict) and "noise_model" in source_section:
            source_section = source_section.get("noise_model")

        source_cfg = FlowPKExperimentConfig._merge_dicts(
            source_section, config_dict.get("source_process")
        )

        # -----------------------------------------------------------------
        # Choose MetaStudy class dynamically (simple vs full)
        # -----------------------------------------------------------------
        if meta_study_cfg.get("simple_mode", False):
            meta_study_instance = SimpleMetaStudyConfig(**meta_study_cfg)
        else:
            meta_study_instance = MetaStudyConfig(**meta_study_cfg)

        train_cfg = TrainingConfig._filter_kwargs(train_cfg)

        return FlowPKExperimentConfig(
            experiment_type=str(config_dict.get("experiment_type", "flowpk")).lower(),
            name_str=config_dict.get("name_str", "FlowPK"),
            tags=config_dict.get("tags", ["flow-pk", "B-0"]),
            experiment_name=config_dict.get("experiment_name", "flow_pk_compartments"),
            experiment_indentifier=config_dict.get("experiment_indentifier", None),
            my_results_path=config_dict.get("my_results_path", None),
            experiment_dir=config_dict.get("experiment_dir", None),
            comet_ai_key=config_dict.get("comet_ai_key", None),
            hugging_face_token=config_dict.get("hugging_face_token", None),
            upload_to_hf_hub=config_dict.get("upload_to_hf_hub", True),
            hf_model_name=config_dict.get("hf_model_name", "FlowPK_test"),
            hf_model_card_path=tuple(
                config_dict.get("hf_model_card_path", ("hf_model_card", "CVAE_Readme.md"))
            ),
            debug_test=config_dict.get("debug_test", False),
            flow_num_steps=int(config_dict.get("flow_num_steps", 50)),
            vector_field=VectorFieldPKConfig(**vector_field_cfg),
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
                loaded = FlowPKExperimentConfig._load_ref_yaml(ref_value, base_dir)
                return loaded or {}
            return section

        if isinstance(section, str):
            loaded = FlowPKExperimentConfig._load_ref_yaml(section, base_dir)
            return loaded or {}

        raise TypeError(
            f"Expected configuration section '{key}' to be a mapping or string reference."
        )

    def to_yaml(self, file_path: str):
        """Saves the class to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file, default_flow_style=False)


class HFFlowPKConfig(PretrainedConfig):
    """
    HF config wrapping FlowPKExperimentConfig plus tracked metrics.

    Canonical storage:
        self.tracking: dict with shape
            {
              "best": { "<metric_name>": {"value": float, "step": int|None, "epoch": int|None} },
              "meta": { ...optional... }
            }

    Backward compat:
        - Accepts legacy keys like best_val_loss / best_val_rmse.
        - Mirrors best["val_rmse"] to `best_val_loss` if you still use that elsewhere.
    """

    model_type = "flow_pk"

    def __init__(self, **kwargs):
        # --- extract tracking / legacy keys before super().__init__ ---
        tracking = kwargs.pop("tracking", None)

        # legacy keys (accept either; normalize into tracking)
        legacy_best_val_loss = kwargs.pop("best_val_loss", None)
        legacy_best_val_rmse = kwargs.pop("best_val_rmse", None)

        super().__init__(**kwargs)

        # copy remaining config fields
        for k, v in kwargs.items():
            setattr(self, k, v)

        # initialize tracking
        if tracking is None or not isinstance(tracking, dict):
            tracking = {"best": {}, "meta": {}}
        tracking.setdefault("best", {})
        tracking.setdefault("meta", {})
        self.tracking: Dict[str, Any] = tracking

        # fold legacy into canonical schema if present
        legacy = legacy_best_val_loss if legacy_best_val_loss is not None else legacy_best_val_rmse
        if legacy is not None:
            # choose a canonical metric name; I'd recommend "val_rmse" if that’s what it is.
            self.set_best("val_rmse", legacy)

        # optional alias for older codepaths
        self._sync_legacy_aliases()

    # --------- public API ----------
    def set_best(
        self,
        metric_name: str,
        value: Any,
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        v = _to_float(value)
        self.tracking["best"][metric_name] = {"value": v, "step": step, "epoch": epoch}
        self._sync_legacy_aliases()

    def get_best(self, metric_name: str, default: float = math.inf) -> float:
        d = self.tracking.get("best", {}).get(metric_name)
        if not d:
            return float(default)
        return _to_float(d.get("value", default))

    def is_better(
        self,
        metric_name: str,
        candidate_value: Any,
        *,
        higher_is_better: bool = False,
    ) -> bool:
        cand = _to_float(candidate_value)
        best = self.get_best(metric_name, default=(-math.inf if higher_is_better else math.inf))
        return cand > best if higher_is_better else cand < best

    def update_if_better(
        self,
        metric_name: str,
        candidate_value: Any,
        *,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        higher_is_better: bool = False,
    ) -> bool:
        if self.is_better(metric_name, candidate_value, higher_is_better=higher_is_better):
            self.set_best(metric_name, candidate_value, step=step, epoch=epoch)
            return True
        return False

    # --------- construction ----------
    @classmethod
    def from_flowpk(cls, flowpk_cfg, **tracked_best: float) -> "HFFlowPKConfig":
        """
        tracked_best: e.g. val_rmse=..., val_nll=..., val_crps=...
        """
        cfg_dict = asdict(flowpk_cfg)
        cfg = cls(**cfg_dict)
        for k, v in tracked_best.items():
            cfg.set_best(k, v)
        return cfg

    # --------- internal ----------
    def _sync_legacy_aliases(self) -> None:
        """
        Keep a legacy scalar field for older code that expects `best_val_loss`.
        Here we mirror it to `best["val_rmse"]` by convention.
        """
        # if val_rmse exists, mirror it; otherwise inf
        self.best_val_loss = self.get_best("val_rmse", default=math.inf)


FlowPKConfig = FlowPKExperimentConfig
