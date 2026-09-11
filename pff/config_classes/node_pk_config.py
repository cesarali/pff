import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml  # type: ignore

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


@dataclass
class EncoderDecoderNetworkConfig:
    """
    Configuration for the encoder-decoder network.
    """

    # Encoder configuration
    individual_encoder_name: str = "RNNContextEncoder"
    time_obs_encoder_hidden_dim: int = 200
    time_obs_encoder_output_dim: int = 200
    rnn_individual_encoder_number_of_layers: int = 2
    individual_encoder_number_of_heads: int = 4
    encoder_rnn_hidden_dim: int = 128
    input_encoding_hidden_dim: int = 128
    zi_latent_dim: int = 200
    z_s_latent_dim: Optional[int] = None
    z_i_latent_dim: Optional[int] = None
    use_attention: bool = True
    use_self_attention: bool = False
    use_time_deltas: bool = True

    # Decoder configuration
    decoder_name: str = "RNNDecoder"
    decoder_num_layers: int = 2
    decoder_attention_layers: int = 2
    decoder_hidden_dim: int = 128
    decoder_rnn_hidden_dim: int = 200
    rnn_decoder_number_of_layers: int = 4
    node_step: bool = True
    exclusive_node_step: bool = False
    cov_proj_dim: int = 16  # p in the paper
    ignore_logvar: bool = True  # sampling

    # Aggregator
    aggregator_type: str = "attention"  # attention, mean
    aggregator_num_heads: int = 8

    # Control reconstruction vs prediction losses
    prediction_only: bool = False
    reconstruction_only: bool = False

    # Deterministic study latent (disable sampling)
    study_latent_deterministic: bool = False

    # Deterministic individual latent for prediction
    prediction_latent_deterministic: bool = False

    # How to combine study and individual latents
    combine_latent_mode: str = "mlp"  # Options: "mlp", "sum"

    # MLP configurations (used in init_hidden, output heads, drift)
    init_hidden_num_layers: int = 2
    output_head_num_layers: int = 2
    drift_num_layers: int = 3
    dropout: float = 0.1
    activation: str = "ReLU"  # For init/logvar/mean
    drift_activation: str = "Tanh"
    norm: str = "layer"  # Options: "layer", "batch", None

    # Loss
    loss_name: str = "nll"  # Options: "nll", "log_nll", "rmse", mv_nll

    # latent node pk
    kl_weight: float = 1.0

    # KL regularisation flags
    use_kl_s: bool = True
    use_kl_i: bool = True
    use_kl_i_np: bool = True
    use_kl_init: bool = True
    use_invariance_loss: bool = True

    # Optional scaling for dosing amount inputs (route types remain unscaled)
    scale_dosing_amounts: bool = True

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "EncoderDecoderNetworkConfig":
        """Instantiate the network configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict) and "network" in config_dict:
            config_dict = config_dict.get("network") or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected 'network' section in YAML to be a mapping.")

        return cls(**config_dict)


@dataclass
class NodePKExperimentConfig:
    """Experiment configuration for NodePK-family models."""

    experiment_type: str = "nodepk"
    name_str: str = "NodePK"
    comet_ai_key: str = None
    experiment_name: str = "node_pk_compartments"
    hugging_face_token: str = None
    upload_to_hf_hub: bool = True
    hf_model_name: str = "NodePK_test"
    hf_model_card_path: Tuple[str, str, str] = ("hf_model_card", "CVAE_Readme.md")

    tags: List[str] = field(default_factory=lambda: ["node-pk", "B-0"])
    experiment_indentifier: str = None
    my_results_path: str = None
    experiment_dir: str = None
    verbose: bool = False
    run_index: int = 0
    debug_test: bool = False

    network: EncoderDecoderNetworkConfig = field(default_factory=EncoderDecoderNetworkConfig)
    mix_data: MixDataConfig = field(default_factory=MixDataConfig)

    context_observations: ObservationsConfig = field(default_factory=ObservationsConfig)
    target_observations: ObservationsConfig = field(default_factory=ObservationsConfig)

    meta_study: MetaStudyConfig = field(default_factory=MetaStudyConfig)
    dosing: MetaDosingConfig = field(default_factory=MetaDosingConfig)

    train: TrainingConfig = field(default_factory=TrainingConfig)

    @staticmethod
    def from_yaml(file_path: str) -> "NodePKExperimentConfig":
        """Initializes the class from a YAML file.

        Supports both monolithic experiment YAML files as well as files that
        reference dedicated data, training, and model configuration YAMLs.
        """

        with open(file_path, "r") as file:
            config_dict = yaml.load(file, Loader=TupleSafeLoader) or {}

        exp_type = None
        if isinstance(config_dict, dict):
            exp_type = config_dict.get("experiment_type")
        if exp_type is not None and str(exp_type).lower() != "nodepk":
            raise ValueError(
                f"Expected experiment_type 'nodepk' for NodePKExperimentConfig, got {exp_type!r}."
            )

        base_dir = os.path.dirname(os.path.abspath(file_path))

        data_cfg_dict = (
            NodePKExperimentConfig._load_ref_yaml(config_dict.get("data_config"), base_dir) or {}
        )
        training_cfg_dict = (
            NodePKExperimentConfig._load_ref_yaml(config_dict.get("training_config"), base_dir)
            or {}
        )
        model_cfg_dict = (
            NodePKExperimentConfig._load_ref_yaml(config_dict.get("model_config"), base_dir) or {}
        )


        observations_section = NodePKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "observations_config"
        )
        if observations_section is None:
            observations_section = NodePKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "observations_config"
            )
        if observations_section is not None:
            context_observations_base = observations_section.get("context_observations")
            target_observations_base = observations_section.get("target_observations")
        else:
            context_observations_base = data_cfg_dict.get("context_observations")
            target_observations_base = data_cfg_dict.get("target_observations")

        mix_data_section = NodePKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "mix_data_config"
        )
        if mix_data_section is None:
            mix_data_section = NodePKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "mix_data_config"
            )
        if mix_data_section is None:
            mix_data_section = data_cfg_dict.get("mix_data")

        meta_study_section = NodePKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_study_config"
        )
        if meta_study_section is None:
            meta_study_section = NodePKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_study_config"
            )
        meta_dosing_section = NodePKExperimentConfig._resolve_config_section(
            config_dict, base_dir, "meta_dosing_config"
        )
        if meta_dosing_section is None:
            meta_dosing_section = NodePKExperimentConfig._resolve_config_section(
                data_cfg_dict, base_dir, "meta_dosing_config"
            )

        meta_study_base = NodePKExperimentConfig._extract_config_mapping(
            meta_study_section, "meta_study"
        )
        if meta_study_base is None and meta_dosing_section is not None:
            meta_study_base = NodePKExperimentConfig._extract_config_mapping(
                meta_dosing_section, "meta_study"
            )
        if meta_study_base is None:
            meta_study_base = data_cfg_dict.get("meta_study")

        dosing_base = NodePKExperimentConfig._extract_config_mapping(meta_dosing_section, "dosing")
        if dosing_base is None:
            dosing_base = data_cfg_dict.get("dosing")

        mix_data_cfg = NodePKExperimentConfig._merge_dicts(
            mix_data_section, config_dict.get("mix_data")
        )
        context_obs_cfg = NodePKExperimentConfig._merge_dicts(
            context_observations_base, config_dict.get("context_observations")
        )
        target_obs_cfg = NodePKExperimentConfig._merge_dicts(
            target_observations_base, config_dict.get("target_observations")
        )
        meta_study_cfg = NodePKExperimentConfig._merge_dicts(
            meta_study_base, config_dict.get("meta_study")
        )
        dosing_cfg = NodePKExperimentConfig._merge_dicts(dosing_base, config_dict.get("dosing"))

        train_section = training_cfg_dict.get("train", training_cfg_dict)
        train_cfg = NodePKExperimentConfig._merge_dicts(train_section, config_dict.get("train"))

        network_section = model_cfg_dict.get("network", model_cfg_dict)
        network_cfg = NodePKExperimentConfig._merge_dicts(
            network_section, config_dict.get("network")
        )

        # -----------------------------------------------------------------
        # Choose MetaStudy class dynamically (simple vs full)
        # -----------------------------------------------------------------
        if meta_study_cfg.get("simple_mode", False):
            meta_study_instance = SimpleMetaStudyConfig(**meta_study_cfg)
        else:
            meta_study_instance = MetaStudyConfig(**meta_study_cfg)

        train_cfg = TrainingConfig._filter_kwargs(train_cfg)

        return NodePKExperimentConfig(
            experiment_type=str(config_dict.get("experiment_type", "nodepk")).lower(),
            name_str=config_dict.get("name_str", "ExampleModel"),
            tags=config_dict.get("tags", ["node-pk", "B-0"]),
            experiment_name=config_dict.get("experiment_name", "aicme_compartments"),
            experiment_indentifier=config_dict.get("experiment_indentifier", None),
            my_results_path=config_dict.get("my_results_path", None),
            experiment_dir=config_dict.get("experiment_dir", None),
            comet_ai_key=config_dict.get("comet_ai_key", None),
            hugging_face_token=config_dict.get("hugging_face_token", None),
            upload_to_hf_hub=config_dict.get("upload_to_hf_hub", True),
            hf_model_name=config_dict.get("hf_model_name", "NodePK_test"),
            hf_model_card_path=tuple(
                config_dict.get("hf_model_card_path", ("hf_model_card", "CVAE_Readme.md"))
            ),
            debug_test=config_dict.get("debug_test", False),
            network=EncoderDecoderNetworkConfig(**network_cfg),
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
                loaded = NodePKExperimentConfig._load_ref_yaml(ref_value, base_dir)
                return loaded or {}
            return section

        if isinstance(section, str):
            loaded = NodePKExperimentConfig._load_ref_yaml(section, base_dir)
            return loaded or {}

        raise TypeError(
            f"Expected configuration section '{key}' to be a mapping or string reference."
        )

    def to_yaml(self, file_path: str):
        """Saves the class to a YAML file."""
        with open(file_path, "w") as file:
            yaml.dump(asdict(self), file, default_flow_style=False)


NodePKConfig = NodePKExperimentConfig


class HFNodePKConfig(PretrainedConfig):
    """
    HF config wrapping NodePKConfig plus tracked metrics.

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

    model_type = "node_pk"

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
    def from_nodepk(cls, nodepk_cfg, **tracked_best: float) -> "HFNodePKConfig":
        """
        tracked_best: e.g. val_rmse=..., val_nll=..., val_crps=...
        """
        cfg_dict = asdict(nodepk_cfg)
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
