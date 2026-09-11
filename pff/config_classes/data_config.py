import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
import warnings

try:  # pragma: no cover - exercised indirectly via configuration loading
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml

try:  # pragma: no cover - optional dependency for downstream modules
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - torch is not required for configuration loading
    torch = None  # type: ignore

@dataclass
class SimpleMetaStudyConfig:
    """
    Minimal configuration for the synthetic (non-mechanistic) PK simulator.
    Used when `simple_mode=True` is detected in the YAML file.
    """

    simple_mode: bool = True

    # --- keep same naming as MetaStudyConfig for compatibility ---
    num_individuals: int = 16
    num_individuals_range: Tuple[int, int] = (16, 16)  # <== added to avoid downstream errors

    time_start: float = 0.0
    time_stop: float = 24.0
    time_num_steps: int = 40

    band_scale_range: Tuple[float, float] = (0.1, 0.3)
    baseline_range: Tuple[float, float] = (0.0, 0.1)
    decay_rate_range: Tuple[float, float] = (0.3, 0.6)
    p1: float = 0.5  # of runs use the exponential, 65% use the pulse
    num_peripherals_range: Tuple[int, int] = (1, 3)

    solver_method: str = "dummy"
    drug_id_options: List[str] = field(default_factory=lambda: ["DummyDrug"])

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "SimpleMetaStudyConfig":
        with open(file_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        cfg = cfg.get("meta_study", cfg)

        # Ensure backward compatibility if YAML only defines num_individuals
        if "num_individuals_range" not in cfg and "num_individuals" in cfg:
            n = cfg["num_individuals"]
            cfg["num_individuals_range"] = (n, n)

        return cls(**cfg)


@dataclass
class MetaStudyConfig:
    """
    This class contains the configuration for the compartment study.
    i.e. it specifies the parameters to sample the population which
    in turns will sample the individuals.
    """

    drug_id_options: List[str] = field(default_factory=lambda: ["Drug_A", "Drug_B", "Drug_C"])
    num_individuals_range: Tuple[int, int] = (20, 20)

    num_peripherals_range: Tuple[int, int] = (1, 3)
    log_k_a_mean_range: Tuple[float, float] = (-1.5, 1.5)
    log_k_a_std_range: Tuple[float, float] = (0.1, 0.5)
    k_a_tmag_range: Tuple[float, float] = (0.01, 0.1)
    k_a_tscl_range: Tuple[float, float] = (1.0, 5.0)
    log_k_e_mean_range: Tuple[float, float] = (-1.5, 1.5)
    log_k_e_std_range: Tuple[float, float] = (0.1, 0.5)
    k_e_tmag_range: Tuple[float, float] = (0.01, 0.1)
    k_e_tscl_range: Tuple[float, float] = (1.0, 5.0)
    log_V_mean_range: Tuple[float, float] = (-1.5, 1.5)
    log_V_std_range: Tuple[float, float] = (0.1, 0.5)
    V_tmag_range: Tuple[float, float] = (0.01, 0.1)
    V_tscl_range: Tuple[float, float] = (1.0, 5.0)
    log_k_1p_mean_range: Tuple[float, float] = (-1.5, 1.5)
    log_k_1p_std_range: Tuple[float, float] = (0.1, 0.5)
    k_1p_tmag_range: Tuple[float, float] = (0.01, 0.1)
    k_1p_tscl_range: Tuple[float, float] = (1.0, 5.0)
    log_k_p1_mean_range: Tuple[float, float] = (-1.5, 1.5)
    log_k_p1_std_range: Tuple[float, float] = (0.1, 0.5)
    k_p1_tmag_range: Tuple[float, float] = (0.01, 0.1)
    k_p1_tscl_range: Tuple[float, float] = (1.0, 5.0)

    # Parameters for observation noise
    rel_ruv_range: Tuple[float, float] = (0.05, 0.3)

    # Parameters for generating time_points
    time_start: float = 0.0
    time_stop: float = 10.0
    time_num_steps: int = 100

    # parameters for solver
    solver_method: str = "rk4"

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "MetaStudyConfig":
        """Instantiate the meta-study configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict) and "meta_study" in config_dict:
            config_dict = config_dict.get("meta_study") or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected 'meta_study' section in YAML to be a mapping.")

        return cls(**config_dict)


@dataclass
class ObservationsConfig:
    """High-level knobs describing an observation strategy."""

    # ``None`` (e.g. YAML ``type: null``) is treated as the legacy
    # ``pk_peak_half_life`` strategy by the observation factory.
    type: Optional[str] = "pk_peak_half_life"
    add_rem: bool = True
    split_past_future: bool = False
    min_past: Optional[int] = None
    max_past: Optional[int] = None
    max_num_obs: int = 10
    empirical_number_of_obs: int = 2
    # When True, entries at non-positive times are excluded from sampled
    # observations (e.g. concentration at dosing time t=0).
    drop_time_zero_observations: bool = True
    # For ``fixed_regular_grid`` only: skip this many solver-grid steps before
    # building the deterministic evenly spaced subset. ``0`` preserves the
    # legacy behaviour where the first selected index is exactly ``t=0``.
    fixed_grid_start_index: int = 0

    # Strategy specific semantic controls (do not affect tensor shapes directly)
    past_time_ratio: float = 0.1  # Used by random strategies with fixed boundary
    # Sampling policy for split-past/future strategies:
    # - False: sample uniformly in [min_past, max_past]
    # - True: sample 0 with prob. 0.5, otherwise sample uniformly
    #   in [max(1, min_past), max_past]
    generative_bias: bool = False

    def __post_init__(self):
        if not isinstance(self.generative_bias, bool):
            raise ValueError("generative_bias must be a boolean (true/false)")
        if self.fixed_grid_start_index < 0:
            raise ValueError("fixed_grid_start_index must be non-negative")

        if self.split_past_future:
            if self.min_past is None or self.max_past is None:
                raise ValueError(
                    "min_past and max_past must be provided when split_past_future=True"
                )
            if self.min_past < 0:
                raise ValueError("min_past must be non-negative")
            if self.max_past < self.min_past:
                raise ValueError("max_past must be >= min_past")
            self.add_rem = True

    @classmethod
    def from_yaml(
        cls,
        file_path: Union[str, os.PathLike],
        section: Optional[str] = None,
    ) -> "ObservationsConfig":
        """Instantiate an observation configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected YAML content to be a mapping.")

        if section is not None:
            if section not in config_dict:
                raise KeyError(f"Section '{section}' not found in YAML file '{file_path}'.")
            config_dict = config_dict.get(section) or {}
        else:
            potential_sections = [
                key for key in ("context_observations", "target_observations") if key in config_dict
            ]
            if len(potential_sections) > 1:
                raise ValueError(
                    "Multiple observation sections found; specify which one to load using the 'section' argument."
                )
            if potential_sections:
                config_dict = config_dict.get(potential_sections[0]) or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected observation configuration to be provided as a mapping.")

        return cls(**config_dict)


@dataclass
class MixDataConfig:
    """
    Here we specify how do we construct the  mix databatch,
    i.e. if we treat as a the decoder variable one full path
         or if we treat as the decoder variable the future steps of a paht
    """

    test_empirical_datasets: List[str] = field(default_factory=lambda: ["cesarali/lenuzza-2016"])
    # Deprecated fields removed (unused in current training flow):
    # pretraining_*, val_protocol, test_protocol, split_strategy, split_seed.
    evaluate_prediction_steps_past: int = 4  # lenght of past is kept fix for evaluation
    sample_size_for_generative_evaluation_val: Optional[int] = None
    # Number of generative samples (S) used for validation-time callback
    # evaluation (new individuals and VPC/NPDE consumers). Defaults to 10.
    sample_size_for_generative_evaluation_end_of_training: Optional[int] = None
    # Number of generative samples (S) used for end-of-training callback
    # evaluation (empirical end hooks). Defaults to 500.
    sample_size_for_generative_evaluation: Optional[int] = None
    # Deprecated legacy alias for both values above. When set and the new
    # fields are not provided, the same value is applied to both stages.
    # Value/time normalization flags consumed by PKScaler.
    # Precedence for value scaling:
    # 1) log_and_z=True -> "log_and_z"
    # 2) log_and_max=True -> "log_and_max"
    # 3) log_transform=True -> "log"
    # 4) z_score_normalization=True -> "zscore"
    # 5) normalize_by_max=True -> "max"
    # 6) otherwise -> "none"
    z_score_normalization: bool = False
    # Explicit single switch for log + z-score scaling in PKScaler.
    log_and_z: bool = False
    # Explicit single switch for log + max scaling in PKScaler.
    log_and_max: bool = False
    normalize_by_max: bool = True
    normalize_time: bool = True

    n_of_permutations: int = 1
    n_of_databatches: Optional[int] = None  # deprecated alias
    n_of_target_individuals: int = 1  # ignored for LOO/NO_TARGET
    # Log-only transform flag consumed by PKScaler (value_method="log").
    # This is no longer handled in the dataset/datamodule path.
    log_transform: bool = False  # Matches node-pk-1804.yaml

    store_in_tempfile: bool = False  # When True dataset is generated and saved to a temporary file
    keep_tempfile: bool = False  # Don't delete the temporary file on cleanup
    recreate_tempfile: bool = False  # Regenerate file even if it already exists

    tempfile_path: Tuple[str, str] = (
        "preprocessed",
        "simulated_ou_as_rates.tr",
    )

    tqdm_progress: bool = False  # Show progress bar when generating temp files
    # DATA SIZES
    train_size: int = 1000
    val_size: int = 100
    test_size: int = 100

    def __post_init__(self) -> None:
        if self.n_of_databatches is not None and self.n_of_permutations == 1:
            self.n_of_permutations = self.n_of_databatches
            warnings.warn(
                "n_of_databatches is deprecated; use n_of_permutations",
                DeprecationWarning,
            )
        legacy_sample_size = self.sample_size_for_generative_evaluation
        if (
            self.sample_size_for_generative_evaluation_val is None
            and legacy_sample_size is not None
        ):
            self.sample_size_for_generative_evaluation_val = int(legacy_sample_size)
        if (
            self.sample_size_for_generative_evaluation_end_of_training is None
            and legacy_sample_size is not None
        ):
            self.sample_size_for_generative_evaluation_end_of_training = int(legacy_sample_size)

        if self.sample_size_for_generative_evaluation_val is None:
            self.sample_size_for_generative_evaluation_val = 10
        if self.sample_size_for_generative_evaluation_end_of_training is None:
            self.sample_size_for_generative_evaluation_end_of_training = 500

        if int(self.sample_size_for_generative_evaluation_val) < 1:
            raise ValueError("sample_size_for_generative_evaluation_val must be >= 1")
        if int(self.sample_size_for_generative_evaluation_end_of_training) < 1:
            raise ValueError("sample_size_for_generative_evaluation_end_of_training must be >= 1")

        self.sample_size_for_generative_evaluation_val = int(
            self.sample_size_for_generative_evaluation_val
        )
        self.sample_size_for_generative_evaluation_end_of_training = int(
            self.sample_size_for_generative_evaluation_end_of_training
        )

        if legacy_sample_size is not None:
            warnings.warn(
                "sample_size_for_generative_evaluation is deprecated; use "
                "sample_size_for_generative_evaluation_val and "
                "sample_size_for_generative_evaluation_end_of_training",
                DeprecationWarning,
            )
        if self.n_of_permutations < 1:
            raise ValueError("n_of_permutations must be >= 1")

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "MixDataConfig":
        """Instantiate the mix-data configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict):
            for key in ("mix_data", "mix_data_config"):
                if key in config_dict and isinstance(config_dict[key], dict):
                    config_dict = config_dict[key]
                    break

        if not isinstance(config_dict, dict):
            raise TypeError("Expected mix data configuration to be provided as a mapping.")

        return cls(**config_dict)


@dataclass
class MetaDosingConfig:
    """
    Config for specifying meta dosing information.
    """

    num_individuals: int = 10
    same_route: bool = True
    logdose_mean_range: Tuple[float, float] = (-2, 2)
    logdose_std_range: Tuple[float, float] = (0.1, 0.5)
    route_options: List[str] = field(default_factory=lambda: ["oral", "iv"])
    route_weights: List[float] = field(default_factory=lambda: [0.8, 0.2])
    time: float = 0.0

    @classmethod
    def from_yaml(cls, file_path: Union[str, os.PathLike]) -> "MetaDosingConfig":
        """Instantiate the meta-dosing configuration from a YAML file."""

        with open(file_path, "r", encoding="utf-8") as handle:
            config_dict = yaml.safe_load(handle) or {}

        if isinstance(config_dict, dict) and "dosing" in config_dict:
            config_dict = config_dict.get("dosing") or {}

        if not isinstance(config_dict, dict):
            raise TypeError("Expected 'dosing' section in YAML to be a mapping.")

        return cls(**config_dict)

@dataclass
class MetaDosingWithDurationConfig(MetaDosingConfig):
    """
    Config for specifying meta dosing information including iv infusions.
    """

    route_duration_weights:  Dict[str,float] = field(default_factory=lambda: {"oral": 0.0, "iv": 0.5}) # no duration for oral, 50% chance of infusion for iv
    duration_range: Tuple[float, float] = (0.5, 2.0)  # Duration of infusion; 0.0 means bolus

@dataclass
class DosingConfig:
    """
    Config for specifying dosing information. For now, it just holds the amount D of a single oral dose
    given at time t = 0.
    """

    dose: float = 1.0
    route: str = "oral"
    time: float = 0.0


@dataclass
class DosingWithDurationConfig:
    """
    Config for specifying dosing information. It holds the amount D of a dose
    given at time t = 0, optionally with an infusion duration.
    """

    dose: float = 1.0
    route: str = "oral"
    time: float = 0.0
    duration: float = 0.0  # Duration of infusion; 0.0 means bolus
