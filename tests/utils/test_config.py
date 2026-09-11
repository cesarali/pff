from pff import config_dir
from pff.config_classes.data_config import (
    MetaDosingConfig,
    MetaStudyConfig,
    MixDataConfig,
    ObservationsConfig,
)
from pff.config_classes.node_pk_config import (
    EncoderDecoderNetworkConfig,
    NodePKExperimentConfig,
)
from pff.config_classes.training_config import TrainingConfig


def test_sub_configs_load_directly_from_yaml():
    """Each sub-configuration should instantiate from its standalone YAML file."""

    experiment_dir = config_dir / "experiment_configs" / "node-pk"

    network_cfg = EncoderDecoderNetworkConfig.from_yaml(
        experiment_dir / "base-homogeneous.model.yaml"
    )
    assert isinstance(network_cfg, EncoderDecoderNetworkConfig)

    mix_data_cfg = MixDataConfig.from_yaml(experiment_dir / "base-homogeneous.mix_data.yaml")
    assert isinstance(mix_data_cfg, MixDataConfig)

    observations_path = experiment_dir / "base-homogeneous.observations.yaml"
    context_cfg = ObservationsConfig.from_yaml(observations_path, section="context_observations")
    target_cfg = ObservationsConfig.from_yaml(observations_path, section="target_observations")
    assert isinstance(context_cfg, ObservationsConfig)
    assert isinstance(target_cfg, ObservationsConfig)

    meta_study_cfg = MetaStudyConfig.from_yaml(experiment_dir / "base-homogeneous.meta_study.yaml")
    assert isinstance(meta_study_cfg, MetaStudyConfig)

    dosing_cfg = MetaDosingConfig.from_yaml(experiment_dir / "base-homogeneous.dosing.yaml")
    assert isinstance(dosing_cfg, MetaDosingConfig)

    training_cfg = TrainingConfig.from_yaml(experiment_dir / "base-homogeneous.training.yaml")
    assert isinstance(training_cfg, TrainingConfig)


def test_full_nodepk_config_loads_from_yaml():
    """Ensure the entire NodePKConfig can be instantiated from a full YAML experiment file."""
    experiment_dir = config_dir / "experiment_configs" / "node-pk"
    full_yaml_path = experiment_dir / "base-homogeneous.yaml"

    cfg = NodePKExperimentConfig.from_yaml(full_yaml_path)

    # Sanity checks for top-level fields
    assert isinstance(cfg, NodePKExperimentConfig)
    assert cfg.name_str == "AICMEPK"
    assert isinstance(cfg.network, EncoderDecoderNetworkConfig)
    assert isinstance(cfg.mix_data, MixDataConfig)
    assert isinstance(cfg.context_observations, ObservationsConfig)
    assert isinstance(cfg.target_observations, ObservationsConfig)
    assert isinstance(cfg.meta_study, MetaStudyConfig)
    assert isinstance(cfg.dosing, MetaDosingConfig)
    assert isinstance(cfg.train, TrainingConfig)

    # Optional: confirm nested values exist
    assert cfg.train.epochs > 0
    assert hasattr(cfg.network, "individual_encoder_name")
    assert hasattr(cfg.meta_study, "time_stop")
    assert hasattr(cfg.dosing, "route_options")
