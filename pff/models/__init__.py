"""Public FlowPK model registry and configuration loading utilities."""

from pathlib import Path
from typing import Type

try:  # pragma: no cover - exercised indirectly via configuration loading
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml

from pff.config_classes.flow_pk_config import FlowPKExperimentConfig
from pff.config_classes.utils import TupleSafeLoader
from pff.models.amortized_inference.flows_pk import FlowPK


def get_model_class(
    model_config: FlowPKExperimentConfig | None = None,
    name_str: str | None = None,
) -> Type[FlowPK]:
    """Return the only supported model class, :class:`FlowPK`."""

    resolved_name = model_config.name_str if model_config is not None else name_str
    if resolved_name == "FlowPK":
        return FlowPK
    raise ValueError(f"PFF supports only the FlowPK model, got {resolved_name!r}.")


def get_model_config(yaml_path: str | Path) -> FlowPKExperimentConfig:
    """Load and validate a FlowPK experiment configuration from YAML."""

    with Path(yaml_path).open("r", encoding="utf-8") as file:
        config_dict = yaml.load(file, Loader=TupleSafeLoader) or {}

    if not isinstance(config_dict, dict):
        raise TypeError("Expected experiment YAML to be a mapping.")

    experiment_type = str(config_dict.get("experiment_type", "")).lower()
    name_str = config_dict.get("name_str")
    if experiment_type != "flowpk" or name_str not in (None, "FlowPK"):
        raise ValueError(
            "PFF requires 'experiment_type: flowpk' and, when provided, "
            f"'name_str: FlowPK'; got experiment_type={experiment_type!r}, name_str={name_str!r}."
        )
    if "network" in config_dict:
        raise ValueError(
            "FlowPK configs require a 'vector_field' section; rename the legacy 'network' section."
        )
    return FlowPKExperimentConfig.from_yaml(str(yaml_path))


__all__ = ["FlowPK", "get_model_class", "get_model_config"]
