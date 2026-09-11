"""Train PK models using the simplified Lightning experiment runner."""

from __future__ import annotations

import comet_ml  # noqa: F401  # Import early so Comet can patch downstream ML imports.
import os
from pathlib import Path

from pff import config_dir
from pff.config_classes.data_config import MetaStudyConfig, SimpleMetaStudyConfig

try:  # pragma: no cover - exercised indirectly via CLI configuration loading
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    from pff.config_classes import yaml_fallback as yaml
from pff.models import get_model_config
from pff.training.basic_experiment import BasicLightningExperiment
from pff.training.utils_parsing_args import (
    parse_args,
    parse_override_list,
    update_config_with_overrides,
)


def _resolve_meta_study_path(config_path: str, meta_study_name: str) -> Path:
    """Resolve meta-study path relative to the experiment config directory."""
    candidate = Path(meta_study_name).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(config_path).resolve().parent / candidate


def _load_meta_study_config(meta_study_path: Path):
    """Load MetaStudyConfig or SimpleMetaStudyConfig from YAML content."""
    with meta_study_path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if not isinstance(parsed, dict):
        raise TypeError(f"Expected a mapping in meta-study YAML: {meta_study_path}")

    meta_payload = parsed.get("meta_study", parsed)
    if not isinstance(meta_payload, dict):
        raise TypeError(f"Expected 'meta_study' section to be a mapping: {meta_study_path}")

    if bool(meta_payload.get("simple_mode", False)):
        return SimpleMetaStudyConfig(**meta_payload)
    return MetaStudyConfig(**meta_payload)


def train() -> None:
    """Load configuration, apply overrides, and launch training."""

    default_yaml = os.path.join(
        config_dir,
        "experiment_configs",
        "UAI",
        "Rebuttal",
        "Potsdam",
        "aicme-t-pk-cluster",
        "base.yaml",
    )

    args = parse_args(default_yaml)

    model_cfg = get_model_config(args.config_path)
    if args.meta_study_name:
        if not hasattr(model_cfg, "meta_study"):
            raise TypeError(
                "The --meta_study_name option requires a config that defines `meta_study`."
            )
        meta_study_path = _resolve_meta_study_path(args.config_path, args.meta_study_name)
        if not meta_study_path.exists():
            raise FileNotFoundError(f"Meta-study file not found: {meta_study_path}")
        model_cfg.meta_study = _load_meta_study_config(meta_study_path)

    overrides = parse_override_list(args.override)
    if overrides:
        model_cfg = update_config_with_overrides(model_cfg, overrides)

    experiment = BasicLightningExperiment.from_config(exp_config=model_cfg)
    experiment.train()


if __name__ == "__main__":
    train()
