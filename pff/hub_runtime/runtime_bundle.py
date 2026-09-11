"""Manual export path for consumer-facing Hugging Face runtime bundles."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Sequence

import torch
from huggingface_hub import HfApi, create_repo

from pff import config_dir, project_dir
from pff.hub_runtime.configuration_pff import PKHubConfig
from pff.hub_runtime.modeling_pff import PKHubModel
from pff.hub_runtime.runtime_contract import (
    build_runtime_config_payload,
    resolve_model_card_text,
    runtime_readme_text,
)

ROOT_CONFIGURATION_FILENAME = "configuration_pff.py"
ROOT_MODELING_FILENAME = "modeling_pff.py"
_HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")
_COMET_KEY_ASSIGNMENT_PATTERN = re.compile(r"(COMET_API_KEY\s*=\s*)(['\"]).*?\2")
_HF_KEY_ASSIGNMENT_PATTERN = re.compile(r"(HF_KEYS\s*=\s*)(['\"]).*?\2")


@dataclass
class RuntimeBundleArtifacts:
    """Return metadata for a staged runtime bundle."""

    bundle_dir: Path
    runtime_repo_id: str
    original_repo_id: Optional[str]
    readme_path: Path


def default_runtime_repo_id(experiment, *, suffix: str = "-runtime") -> str:
    """Resolve the default runtime bundle repo id for a loaded experiment."""

    if getattr(experiment, "exp_config", None) is None:
        raise RuntimeError("Experiment config is not loaded.")
    if getattr(experiment, "hf_token", None) is None:
        raise RuntimeError(
            "No Hugging Face token available. Set hugging_face_token in the config or KEYS.txt."
        )

    user = HfApi().whoami(token=experiment.hf_token)["name"]
    return f"{user}/{experiment.exp_config.hf_model_name}{suffix}"


def _default_original_repo_id(experiment) -> Optional[str]:
    """Infer the legacy/native Hub repo id if enough metadata is available."""

    if getattr(experiment, "exp_config", None) is None:
        return None
    if getattr(experiment, "hf_token", None) is None:
        return None
    user = HfApi().whoami(token=experiment.hf_token)["name"]
    return f"{user}/{experiment.exp_config.hf_model_name}"


def _validate_loaded_experiment(experiment) -> None:
    """Ensure the loaded experiment has the minimum state needed for manual export."""

    if getattr(experiment, "model", None) is None:
        raise RuntimeError("Experiment model is not loaded.")
    if getattr(experiment, "exp_config", None) is None:
        raise RuntimeError("Experiment config is not loaded.")
    if getattr(experiment, "experiment_dir", None) is None:
        raise RuntimeError("Experiment directory is required before pushing.")
    if getattr(experiment, "hf_token", None) is None:
        raise RuntimeError(
            "No Hugging Face token available. Set hugging_face_token in the config or KEYS.txt."
        )


def _copy_runtime_support_files(bundle_dir: Path) -> None:
    """Copy the local package and root remote-code entrypoints into the bundle."""

    package_src = project_dir / "pff"
    package_dst = bundle_dir / "pff"
    shutil.copytree(package_src, package_dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))

    root_config_src = package_src / "hub_runtime" / ROOT_CONFIGURATION_FILENAME
    root_modeling_src = package_src / "hub_runtime" / ROOT_MODELING_FILENAME
    shutil.copy2(root_config_src, bundle_dir / ROOT_CONFIGURATION_FILENAME)
    shutil.copy2(root_modeling_src, bundle_dir / ROOT_MODELING_FILENAME)

    for extra_name in ("requirements.txt", "LICENSE"):
        extra_src = project_dir / extra_name
        if extra_src.is_file():
            shutil.copy2(extra_src, bundle_dir / extra_name)

    _scrub_runtime_bundle_secrets(bundle_dir)
    _validate_no_hf_secrets(bundle_dir)


def _scrub_runtime_bundle_secrets(bundle_dir: Path) -> None:
    """Remove token-like secrets from copied source files before Hub upload."""

    candidate_files = [
        *bundle_dir.rglob("*.py"),
        *bundle_dir.rglob("*.md"),
        *bundle_dir.rglob("*.txt"),
        *bundle_dir.rglob("*.json"),
    ]
    for path in candidate_files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        updated = text
        updated = _HF_TOKEN_PATTERN.sub("hf_REDACTED", updated)
        updated = _COMET_KEY_ASSIGNMENT_PATTERN.sub(r"\1\2REDACTED\2", updated)
        updated = _HF_KEY_ASSIGNMENT_PATTERN.sub(r"\1\2REDACTED\2", updated)

        if path.as_posix().endswith("pff/utils/__init__.py"):
            updated = (
                "PASCAL_BASE_DIR = ''\n"
                "NERSC_BASE_DIR = ''\n"
                "NERSC_EXPERIMENT_DIR = ''\n"
                "COMET_API_KEY = 'REDACTED'\n"
                "HF_KEYS = 'REDACTED'\n"
                "WORKSPACE = ''\n"
                "PROJECT = ''\n"
            )

        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _validate_no_hf_secrets(bundle_dir: Path) -> None:
    """Fail fast if token-like Hugging Face secrets remain after scrubbing."""

    offending_files: list[str] = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md", ".txt", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _HF_TOKEN_PATTERN.search(text):
            offending_files.append(str(path.relative_to(bundle_dir)))

    if offending_files:
        raise RuntimeError(
            "Refusing to upload runtime bundle because token-like Hugging Face secrets "
            f"remain after scrubbing: {offending_files}"
        )


def build_runtime_bundle_dir(
    *,
    experiment,
    bundle_dir: Path,
    model_card_path: Optional[Sequence[str]] = None,
    hf_repo_id: Optional[str] = None,
    original_repo_id: Optional[str] = None,
) -> RuntimeBundleArtifacts:
    """Stage a self-contained runtime bundle in ``bundle_dir`` without uploading it."""

    _validate_loaded_experiment(experiment)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    runtime_repo_id = hf_repo_id or default_runtime_repo_id(experiment)
    native_repo_id = original_repo_id or _default_original_repo_id(experiment)

    normalized_model_card_path = tuple(
        model_card_path
        if model_card_path is not None
        else getattr(experiment.exp_config, "hf_model_card_path", ("hf_model_cards", "README.md"))
    )
    local_model_card_path = Path(config_dir).joinpath(*normalized_model_card_path)
    base_model_card = resolve_model_card_text(local_model_card_path)

    runtime_payload = build_runtime_config_payload(
        backbone=experiment.model,
        exp_config=experiment.exp_config,
        original_repo_id=native_repo_id,
        runtime_repo_id=runtime_repo_id,
    )
    runtime_config = PKHubConfig(
        **runtime_payload,
        auto_map={
            "AutoConfig": f"{ROOT_CONFIGURATION_FILENAME[:-3]}.PKHubConfig",
            "AutoModel": f"{ROOT_MODELING_FILENAME[:-3]}.PKHubModel",
        },
        architectures=["PKHubModel"],
    )

    runtime_model = PKHubModel(runtime_config, backbone=experiment.model)
    state_dict = {name: tensor.detach().cpu() for name, tensor in runtime_model.state_dict().items()}
    torch.save(state_dict, bundle_dir / "pytorch_model.bin")
    runtime_config.save_pretrained(str(bundle_dir))

    _copy_runtime_support_files(bundle_dir)

    readme_text = runtime_readme_text(
        base_model_card=base_model_card,
        runtime_repo_id=runtime_repo_id,
        original_repo_id=native_repo_id,
        supported_tasks=runtime_config.supported_tasks,
        default_task=runtime_config.default_task,
    )
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")

    return RuntimeBundleArtifacts(
        bundle_dir=bundle_dir,
        runtime_repo_id=runtime_repo_id,
        original_repo_id=native_repo_id,
        readme_path=readme_path,
    )


def push_loaded_model_runtime_bundle(
    experiment,
    model_card_path: Optional[Sequence[str]] = None,
    hf_repo_id: Optional[str] = None,
    alias_name: str = "runtime_bundle_hf",
    commit_message: str = "manual runtime bundle push",
    *,
    original_repo_id: Optional[str] = None,
    exist_ok: bool = True,
) -> str:
    """Build and upload the consumer-facing runtime bundle for a loaded experiment."""

    _validate_loaded_experiment(experiment)
    runtime_repo_id = hf_repo_id or default_runtime_repo_id(experiment)
    create_repo(runtime_repo_id, exist_ok=exist_ok, token=experiment.hf_token)

    bundle_root = Path(experiment.experiment_dir) / alias_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=str(bundle_root), prefix="hf_runtime_bundle_") as temp_dir:
        staged_dir = Path(temp_dir)
        build_runtime_bundle_dir(
            experiment=experiment,
            bundle_dir=staged_dir,
            model_card_path=model_card_path,
            hf_repo_id=runtime_repo_id,
            original_repo_id=original_repo_id,
        )

        api = HfApi(token=experiment.hf_token)
        api.upload_folder(
            folder_path=str(staged_dir),
            repo_id=runtime_repo_id,
            commit_message=commit_message,
            token=experiment.hf_token,
        )

    return runtime_repo_id


__all__ = [
    "RuntimeBundleArtifacts",
    "build_runtime_bundle_dir",
    "default_runtime_repo_id",
    "push_loaded_model_runtime_bundle",
]
