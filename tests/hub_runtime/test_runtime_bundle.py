"""Tests for the parallel Hugging Face runtime-bundle export path."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import AutoModel

from pff.hub_runtime import build_runtime_bundle_dir, default_runtime_repo_id
from pff.models.amortized_inference.aicme import AICMEPK
from pff.config_classes.node_pk_config import NodePKExperimentConfig


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _runtime_config() -> NodePKExperimentConfig:
    """Return a compact AICME config suitable for runtime-bundle tests."""

    cfg = NodePKExperimentConfig()
    cfg.hf_model_name = "AICMEPK_test"
    cfg.train = replace(
        cfg.train,
        batch_size=1,
        num_workers=0,
        persistent_workers=False,
        epochs=1,
    )
    cfg.mix_data = replace(
        cfg.mix_data,
        train_size=1,
        val_size=1,
        test_size=1,
        n_of_permutations=1,
        n_of_target_individuals=1,
        test_empirical_datasets=[],
    )
    cfg.meta_study = replace(cfg.meta_study, num_individuals_range=(2, 2))
    cfg.network = replace(cfg.network, aggregator_type="mean")
    cfg.context_observations = replace(
        cfg.context_observations,
        split_past_future=False,
        add_rem=True,
        max_num_obs=6,
    )
    cfg.target_observations = replace(
        cfg.target_observations,
        split_past_future=True,
        add_rem=True,
        min_past=1,
        max_past=2,
        max_num_obs=6,
    )
    return cfg


def _loaded_experiment(tmp_path: Path):
    """Build a lightweight loaded-experiment namespace for manual export tests."""

    cfg = _runtime_config()
    model = AICMEPK(cfg)
    return SimpleNamespace(
        model=model,
        exp_config=cfg,
        experiment_dir=str(tmp_path),
        hf_token="test-token",
    )


def test_default_runtime_repo_id_uses_runtime_suffix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The runtime bundle should default to a separate `-runtime` repo name."""

    experiment = _loaded_experiment(tmp_path)

    class _FakeApi:
        def whoami(self, token=None):
            return {"name": "tester"}

    monkeypatch.setattr("pff.hub_runtime.runtime_bundle.HfApi", lambda *a, **k: _FakeApi())
    repo_id = default_runtime_repo_id(experiment)
    assert repo_id == "tester/AICMEPK_test-runtime"


def test_build_runtime_bundle_dir_writes_expected_files(tmp_path: Path):
    """Staging the runtime bundle should write weights, config, code, and README."""

    experiment = _loaded_experiment(tmp_path)
    artifacts = build_runtime_bundle_dir(
        experiment=experiment,
        bundle_dir=tmp_path / "bundle",
        hf_repo_id="tester/aicmepk-runtime",
        original_repo_id="tester/aicmepk-native",
    )

    assert artifacts.runtime_repo_id == "tester/aicmepk-runtime"
    assert (artifacts.bundle_dir / "pytorch_model.bin").is_file()
    assert (artifacts.bundle_dir / "config.json").is_file()
    assert (artifacts.bundle_dir / "README.md").is_file()
    assert (artifacts.bundle_dir / "configuration_pff.py").is_file()
    assert (artifacts.bundle_dir / "modeling_pff.py").is_file()
    assert (artifacts.bundle_dir / "pff" / "hub_runtime" / "runtime_contract.py").is_file()

    config_payload = json.loads((artifacts.bundle_dir / "config.json").read_text(encoding="utf-8"))
    assert config_payload["auto_map"]["AutoModel"] == "modeling_pff.PKHubModel"
    assert config_payload["runtime_repo_id"] == "tester/aicmepk-runtime"
    assert config_payload["original_repo_id"] == "tester/aicmepk-native"

    readme_text = (artifacts.bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "consumer-facing runtime bundle" in readme_text
    assert "trust_remote_code=True" in readme_text


def test_runtime_model_run_task_contracts(tmp_path: Path):
    """The Hub wrapper should expose both generate and predict StudyJSON contracts."""

    experiment = _loaded_experiment(tmp_path)
    artifacts = build_runtime_bundle_dir(
        experiment=experiment,
        bundle_dir=tmp_path / "bundle",
        hf_repo_id="tester/aicmepk-runtime",
        original_repo_id="tester/aicmepk-native",
    )
    model = AutoModel.from_pretrained(str(artifacts.bundle_dir), trust_remote_code=True)
    assert model.__class__.__name__ == "PKHubModel"

    generate_outputs = model.run_task(
        task="generate",
        studies=[_load_fixture("study_ctx_only.json")],
        num_samples=2,
    )
    assert generate_outputs["task"] == "generate"
    assert generate_outputs["io_schema_version"] == "studyjson-v1"
    assert len(generate_outputs["results"]) == 1
    assert len(generate_outputs["results"][0]["samples"]) == 2

    predict_outputs = model.run_task(
        task="predict",
        studies=[_load_fixture("study_ctx_tgt.json")],
        num_samples=2,
    )
    assert predict_outputs["task"] == "predict"
    assert len(predict_outputs["results"]) == 1
    assert len(predict_outputs["results"][0]["samples"]) == 2
    for sampled_study in predict_outputs["results"][0]["samples"]:
        assert len(sampled_study["target"]) == 1
        assert len(sampled_study["target"][0]["prediction_samples"]) == 1


def test_runtime_model_rejects_over_capacity_inputs(tmp_path: Path):
    """Inputs that exceed stored capacities must raise instead of truncating."""

    experiment = _loaded_experiment(tmp_path)
    artifacts = build_runtime_bundle_dir(
        experiment=experiment,
        bundle_dir=tmp_path / "bundle",
        hf_repo_id="tester/aicmepk-runtime",
        original_repo_id="tester/aicmepk-native",
    )
    model = AutoModel.from_pretrained(str(artifacts.bundle_dir), trust_remote_code=True)

    oversized = _load_fixture("study_ctx_only.json")
    oversized["context"][0]["observations"] = [0.1] * 20
    oversized["context"][0]["observation_times"] = [float(i) for i in range(20)]

    with pytest.raises(ValueError, match="observation capacity"):
        model.run_task(task="generate", studies=[oversized], num_samples=1)


def test_runtime_bundle_loads_from_local_bundle_without_repo_imports(tmp_path: Path):
    """A clean Python process should load the bundle using the copied package source."""

    experiment = _loaded_experiment(tmp_path)
    artifacts = build_runtime_bundle_dir(
        experiment=experiment,
        bundle_dir=tmp_path / "bundle",
        hf_repo_id="tester/aicmepk-runtime",
        original_repo_id="tester/aicmepk-native",
    )

    study_json = json.dumps([_load_fixture("study_ctx_only.json")])
    script = f"""
import json
import sys
from pathlib import Path

project_root = {str(PROJECT_ROOT)!r}
sys.path = [entry for entry in sys.path if project_root not in str(entry)]

from transformers import AutoModel

model = AutoModel.from_pretrained({str(artifacts.bundle_dir)!r}, trust_remote_code=True, local_files_only=True)
outputs = model.run_task(task="generate", studies=json.loads({study_json!r}), num_samples=2)
import pff

print(json.dumps({{
    "pff_file": pff.__file__,
    "sample_count": len(outputs["results"][0]["samples"]),
    "supported_tasks": outputs["model_info"]["supported_tasks"],
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout.strip())
    assert payload["sample_count"] == 2
    assert payload["supported_tasks"] == ["generate", "predict"]
    assert str(artifacts.bundle_dir) in payload["pff_file"]
