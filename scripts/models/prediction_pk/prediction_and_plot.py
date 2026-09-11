"""Generate prediction StudyJSONs from the active ``PredictionPK`` config."""

from __future__ import annotations

from pathlib import Path

from pff import config_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import AICMECompartmentsDataModule
from pff.models.amortized_inference.prediction_pk import PredictionPK


def _prediction_config_from_file() -> NodePKExperimentConfig:
    """Load the maintained AISTATS ``PredictionPK`` configuration."""

    default_yaml = Path(config_dir) / "experiment_configs" / "AISTATS" / "node-pk" / "base.yaml"
    return NodePKExperimentConfig.from_yaml(str(default_yaml))


def _first_batch_list(dm: AICMECompartmentsDataModule):
    """Return the first permutation list from the training dataloader on CPU."""

    dm.prepare_data()
    dm.setup()
    batch_list = next(iter(dm.train_dataloader()))
    return [batch.to_device("cpu") for batch in batch_list]


def main() -> None:
    """Sample predictive trajectories and print a short summary."""

    cfg = _prediction_config_from_file()
    dm = AICMECompartmentsDataModule(cfg)
    batch_list = _first_batch_list(dm)
    model = PredictionPK(cfg)
    studies = model.sample_individual_prediction_from_batch_list_to_studyjson(
        batch_list[:1],
        sample_size=8,
    )

    print(f"Generated {len(studies)} prediction study-json group(s).")
    if studies and studies[0]:
        first_study = studies[0][0]
        substance = first_study.get("meta_data", {}).get("substance_name", "")
        print(f"First study substance: {substance}")
        print(f"Predicted target individuals: {len(first_study.get('target', []))}")


if __name__ == "__main__":
    main()
