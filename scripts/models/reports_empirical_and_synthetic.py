from pathlib import Path
from typing import List

from torchtyping import TensorType as TT

from pff import config_dir, data_dir, reports_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_empirical import (
    load_empirical_hf_batches_as_dm,
)
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.models.amortized_inference.aicme import AICMEPK
from pff.data.data_empirical.json_schema import StudyJSON
from pathlib import Path
from dataclasses import dataclass

from dataclasses import dataclass
from pathlib import Path
from typing import List


# --- Config class -------------------------------------------------------
@dataclass
class NotebookConfig:
    yaml: Path
    split: str
    json: Path
    out: Path
    samples: int

    @classmethod
    def default(cls, config_dir: Path, data_dir: Path, reports_dir: Path):
        default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
        default_json = Path(data_dir) / "preprocessed" / "lenuzza_2016.json"
        default_out = Path(reports_dir) / "empirical_prediction_plot.png"

        return cls(
            yaml=default_yaml,
            split="train",
            json=default_json,
            out=default_out,
            samples=8,
        )


# --- Main workflow ------------------------------------------------------
def run_experiment(config_dir: Path, data_dir: Path, reports_dir: Path):
    # 1) Get defaults
    args = NotebookConfig.default(config_dir, data_dir, reports_dir)

    # 2) Load configuration
    cfg: NodePKExperimentConfig = NodePKExperimentConfig.from_yaml(str(args.yaml))

    # 3) Build synthetic data module and model
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    model = AICMEPK(cfg)
    model.eval()

    loader = dm.train_dataloader()

    # 4) Load empirical batches
    empirical_batch_list = load_empirical_hf_batches_as_dm(
        "cesarali/lenuzza-2016",
        meta_dosing=cfg.dosing,
        datamodule=dm,
    )
    empirical_outputs = model(empirical_batch_list, return_forward_report=True)
    loss_outputs = empirical_outputs.to_dict()

    # 5) Load synthetic batches
    synthetic_batch_list: List[AICMECompartmentsDataBatch] = next(iter(loader))
    synthetic_outputs = model(synthetic_batch_list, return_forward_report=True)
    loss_outputs = synthetic_outputs.to_dict()

    return loss_outputs


# --- Entry point --------------------------------------------------------
if __name__ == "__main__":
    # Replace with your actual paths
    run_experiment(config_dir, data_dir, reports_dir)
