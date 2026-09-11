"""Sample predictions on synthetic and empirical batches, print shapes, and plot.

The utility first draws a batch list from a synthetic ``AICME`` data module
to compare tensor dimensions against an empirical ``StudyJSON`` dataset. For
both sources it samples stochastic predictions using
``model.sample_individual_prediction`` and reports the shapes of the context,
target, and predicted trajectories before rendering the empirical predictions
as a grid of ``StudyJSON`` records.

The shapes follow the project's conventions:
- Each batch has ``target_obs`` and ``context_obs`` tensors shaped
  ``[B, I, T, 1]``.
- ``model.sample_individual_prediction`` returns ``prediction_sample`` and
  ``prediction_time`` shaped ``[S, B, It, Tr, 1]``.
- Empirical predictions are converted to ``StudyJSON`` via
  :func:`prediction_to_study_jsons` and rendered as a grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import sys

from torchtyping import TensorType as TT

# Ensure local package is importable when running from repository root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pff import config_dir, data_dir, reports_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.data_empirical import (
    load_empirical_json_batches,
    load_empirical_json_batches_as_dm,
    prediction_to_study_jsons,
)
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.data.data_empirical.json_schema import StudyJSON
from pff.utils.plots.databatch_plot import plot_list_list_study_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    default_json = Path(data_dir) / "preprocessed" / "lenuzza_2016.json"
    parser.add_argument(
        "--yaml",
        type=Path,
        default=default_yaml,
        help="Path to NodePK YAML config",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
        help="Which synthetic split to draw a comparison batch from",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=default_json,
        help="Path to empirical StudyJSON file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(reports_dir) / "empirical_prediction_plot.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=8,
        help="Number of stochastic prediction samples S",
    )
    args = parser.parse_args()

    # 1) Load configuration ---------------------------------------------------
    cfg: NodePKExperimentConfig = NodePKExperimentConfig.from_yaml(str(args.yaml))

    # 2) Build synthetic data module and model -------------------------------
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    # 3) Pull a batch list from requested split and inspect shapes ----------
    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    synthetic_batch_list: List[AICMECompartmentsDataBatch] = next(iter(loader))
    print("Synthetic batch list:")
    for p, batch in enumerate(synthetic_batch_list[:1]):
        print(f"  Permutation {p}:")
        context_obs: TT["B", "Ic", "T", 1] = batch.context_obs
        target_obs: TT["B", "It", "T", 1] = batch.target_obs
        context_rem_sim: TT["B", "It", "T", 1] = batch.context_rem_sim
        target_rem_sim: TT["B", "Ic", "T", 1] = batch.target_rem_sim
        print(f"  context_obs shape: {context_obs.shape}")
        print(f"  target_obs shape: {target_obs.shape}")
        print(f"  context_rem_sim shape: {context_rem_sim.shape}")
        print(f"  target_rem_sim shape: {target_rem_sim.shape}")

    # 4) Load empirical batches ----------------------------------------------
    # Shapes from the synthetic datamodule ``dm`` are reused when padding
    # empirical observations.
    # batches: List[AICMECompartmentsDataBatch] = load_empirical_json_batches(
    #    args.json, meta_dosing=cfg.dosing, datamodule=dm
    # )  # list length P; each batch shaped [B, ...]
    print("\n\n")
    print("Empirical batch list:")

    batches: List[AICMECompartmentsDataBatch] = load_empirical_json_batches_as_dm(
        args.json, meta_dosing=cfg.dosing, datamodule=dm
    )
    # 5) Sample predictions ---------------------------------------------------
    for p, batch in enumerate(batches[:1]):
        print(f"Permutation {p}:")
        context_obs: TT["B", "Ic", "T", 1] = batch.context_obs
        target_obs: TT["B", "It", "T", 1] = batch.target_obs
        context_rem_sim: TT["B", "It", "T", 1] = batch.context_rem_sim
        target_rem_sim: TT["B", "Ic", "T", 1] = batch.target_rem_sim
        print(f"  context_obs shape: {context_obs.shape}")
        print(f"  target_obs shape: {target_obs.shape}")
        print(f"  context_rem_sim shape: {context_rem_sim.shape}")
        print(f"  target_rem_sim shape: {target_rem_sim.shape}")


if __name__ == "__main__":
    main()
