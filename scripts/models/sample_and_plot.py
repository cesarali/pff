"""Sample NewAICMEPK predictions and plot them.

This utility loads a NewAICMEPK model from a NodePK YAML configuration,
constructs its corresponding data module, samples a random batch list
from one of the data splits and draws individual predictions. The
resulting StudyJSON records are rendered to a grid and stored under the
``reports`` directory.

The script mirrors the shapes used throughout the project:
- Each ``AICMECompartmentsDataBatch`` contains tensors shaped like
  ``target_obs: [B, I_t, T, 1]`` and ``context_obs: [B, I_c, T, 1]``.
- ``model.sample_individual_prediction_from_batch_list_to_studyjson``
  returns ``List[List[StudyJSON]]`` where the outer dimension indexes
  permutations ``P`` and the inner dimension indexes the batch ``B``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
import argparse
import random
import sys

# Ensure local package is importable when running from repository root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pff import config_dir, reports_dir
from pff.config_classes.node_pk_config import NodePKExperimentConfig
from pff.data.datasets.aicme_datasets import (
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.models.amortized_inference.aicme import AICMEPK
from pff.utils.plots.databatch_plot import plot_list_list_study_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
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
        help="Which split to sample from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(reports_dir) / "model_prediction_plot.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    # 1) Load configuration ---------------------------------------------------
    cfg: NodePKExperimentConfig = NodePKExperimentConfig.from_yaml(str(args.yaml))

    # 2) Build data module and model -----------------------------------------
    dm = AICMECompartmentsDataModule(cfg)
    dm.prepare_data()
    dm.setup()

    model = AICMEPK(cfg)
    model.eval()

    # 3) Pull a random batch list from the requested split -------------------
    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    batch_list: List[AICMECompartmentsDataBatch] = next(iter(loader))
    random.shuffle(batch_list)  # ensure randomness in permutation order

    # 4) Predict and convert to StudyJSON ------------------------------------
    # studies = model.sample_individual_prediction_from_batch_list_to_studyjson(
    #    batch_list, cfg.dosing
    # )

    # 5) Plot grid of predictions --------------------------------------------
    # args.out.parent.mkdir(parents=True, exist_ok=True)
    # plot_list_list_study_json(studies, file_name=str(args.out))
    # print(f"Saved plot → {args.out}")

    # 5) Predict and convert to StudyJSON ------------------------------------
    batch = batch_list[0]
    sampled_studies = model.sample_new_individuals_to_studyjson(batch)

    plot_list_list_study_json([sampled_studies], file_name=str(args.out))


if __name__ == "__main__":
    main()
