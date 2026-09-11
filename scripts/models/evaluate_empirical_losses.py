"""Evaluate NodePK model losses on the empirical Lenuzza dataset.

This utility mirrors the empirical batch loading performed by other scripts in
``scripts/models``.  It loads the default ``NodePK`` configuration and empirical
``StudyJSON`` file, runs a forward pass of :class:`~pff.models.amortized_inference.new_aicme.NewAICMEPK`
without gradient tracking, and reports the per-substance metrics produced by
:class:`pff.models.amortized_inference.new_aicme.AICMEForwardOutputs`.  When
per-individual metrics are available they can optionally be exported as well.

Results are rendered as :mod:`pandas` tables on stdout, and optional CSV
exports can be requested via command line arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence
import sys

import pandas as pd
import torch

# Ensure local package is importable when running from repository root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pff import config_dir, data_dir  # noqa: E402
from pff.config_classes.node_pk_config import NodePKExperimentConfig  # noqa: E402
from pff.data.data_empirical import load_empirical_hf_batches_as_dm  # noqa: E402
from pff.data.datasets.aicme_datasets import (  # noqa: E402
    AICMECompartmentsDataBatch,
    AICMECompartmentsDataModule,
)
from pff.models.amortized_inference.aicme import (  # noqa: E402
    AICMEForwardOutputs,
    AICMEPK,
)


def _load_model(cfg: NodePKExperimentConfig, checkpoint: Path | None) -> AICMEPK:
    """Instantiate ``NewAICMEPK`` and optionally restore a checkpoint."""

    model = AICMEPK(cfg)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _forward_loss(
    model: AICMEPK,
    batches: Sequence[AICMECompartmentsDataBatch],
) -> AICMEForwardOutputs:
    """Run a forward pass on ``batches`` and collect ``AICMEForwardOutputs``."""

    with torch.no_grad():
        outputs = model(batches, return_forward_report=True)
    return outputs


def _per_individual_table(per_individual: List[dict[str, object]]) -> pd.DataFrame:
    """Convert nested per-individual metrics to a tidy :class:`DataFrame`."""

    if not per_individual:
        return pd.DataFrame()

    # ``pd.json_normalize`` flattens the ``metrics`` dictionary into columns.
    records = pd.json_normalize(per_individual)
    records.rename(columns=lambda col: col.replace("metrics.", ""), inplace=True)
    return records


def _per_substance_table(per_substance: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Convert per-substance aggregates to a :class:`DataFrame`."""

    if not per_substance:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(per_substance, orient="index")
    df.index.name = "substance"
    df.reset_index(inplace=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_yaml = Path(config_dir) / "experiment_configs" / "node-pk" / "base-homogeneous.yaml"
    default_json = Path(data_dir) / "preprocessed" / "lenuzza_2016.json"
    parser.add_argument(
        "--yaml",
        type=Path,
        default=default_yaml,
        help="Path to the NodePK YAML configuration file.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=default_json,
        help="Path to the empirical StudyJSON dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a model checkpoint to restore before evaluation.",
    )
    parser.add_argument(
        "--per-individual-csv",
        type=Path,
        default=None,
        help="Optional CSV path to export the per-individual metrics.",
    )
    parser.add_argument(
        "--per-substance-csv",
        type=Path,
        default=None,
        help="Optional CSV path to export the per-substance aggregates.",
    )
    args = parser.parse_args()

    cfg: NodePKExperimentConfig = NodePKExperimentConfig.from_yaml(str(args.yaml))
    datamodule = AICMECompartmentsDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    batches: List[AICMECompartmentsDataBatch] = load_empirical_hf_batches_as_dm(
        "cesarali/lenuzza-2016", meta_dosing=cfg.dosing, datamodule=datamodule
    )

    model = _load_model(cfg, args.checkpoint)
    outputs = _forward_loss(model, batches)

    print("Aggregated losses:")
    agg_series = pd.Series({k: float(v) for k, v in outputs.to_dict().items()})
    print(agg_series.to_string())
    print()

    per_individual_records = getattr(outputs, "per_individual", [])
    per_individual_df = _per_individual_table(per_individual_records)
    if per_individual_records:
        per_individual_df.sort_values(["study", "substance", "subject"], inplace=True)
        print("Per-individual metrics:")
        print(per_individual_df.to_string(index=False))
    else:
        print("Per-individual metrics: none available.")
    print()

    per_substance_df = _per_substance_table(getattr(outputs, "per_substance", {}))
    if not per_substance_df.empty:
        per_substance_df.sort_values("substance", inplace=True)
        print("Per-substance metrics:")
        print(per_substance_df.to_string(index=False))
    else:
        print("Per-substance metrics: none available.")

    if args.per_individual_csv is not None and per_individual_records:
        per_individual_df.to_csv(args.per_individual_csv, index=False)
        print(f"Saved per-individual metrics to {args.per_individual_csv}")

    if args.per_substance_csv is not None and not per_substance_df.empty:
        per_substance_df.to_csv(args.per_substance_csv, index=False)
        print(f"Saved per-substance metrics to {args.per_substance_csv}")


if __name__ == "__main__":
    main()
