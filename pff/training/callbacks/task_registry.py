"""Registry of locally supported scheduler task functions."""

from __future__ import annotations

from typing import Callable, Dict

from pff.training.callbacks.pk_task_empirical import (
    task_empirical_summary,
    task_empirical_heldout_generated_classifier,
    task_empirical_predictive_metrics,
)
from pff.training.callbacks.pk_task_synthetic import (
    task_diverse_synthetic_experiment_sample_distances,
    task_generative_images,
    task_generative_metrics,
    task_predictive_images,
    task_synthetic_vpc_paired_images,
    task_vpc_images,
    task_vpc_npde_pvalues,
)

TASK_REGISTRY: Dict[str, Callable] = {
    "pk.empirical.predictive.metrics": task_empirical_predictive_metrics,
    "pk.empirical.heldout_generated_classifier": task_empirical_heldout_generated_classifier,
    "pk.predictive.images": task_predictive_images,
    "pk.generative.metrics": task_generative_metrics,
    "pk.generative.images": task_generative_images,
    "pk.vpc.npde_pvalues": task_vpc_npde_pvalues,
    "pk.vpc.images": task_vpc_images,
    "pk.synthetic.vpc.paired_images": task_synthetic_vpc_paired_images,
    "pk.empirical.summary": task_empirical_summary,
    "pk.diverse_experiment.distances": task_diverse_synthetic_experiment_sample_distances,
    "pk.diverse_synthetic_experiment.sample_distances": (
        task_diverse_synthetic_experiment_sample_distances
    ),
}

__all__ = ["TASK_REGISTRY"]
