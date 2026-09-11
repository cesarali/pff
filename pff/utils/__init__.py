"""Utility package constants and compatibility aliases.

The package-level ``pff`` module is the single source of truth for
credential loading. This module only re-exports those keys under the legacy
names still used by older training utilities.
"""

from .. import COMET_KEY, HUGGINGFACE_KEY

PASCAL_BASE_DIR = "/home/df630/pff"
NERSC_BASE_DIR = "/global/homes/d/dfarough/pff"
NERSC_EXPERIMENT_DIR = "/pscratch/sd/d/dfarough/pff"
COMET_API_KEY = COMET_KEY
HF_KEYS = HUGGINGFACE_KEY
WORKSPACE = "dfaroughy"
PROJECT = "Pharma"

__all__ = [
    "COMET_API_KEY",
    "HF_KEYS",
    "NERSC_BASE_DIR",
    "NERSC_EXPERIMENT_DIR",
    "PASCAL_BASE_DIR",
    "PROJECT",
    "WORKSPACE",
]
