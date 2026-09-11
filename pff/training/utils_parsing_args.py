from __future__ import annotations
import lightning.pytorch as pl
import argparse
import os
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

def parse_args(default_cfg_path: str):
    """Minimal CLI: a YAML path + any number of --override key=value pairs."""
    p = argparse.ArgumentParser(description="Train a Generative‑PK model with optional overrides")
    p.add_argument("--config_path", type=str, default=default_cfg_path)
    p.add_argument(
        "--meta_study_name",
        type=str,
        default=None,
        help=(
            "Optional meta-study YAML file name/path. If provided, it replaces the "
            "meta-study config loaded from --config_path."
        ),
    )

    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Overrides as key=value, e.g. train.epochs=500 mix_data.train_size=2000",
    )
    return p.parse_args()

def parse_override_list(items: list[str]) -> Dict[str, Any]:
    """Convert ["a.b=3", "x.y=True"] → {"a.b": 3, "x.y": True}."""
    out: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item!r} (expected key=value)")
        key, val = item.split("=", 1)
        # Try to interpret literals (int, float, bool, list, etc.)
        try:
            out[key] = eval(val)
        except Exception:
            out[key] = val  # leave as string
    return out

# ──────────────────────────────────────────────────────────────────────────────
# 2. Safe override of *nested* dataclass configs
# ──────────────────────────────────────────────────────────────────────────────

def _deep_set(d: dict, dotted_key: str, value: Any):
    """Insert *value* into nested dict *d* following dotted path."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def _rebuild(template: Any, payload: Any):
    """Recursively rebuild *template* dataclass using data from *payload* (a dict)."""
    if is_dataclass(template):
        kwargs = {
            f.name: _rebuild(getattr(template, f.name), payload[f.name])
            for f in template.__dataclass_fields__.values()
        }
        return template.__class__(**kwargs)
    return payload

def update_config_with_overrides(cfg, overrides: Dict[str, Any]):
    """Return **new** config where dotted overrides are applied consistently."""
    cfg_dict = asdict(cfg)  # deep copy into plain Python types
    for dotted, val in overrides.items():
        _deep_set(cfg_dict, dotted, val)
    return _rebuild(cfg, cfg_dict)
