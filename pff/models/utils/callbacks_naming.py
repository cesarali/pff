"""Shared naming helpers for callback metric/image output labels."""

from __future__ import annotations

import re


def safe_substance_name(raw_name: object, *, fallback: str) -> str:
    """Return a filesystem-safe substance token for output filenames."""

    if isinstance(raw_name, str):
        candidate = raw_name.strip()
    elif raw_name is None:
        candidate = ""
    else:
        candidate = str(raw_name).strip()

    safe = re.sub(r"[^0-9A-Za-z]+", "_", candidate).strip("_")
    return safe or fallback


def display_substance_name(raw_name: object, *, fallback: str) -> str:
    """Return a human-readable substance label for plot titles."""

    if isinstance(raw_name, str):
        candidate = raw_name.strip()
        return candidate or fallback
    if raw_name is None:
        return fallback
    candidate = str(raw_name).strip()
    return candidate or fallback

