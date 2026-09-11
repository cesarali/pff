"""Minimal YAML loader fallback used when PyYAML is unavailable."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List


class SafeLoader:
    """Compatibility stub mimicking :class:`yaml.SafeLoader`."""

    _constructors: Dict[str, Any] = {}

    @classmethod
    def add_constructor(cls, tag: str, constructor: Any) -> None:
        cls._constructors[tag] = constructor


Loader = SafeLoader


def _convert_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None

    if value.startswith("[") or value.startswith("{") or value.startswith("("):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            pass

    if value.startswith("\"") and value.endswith("\""):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]

    try:
        if "." in value or "e" in lowered:
            return float(value)
        return int(value)
    except ValueError:
        pass

    return value


def _parse_lines(lines: List[str], indent: int = 0) -> Any:
    mapping: Dict[str, Any] = {}
    sequence: List[Any] = []
    is_list: bool | None = None

    while lines:
        line = lines[0]
        stripped = line.lstrip()

        if not stripped or stripped.startswith("#"):
            lines.pop(0)
            continue

        current_indent = len(line) - len(stripped)

        if current_indent < indent and not stripped.startswith("- "):
            break

        if stripped.startswith("- "):
            if is_list is False:
                raise ValueError("Mixed mapping and sequence at the same level is unsupported.")
            is_list = True

            lines.pop(0)
            item_value = stripped[2:].strip()

            if not item_value:
                sequence.append(_parse_lines(lines, current_indent + 2))
                continue

            if item_value.endswith(":"):
                key = item_value[:-1].strip()
                value = _parse_lines(lines, current_indent + 2)
                sequence.append({key: value})
                continue

            sequence.append(_convert_scalar(item_value))
            continue

        if is_list is True:
            raise ValueError("Mixed mapping and sequence at the same level is unsupported.")

        is_list = False

        lines.pop(0)
        if ":" not in stripped:
            raise ValueError(f"Invalid mapping entry: '{stripped}'.")

        key, value_part = stripped.split(":", 1)
        key = key.strip()
        value_part = value_part.strip()

        if value_part:
            mapping[key] = _convert_scalar(value_part)
        else:
            mapping[key] = _parse_lines(lines, current_indent + 2)

    if is_list:
        return sequence
    return mapping


def safe_load(stream: Any) -> Any:
    """Parse YAML content from ``stream`` and return Python data structures."""

    if hasattr(stream, "read"):
        content = stream.read()
    else:
        content = stream

    if not isinstance(content, str):
        raise TypeError("YAML content must be a string or text stream.")

    raw_lines = content.splitlines()
    return _parse_lines(raw_lines.copy()) if raw_lines else None


def load(stream: Any, Loader: Any | None = None) -> Any:  # noqa: N803 - API compatibility
    """Compatibility wrapper mirroring :func:`yaml.load`."""

    return safe_load(stream)


def dump(data: Any, stream: Any | None = None, default_flow_style: bool | None = None) -> str:
    """Serialise ``data`` to YAML (JSON style in the fallback implementation)."""

    text = json.dumps(data, indent=2)
    if stream is not None:
        stream.write(text)
        return ""
    return text
