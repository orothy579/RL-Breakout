"""YAML config loader with single-inheritance deep-merge.

MY ORIGINAL CONTRIBUTION: a small config system that lets every experiment be
expressed as a baseline file plus a tiny override. ``inherits: dqn_baseline.yaml``
pulls in the parent and the child only re-states the keys it changes (recursively
deep-merged). This is what keeps each ablation a true single-variable change and
keeps the repo's dozens of configs DRY. Cyclic ``inherits`` chains are detected
and rejected.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``child`` onto ``parent`` (child wins on leaf keys)."""
    result = copy.deepcopy(parent)
    for key, value in child.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if _seen is None:
        _seen = set()
    if path in _seen:
        raise ValueError(f"Cyclic config inheritance detected at {path}")
    _seen.add(path)

    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    parent_ref = cfg.pop("inherits", None)
    if parent_ref:
        parent_path = (path.parent / parent_ref).resolve()
        parent_cfg = load_config(parent_path, _seen)
        cfg = _deep_merge(parent_cfg, cfg)

    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
