from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    config["_config_path"] = str(config_path)
    return config


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base = Path(base_dir) if base_dir is not None else project_root()
    return (base / path).resolve()


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def copy_config_to_run(config: dict[str, Any], run_dir: Path) -> None:
    src = config.get("_config_path")
    if src and Path(src).exists():
        shutil.copy2(src, run_dir / "used_config.yaml")
    else:
        save_yaml(config, run_dir / "used_config.yaml")


def anchor_dict(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    anchors = config.get("anchors", {})
    normalized: dict[str, dict[str, float]] = {}
    for anchor_id, values in anchors.items():
        anchor_id = str(anchor_id)
        normalized[anchor_id] = {
            "x": float(values["x"]),
            "y": float(values["y"]),
            "bias": float(values.get("bias", 0.0)),
        }
    if not normalized:
        raise ValueError("Config must define at least one anchor under 'anchors'.")
    return normalized


def anchor_ids(config_or_anchors: dict[str, Any]) -> list[str]:
    anchors = config_or_anchors.get("anchors", config_or_anchors)
    return [str(anchor_id) for anchor_id in anchors.keys()]

