from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import anchor_ids, resolve_path, save_json


STANDARD_REQUIRED = ["gt_x", "gt_y"]

DEFAULT_PRESERVED_COLUMNS = [
    "gt_loop_id",
    "gt_segment",
    "gt_segment_start_s",
    "gt_segment_end_s",
    "gt_segment_progress",
    "gt_source",
    "loop_count_assumption",
    "source_raw_file",
]


def _as_candidates(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _find_column(df: pd.DataFrame, candidates: list[str], required_name: str, path: Path, required: bool) -> str | None:
    lower_lookup = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_lookup:
            return lower_lookup[candidate.lower()]
    if required:
        raise ValueError(
            f"{path} is missing required column for '{required_name}'. "
            f"Tried candidates: {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def _clean_trajectory_name(path: Path) -> str:
    stem = path.stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    return stem


def _metadata_for_track(path: Path, config: dict) -> dict[str, Any]:
    name = _clean_trajectory_name(path)
    static_names = {str(item).lower() for item in config.get("data", {}).get("static_calibration_tracks", [])}
    return {
        "source_file": path.name,
        "trajectory": name,
        "segment_id": 0,
        "is_static_calibration": path.name.lower() in static_names or name.lower() in static_names,
    }


def normalize_track(path: str | Path, config: dict) -> pd.DataFrame:
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"CSV file is empty: {path}")

    columns_cfg = config.get("columns", {})
    anchors = anchor_ids(config)
    rename_map: dict[str, str] = {}

    time_col = _find_column(raw, _as_candidates(columns_cfg.get("time", ["time"])), "time", path, required=False)
    if time_col:
        rename_map[time_col] = "time"

    gt_cfg = columns_cfg.get("ground_truth", {})
    rename_map[
        _find_column(raw, _as_candidates(gt_cfg.get("x", ["x_true", "gt_x"])), "ground_truth.x", path, True)
    ] = "gt_x"
    rename_map[
        _find_column(raw, _as_candidates(gt_cfg.get("y", ["y_true", "gt_y"])), "ground_truth.y", path, True)
    ] = "gt_y"

    range_cfg = columns_cfg.get("ranges", {})
    for anchor_id in anchors:
        candidates = _as_candidates(range_cfg.get(anchor_id, [f"d{anchor_id}", f"range_{anchor_id}"]))
        source = _find_column(raw, candidates, f"ranges.{anchor_id}", path, required=True)
        rename_map[source] = f"range_{anchor_id}"

    optional_groups = {
        "raw_position": ("raw_x", "raw_y", ["x"], ["y"]),
        "legacy_kf_position": ("legacy_kf_x", "legacy_kf_y", ["x_kf"], ["y_kf"]),
    }
    for group_name, (std_x, std_y, default_x, default_y) in optional_groups.items():
        group_cfg = columns_cfg.get(group_name, {})
        source_x = _find_column(raw, _as_candidates(group_cfg.get("x", default_x)), f"{group_name}.x", path, False)
        source_y = _find_column(raw, _as_candidates(group_cfg.get("y", default_y)), f"{group_name}.y", path, False)
        if source_x and source_y:
            rename_map[source_x] = std_x
            rename_map[source_y] = std_y

    normalized = raw.rename(columns=rename_map)
    preserve_cols = config.get("data", {}).get("preserve_columns", DEFAULT_PRESERVED_COLUMNS)
    keep_cols = set(rename_map.values())
    for col in preserve_cols:
        if col in normalized.columns:
            keep_cols.add(col)
    keep_cols = sorted(keep_cols)
    normalized = normalized[keep_cols].copy()

    string_cols = {"source_file", "trajectory", "gt_segment", "gt_source", "source_raw_file"}
    numeric_cols = [col for col in normalized.columns if col not in string_cols]
    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    required = STANDARD_REQUIRED + [f"range_{anchor_id}" for anchor_id in anchors]
    before = len(normalized)
    normalized = normalized.dropna(subset=required).reset_index(drop=True)
    dropped = before - len(normalized)

    if config.get("data", {}).get("sort_by_time", True) and "time" in normalized.columns:
        normalized = normalized.sort_values("time", kind="mergesort").reset_index(drop=True)

    for key, value in _metadata_for_track(path, config).items():
        normalized[key] = value
    normalized["sample_index"] = np.arange(len(normalized), dtype=int)
    normalized["dropped_required_nan_rows"] = dropped

    return normalized


def load_all_tracks(config: dict) -> pd.DataFrame:
    paths = config.get("data", {}).get("csv_paths", [])
    if not paths:
        raise ValueError("Config data.csv_paths is empty.")
    frames = [normalize_track(path, config) for path in paths]
    return pd.concat(frames, ignore_index=True)


def assign_splits(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    split_cfg = config.get("split", {})
    strategy = split_cfg.get("strategy", "by_track")
    df = df.copy()
    df["split"] = ""

    if strategy == "chronological_by_file":
        train_ratio = float(split_cfg.get("train_ratio", 0.7))
        val_ratio = float(split_cfg.get("val_ratio", 0.15))
        for _, part in df.groupby("trajectory", sort=False):
            idx = part.index.to_numpy()
            train_end = int(len(idx) * train_ratio)
            val_end = int(len(idx) * (train_ratio + val_ratio))
            df.loc[idx[:train_end], "split"] = "train"
            df.loc[idx[train_end:val_end], "split"] = "val"
            df.loc[idx[val_end:], "split"] = "test"
    elif strategy == "by_track":
        val_tracks = {str(item).lower() for item in split_cfg.get("validation_tracks", [])}
        test_tracks = {str(item).lower() for item in split_cfg.get("test_tracks", [])}
        train_tracks = {str(item).lower() for item in split_cfg.get("train_tracks", [])}
        val_ratio = float(split_cfg.get("validation_within_train_ratio", 0.0))

        for trajectory, part in df.groupby("trajectory", sort=False):
            source = str(part["source_file"].iloc[0]).lower()
            name = str(trajectory).lower()
            if name in test_tracks or source in test_tracks:
                split = "test"
            elif name in val_tracks or source in val_tracks:
                split = "val"
            elif train_tracks and not (name in train_tracks or source in train_tracks):
                split = "ignore"
            else:
                split = "train"
            df.loc[part.index, "split"] = split

        if not val_tracks and val_ratio > 0:
            train_indices = df.index[df["split"] == "train"].to_numpy()
            df.loc[train_indices, "split"] = ""
            for _, part in df.loc[train_indices].groupby("trajectory", sort=False):
                idx = part.index.to_numpy()
                val_start = int(len(idx) * (1 - val_ratio))
                df.loc[idx[:val_start], "split"] = "train"
                df.loc[idx[val_start:], "split"] = "val"
    elif strategy == "leave_one_track_out":
        leave_out = str(split_cfg.get("leave_out_track", "")).lower()
        val_tracks = {str(item).lower() for item in split_cfg.get("validation_tracks", [])}
        if not leave_out:
            raise ValueError("leave_one_track_out requires split.leave_out_track.")
        for trajectory, part in df.groupby("trajectory", sort=False):
            source = str(part["source_file"].iloc[0]).lower()
            name = str(trajectory).lower()
            if name == leave_out or source == leave_out:
                split = "test"
            elif name in val_tracks or source in val_tracks:
                split = "val"
            else:
                split = "train"
            df.loc[part.index, "split"] = split
    else:
        raise ValueError(f"Unknown split.strategy: {strategy}")

    df = df[df["split"] != "ignore"].reset_index(drop=True)
    missing = {"train", "val", "test"} - set(df["split"].unique())
    if missing:
        raise ValueError(f"Split strategy '{strategy}' did not create required split(s): {sorted(missing)}")
    return df


def write_prepared_dataset(df: pd.DataFrame, out_dir: Path, config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "all_samples.csv", index=False)
    manifest_rows = []
    for (split, trajectory), part in df.groupby(["split", "trajectory"], sort=False):
        part.to_csv(out_dir / f"{split}_{trajectory.replace(' ', '_')}.csv", index=False)
        manifest_rows.append(
            {
                "split": split,
                "trajectory": trajectory,
                "source_file": part["source_file"].iloc[0],
                "rows": len(part),
                "time_min": float(part["time"].min()) if "time" in part.columns else None,
                "time_max": float(part["time"].max()) if "time" in part.columns else None,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "split_manifest.csv", index=False)
    save_json(
        {
            "columns": list(df.columns),
            "split_counts": df["split"].value_counts().to_dict(),
            "tracks": sorted(df["trajectory"].unique().tolist()),
            "split_strategy": config.get("split", {}).get("strategy"),
        },
        out_dir / "schema.json",
    )
