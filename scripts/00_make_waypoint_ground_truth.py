from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_FILES = [
    "Data hasil/10lup+trilat.csv",
    "Data hasil/10lup1+trilat.csv",
    "Data hasil/10lup2+trilat.csv",
    "Data hasil/trilat5lup.csv",
    "Data hasil/trilat5lup1.csv",
]

WAYPOINTS = [
    {"name": "P1", "x": 1.0, "y": 1.0},
    {"name": "P2", "x": 3.0, "y": 1.0},
    {"name": "P3", "x": 3.0, "y": 3.0},
    {"name": "P4", "x": 1.0, "y": 3.0},
]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "track"


def output_name_for(path: Path) -> str:
    return f"{slugify(path.stem)}_gt.csv"


def infer_loop_count(path: Path) -> int:
    match = re.search(r"(\d+)\s*lup", path.stem.lower())
    if not match:
        match = re.search(r"(\d+)\s*loop", path.stem.lower())
    if match:
        return max(int(match.group(1)), 1)
    return 1


def default_waypoint_events(df: pd.DataFrame, loop_count: int) -> list[dict[str, Any]]:
    start_time = float(df["time"].iloc[0])
    end_time = float(df["time"].iloc[-1])
    segment_count = loop_count * len(WAYPOINTS)
    segment_dt = (end_time - start_time) / segment_count

    events = []
    for event_index in range(segment_count + 1):
        waypoint = WAYPOINTS[event_index % len(WAYPOINTS)]
        events.append(
            {
                "time_s": round(start_time + event_index * segment_dt, 6),
                "waypoint": waypoint["name"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "loop": min(event_index // len(WAYPOINTS) + 1, loop_count),
            }
        )
    return events


def waypoint_for_event_index(index: int) -> dict[str, float | str]:
    return WAYPOINTS[index % len(WAYPOINTS)]


def events_from_waypoint_times(df: pd.DataFrame, times_s: list[float], loop_count: int) -> list[dict[str, Any]]:
    if len(times_s) < 2:
        raise ValueError("waypoint_times_s must contain at least two entries.")

    events: list[dict[str, Any]] = []
    first_data_time = float(df["time"].iloc[0])
    last_data_time = float(df["time"].iloc[-1])

    if first_data_time < float(times_s[0]):
        first_wp = waypoint_for_event_index(0)
        events.append(
            {
                "time_s": round(first_data_time, 6),
                "waypoint": first_wp["name"],
                "x": first_wp["x"],
                "y": first_wp["y"],
                "loop": 1,
            }
        )

    for index, time_s in enumerate(times_s):
        waypoint = waypoint_for_event_index(index)
        events.append(
            {
                "time_s": round(float(time_s), 6),
                "waypoint": waypoint["name"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "loop": min(index // len(WAYPOINTS) + 1, loop_count),
            }
        )

    if last_data_time > float(times_s[-1]):
        last_index = len(times_s) - 1
        waypoint = waypoint_for_event_index(last_index)
        events.append(
            {
                "time_s": round(last_data_time, 6),
                "waypoint": waypoint["name"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "loop": loop_count,
            }
        )

    return events


def events_from_movement_intervals(df: pd.DataFrame, intervals_s: list[list[float]], loop_count: int) -> list[dict[str, Any]]:
    if not intervals_s:
        raise ValueError("movement_intervals_s must not be empty.")

    events: list[dict[str, Any]] = []
    first_data_time = float(df["time"].iloc[0])
    last_data_time = float(df["time"].iloc[-1])

    def add_event(time_s: float, waypoint_index: int) -> None:
        waypoint = waypoint_for_event_index(waypoint_index)
        if events and abs(float(events[-1]["time_s"]) - time_s) < 1e-9:
            events[-1] = {
                **events[-1],
                "waypoint": waypoint["name"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "loop": min(waypoint_index // len(WAYPOINTS) + 1, loop_count),
            }
            return
        events.append(
            {
                "time_s": round(float(time_s), 6),
                "waypoint": waypoint["name"],
                "x": waypoint["x"],
                "y": waypoint["y"],
                "loop": min(waypoint_index // len(WAYPOINTS) + 1, loop_count),
            }
        )

    first_start = float(intervals_s[0][0])
    if first_data_time < first_start:
        add_event(first_data_time, 0)

    for segment_index, interval in enumerate(intervals_s):
        if len(interval) != 2:
            raise ValueError(f"Invalid movement interval at index {segment_index}: {interval}")
        start_s, end_s = float(interval[0]), float(interval[1])
        if end_s <= start_s:
            raise ValueError(f"Movement interval end must be after start: {interval}")
        add_event(start_s, segment_index)
        add_event(end_s, segment_index + 1)

    last_waypoint_index = len(intervals_s)
    if last_data_time > float(intervals_s[-1][1]):
        add_event(last_data_time, last_waypoint_index)

    return events


def load_manual_specs(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data.get("files", data)


def apply_manual_specs(manifest: dict[str, Any], specs: dict[str, Any]) -> dict[str, Any]:
    files = manifest.setdefault("files", {})
    for key, spec in specs.items():
        input_path = resolve(key)
        if not input_path.exists():
            raise FileNotFoundError(f"Manual timing input CSV not found: {input_path}")
        df = pd.read_csv(input_path)
        loop_count = int(spec.get("loop_count", infer_loop_count(input_path)))
        output_name = spec.get("output_name", output_name_for(input_path))
        mode = spec.get("mode", "waypoint_times")
        if mode == "waypoint_times":
            events = events_from_waypoint_times(df, [float(v) for v in spec["waypoint_times_s"]], loop_count)
            source = "manual_waypoint_times_from_notes"
        elif mode == "movement_intervals":
            events = events_from_movement_intervals(
                df,
                [[float(v[0]), float(v[1])] for v in spec["movement_intervals_s"]],
                loop_count,
            )
            source = "manual_movement_intervals_from_notes"
        else:
            raise ValueError(f"Unknown manual timing mode for {key}: {mode}")
        files[key] = {
            "loop_count": loop_count,
            "output_name": output_name,
            "ground_truth_source": spec.get("ground_truth_source", source),
            "waypoint_events": events,
        }
    return manifest


def infer_waypoint_events_from_raw(df: pd.DataFrame, loop_count: int) -> list[dict[str, Any]]:
    if "x" not in df.columns or "y" not in df.columns:
        return default_waypoint_events(df, loop_count)

    default_events = default_waypoint_events(df, loop_count)
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    raw_x = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
    raw_y = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(times) & np.isfinite(raw_x) & np.isfinite(raw_y)
    if finite.sum() < len(default_events):
        return default_events

    times = times[finite]
    raw_x = raw_x[finite]
    raw_y = raw_y[finite]
    segment_dt = (float(df["time"].iloc[-1]) - float(df["time"].iloc[0])) / (loop_count * len(WAYPOINTS))
    search_half_width = max(segment_dt * 0.48, 0.25)
    inferred: list[dict[str, Any]] = []
    previous_time = -np.inf

    for index, event in enumerate(default_events):
        expected_time = float(event["time_s"])
        waypoint_x = float(event["x"])
        waypoint_y = float(event["y"])

        if index == 0:
            chosen_time = max(float(times[0]), expected_time)
        elif index == len(default_events) - 1:
            chosen_time = min(float(times[-1]), expected_time)
        else:
            window_start = max(float(times[0]), expected_time - search_half_width, previous_time + 0.05)
            window_end = min(float(times[-1]), expected_time + search_half_width)
            in_window = (times >= window_start) & (times <= window_end)
            if in_window.any():
                distance = np.hypot(raw_x[in_window] - waypoint_x, raw_y[in_window] - waypoint_y)
                local_times = times[in_window]
                chosen_time = float(local_times[int(np.argmin(distance))])
            else:
                chosen_time = expected_time

        if chosen_time <= previous_time:
            chosen_time = previous_time + 0.05
        previous_time = chosen_time

        inferred.append(
            {
                **event,
                "time_s": round(chosen_time, 6),
                "time_s_initial_equal_segment": event["time_s"],
            }
        )

    return inferred


def load_or_create_manifest(manifest_path: Path, input_paths: list[Path]) -> dict[str, Any]:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "notes": [
                "Edit waypoint_events.time_s if manual waypoint timestamps are available.",
                "The generated defaults assume constant speed and equal duration per rectangle side.",
                "Ground truth must come from measured waypoint times for the final thesis result.",
            ],
            "waypoints": WAYPOINTS,
            "files": {},
        }

    files = manifest.setdefault("files", {})
    changed = False
    for path in input_paths:
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        if key in files:
            continue
        df = pd.read_csv(path)
        loop_count = infer_loop_count(path)
        files[key] = {
            "loop_count": loop_count,
            "output_name": output_name_for(path),
            "ground_truth_source": "estimated_equal_segment_duration",
            "waypoint_events": default_waypoint_events(df, loop_count),
        }
        changed = True

    if changed or not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def refresh_manifest_from_raw(manifest: dict[str, Any], input_paths: list[Path]) -> dict[str, Any]:
    files = manifest.setdefault("files", {})
    for path in input_paths:
        key = str(path.relative_to(ROOT)).replace("\\", "/")
        df = pd.read_csv(path)
        loop_count = int(files.get(key, {}).get("loop_count", infer_loop_count(path)))
        files[key] = {
            **files.get(key, {}),
            "loop_count": loop_count,
            "output_name": files.get(key, {}).get("output_name", output_name_for(path)),
            "ground_truth_source": "estimated_from_raw_trilateration_waypoint_times",
            "waypoint_events": infer_waypoint_events_from_raw(df, loop_count),
        }
    return manifest


def interpolate_ground_truth(times: np.ndarray, events: list[dict[str, Any]]) -> pd.DataFrame:
    events = sorted(events, key=lambda item: float(item["time_s"]))
    event_times = np.asarray([float(item["time_s"]) for item in events], dtype=float)
    event_x = np.asarray([float(item["x"]) for item in events], dtype=float)
    event_y = np.asarray([float(item["y"]) for item in events], dtype=float)

    if len(events) < 2:
        raise ValueError("At least two waypoint events are required.")
    if np.any(np.diff(event_times) <= 0):
        raise ValueError("Waypoint event times must be strictly increasing.")

    segment_idx = np.searchsorted(event_times, times, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, len(events) - 2)
    start_t = event_times[segment_idx]
    end_t = event_times[segment_idx + 1]
    denom = np.maximum(end_t - start_t, 1e-12)
    alpha = np.clip((times - start_t) / denom, 0.0, 1.0)

    gt_x = event_x[segment_idx] + alpha * (event_x[segment_idx + 1] - event_x[segment_idx])
    gt_y = event_y[segment_idx] + alpha * (event_y[segment_idx + 1] - event_y[segment_idx])
    segment_name = [
        f"{events[i]['waypoint']}_to_{events[i + 1]['waypoint']}"
        for i in segment_idx
    ]
    loop_id = [int(events[i].get("loop", 1)) for i in segment_idx]

    return pd.DataFrame(
        {
            "gt_x": gt_x,
            "gt_y": gt_y,
            "x_true": gt_x,
            "y_true": gt_y,
            "gt_loop_id": loop_id,
            "gt_segment": segment_name,
            "gt_segment_start_s": start_t,
            "gt_segment_end_s": end_t,
            "gt_segment_progress": alpha,
        }
    )


def write_ground_truth_files(manifest: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for key, spec in manifest["files"].items():
        input_path = resolve(key)
        if not input_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {input_path}")

        df = pd.read_csv(input_path)
        if "time" not in df.columns:
            raise ValueError(f"{input_path} does not contain a 'time' column.")

        gt = interpolate_ground_truth(df["time"].to_numpy(dtype=float), spec["waypoint_events"])
        out = pd.concat([df.reset_index(drop=True), gt], axis=1)
        out["gt_source"] = spec.get("ground_truth_source", "manual_waypoint_interpolation")
        out["loop_count_assumption"] = int(spec.get("loop_count", 1))
        out["source_raw_file"] = input_path.name

        output_name = spec.get("output_name") or output_name_for(input_path)
        output_path = output_dir / output_name
        out.to_csv(output_path, index=False)

        summary_rows.append(
            {
                "input_file": key,
                "output_file": str(output_path.relative_to(ROOT)).replace("\\", "/"),
                "rows": len(out),
                "loop_count": int(spec.get("loop_count", 1)),
                "gt_source": out["gt_source"].iloc[0],
                "time_start_s": float(out["time"].iloc[0]),
                "time_end_s": float(out["time"].iloc[-1]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "manifest_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate waypoint-interpolated ground truth for latest UWB CSV files.")
    parser.add_argument("--manifest", default="configs/latest_waypoint_manifest.json")
    parser.add_argument("--manual-times", default=None)
    parser.add_argument("--output-dir", default="Data eksperimen/latest_waypoint_ground_truth")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUT_FILES)
    parser.add_argument(
        "--infer-events-from-raw",
        action="store_true",
        help="Refresh waypoint event times from nearest raw trilateration positions in each expected segment window.",
    )
    args = parser.parse_args()

    input_paths = [resolve(path) for path in args.inputs]
    missing = [path for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input CSV files: {missing}")

    manifest_path = resolve(args.manifest)
    output_dir = resolve(args.output_dir)
    manifest = load_or_create_manifest(manifest_path, input_paths)
    manual_specs = load_manual_specs(resolve(args.manual_times) if args.manual_times else None)
    if manual_specs:
        manifest = apply_manual_specs(manifest, manual_specs)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.infer_events_from_raw:
        manifest = refresh_manifest_from_raw(manifest, input_paths)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_ground_truth_files(manifest, output_dir)

    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")
    print("Generated waypoint ground-truth CSV files.")


if __name__ == "__main__":
    main()
