from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "dataset_baru"
OUTPUT_DIR = ROOT / "Data eksperimen" / "dataset_baru_target_ground_truth"

STATIC_POINTS = {
    "1": (1.0, 1.0),
    "2": (3.0, 1.0),
    "3": (3.0, 3.0),
    "4": (1.0, 3.0),
    "5": (2.0, 2.0),
}


def add_static_ground_truth(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    point_id = path.stem.split("_")[1]
    x, y = STATIC_POINTS[point_id]

    df["gt_x"] = x
    df["gt_y"] = y
    df["x_true"] = x
    df["y_true"] = y
    df["gt_segment"] = f"static_P{point_id}"
    df["gt_loop_id"] = 0
    df["gt_source"] = "static_point_from_filename"
    df["source_raw_file"] = path.name

    output_path = OUTPUT_DIR / f"{path.stem}_gt.csv"
    df.to_csv(output_path, index=False)
    return {
        "input": str(path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "rows": len(df),
        "events": 1,
        "mode": "static",
        "gt_source": "static_point_from_filename",
    }


def add_target_event_ground_truth(path: Path, segments_per_loop: int) -> dict[str, object]:
    df = pd.read_csv(path)
    events = df[df["target_x"].notna() & df["target_y"].notna()][["time", "target_x", "target_y"]].copy()
    if len(events) < 2:
        raise ValueError(f"Not enough target_x/target_y waypoint events: {path}")

    events = events.astype(float).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    times = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    event_t = events["time"].to_numpy(dtype=float)
    event_x = events["target_x"].to_numpy(dtype=float)
    event_y = events["target_y"].to_numpy(dtype=float)

    segment_idx = np.searchsorted(event_t, times, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, len(event_t) - 2)
    start_t = event_t[segment_idx]
    end_t = event_t[segment_idx + 1]
    alpha = np.clip((times - start_t) / np.maximum(end_t - start_t, 1e-12), 0.0, 1.0)

    gt_x = event_x[segment_idx] + alpha * (event_x[segment_idx + 1] - event_x[segment_idx])
    gt_y = event_y[segment_idx] + alpha * (event_y[segment_idx + 1] - event_y[segment_idx])
    before_first = times <= event_t[0]
    after_last = times >= event_t[-1]
    gt_x[before_first] = event_x[0]
    gt_y[before_first] = event_y[0]
    gt_x[after_last] = event_x[-1]
    gt_y[after_last] = event_y[-1]

    df["gt_x"] = gt_x
    df["gt_y"] = gt_y
    df["x_true"] = gt_x
    df["y_true"] = gt_y
    df["gt_segment"] = [
        f"P{int(i % segments_per_loop) + 1}_to_P{int((i + 1) % segments_per_loop) + 1}" for i in segment_idx
    ]
    df["gt_loop_id"] = (segment_idx // segments_per_loop + 1).astype(int)
    df["gt_source"] = "interpolated_from_target_xy_waypoint_events"
    df["source_raw_file"] = path.name

    output_path = OUTPUT_DIR / f"{path.stem}_gt.csv"
    df.to_csv(output_path, index=False)
    return {
        "input": str(path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "rows": len(df),
        "events": len(events),
        "mode": "target_xy_events",
        "gt_source": "interpolated_from_target_xy_waypoint_events",
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for path in sorted((INPUT_DIR / "Diam").glob("*.csv")):
        summary.append(add_static_ground_truth(path))
    for path in sorted((INPUT_DIR / "MAJU (L)").glob("*.csv")):
        summary.append(add_target_event_ground_truth(path, segments_per_loop=4))
    for path in sorted((INPUT_DIR / "segitiga").glob("*.csv")):
        summary.append(add_target_event_ground_truth(path, segments_per_loop=3))

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "manifest_summary.csv", index=False)
    print(f"Wrote {len(summary_df)} ground-truth CSV files to {OUTPUT_DIR}")
    print(summary_df[["input", "rows", "events", "mode"]].to_string(index=False))


if __name__ == "__main__":
    main()
