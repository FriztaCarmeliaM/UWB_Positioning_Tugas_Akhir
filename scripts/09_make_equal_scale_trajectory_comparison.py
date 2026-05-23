from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwb_localization import plotting
from uwb_localization.config import load_config


def main() -> None:
    config = load_config(ROOT / "configs/uwb_pipeline_10loop_moretrain.yaml")
    out_dir = ROOT / "docs/results/previous_trajectory_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [
        (
            ROOT / "outputs/uwb_10loop_final_pipeline/20260518_212851/06_evaluation/predictions.csv",
            "10lup2_trilat_gt",
            out_dir / "before_10loop_baseline_test_equal_scale.png",
            "Before Final - 10-loop Baseline Test",
        ),
        (
            ROOT / "outputs/uwb_10loop_moretrain_pipeline/20260518_213455/06_evaluation/predictions.csv",
            "10lup2_trilat_gt",
            out_dir / "after_10loop_moretrain_test_equal_scale.png",
            "After Final - 10-loop More-train Test",
        ),
    ]

    for csv_path, trajectory, output_path, title in items:
        df = pd.read_csv(csv_path)
        part = df[df["trajectory"] == trajectory].copy()
        plotting.plot_trajectory(
            part,
            output_path,
            title,
            config,
            fixed_bounds=(0.5, 4.0, 0.5, 4.0),
        )
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
