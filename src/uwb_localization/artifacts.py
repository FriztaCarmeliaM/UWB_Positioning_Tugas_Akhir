from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import copy_config_to_run, resolve_path


def pipeline_base_dir(config: dict) -> Path:
    output_cfg = config.get("output", {})
    return resolve_path(output_cfg.get("base_dir", "outputs/uwb_calibrated_pipeline"))


def latest_run_file(config: dict) -> Path:
    return pipeline_base_dir(config) / "latest_run.txt"


def create_run_dir(config: dict) -> Path:
    base_dir = pipeline_base_dir(config)
    base_dir.mkdir(parents=True, exist_ok=True)
    run_name = config.get("output", {}).get("run_name")
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    suffix = 1
    original = run_dir
    while run_dir.exists():
        suffix += 1
        run_dir = original.with_name(f"{original.name}_{suffix:02d}")
    run_dir.mkdir(parents=True)
    (base_dir / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")
    copy_config_to_run(config, run_dir)
    return run_dir


def get_run_dir(config: dict, run_dir: str | Path | None = None, create: bool = False) -> Path:
    if run_dir is not None:
        path = resolve_path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if create:
        return create_run_dir(config)

    latest_path = latest_run_file(config)
    if not latest_path.exists():
        raise FileNotFoundError(
            f"No latest run found at {latest_path}. Run scripts/01_prepare_dataset.py first "
            "or pass --run-dir explicitly."
        )
    path = Path(latest_path.read_text(encoding="utf-8").strip())
    if not path.exists():
        raise FileNotFoundError(f"Latest run directory does not exist: {path}")
    return path


def stage_dir(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    path.mkdir(parents=True, exist_ok=True)
    return path

