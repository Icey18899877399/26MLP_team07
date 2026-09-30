"""Commands and output inspection for the original, unmodified figure scripts."""

from __future__ import annotations

import sys
from pathlib import Path

from .original_catalog import list_original_experiments, original_experiment_spec, original_root


def build_original_command(experiment_id: str, output_dir: Path) -> list[str]:
    """Build a full-default command with only the original data and output CLI flags."""
    _, _, paths, _, _ = original_experiment_spec(experiment_id)
    base = [sys.executable, "-m", f"visualization.{experiment_id}_figures"]
    root = original_root()
    for path in paths:
        if not (root / path).is_file():
            raise FileNotFoundError(f"Original source data unavailable: {path}")
    if experiment_id == "one_class_svm":
        return [*base, "--data-dir", str(root / "data" / "anomaly"), "--output-dir", str(output_dir)]
    if experiment_id == "isolation_forest":
        return [*base, "--cardio", str(root / paths[0]), "--mammography", str(root / paths[1]), "--output", str(output_dir)]
    return [*base, "--data", str(root / paths[0]), "--output", str(output_dir)]


def resolve_original_run_asset(output_dir: Path, filename: str) -> Path:
    """Constrain downloaded run files to regular PNG/CSV files below output_dir."""
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or "\\" in filename or relative.suffix.lower() not in {".png", ".csv"}:
        raise ValueError("Invalid run asset")
    root = output_dir.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Run asset unavailable")
    return target


def list_original_run_assets(experiment_id: str, output_dir: Path, run_id: str) -> tuple[list[dict], list[dict]]:
    """Describe verified output files using the HTTP job artifact URL format."""
    original_experiment_spec(experiment_id)
    prefix = f"/api/original-runs/{run_id}/assets/"
    figures = []
    source_data = []
    if not output_dir.is_dir():
        return figures, source_data
    for candidate in sorted(output_dir.rglob("*")):
        if candidate.suffix.lower() not in {".png", ".csv"} or not candidate.is_file():
            continue
        name = candidate.relative_to(output_dir).as_posix()
        try:
            resolve_original_run_asset(output_dir, name)
        except ValueError:
            continue
        descriptor = {"name": name, "url": prefix + name}
        if candidate.suffix.lower() == ".png":
            descriptor["title"] = candidate.stem.split("_", 1)[-1].replace("_", " ").title()
            figures.append(descriptor)
        else:
            source_data.append(descriptor)
    return figures, source_data
