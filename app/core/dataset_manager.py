"""Portable project dataset management."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


SUPPORTED_DATASET_EXTENSIONS = {
    ".csv",
    ".xlsx",
    ".parquet",
    ".feather",
    ".fst",
}
DUPLICATE_APPEND_TIMESTAMP = "append_timestamp"
DUPLICATE_OVERWRITE = "overwrite"
DUPLICATE_CANCEL = "cancel"
DUPLICATE_POLICIES = {
    DUPLICATE_APPEND_TIMESTAMP,
    DUPLICATE_OVERWRITE,
    DUPLICATE_CANCEL,
}


def copy_dataset_into_project(
    source: str | Path,
    project_dir: str | Path,
    *,
    duplicate_policy: str = DUPLICATE_APPEND_TIMESTAMP,
    now: Callable[[], datetime] = datetime.now,
) -> dict[str, Any]:
    """Copy a supported dataset into ``data/`` and return project metadata."""

    source_path = Path(source).resolve()
    project_path = Path(project_dir).resolve()
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {source_path}")
    extension = source_path.suffix.casefold()
    if extension not in SUPPORTED_DATASET_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DATASET_EXTENSIONS))
        raise ValueError(
            f"Unsupported dataset format '{extension}'. Supported formats: {supported}."
        )
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise ValueError(f"Unknown duplicate dataset policy: {duplicate_policy}")

    data_dir = project_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / source_path.name
    same_file = _same_file(source_path, destination)
    if destination.exists() and not same_file:
        if duplicate_policy == DUPLICATE_CANCEL:
            raise FileExistsError(f"Dataset copy cancelled: {destination} already exists.")
        if duplicate_policy == DUPLICATE_APPEND_TIMESTAMP:
            destination = _timestamped_destination(destination, now())
    if not same_file:
        shutil.copy2(source_path, destination)

    relative_path = destination.relative_to(project_path).as_posix()
    return {
        "project_relative_path": relative_path,
        "original_source_path": str(source_path),
        "copied_project_path": relative_path,
        "copied_to_project": True,
        "file_size": int(destination.stat().st_size),
        "copy_timestamp": now().isoformat(timespec="seconds"),
    }


def project_dataset_path(config: Any) -> Path | None:
    """Resolve the current project-owned dataset path."""

    dataset = dict(getattr(config, "dataset", {}) or {})
    relative_path = str(dataset.get("project_relative_path", "")).strip()
    if relative_path:
        return (Path(config.project_dir) / Path(relative_path)).resolve()
    input_file = str(getattr(config, "input_file", "")).strip()
    return Path(input_file).resolve() if input_file else None


def _timestamped_destination(destination: Path, timestamp: datetime) -> Path:
    suffix = timestamp.strftime("%Y%m%d_%H%M%S")
    candidate = destination.with_name(
        f"{destination.stem}_{suffix}{destination.suffix}"
    )
    counter = 2
    while candidate.exists():
        candidate = destination.with_name(
            f"{destination.stem}_{suffix}_{counter}{destination.suffix}"
        )
        counter += 1
    return candidate


def _same_file(source: Path, destination: Path) -> bool:
    try:
        return source.samefile(destination)
    except (FileNotFoundError, OSError):
        return source == destination.resolve()
