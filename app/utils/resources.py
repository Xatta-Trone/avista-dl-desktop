"""Application resource path resolution for development and packaged builds."""

from __future__ import annotations

import sys
from pathlib import Path


def is_packaged_application() -> bool:
    """Return whether AVISTA is running from a supported packaged build."""

    return bool(
        getattr(sys, "frozen", False)
        or globals().get("__compiled__") is not None
    )


def get_app_resource_path(
    relative_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> Path:
    """Resolve a project/app resource in development or packaged builds."""

    relative = Path(relative_path)
    if relative.is_absolute():
        return relative.resolve()

    candidates: list[Path] = []
    if project_dir is not None:
        candidates.append(Path(project_dir).resolve())
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        candidates.append(Path(pyinstaller_root))
    if is_packaged_application():
        candidates.append(Path(sys.executable).resolve().parent)
    candidates.append(Path(__file__).resolve().parents[2])

    for root in candidates:
        candidate = (root / relative).resolve()
        if candidate.exists():
            return candidate
    return (candidates[0] / relative).resolve()
