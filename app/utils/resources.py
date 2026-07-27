"""Application resource path resolution for development and packaged builds."""

from __future__ import annotations

import sys
from pathlib import Path


TABPFN_CHECKPOINT_FILENAME = (
    "tabpfn-v2.5-classifier-v2.5_default.ckpt"
)


def is_packaged_application() -> bool:
    """Return whether AVISTA is running from a supported packaged build."""

    main_module = sys.modules.get("__main__")
    return bool(
        getattr(sys, "frozen", False)
        or getattr(sys, "_MEIPASS", None)
        or globals().get("__compiled__") is not None
        or getattr(main_module, "__compiled__", None) is not None
    )


def packaged_resource_root() -> Path:
    """Return the directory containing collected packaged resources."""

    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        return Path(pyinstaller_root).resolve()
    if is_packaged_application():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def app_resource_candidates(
    relative_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> list[Path]:
    """Return ordered, de-duplicated candidates for an app resource."""

    relative = Path(relative_path)
    if relative.is_absolute():
        return [relative.resolve()]

    roots: list[Path] = []
    if project_dir is not None:
        roots.append(Path(project_dir).resolve())
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        roots.append(Path(pyinstaller_root).resolve())
    if is_packaged_application():
        roots.append(Path(sys.executable).resolve().parent)
    roots.append(Path(__file__).resolve().parents[2])

    candidates: list[Path] = []
    for root in roots:
        candidate = (root / relative).resolve()
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def get_app_resource_path(
    relative_path: str | Path,
    *,
    project_dir: str | Path | None = None,
) -> Path:
    """Resolve a project/app resource in development or packaged builds."""

    candidates = app_resource_candidates(
        relative_path,
        project_dir=project_dir,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def tabpfn_checkpoint_candidates() -> list[Path]:
    """Return supported source and packaged TabPFN checkpoint locations."""

    candidates = app_resource_candidates(
        Path("app") / "assets" / TABPFN_CHECKPOINT_FILENAME
    )
    for candidate in app_resource_candidates(
        Path("assets") / TABPFN_CHECKPOINT_FILENAME
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_tabpfn_checkpoint() -> Path:
    """Resolve the bundled TabPFN checkpoint or report every checked path."""

    candidates = tabpfn_checkpoint_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Bundled TabPFN checkpoint was not found. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def expected_packaged_xgboost_dll() -> Path:
    """Return the preferred installed XGBoost native-library location."""

    return (
        packaged_resource_root()
        / "xgboost"
        / "lib"
        / "xgboost.dll"
    ).resolve()
