"""Dependency validation helpers."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.environment_manager import resolve_environment_path


_PACKAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_imports(package_names: list[str]) -> dict[str, Any]:
    """Check whether packages can be resolved without importing them."""

    results = {}
    missing = []

    for package_name in package_names:
        try:
            found = importlib.util.find_spec(package_name) is not None
        except (ImportError, AttributeError, ValueError):
            found = False

        results[package_name] = found
        if not found:
            missing.append(package_name)

    return {
        "success": not missing,
        "packages": results,
        "missing": missing,
    }


def check_optional_packages(
    package_names: list[str],
    *,
    project_dir: str | Path,
    environment_mode: str,
    app_root: str | Path,
) -> dict[str, Any]:
    """Check package availability in the configured active environment."""

    packages = list(dict.fromkeys(package_names))
    environment = resolve_environment_path(project_dir, app_root, environment_mode)
    python_path = Path(environment["python"])
    if not python_path.exists():
        return {
            "success": False,
            "packages": {package: False for package in packages},
            "missing": packages,
            "python": str(python_path),
            "error": f"Python executable not found: {python_path}",
        }

    script = (
        "import importlib.util, json, sys; "
        "names=json.loads(sys.argv[1]); "
        "print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", script, json.dumps(packages)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "Dependency check failed.")
        statuses = json.loads(result.stdout.strip())
        missing = [package for package in packages if not statuses.get(package, False)]
        return {
            "success": not missing,
            "packages": statuses,
            "missing": missing,
            "python": str(python_path),
            "error": None,
        }
    except Exception as exc:
        return {
            "success": False,
            "packages": {package: False for package in packages},
            "missing": packages,
            "python": str(python_path),
            "error": str(exc),
        }


def install_optional_package(
    package_name: str,
    *,
    project_dir: str | Path,
    environment_mode: str,
    app_root: str | Path,
) -> dict[str, Any]:
    """Install one package into the configured environment and log pip output."""

    if not _PACKAGE_NAME_PATTERN.fullmatch(package_name):
        raise ValueError(f"Invalid package name '{package_name}'.")

    project_path = Path(project_dir)
    environment = resolve_environment_path(project_path, app_root, environment_mode)
    python_path = Path(environment["python"])
    log_path = project_path / "install_log.txt"
    started_at = datetime.now(timezone.utc).isoformat()
    command = [str(python_path), "-m", "pip", "install", package_name]

    if not python_path.exists():
        error = f"Python executable not found: {python_path}"
        _append_package_install_log(log_path, started_at, command, None, "", error)
        return {
            "success": False,
            "package": package_name,
            "python": str(python_path),
            "log_path": str(log_path),
            "returncode": None,
            "error": error,
        }

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        error = None
        if result.returncode != 0:
            error = result.stderr.strip() or result.stdout.strip() or "pip install failed"
        _append_package_install_log(
            log_path,
            started_at,
            command,
            result.returncode,
            result.stdout,
            result.stderr,
        )
        return {
            "success": result.returncode == 0,
            "package": package_name,
            "python": str(python_path),
            "log_path": str(log_path),
            "returncode": result.returncode,
            "error": error,
        }
    except Exception as exc:
        _append_package_install_log(log_path, started_at, command, None, "", str(exc))
        return {
            "success": False,
            "package": package_name,
            "python": str(python_path),
            "log_path": str(log_path),
            "returncode": None,
            "error": str(exc),
        }


def _append_package_install_log(
    log_path: Path,
    started_at: str,
    command: list[str],
    returncode: int | None,
    stdout: str,
    stderr: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            "\n".join(
                [
                    "",
                    f"started_at={started_at}",
                    f"command={' '.join(command)}",
                    f"returncode={returncode}",
                    "",
                    "STDOUT:",
                    stdout,
                    "",
                    "STDERR:",
                    stderr,
                    "",
                ]
            )
        )
