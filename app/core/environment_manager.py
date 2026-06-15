"""Environment management helpers for AVISTA projects."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGED_RUNTIME = "packaged_runtime"
SHARED_CPU_ENV = "shared_cpu_env"
SHARED_GPU_ENV = "shared_gpu_env"
PROJECT_ISOLATED_ENV = "project_isolated_env"
ENVIRONMENT_MODES = {
    PACKAGED_RUNTIME,
    SHARED_CPU_ENV,
    SHARED_GPU_ENV,
    PROJECT_ISOLATED_ENV,
}


def _venv_dir(project_dir: str | Path) -> Path:
    return Path(project_dir) / ".venv"


def _is_windows() -> bool:
    return platform.system().lower() == "windows"


def get_venv_python(project_dir: str | Path) -> Path:
    """Return the expected Python executable path for a project venv."""

    venv_dir = _venv_dir(project_dir)
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def get_venv_pip(project_dir: str | Path) -> Path:
    """Return the expected pip executable path for a project venv."""

    venv_dir = _venv_dir(project_dir)
    if _is_windows():
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def get_managed_env_root(app_root: str | Path) -> Path:
    """Return the root directory for shared managed environments."""

    return Path(app_root) / "managed_envs"


def get_shared_cpu_env_path(app_root: str | Path) -> Path:
    """Return the shared CPU environment directory."""

    return get_managed_env_root(app_root) / "cpu_env"


def get_shared_gpu_env_path(app_root: str | Path) -> Path:
    """Return the shared GPU environment directory."""

    return get_managed_env_root(app_root) / "gpu_env"


def resolve_environment_path(
    project_dir: str | Path,
    app_root: str | Path,
    environment_mode: str,
) -> dict[str, Any]:
    """Resolve the Python runtime path for a configured environment mode."""

    mode = str(environment_mode or PACKAGED_RUNTIME).strip().lower()
    if mode not in ENVIRONMENT_MODES:
        raise ValueError(f"Unsupported environment_mode '{environment_mode}'.")

    if mode == PACKAGED_RUNTIME:
        return {
            "environment_mode": mode,
            "env_dir": None,
            "python": Path(sys.executable),
            "pip": None,
        }

    if mode == SHARED_CPU_ENV:
        env_dir = get_shared_cpu_env_path(app_root)
    elif mode == SHARED_GPU_ENV:
        env_dir = get_shared_gpu_env_path(app_root)
    else:
        env_dir = _venv_dir(project_dir)

    return {
        "environment_mode": mode,
        "env_dir": env_dir,
        "python": _env_python(env_dir),
        "pip": _env_pip(env_dir),
    }


def create_venv(project_dir: str | Path) -> dict[str, Any]:
    """Create a project-local .venv if needed."""

    project_path = Path(project_dir)
    venv_path = _venv_dir(project_path)
    python_path = get_venv_python(project_path)

    try:
        project_path.mkdir(parents=True, exist_ok=True)
        if not python_path.exists():
            venv.create(venv_path, with_pip=True)

        return {
            "success": python_path.exists(),
            "venv_dir": str(venv_path),
            "python": str(python_path),
            "pip": str(get_venv_pip(project_path)),
            "error": None if python_path.exists() else "Virtual environment Python was not created.",
        }
    except Exception as exc:
        return {
            "success": False,
            "venv_dir": str(venv_path),
            "python": str(python_path),
            "pip": str(get_venv_pip(project_path)),
            "error": str(exc),
        }


def install_requirements(project_dir: str | Path, requirements_file: str | Path) -> dict[str, Any]:
    """Install a requirements file into the project venv and write install_log.txt."""

    project_path = Path(project_dir)
    requirements_path = Path(requirements_file)
    pip_path = get_venv_pip(project_path)
    log_path = project_path / "install_log.txt"
    started_at = datetime.now(timezone.utc).isoformat()

    if not pip_path.exists():
        message = f"pip executable not found: {pip_path}"
        _write_install_log(log_path, started_at, requirements_path, message)
        return {"success": False, "log_path": str(log_path), "error": message, "returncode": None}

    if not requirements_path.exists():
        message = f"requirements file not found: {requirements_path}"
        _write_install_log(log_path, started_at, requirements_path, message)
        return {"success": False, "log_path": str(log_path), "error": message, "returncode": None}

    try:
        result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        log_text = "\n".join(
            [
                f"started_at={started_at}",
                f"requirements_file={requirements_path}",
                f"returncode={result.returncode}",
                "",
                "STDOUT:",
                result.stdout,
                "",
                "STDERR:",
                result.stderr,
            ]
        )
        log_path.write_text(log_text, encoding="utf-8")
        return {
            "success": result.returncode == 0,
            "log_path": str(log_path),
            "error": None if result.returncode == 0 else result.stderr.strip() or "pip install failed",
            "returncode": result.returncode,
        }
    except Exception as exc:
        _write_install_log(log_path, started_at, requirements_path, str(exc))
        return {"success": False, "log_path": str(log_path), "error": str(exc), "returncode": None}


def save_environment_info(project_dir: str | Path, info: dict[str, Any]) -> Path:
    """Save environment metadata to environment_info.json."""

    output_path = Path(project_dir) / "logs" / "environment_info.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return output_path


def collect_environment_info(
    extra_info: dict[str, Any] | None = None,
    *,
    project_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Collect runtime, CPU, memory, and optional project-drive metadata."""

    info: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "system": platform.system(),
        "architecture": platform.machine() or platform.architecture()[0],
        "python_version": sys.version,
        "python_executable": sys.executable,
        "psutil_available": False,
        "system_info_error": None,
    }
    try:
        import psutil

        info.update(_collect_psutil_info(psutil, project_dir))
        info["psutil_available"] = True
    except Exception as exc:
        info.update(
            {
                "cpu_name": platform.processor() or "Unknown",
                "physical_cores": None,
                "logical_cores": None,
                "cpu_usage_percent": None,
                "ram_total_bytes": None,
                "ram_available_bytes": None,
                "ram_used_percent": None,
                "disk_total_bytes": None,
                "disk_free_bytes": None,
                "system_info_error": f"psutil unavailable: {exc}",
            }
        )
    if extra_info:
        info.update(extra_info)
    return info


def _collect_psutil_info(psutil_module: Any, project_dir: str | Path | None) -> dict[str, Any]:
    memory = psutil_module.virtual_memory()
    disk_path = Path(project_dir).resolve() if project_dir else Path.cwd().resolve()
    disk = psutil_module.disk_usage(str(disk_path))
    return {
        "cpu_name": platform.processor() or platform.machine() or "Unknown",
        "physical_cores": psutil_module.cpu_count(logical=False),
        "logical_cores": psutil_module.cpu_count(logical=True),
        "cpu_usage_percent": float(psutil_module.cpu_percent(interval=None)),
        "ram_total_bytes": int(memory.total),
        "ram_available_bytes": int(memory.available),
        "ram_used_percent": float(memory.percent),
        "disk_total_bytes": int(disk.total),
        "disk_free_bytes": int(disk.free),
    }


def _write_install_log(log_path: Path, started_at: str, requirements_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"started_at={started_at}",
                f"requirements_file={requirements_path}",
                "",
                message,
            ]
        ),
        encoding="utf-8",
    )


def _env_python(env_dir: Path) -> Path:
    if _is_windows():
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def _env_pip(env_dir: Path) -> Path:
    if _is_windows():
        return env_dir / "Scripts" / "pip.exe"
    return env_dir / "bin" / "pip"
