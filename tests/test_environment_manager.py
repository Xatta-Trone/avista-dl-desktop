import json
import subprocess
from pathlib import Path

from app.core.dependency_manager import check_optional_packages, install_optional_package
from app.core.environment_manager import (
    get_managed_env_root,
    get_shared_cpu_env_path,
    get_shared_gpu_env_path,
    get_venv_pip,
    get_venv_python,
    resolve_environment_path,
    save_environment_info,
    collect_environment_info,
)


def test_get_venv_paths_for_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.platform.system", lambda: "Windows")

    assert get_venv_python(tmp_path) == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert get_venv_pip(tmp_path) == tmp_path / ".venv" / "Scripts" / "pip.exe"


def test_get_venv_paths_for_posix(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.platform.system", lambda: "Linux")

    assert get_venv_python(tmp_path) == tmp_path / ".venv" / "bin" / "python"
    assert get_venv_pip(tmp_path) == tmp_path / ".venv" / "bin" / "pip"


def test_save_environment_info(tmp_path):
    info = {"python_version": "3.12", "cuda_available": False}

    saved_path = save_environment_info(tmp_path, info)

    assert saved_path == Path(tmp_path) / "logs" / "environment_info.json"
    assert json.loads(saved_path.read_text(encoding="utf-8")) == info


def test_collect_environment_info_includes_cpu_memory_and_disk(tmp_path):
    info = collect_environment_info(project_dir=tmp_path)

    assert info["psutil_available"] is True
    assert info["cpu_name"]
    assert info["physical_cores"] is not None
    assert info["logical_cores"] is not None
    assert 0 <= info["cpu_usage_percent"] <= 100
    assert info["architecture"]
    assert info["ram_total_bytes"] > 0
    assert info["ram_available_bytes"] > 0
    assert 0 <= info["ram_used_percent"] <= 100
    assert info["disk_free_bytes"] >= 0


def test_collect_environment_info_falls_back_when_psutil_is_missing(
    monkeypatch, tmp_path
):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("missing psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    info = collect_environment_info(project_dir=tmp_path)

    assert info["psutil_available"] is False
    assert "psutil unavailable" in info["system_info_error"]
    assert info["architecture"]


def test_managed_env_paths(tmp_path):
    app_root = tmp_path / "app"

    assert get_managed_env_root(app_root) == app_root / "managed_envs"
    assert get_shared_cpu_env_path(app_root) == app_root / "managed_envs" / "cpu_env"
    assert get_shared_gpu_env_path(app_root) == app_root / "managed_envs" / "gpu_env"


def test_resolve_packaged_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.sys.executable", "C:/Python/python.exe")

    resolved = resolve_environment_path(tmp_path / "project", tmp_path / "app", "packaged_runtime")

    assert resolved["environment_mode"] == "packaged_runtime"
    assert resolved["env_dir"] is None
    assert resolved["python"] == Path("C:/Python/python.exe")
    assert resolved["pip"] is None


def test_resolve_shared_cpu_env(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.platform.system", lambda: "Windows")
    app_root = tmp_path / "app"

    resolved = resolve_environment_path(tmp_path / "project", app_root, "shared_cpu_env")

    assert resolved["env_dir"] == app_root / "managed_envs" / "cpu_env"
    assert resolved["python"] == app_root / "managed_envs" / "cpu_env" / "Scripts" / "python.exe"
    assert resolved["pip"] == app_root / "managed_envs" / "cpu_env" / "Scripts" / "pip.exe"


def test_resolve_shared_gpu_env(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.platform.system", lambda: "Linux")
    app_root = tmp_path / "app"

    resolved = resolve_environment_path(tmp_path / "project", app_root, "shared_gpu_env")

    assert resolved["env_dir"] == app_root / "managed_envs" / "gpu_env"
    assert resolved["python"] == app_root / "managed_envs" / "gpu_env" / "bin" / "python"
    assert resolved["pip"] == app_root / "managed_envs" / "gpu_env" / "bin" / "pip"


def test_resolve_project_isolated_env(monkeypatch, tmp_path):
    monkeypatch.setattr("app.core.environment_manager.platform.system", lambda: "Windows")
    project_dir = tmp_path / "project"

    resolved = resolve_environment_path(project_dir, tmp_path / "app", "project_isolated_env")

    assert resolved["env_dir"] == project_dir / ".venv"
    assert resolved["python"] == project_dir / ".venv" / "Scripts" / "python.exe"
    assert resolved["pip"] == project_dir / ".venv" / "Scripts" / "pip.exe"


def test_check_optional_packages_uses_active_environment_python(monkeypatch, tmp_path):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.touch()

    def fake_run(command, **_kwargs):
        assert command[0] == str(python_path)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout='{"xgboost": true, "tabpfn": false}',
            stderr="",
        )

    monkeypatch.setattr("app.core.dependency_manager.subprocess.run", fake_run)
    result = check_optional_packages(
        ["xgboost", "tabpfn"],
        project_dir=tmp_path,
        environment_mode="project_isolated_env",
        app_root=tmp_path,
    )

    assert result["packages"] == {"xgboost": True, "tabpfn": False}
    assert result["missing"] == ["tabpfn"]


def test_install_optional_package_logs_pip_output(monkeypatch, tmp_path):
    python_path = tmp_path / ".venv" / "Scripts" / "python.exe"
    python_path.parent.mkdir(parents=True)
    python_path.touch()

    def fake_run(command, **_kwargs):
        assert command == [str(python_path), "-m", "pip", "install", "xgboost"]
        return subprocess.CompletedProcess(command, 0, stdout="installed", stderr="")

    monkeypatch.setattr("app.core.dependency_manager.subprocess.run", fake_run)
    result = install_optional_package(
        "xgboost",
        project_dir=tmp_path,
        environment_mode="project_isolated_env",
        app_root=tmp_path,
    )

    assert result["success"] is True
    log_text = (tmp_path / "install_log.txt").read_text(encoding="utf-8")
    assert "pip install xgboost" in log_text
    assert "installed" in log_text
