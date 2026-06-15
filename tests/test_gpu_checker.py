import builtins
import types

from app.core.gpu_checker import (
    _python_for_project,
    check_gpu,
    check_nvidia_smi,
    repair_gpu_torch,
    validate_torch_cuda,
)


def test_check_gpu_fallback_when_torch_unavailable(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("No module named torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(
        "app.core.gpu_checker.check_nvidia_smi",
        lambda: {
            "available": False,
            "gpu_detected": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": "nvidia-smi not found",
        },
    )

    info = check_gpu()

    assert info["cuda_available"] is False
    assert info["gpu_count"] == 0
    assert info["gpu_name"] is None
    assert info["torch_cuda_version"] is None
    assert info["tensor_test_passed"] is False
    assert "PyTorch unavailable" in info["error"]


def test_validate_torch_cuda_available(monkeypatch):
    fake_torch = types.SimpleNamespace(
        __version__="2.5.0+cu126",
        version=types.SimpleNamespace(cuda="12.6"),
        backends=types.SimpleNamespace(
            cudnn=types.SimpleNamespace(version=lambda: 90100)
        ),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: "Test GPU",
        ),
        tensor=lambda values, device=None: types.SimpleNamespace(item=lambda: 1.0),
    )
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake_torch
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    info = validate_torch_cuda()

    assert info["cuda_available"] is True
    assert info["gpu_count"] == 1
    assert info["gpu_name"] == "Test GPU"
    assert info["torch_cuda_version"] == "12.6"
    assert info["torch_installed"] is True
    assert info["cudnn_version"] == 90100
    assert info["tensor_test_passed"] is True


def test_check_gpu_cuda_unavailable_but_nvidia_smi_available(monkeypatch):
    monkeypatch.setattr(
        "app.core.gpu_checker.validate_torch_cuda",
        lambda: {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": None,
            "torch_cuda_version": None,
            "torch_version": "2.5.0+cpu",
            "tensor_test_passed": False,
            "error": None,
            "repair_recommended": False,
        },
    )
    monkeypatch.setattr(
        "app.core.gpu_checker.check_nvidia_smi",
        lambda: {
            "available": True,
            "gpu_detected": True,
            "returncode": 0,
            "stdout": "NVIDIA RTX",
            "stderr": "",
            "error": None,
        },
    )

    info = check_gpu()

    assert info["cuda_available"] is False
    assert info["nvidia_gpu_detected"] is True
    assert info["repair_recommended"] is True
    assert "CPU-only or CUDA-mismatched" in info["gpu_status_message"]


def test_check_nvidia_smi_reports_driver_and_memory(monkeypatch):
    def fake_run(command, **_kwargs):
        assert "driver_version" in command[1]
        return types.SimpleNamespace(
            returncode=0,
            stdout="NVIDIA RTX 4090, 555.99, 24564, 1024, 23540\n",
            stderr="",
        )

    monkeypatch.setattr("app.core.gpu_checker.subprocess.run", fake_run)

    info = check_nvidia_smi()

    assert info["gpu_detected"] is True
    assert info["gpu_name"] == "NVIDIA RTX 4090"
    assert info["driver_version"] == "555.99"
    assert info["gpu_memory_total_mb"] == 24564
    assert info["gpu_memory_used_mb"] == 1024
    assert info["gpu_memory_free_mb"] == 23540


def test_check_nvidia_smi_unavailable(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("app.core.gpu_checker.subprocess.run", fake_run)

    info = check_nvidia_smi()

    assert info["available"] is False
    assert info["gpu_detected"] is False
    assert info["error"] == "nvidia-smi not found"


def test_repair_gpu_torch_falls_back_to_cuda_118(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(
        "app.core.gpu_checker.validate_torch_cuda",
        lambda: {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": None,
            "torch_cuda_version": None,
            "torch_version": "2.5.0+cpu",
            "tensor_test_passed": False,
            "error": None,
        },
    )
    monkeypatch.setattr(
        "app.core.gpu_checker.check_nvidia_smi",
        lambda: {
            "available": True,
            "gpu_detected": True,
            "returncode": 0,
            "stdout": "NVIDIA RTX",
            "stderr": "",
            "error": None,
        },
    )
    monkeypatch.setattr("app.core.gpu_checker._python_for_project", lambda project_dir: tmp_path / "python.exe")

    def fake_run(command, capture_output=True, text=True, check=False):
        commands.append(command)
        index_url = command[-1] if "--index-url" in command else ""
        returncode = 1 if index_url.endswith("/cu126") else 0
        return types.SimpleNamespace(returncode=returncode, stdout="ok", stderr="failed" if returncode else "")

    monkeypatch.setattr("app.core.gpu_checker.subprocess.run", fake_run)

    result = repair_gpu_torch(tmp_path)

    assert result["success"] is False
    assert any("uninstall" in command for command in commands)
    assert any("https://download.pytorch.org/whl/cu126" in command for command in commands)
    assert any("https://download.pytorch.org/whl/cu118" in command for command in commands)
    assert (tmp_path / "install_log.txt").exists()


def test_gpu_repair_uses_active_avista_runtime_python(monkeypatch, tmp_path):
    active_python = tmp_path / "AVISTA" / "python.exe"
    monkeypatch.setattr("app.core.gpu_checker.sys.executable", str(active_python))
    project_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    project_python.parent.mkdir(parents=True)
    project_python.touch()

    assert _python_for_project(tmp_path) == active_python
