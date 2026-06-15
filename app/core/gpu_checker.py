"""GPU availability checks and explicit CUDA PyTorch repair helpers."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CUDA_INDEX_URLS = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu118": "https://download.pytorch.org/whl/cu118",
}


def check_gpu() -> dict[str, Any]:
    """Return CUDA/GPU information without installing or uninstalling packages."""

    info = validate_torch_cuda()
    smi_info = check_nvidia_smi()
    info["nvidia_smi_available"] = smi_info["available"]
    info["nvidia_gpu_detected"] = smi_info["gpu_detected"]
    info["nvidia_smi_output"] = smi_info["stdout"]
    info["nvidia_smi_gpu_name"] = smi_info.get("gpu_name")
    info["driver_version"] = smi_info.get("driver_version")
    if not info.get("gpu_name"):
        info["gpu_name"] = smi_info.get("gpu_name")
    if info.get("gpu_memory_total_mb") is None:
        info["gpu_memory_total_mb"] = smi_info.get("gpu_memory_total_mb")
    if info.get("gpu_memory_free_mb") is None:
        info["gpu_memory_free_mb"] = smi_info.get("gpu_memory_free_mb")
    info["gpu_memory_used_mb"] = _used_memory(
        info.get("gpu_memory_total_mb"),
        info.get("gpu_memory_free_mb"),
    )

    if info["cuda_available"] and info["tensor_test_passed"]:
        info["gpu_status_message"] = "CUDA is available and the tensor test passed."
        info["repair_recommended"] = False
    elif smi_info["gpu_detected"]:
        info["gpu_status_message"] = (
            "GPU detected, but current PyTorch is CPU-only or CUDA-mismatched."
        )
        info["repair_recommended"] = True
    else:
        info["gpu_status_message"] = "No NVIDIA GPU/driver detected."
        info["repair_recommended"] = False

    return info


def validate_torch_cuda() -> dict[str, Any]:
    """Validate PyTorch CUDA availability and run a small CUDA tensor test."""

    info: dict[str, Any] = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "torch_cuda_version": None,
        "torch_version": None,
        "torch_installed": False,
        "cudnn_version": None,
        "tensor_test_passed": False,
        "error": None,
        "nvidia_smi_available": None,
        "nvidia_gpu_detected": None,
        "nvidia_smi_output": None,
        "gpu_status_message": None,
        "repair_recommended": False,
        "gpu_memory_total_mb": None,
        "gpu_memory_free_mb": None,
        "gpu_memory_used_mb": None,
        "driver_version": None,
        "nvidia_smi_gpu_name": None,
    }

    try:
        import torch
    except Exception as exc:
        info["error"] = f"PyTorch unavailable: {exc}"
        return info

    try:
        info["torch_installed"] = True
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["gpu_count"] = int(torch.cuda.device_count()) if info["cuda_available"] else 0
        info["torch_cuda_version"] = torch.version.cuda
        backends = getattr(torch, "backends", None)
        cudnn = getattr(backends, "cudnn", None)
        if cudnn is not None:
            info["cudnn_version"] = cudnn.version()

        if info["cuda_available"] and info["gpu_count"] > 0:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            memory_info = _torch_memory_info(torch)
            info["gpu_memory_total_mb"] = memory_info["gpu_memory_total_mb"]
            info["gpu_memory_free_mb"] = memory_info["gpu_memory_free_mb"]
            info["gpu_memory_used_mb"] = _used_memory(
                info["gpu_memory_total_mb"],
                info["gpu_memory_free_mb"],
            )
            test_tensor = torch.tensor([1.0], device="cuda")
            info["tensor_test_passed"] = bool(test_tensor.item() == 1.0)

        return info
    except Exception as exc:
        info["error"] = str(exc)
        return info


def check_nvidia_smi() -> dict[str, Any]:
    """Run nvidia-smi to detect whether an NVIDIA GPU/driver is present."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "available": False,
            "gpu_detected": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": "nvidia-smi not found",
        }
    except Exception as exc:
        return {
            "available": False,
            "gpu_detected": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "error": str(exc),
        }

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    gpu_detected = result.returncode == 0 and bool(stdout.strip())
    parsed = _parse_nvidia_smi_query(stdout) if gpu_detected else {}
    return {
        "available": result.returncode == 0,
        "gpu_detected": gpu_detected,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "gpu_name": parsed.get("gpu_name"),
        "driver_version": parsed.get("driver_version"),
        "gpu_memory_total_mb": parsed.get("gpu_memory_total_mb"),
        "gpu_memory_used_mb": parsed.get("gpu_memory_used_mb"),
        "gpu_memory_free_mb": parsed.get("gpu_memory_free_mb"),
        "error": None if result.returncode == 0 else stderr.strip() or "nvidia-smi failed",
    }


def install_gpu_torch(project_dir: str | Path, cuda_version: str = "cu126") -> dict[str, Any]:
    """Install CUDA PyTorch into the project environment for a specific CUDA wheel index."""

    if cuda_version not in CUDA_INDEX_URLS:
        raise ValueError(f"Unsupported CUDA version '{cuda_version}'. Expected one of {sorted(CUDA_INDEX_URLS)}.")

    python_path = _python_for_project(project_dir)
    command = [
        str(python_path),
        "-m",
        "pip",
        "install",
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        CUDA_INDEX_URLS[cuda_version],
    ]
    return _run_logged_command(project_dir, command, f"install CUDA PyTorch {cuda_version}")


def repair_gpu_torch(project_dir: str | Path) -> dict[str, Any]:
    """Explicitly repair PyTorch CUDA when an NVIDIA GPU exists but CUDA is inactive."""

    log_entries: list[dict[str, Any]] = []
    initial_validation = validate_torch_cuda()
    if initial_validation["cuda_available"]:
        return {
            "success": True,
            "message": "PyTorch CUDA is already available.",
            "initial_validation": initial_validation,
            "nvidia_smi": None,
            "steps": log_entries,
            "final_validation": initial_validation,
        }

    smi_info = check_nvidia_smi()
    if not smi_info["gpu_detected"]:
        _append_install_log(project_dir, "No NVIDIA GPU/driver detected. Keeping CPU PyTorch.", smi_info)
        return {
            "success": False,
            "message": "No NVIDIA GPU/driver detected. Keeping CPU PyTorch.",
            "initial_validation": initial_validation,
            "nvidia_smi": smi_info,
            "steps": log_entries,
            "final_validation": initial_validation,
        }

    uninstall = _run_logged_command(
        project_dir,
        [str(_python_for_project(project_dir)), "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        "uninstall CPU PyTorch",
    )
    log_entries.append(uninstall)

    cu126 = install_gpu_torch(project_dir, "cu126")
    log_entries.append(cu126)
    if cu126["success"]:
        final_validation = validate_torch_cuda()
        if final_validation["cuda_available"]:
            return _repair_result(True, "CUDA PyTorch repair succeeded with CUDA 12.6.", initial_validation, smi_info, log_entries, final_validation)

    cu118 = install_gpu_torch(project_dir, "cu118")
    log_entries.append(cu118)
    final_validation = validate_torch_cuda()
    success = cu118["success"] and final_validation["cuda_available"]
    message = (
        "CUDA PyTorch repair succeeded with CUDA 11.8 fallback."
        if success
        else "CUDA PyTorch repair failed after CUDA 12.6 and CUDA 11.8 attempts."
    )
    return _repair_result(success, message, initial_validation, smi_info, log_entries, final_validation)


def _repair_result(
    success: bool,
    message: str,
    initial_validation: dict[str, Any],
    smi_info: dict[str, Any],
    steps: list[dict[str, Any]],
    final_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "initial_validation": initial_validation,
        "nvidia_smi": smi_info,
        "steps": steps,
        "final_validation": final_validation,
    }


def _torch_memory_info(torch_module: Any) -> dict[str, Any]:
    if hasattr(torch_module.cuda, "mem_get_info"):
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        return {
            "gpu_memory_total_mb": round(total_bytes / (1024 * 1024), 2),
            "gpu_memory_free_mb": round(free_bytes / (1024 * 1024), 2),
        }

    if hasattr(torch_module.cuda, "get_device_properties"):
        properties = torch_module.cuda.get_device_properties(0)
        return {
            "gpu_memory_total_mb": round(properties.total_memory / (1024 * 1024), 2),
            "gpu_memory_free_mb": None,
        }

    return {"gpu_memory_total_mb": None, "gpu_memory_free_mb": None}


def _parse_nvidia_smi_query(stdout: str) -> dict[str, Any]:
    first_line = next((line.strip() for line in stdout.splitlines() if line.strip()), "")
    if not first_line:
        return {}

    parts = [part.strip() for part in first_line.split(",")]
    parsed: dict[str, Any] = {"gpu_name": parts[0] if parts else None}
    if len(parts) >= 2:
        parsed["driver_version"] = parts[1]
    if len(parts) >= 3:
        parsed["gpu_memory_total_mb"] = _safe_float(parts[2])
    if len(parts) >= 4:
        parsed["gpu_memory_used_mb"] = _safe_float(parts[3])
    if len(parts) >= 5:
        parsed["gpu_memory_free_mb"] = _safe_float(parts[4])
    return parsed


def _used_memory(total: Any, free: Any) -> float | None:
    if total is None or free is None:
        return None
    return max(0.0, float(total) - float(free))


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _python_for_project(project_dir: str | Path) -> Path:
    """Return the active AVISTA runtime Python used by this process."""

    return Path(sys.executable)


def _run_logged_command(project_dir: str | Path, command: list[str], label: str) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        payload = {
            "label": label,
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
            "error": None if result.returncode == 0 else result.stderr.strip() or f"{label} failed",
        }
    except Exception as exc:
        payload = {
            "label": label,
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "success": False,
            "error": str(exc),
        }

    _append_install_log(project_dir, label, payload)
    return payload


def _append_install_log(project_dir: str | Path, label: str, payload: dict[str, Any]) -> None:
    log_path = Path(project_dir) / "install_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n[{timestamp}] {label}\n")
        for key, value in payload.items():
            log_file.write(f"{key}: {value}\n")
