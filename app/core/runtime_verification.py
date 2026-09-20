"""Runtime inventory saved with startup environment diagnostics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from app.__version__ import (
    APP_DESCRIPTION,
    APP_NAME,
    RELEASE_DATE,
    __version__,
)
from app.core.tabpfn_model_manager import get_tabpfn_model_status
from app.utils.resources import (
    get_app_resource_path,
    is_packaged_application,
)


APP_LOGO = "app/assets/logo.png"


def collect_runtime_verification(
    gpu_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect packaged-runtime, optional-package, and asset availability."""

    gpu = gpu_info or {}
    tabpfn_status = get_tabpfn_model_status()
    checkpoint = tabpfn_status.cache_path
    logo = get_app_resource_path(APP_LOGO)
    return {
        "app_name": APP_NAME,
        "app_description": APP_DESCRIPTION,
        "app_version": __version__,
        "app_release_date": RELEASE_DATE,
        "bundled_python_path": sys.executable,
        "packaged_runtime": is_packaged_application(),
        "torch_version": gpu.get("torch_version"),
        "cuda_available": bool(gpu.get("cuda_available")),
        "gpu_name": gpu.get("gpu_name"),
        "xgboost_available": _module_available("xgboost"),
        "tabpfn_available": _module_available("tabpfn"),
        "tabpfn_checkpoint_path": str(checkpoint),
        "tabpfn_checkpoint_exists": checkpoint.is_file(),
        "tabpfn_checkpoint_source": tabpfn_status.active_checkpoint_source,
        "logo_path": str(logo),
        "logo_exists": logo.is_file(),
    }


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False
