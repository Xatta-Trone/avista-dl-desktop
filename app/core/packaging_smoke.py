"""Small source and frozen-runtime model checks used by release packaging."""

from __future__ import annotations

import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from app.__version__ import __version__
from app.utils.resources import is_packaged_application


def run_packaging_smoke(kind: str, output_path: str | Path) -> int:
    """Run one bounded model smoke check and persist a structured result."""

    normalized = kind.strip().casefold()
    result: dict[str, Any] = {
        "kind": normalized,
        "avista_version": __version__,
        "packaged": is_packaged_application(),
        "executable": sys.executable,
        "python_version": platform.python_version(),
        "python_architecture": platform.machine(),
    }
    try:
        if normalized == "xgboost":
            result.update(_xgboost_smoke())
        elif normalized == "tabpfn":
            result.update(_tabpfn_smoke())
        else:
            raise ValueError(f"Unsupported packaging smoke kind: {kind}")
        result["status"] = "passed"
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        exit_code = 1

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, default=str),
        encoding="utf-8",
    )
    return exit_code


def _xgboost_smoke() -> dict[str, Any]:
    import xgboost
    from xgboost import XGBClassifier

    package_path = Path(xgboost.__file__).resolve()
    dll_paths = sorted(
        str(path.resolve())
        for path in package_path.parent.rglob("xgboost.dll")
    )
    if not dll_paths:
        raise FileNotFoundError(
            "The xgboost Python package is importable, but xgboost.dll "
            "is absent from the packaged package directory."
        )
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    target = np.asarray([0, 1, 1, 0, 1, 1], dtype=np.int64)
    model = XGBClassifier(
        n_estimators=2,
        max_depth=2,
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    model.fit(features, target)
    predictions = model.predict(features[:2])
    return {
        "package": "xgboost",
        "package_version": str(getattr(xgboost, "__version__", "unknown")),
        "package_path": str(package_path),
        "native_libraries": dll_paths,
        "predictions": np.asarray(predictions).tolist(),
    }


def _tabpfn_smoke() -> dict[str, Any]:
    import joblib
    import sklearn
    import tabpfn
    import torch
    from tabpfn import TabPFNClassifier

    from app.core.tabpfn_model_manager import get_tabpfn_model_status

    status = get_tabpfn_model_status()
    return {
        "package": "tabpfn",
        "package_version": str(getattr(tabpfn, "__version__", "unknown")),
        "package_path": str(Path(tabpfn.__file__).resolve()),
        "torch_version": str(torch.__version__),
        "sklearn_version": str(sklearn.__version__),
        "joblib_version": str(joblib.__version__),
        "classifier_imported": TabPFNClassifier is not None,
        "checkpoint_path": str(status.cache_path),
        "checkpoint_exists": status.cache_checkpoint_exists,
        "checkpoint_source": status.active_checkpoint_source,
        "checkpoint_required_for_package_smoke": False,
    }
