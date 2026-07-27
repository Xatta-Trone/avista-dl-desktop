"""Fail-fast diagnostics for packages required by the Windows release."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import struct
import sys
from pathlib import Path
from typing import Any


PE_MACHINE_AMD64 = 0x8664


def pe_machine(path: Path) -> int:
    """Read the Windows PE machine value from an executable or DLL."""

    with path.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError(f"Not a PE file: {path}")
        handle.seek(0x3C)
        pe_offset = struct.unpack("<I", handle.read(4))[0]
        handle.seek(pe_offset)
        if handle.read(4) != b"PE\0\0":
            raise ValueError(f"Invalid PE signature: {path}")
        return struct.unpack("<H", handle.read(2))[0]


def collect_packaging_diagnostics() -> dict[str, Any]:
    """Import required packages and inspect architecture/native resources."""

    import joblib
    import sklearn
    import tabpfn
    import torch
    import xgboost

    xgboost_dir = Path(xgboost.__file__).resolve().parent
    xgboost_version_file = xgboost_dir / "VERSION"
    if not xgboost_version_file.is_file():
        raise FileNotFoundError(
            f"xgboost is installed at {xgboost_dir}, but its required "
            "VERSION package data file is missing."
        )
    xgboost_dlls = sorted(xgboost_dir.rglob("xgboost.dll"))
    if not xgboost_dlls:
        raise FileNotFoundError(
            f"xgboost is installed at {xgboost_dir}, but xgboost.dll "
            "was not found recursively."
        )
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("The AVISTA Windows release requires 64-bit Python.")
    dll_machines = {
        str(path): f"0x{pe_machine(path):04X}"
        for path in xgboost_dlls
    }
    wrong_architecture = [
        path
        for path in xgboost_dlls
        if pe_machine(path) != PE_MACHINE_AMD64
    ]
    if wrong_architecture:
        raise RuntimeError(
            "XGBoost wheel contains non-AMD64 native libraries: "
            + ", ".join(str(path) for path in wrong_architecture)
        )

    tabpfn_dir = Path(tabpfn.__file__).resolve().parent
    tabpfn_data = sorted(
        str(path.resolve())
        for path in tabpfn_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() not in {".py", ".pyc"}
    )
    wheel_metadata = (
        importlib.metadata.distribution("xgboost").read_text("WHEEL") or ""
    )
    return {
        "python": {
            "version": platform.python_version(),
            "bits": struct.calcsize("P") * 8,
            "machine": platform.machine(),
            "executable": sys.executable,
        },
        "xgboost": {
            "version": str(xgboost.__version__),
            "package_file": str(Path(xgboost.__file__).resolve()),
            "package_directory": str(xgboost_dir),
            "version_file": str(xgboost_version_file),
            "version_file_value": xgboost_version_file.read_text(
                encoding="utf-8"
            ).strip(),
            "dll_paths": [str(path.resolve()) for path in xgboost_dlls],
            "dll_machines": dll_machines,
            "wheel_metadata": wheel_metadata.strip().splitlines(),
        },
        "tabpfn": {
            "version": str(getattr(tabpfn, "__version__", "unknown")),
            "package_file": str(Path(tabpfn.__file__).resolve()),
            "package_directory": str(tabpfn_dir),
            "package_data": tabpfn_data,
        },
        "dependencies": {
            "torch": str(torch.__version__),
            "sklearn": str(sklearn.__version__),
            "joblib": str(joblib.__version__),
        },
    }


def main() -> int:
    diagnostics = collect_packaging_diagnostics()
    print(json.dumps(diagnostics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
