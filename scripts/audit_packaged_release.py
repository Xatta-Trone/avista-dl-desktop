"""Audit and smoke-test the completed AVISTA PyInstaller distribution."""

from __future__ import annotations

import argparse
import json
import platform
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts.diagnose_packaging_runtime import (
        PE_MACHINE_AMD64,
        pe_machine,
    )
except ImportError:
    from diagnose_packaging_runtime import PE_MACHINE_AMD64, pe_machine


CHECKPOINT_FILENAME = "tabpfn-v2.5-classifier-v2.5_default.ckpt"


def expected_artifacts(dist_dir: Path) -> dict[str, Path]:
    """Return required files in the PyInstaller 6 onedir layout."""

    internal = dist_dir / "_internal"
    return {
        "application": dist_dir / "AVISTA.exe",
        "deep_worker": dist_dir / "AVISTADeepWorker.exe",
        "xgboost_dll": internal / "xgboost" / "lib" / "xgboost.dll",
        "tabpfn_checkpoint": (
            internal / "app" / "assets" / CHECKPOINT_FILENAME
        ),
    }


def audit_artifacts(dist_dir: Path) -> dict[str, Any]:
    """Validate required paths and 64-bit executable/native-library types."""

    dist_dir = dist_dir.resolve()
    artifacts = expected_artifacts(dist_dir)
    missing = [
        f"{name}: {path}"
        for name, path in artifacts.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Packaged artifact audit failed. Missing:\n"
            + "\n".join(missing)
        )
    if struct.calcsize("P") * 8 != 64:
        raise RuntimeError("Artifact audit must run with 64-bit Python.")

    architectures = {}
    for name in ("application", "deep_worker", "xgboost_dll"):
        machine = pe_machine(artifacts[name])
        architectures[name] = f"0x{machine:04X}"
        if machine != PE_MACHINE_AMD64:
            raise RuntimeError(
                f"{name} is not AMD64: {artifacts[name]} "
                f"(machine=0x{machine:04X})"
            )
    return {
        "dist_dir": str(dist_dir),
        "python_architecture": platform.machine(),
        "artifacts": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
            }
            for name, path in artifacts.items()
        },
        "pe_machines": architectures,
    }


def run_packaged_smokes(
    dist_dir: Path,
    *,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """Run tiny XGBoost and TabPFN fits through the packaged executables."""

    artifacts = expected_artifacts(dist_dir.resolve())
    results: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="avista-packaging-smoke-") as temp:
        temp_dir = Path(temp)
        checks = (
            ("xgboost", artifacts["application"]),
            ("tabpfn", artifacts["deep_worker"]),
        )
        for kind, executable in checks:
            output_path = temp_dir / f"{kind}.json"
            completed = subprocess.run(
                [
                    str(executable),
                    "--packaging-smoke-test",
                    kind,
                    "--smoke-output",
                    str(output_path),
                ],
                cwd=dist_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if not output_path.is_file():
                raise RuntimeError(
                    f"{executable.name} did not write {kind} smoke results. "
                    f"Exit code: {completed.returncode}. "
                    f"stdout: {completed.stdout[-2000:]}. "
                    f"stderr: {completed.stderr[-2000:]}."
                )
            result = json.loads(output_path.read_text(encoding="utf-8"))
            result["return_code"] = completed.returncode
            results[kind] = result
            if completed.returncode != 0 or result.get("status") != "passed":
                raise RuntimeError(
                    f"Packaged {kind} smoke test failed: "
                    f"{json.dumps(result, default=str)}"
                )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist-dir", required=True, type=Path)
    parser.add_argument(
        "--skip-model-smokes",
        action="store_true",
        help="Audit files and architecture without executing model fits.",
    )
    args = parser.parse_args()

    audit = audit_artifacts(args.dist_dir)
    if not args.skip_model_smokes:
        audit["model_smokes"] = run_packaged_smokes(args.dist_dir)
    print(json.dumps(audit, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
