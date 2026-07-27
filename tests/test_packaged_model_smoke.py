import importlib.util
import json
from pathlib import Path

import pytest

from app.core.packaging_smoke import run_packaging_smoke
from scripts.audit_packaged_release import expected_artifacts
from scripts.diagnose_packaging_runtime import (
    PE_MACHINE_AMD64,
    collect_packaging_diagnostics,
    pe_machine,
)


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="xgboost is not installed",
)
def test_source_xgboost_packaging_smoke_fits_tiny_dataset(tmp_path):
    output = tmp_path / "xgboost-smoke.json"

    assert run_packaging_smoke("xgboost", output) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["packaged"] is False
    assert any(
        path.casefold().endswith("xgboost\\lib\\xgboost.dll")
        for path in result["native_libraries"]
    )


@pytest.mark.skipif(
    importlib.util.find_spec("tabpfn") is None,
    reason="tabpfn is not installed",
)
def test_source_tabpfn_packaging_smoke_uses_bundled_checkpoint(tmp_path):
    output = tmp_path / "tabpfn-smoke.json"

    assert run_packaging_smoke("tabpfn", output) == 0

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["packaged"] is False
    assert result["checkpoint_exists"] is True
    assert result["checkpoint_size"] > 0
    assert result["device"] == "cpu"
    assert result["n_estimators"] == 2


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None
    or importlib.util.find_spec("tabpfn") is None,
    reason="release packages are not installed",
)
def test_packaging_diagnostics_find_amd64_xgboost_and_tabpfn_data():
    diagnostics = collect_packaging_diagnostics()

    assert diagnostics["python"]["bits"] == 64
    assert diagnostics["xgboost"]["version_file_value"] == (
        diagnostics["xgboost"]["version"]
    )
    assert Path(diagnostics["xgboost"]["version_file"]).is_file()
    assert diagnostics["xgboost"]["dll_paths"]
    assert all(
        machine == "0x8664"
        for machine in diagnostics["xgboost"]["dll_machines"].values()
    )
    assert any(
        path.endswith("tabpfn_col_embedding.pt")
        for path in diagnostics["tabpfn"]["package_data"]
    )
    xgboost_dll = Path(diagnostics["xgboost"]["dll_paths"][0])
    assert pe_machine(xgboost_dll) == PE_MACHINE_AMD64


def test_packaged_artifact_audit_uses_pyinstaller_internal_layout(tmp_path):
    artifacts = expected_artifacts(tmp_path / "AVISTA")

    assert artifacts["application"] == tmp_path / "AVISTA" / "AVISTA.exe"
    assert artifacts["deep_worker"] == (
        tmp_path / "AVISTA" / "AVISTADeepWorker.exe"
    )
    assert artifacts["xgboost_version"] == (
        tmp_path
        / "AVISTA"
        / "_internal"
        / "xgboost"
        / "VERSION"
    )
    assert artifacts["xgboost_dll"] == (
        tmp_path
        / "AVISTA"
        / "_internal"
        / "xgboost"
        / "lib"
        / "xgboost.dll"
    )
    assert artifacts["tabpfn_checkpoint"] == (
        tmp_path
        / "AVISTA"
        / "_internal"
        / "app"
        / "assets"
        / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
