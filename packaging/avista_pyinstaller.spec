# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller standalone build specification for AVISTA."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

from app.__version__ import APP_NAME


project_root = Path(SPECPATH).parent
assets_dir = project_root / "app" / "assets"
version_file = Path(os.environ.get("AVISTA_VERSION_FILE", project_root / "dist" / "avista_version_info.txt"))
worker_version_file = Path(
    os.environ.get(
        "AVISTA_WORKER_VERSION_FILE",
        project_root / "dist" / "avista_deep_worker_version_info.txt",
    )
)
console_enabled = os.environ.get("AVISTA_PYINSTALLER_CONSOLE") == "1"
worker_name = "AVISTADeepWorker"

shared_datas = [
    (str(assets_dir), "app/assets"),
]
for package_name in ("qtawesome", "matplotlib"):
    shared_datas += collect_data_files(package_name)

tabpfn_datas, tabpfn_binaries, tabpfn_hiddenimports = collect_all(
    "tabpfn",
    on_error="raise",
)
(
    tabpfn_utils_datas,
    tabpfn_utils_binaries,
    tabpfn_utils_hiddenimports,
) = collect_all(
    "tabpfn_common_utils",
    on_error="raise",
)
shared_datas += tabpfn_datas
shared_datas += tabpfn_utils_datas

xgboost_version_data = [
    entry
    for entry in collect_data_files("xgboost", includes=["VERSION"])
    if Path(entry[0]).name.casefold() == "version"
]
if not xgboost_version_data:
    raise RuntimeError(
        "The installed xgboost package contains no VERSION data file; "
        "refusing to build an incomplete AVISTA release."
    )
shared_datas += xgboost_version_data

xgboost_binaries = collect_dynamic_libs("xgboost")
xgboost_dlls = [
    source
    for source, _destination in xgboost_binaries
    if Path(source).name.casefold() == "xgboost.dll"
]
if not xgboost_dlls:
    raise RuntimeError(
        "The installed xgboost package contains no discoverable "
        "xgboost.dll; refusing to build an incomplete AVISTA release."
    )
xgboost_binaries = [
    (
        source,
        (
            "xgboost/lib"
            if Path(source).name.casefold() == "xgboost.dll"
            else destination
        ),
    )
    for source, destination in xgboost_binaries
]
shared_binaries = (
    xgboost_binaries
    + tabpfn_binaries
    + tabpfn_utils_binaries
)

gui_hiddenimports = []
for package_name in (
    "app",
    "torch",
    "torchvision",
    "torchaudio",
    "tabpfn",
    "xgboost",
    "lightgbm",
    "sklearn",
    "imblearn",
    "matplotlib",
):
    gui_hiddenimports += collect_submodules(package_name)
gui_hiddenimports += tabpfn_hiddenimports

worker_hiddenimports = []
for package_name in (
    "torch",
    "torchvision",
    "torchaudio",
    "tabpfn",
    "lightgbm",
    "sklearn",
    "joblib",
    "safetensors",
    "einops",
    "huggingface_hub",
    "pydantic",
    "pydantic_settings",
    "tabpfn_common_utils",
    "matplotlib",
):
    worker_hiddenimports += collect_submodules(package_name)
worker_hiddenimports += tabpfn_hiddenimports
worker_hiddenimports += tabpfn_utils_hiddenimports


gui_analysis = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=shared_binaries,
    datas=shared_datas,
    hiddenimports=gui_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

worker_analysis = Analysis(
    [str(project_root / "deep_worker_main.py")],
    pathex=[str(project_root)],
    binaries=shared_binaries,
    datas=shared_datas,
    hiddenimports=worker_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "app.gui"],
    noarchive=False,
    optimize=0,
)

gui_pyz = PYZ(gui_analysis.pure)
worker_pyz = PYZ(worker_analysis.pure)

gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=console_enabled,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(assets_dir / "logo.ico"),
    version=str(version_file),
)

worker_exe = EXE(
    worker_pyz,
    worker_analysis.scripts,
    [],
    exclude_binaries=True,
    name=worker_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(assets_dir / "logo.ico"),
    version=str(worker_version_file),
)

coll = COLLECT(
    gui_exe,
    worker_exe,
    gui_analysis.binaries,
    gui_analysis.datas,
    worker_analysis.binaries,
    worker_analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
