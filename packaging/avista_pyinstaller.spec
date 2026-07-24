# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller standalone build specification for AVISTA."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

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
for package_name in ("qtawesome", "matplotlib", "tabpfn"):
    shared_datas += collect_data_files(package_name)

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

worker_hiddenimports = []
for package_name in (
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
    worker_hiddenimports += collect_submodules(package_name)


gui_analysis = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
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
    binaries=[],
    datas=shared_datas,
    hiddenimports=worker_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "qtawesome", "app.gui"],
    noarchive=False,
    optimize=0,
)

MERGE(
    (gui_analysis, APP_NAME, APP_NAME),
    (worker_analysis, worker_name, worker_name),
)

gui_pyz = PYZ(gui_analysis.pure)
worker_pyz = PYZ(worker_analysis.pure)

gui_exe = EXE(
    gui_pyz,
    gui_analysis.dependencies,
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
    worker_analysis.dependencies,
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
