# -*- mode: python ; coding: utf-8 -*-

"""PyInstaller standalone build specification for AVISTA."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

from app.__version__ import APP_NAME


project_root = Path(SPECPATH).parent
assets_dir = project_root / "app" / "assets"
version_file = Path(os.environ.get("AVISTA_VERSION_FILE", project_root / "dist" / "avista_version_info.txt"))
console_enabled = os.environ.get("AVISTA_PYINSTALLER_CONSOLE") == "1"

datas = [
    (str(assets_dir), "app/assets"),
]
for package_name in ("qtawesome", "matplotlib", "tabpfn"):
    datas += collect_data_files(package_name)

hiddenimports = []
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
    hiddenimports += collect_submodules(package_name)


a = Analysis(
    [str(project_root / "main.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
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

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
