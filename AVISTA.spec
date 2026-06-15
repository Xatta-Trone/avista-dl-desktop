# -*- mode: python ; coding: utf-8 -*-

analysis = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        (
            "app/assets/tabpfn-v2.5-classifier-v2.5_default.ckpt",
            "app/assets",
        ),
        (
            "app/assets/logo.png",
            "app/assets",
        ),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="AVISTA",
    console=False,
)
collection = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="AVISTA",
)
