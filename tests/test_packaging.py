from pathlib import Path

from app.__version__ import APP_DESCRIPTION, RELEASE_DATE
from app.core.runtime_verification import collect_runtime_verification


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_required_packaged_assets_exist():
    logo = PROJECT_ROOT / "app" / "assets" / "logo.png"
    icon = PROJECT_ROOT / "app" / "assets" / "logo.ico"
    checkpoint = (
        PROJECT_ROOT
        / "app"
        / "assets"
        / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )

    assert logo.is_file() and logo.stat().st_size > 0
    assert icon.is_file() and icon.stat().st_size > 0
    assert icon.read_bytes()[:4] == b"\x00\x00\x01\x00"
    assert checkpoint.is_file() and checkpoint.stat().st_size > 0


def test_runtime_verification_reports_versions_packages_and_checkpoint():
    info = collect_runtime_verification(
        {
            "torch_version": "test-torch",
            "cuda_available": True,
            "gpu_name": "Test GPU",
        }
    )

    assert info["app_version"]
    assert info["app_name"] == "AVISTA"
    assert info["app_description"] == APP_DESCRIPTION
    assert info["app_release_date"] == RELEASE_DATE
    assert info["bundled_python_path"]
    assert info["torch_version"] == "test-torch"
    assert info["cuda_available"] is True
    assert info["gpu_name"] == "Test GPU"
    assert isinstance(info["xgboost_available"], bool)
    assert isinstance(info["tabpfn_available"], bool)
    assert info["tabpfn_checkpoint_exists"] is True
    assert info["logo_exists"] is True


def test_pyinstaller_build_uses_clean_environment_and_required_includes():
    script = (PROJECT_ROOT / "packaging" / "build_pyinstaller.ps1").read_text(
        encoding="utf-8"
    )
    spec = (PROJECT_ROOT / "packaging" / "avista_pyinstaller.spec").read_text(
        encoding="utf-8"
    )

    assert '"build_env"' in script
    assert '"requirements_lock.txt"' in script
    assert '"PyInstaller==6.17.0"' in script
    assert '"-m", "PyInstaller"' in script
    assert '"--noconfirm"' in script
    assert '"--clean"' in script
    assert '"--distpath", $DistDir' in script
    assert '"--workpath", $BuildWorkDir' in script
    assert "AVISTA_PYINSTALLER_CONSOLE" in script
    assert "AVISTA_VERSION_FILE" in script
    assert "AVISTA_WORKER_VERSION_FILE" in script
    assert '"AVISTADeepWorker"' in script
    assert "$BuiltWorkerExe" in script
    assert "app\\__version__.py" in script
    assert "Test-ValidIconFile" in script
    assert "create_logo_icon.py" in script
    assert "VSVersionInfo" in script
    assert "APP_DESCRIPTION" in script
    assert "RELEASE_DATE" in script
    assert "'$ApplicationDescription'" in script
    assert "diagnose_packaging_runtime.py" in script
    assert "audit_packaged_release.py" in script
    icon_generator = (
        PROJECT_ROOT / "packaging" / "create_logo_icon.py"
    ).read_text(encoding="utf-8")
    assert "Image.open" in icon_generator
    assert "format=\"ICO\"" in icon_generator
    assert "Invoke-CheckedCommand" in script
    assert "Analysis(" in spec
    assert "COLLECT(" in spec
    assert '"app/assets"' in spec
    assert "deep_worker_main.py" in spec
    assert 'worker_name = "AVISTADeepWorker"' in spec
    assert "MERGE(" not in spec
    assert "worker_exe = EXE(" in spec
    assert "gui_analysis.dependencies" not in spec
    assert "worker_analysis.dependencies" not in spec
    assert "console=True" in spec
    assert '"PySide6", "qtawesome", "app.gui"' in spec
    assert 'collect_submodules(package_name)' in spec
    assert 'collect_dynamic_libs("xgboost")' in spec
    assert 'collect_all(' in spec and '"tabpfn"' in spec
    assert '"tabpfn_common_utils"' in spec
    assert "tabpfn_utils_hiddenimports" in spec
    assert "tabpfn_utils_binaries" in spec
    assert '"xgboost/lib"' in spec
    assert "shared_binaries" in spec
    assert '"huggingface_hub"' in spec
    assert '"safetensors"' in spec
    assert '"torch"' in spec
    assert '"tabpfn"' in spec
    assert 'collect_data_files(package_name)' in spec


def test_locked_cuda_torch_packages_are_compatible_and_available():
    requirements = (PROJECT_ROOT / "requirements_lock.txt").read_text(
        encoding="utf-8"
    )

    assert "torch==2.9.1+cu126" in requirements
    assert "torchvision==0.24.1+cu126" in requirements
    assert "torchaudio==2.9.1+cu126" in requirements
    assert "numpy==1.26.4" in requirements
    assert "pyinstaller==6.17.0" in requirements


def test_release_build_uses_build_pyinstaller_and_documents_file_association():
    build_script = (PROJECT_ROOT / "packaging" / "build_pyinstaller.ps1").read_text(
        encoding="utf-8"
    )
    installer = (PROJECT_ROOT / "packaging" / "avista_installer.iss").read_text(
        encoding="utf-8"
    )
    notes = (
        PROJECT_ROOT / "packaging" / "file_association_notes.md"
    ).read_text(encoding="utf-8")

    assert "avista_pyinstaller.spec" in build_script
    assert "avista_installer.iss" in build_script
    assert "AVISTA_Setup.exe" in build_script
    assert "AVISTADeepWorker.exe" in (
        PROJECT_ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")
    assert "SkipInstaller" in build_script
    assert (
        'Source: "..\\release\\AVISTA\\*"; DestDir: "{app}"; '
        "Flags: ignoreversion recursesubdirs createallsubdirs"
        in installer
    )
    assert 'Subkey: "Software\\Classes\\.avista"' in installer
    assert "AVISTA.Project" in notes
    assert 'AVISTA.exe" "%1' in notes


def test_installer_preserves_existing_install_directory():
    installer = (PROJECT_ROOT / "packaging" / "avista_installer.iss").read_text(
        encoding="utf-8"
    )

    assert "DefaultDirName={code:GetDefaultDirName}" in installer
    assert 'Subkey: "Software\\AVISTA"; ValueType: string; ValueName: "InstallDir"' in installer
    assert "RegQueryStringValue(HKCU, 'Software\\AVISTA', 'InstallDir'" in installer
    assert "RegQueryStringValue(HKLM, 'Software\\AVISTA', 'InstallDir'" in installer
    assert "ExpandConstant('{autopf}\\AVISTA')" in installer


def test_installer_uses_centralized_product_description():
    build_script = (
        PROJECT_ROOT / "packaging" / "build_pyinstaller.ps1"
    ).read_text(encoding="utf-8")
    installer = (
        PROJECT_ROOT / "packaging" / "avista_installer.iss"
    ).read_text(encoding="utf-8")

    assert "APP_DESCRIPTION" in build_script
    assert "/DMyAppDescription=$AppDescription" in build_script
    assert "/DMyAppReleaseDate=$AppReleaseDate" in build_script
    assert "AppComments={#MyAppDescription}" in installer
    assert "VersionInfoDescription={#MyAppDescription}" in installer
    assert "VersionInfoProductTextVersion=" in installer
    assert '#define MyDeepWorkerExeName "AVISTADeepWorker.exe"' in installer
    assert (
        'ValueData: """{app}\\{#MyAppExeName}"" ""%1"""'
        in installer
    )
    assert "MyDeepWorkerExeName" not in installer.split(
        "[Registry]",
        1,
    )[1]


def test_installer_release_documents_exist():
    for name in (
        "LICENSE.txt",
        "THIRD_PARTY_NOTICES.txt",
        "README.md",
        "DEVELOPER_GUIDE.md",
        "CHANGELOG.md",
    ):
        assert (PROJECT_ROOT / name).is_file()


def test_deep_worker_entrypoint_is_gui_free():
    entrypoint = (PROJECT_ROOT / "deep_worker_main.py").read_text(
        encoding="utf-8"
    )
    worker = (
        PROJECT_ROOT / "app" / "training" / "run_torch_model.py"
    ).read_text(encoding="utf-8")
    combined = f"{entrypoint}\n{worker}"

    assert "PySide6" not in combined
    assert "QApplication" not in combined
    assert "MainWindow" not in combined
    assert "update_checker" not in combined
    assert "multiprocessing.freeze_support()" in entrypoint
    assert "flush=True" in worker


def test_github_windows_release_workflow_builds_and_publishes_installer():
    workflow = (
        PROJECT_ROOT / ".github" / "workflows" / "windows-release.yml"
    ).read_text(encoding="utf-8")

    assert 'tags:' in workflow and '"v*"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "release_tag:" in workflow
    assert "runs-on: windows-latest" in workflow
    assert "actions/checkout@v6" in workflow
    assert "actions/setup-python@v6" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "cache: pip" in workflow
    assert "choco install innosetup" in workflow
    assert "./packaging/build_pyinstaller.ps1" in workflow
    assert "tests/test_resources.py" in workflow
    assert "tests/test_main.py" in workflow
    assert "tests/test_version_metadata.py" in workflow
    assert "installer/AVISTA_Setup.exe" in workflow
    assert "release/AVISTA/_internal/xgboost/lib/xgboost.dll" in workflow
    assert (
        "release/AVISTA/_internal/app/assets/"
        "tabpfn-v2.5-classifier-v2.5_default.ckpt"
        in workflow
    )
    assert "scripts/audit_packaged_release.py" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "archive: false" in workflow
    assert "Resolve release tag" in workflow
    assert "github.ref_type" in workflow
    assert "github.ref_name" in workflow
    assert "inputs.release_tag" in workflow
    assert "gh release create" in workflow
    assert "gh release upload" in workflow
    assert "--clobber" in workflow
