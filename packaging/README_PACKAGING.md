# AVISTA Windows Packaging

AVISTA uses PyInstaller onedir mode for the application folder and Inno Setup
6 for the signed installer-ready executable. The standalone folder contains
the CPython runtime, Qt, native scientific libraries, CUDA-enabled PyTorch,
TabPFN, AVISTA assets, and the bundled TabPFN checkpoint. It contains two
entry-point executables:

- `AVISTA.exe`: the windowed PySide6 desktop application.
- `AVISTADeepWorker.exe`: the console-enabled, GUI-free deep-training worker.

The worker is installed beside the desktop executable. It is not a general
Python interpreter, is not used for `.avista` file association, and must not
initialize PySide6 or the AVISTA main window.

## Prerequisites

- Windows 10 or Windows 11 x64.
- Official 64-bit CPython 3.12 available through the `py` launcher. The
  release build uses Python 3.12 because Captum requires NumPy below 2.0.
- Inno Setup 6 from <https://jrsoftware.org/isdl.php>. Install the standard
  Windows package so `ISCC.exe` is available under Program Files.
- At least 30 GB free disk space. Torch, CUDA runtime libraries, TabPFN, and
  PyInstaller's collected application folder make the build large.
- Internet access for the clean environment dependency installation.

Target computers do not need Python or the CUDA Toolkit. NVIDIA users need a
compatible NVIDIA display driver. AVISTA continues in CPU mode when CUDA or an
NVIDIA GPU is unavailable.

## Build

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\packaging\build_pyinstaller.ps1 -Configuration Release -Clean
```

The script creates `build_env` and invokes its Python executable directly,
which provides the same isolation as activation without changing the caller's
PowerShell session. It installs `requirements_lock.txt` plus the pinned
PyInstaller build dependency, builds `dist`, stages the complete application
under `release\AVISTA`, and compiles the installer into `installer`.

If `app\assets\logo.ico` is missing, the script generates it from `logo.png`
with Pillow before invoking PyInstaller.

For a console-enabled troubleshooting build:

```powershell
.\packaging\build_pyinstaller.ps1 -Configuration Debug -Clean -SkipInstaller
```

## Outputs

- `dist\`: PyInstaller onedir output.
- `release\AVISTA\AVISTA.exe`: installer-ready desktop application.
- `release\AVISTA\AVISTADeepWorker.exe`: installed deep-training worker.
- `installer\AVISTA_Setup.exe`: Inno Setup installer.
- `release\AVISTA_Setup.exe`: release copy of the installer.

The PyInstaller specification uses two analyses and entry points, then merges
their shared dependencies into one onedir application folder. Inno Setup's
application-folder rule installs both executables side by side.

## Deep-Worker Launch Modes

Deep-learning command construction is centralized in
`app\training\deep_worker_launcher.py`.

Source runs use:

```text
<active python.exe> -u <absolute path>\app\training\run_torch_model.py ...
```

Packaged runs use:

```text
<installed application directory>\AVISTADeepWorker.exe ...
```

Packaged mode is detected using frozen-runtime markers supported by
PyInstaller and Nuitka. Worker paths are resolved from the installed
executable directory, never the current working directory. The GUI never
launches `AVISTA.exe` with a Python script or `-m` argument.

Each launch writes a project-local diagnostic log under
`logs\training\<model>_worker_<timestamp>.log`. The parent records the
executable, sanitized arguments, working directory, runtime mode, process
status, stdout/stderr, last valid JSON event, and final decimal/hexadecimal
return code. The worker records Python, Torch, CUDA, device, dataset-loading,
model-initialization, and training stages.

## Test The Standalone Build

1. Run `release\AVISTA\AVISTA.exe`.
2. Confirm `release\AVISTA\AVISTADeepWorker.exe` exists beside it.
3. Confirm the Environment page displays the completed startup check.
4. Inspect `%LOCALAPPDATA%\AVISTA\logs\environment_info.json`, or the active
   project's corresponding `logs\environment_info.json`.
5. Confirm `app_version`, `bundled_python_path`, `torch_version`,
   `cuda_available`, `gpu_name`, `xgboost_available`, `tabpfn_available`,
   `tabpfn_checkpoint_exists`, and `logo_exists` are present.
6. Train MambaAttention, FT-Transformer, AutoInt, and TabResNet separately.
   Confirm the GUI remains responsive, results arrive row by row, and no
   additional AVISTA desktop window opens.
7. Verify success, cancel, Python exception, missing checkpoint/asset, and
   native-process failure handling. Confirm every failure points to a complete
   worker log.
8. Test both an NVIDIA system and a CPU-only Windows VM.

To verify command-line project loading:

```powershell
.\release\AVISTA\AVISTA.exe "D:\AVISTA Projects\example.avista"
```

The named project should be loaded on the Project Setup page.

## Create And Test The Installer

The normal build reads `APP_NAME`, `APP_DESCRIPTION`, `__version__`, and
`RELEASE_DATE` from `app\__version__.py`, then passes those values to Windows
executable metadata and Inno Setup. Use the build script rather than compiling
the `.iss` file directly:

```powershell
.\packaging\build_pyinstaller.ps1 -Configuration Release
```

The supported release entry point is `packaging\build_pyinstaller.ps1`. The
script passes the centralized AVISTA name, description, version, and release
date into `packaging\avista_installer.iss`, which defines shortcuts, file
association, and update install-directory behavior.

Install on a clean Windows VM, verify the Program Files installation, desktop
shortcut, Start Menu shortcut, startup environment JSON, CPU fallback, report
export, deep-worker launch, and uninstallation. Confirm both executables are in
the selected application folder and that only `AVISTA.exe` owns shortcuts and
the `.avista` association.

The installer stores the selected folder in `Software\AVISTA\InstallDir` and
uses an existing HKCU or HKLM value as the default on future installs. This
keeps updates pointed at a custom folder such as `D:\AVISTA\` instead of
falling back to Program Files. User projects live outside the application
folder and are not deleted by the installer uninstall rules.

## Test .avista Double-Click

1. Install AVISTA.
2. Create or copy a valid `.avista` project file.
3. Double-click the file in Explorer.
4. Confirm Windows launches `AVISTA.exe "<full project path>"`.
5. Confirm the project name and managed dataset are restored.
6. Uninstall AVISTA and confirm its ProgID and shortcuts are removed.

If Windows cached an older association, use **Open with > Choose another app**
once or sign out and back in after installation.

## Torch And TabPFN Troubleshooting

- Build only from `build_env`; user-site packages can hide missing includes.
- Keep Torch, TorchVision, and TorchAudio on a matched release trio. The
  current CUDA 12.6 lock uses `torch==2.9.1+cu126`,
  `torchvision==0.24.1+cu126`, and `torchaudio==2.9.1+cu126`.
- Keep `numpy==1.26.4` while Captum 0.8.0 is packaged; Captum requires NumPy
  below 2.0, and NumPy 1.26.4 has a Windows wheel for Python 3.12.
- If pip reports no matching TorchAudio distribution, do not continue to
  PyInstaller. The build script treats every native command failure as fatal
  and verifies required imports before compilation.
- Keep hidden imports for `torch`, `torchvision`, `torchaudio`, `tabpfn`,
  `xgboost`, `lightgbm`, `sklearn`, `imblearn`, and `matplotlib`.
- Keep package-data collection for TabPFN, Matplotlib, and QtAwesome.
- Review `build\pyinstaller\warn-avista_pyinstaller.txt` for omitted dynamic
  imports.
- Confirm the checkpoint exists at
  `release\AVISTA\app\assets\tabpfn-v2.5-classifier-v2.5_default.ckpt`.
- CUDA wheels bundle the CUDA runtime required by PyTorch. They do not bundle
  an NVIDIA display driver and do not require a separately installed toolkit.
- A CPU-only target should report CUDA unavailable and continue normally.
- Build and smoke-test onedir mode before changing to any one-file design.

The PyInstaller spec owns Python, Qt, ML package, package-data, icon, and
Windows version-resource collection. Inno Setup owns the installation,
shortcuts, file association, and clean uninstall.

## How To Build Release Using GitHub Actions

The workflow is defined in `.github\workflows\windows-release.yml`.

Manual build:

1. Push the repository to GitHub.
2. Open the repository's **Actions** tab.
3. Select **Build Windows Release**.
4. Choose **Run workflow**.
5. After completion, download the `AVISTA_Setup` artifact. It contains
   `AVISTA_Setup.exe`.

To publish a manually dispatched build to an existing tagged GitHub Release,
enter the tag in the `release_tag` workflow input, for example `vX.Y.Z`.
The workflow creates the release if it does not exist and uploads
`AVISTA_Setup.exe` with overwrite enabled for reruns.

Tagged release:

```powershell
git tag vX.Y.Z
git push origin vX.Y.Z
```

Tags matching `v*` build the installer, create a GitHub Release, generate
release notes, and attach `AVISTA_Setup.exe`.

The workflow uses `windows-latest`, Python 3.12, pip caching, Chocolatey Inno
Setup installation, the focused packaging/resource tests, and the same
`packaging\build_pyinstaller.ps1` used locally. Release publishing uses
GitHub CLI so tagged pushes and manual `release_tag` dispatches follow the
same upload path.

GitHub-hosted Windows runners do not provide an NVIDIA GPU. GPU training and
CUDA execution cannot be tested during the release build. CUDA-enabled
PyTorch can still be downloaded and packaged because its wheels include the
required CUDA runtime libraries. The target computer still needs a compatible
NVIDIA driver. AVISTA performs its runtime GPU check on the user's computer
after installation and continues in CPU mode when no compatible GPU exists.

The workflow verifies that `logo.png`, `logo.ico`, and the bundled TabPFN
checkpoint exist before compilation. It also verifies that both `AVISTA.exe`
and `AVISTADeepWorker.exe` were produced before compiling or publishing the
installer. If the checkpoint is stored with Git LFS, ensure GitHub LFS storage
and bandwidth are available; checkout enables LFS downloads.

## Update Metadata

AVISTA reads update metadata from:

```text
https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json
```

Publish a new update by:

1. Updating `__version__`, `APP_DESCRIPTION`, and `RELEASE_DATE` only in
   `app\__version__.py`.
2. Building `installer\AVISTA_Setup.exe`.
3. Uploading the installer to a GitHub Release such as `vX.Y.Z`.
4. Calculating the installer hash:

   ```powershell
   Get-FileHash .\installer\AVISTA_Setup.exe -Algorithm SHA256
   ```

5. Updating root `updates.json` with `latest_version`, `release_date`,
   `release_notes`, the HTTPS `installer_url`, and the SHA256 value.

The updater verifies `sha256` when provided and refuses to launch the
installer on a mismatch. Leave `sha256` empty only while testing a draft
release asset.
