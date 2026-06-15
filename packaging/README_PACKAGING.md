# AVISTA Windows Packaging

AVISTA uses Nuitka standalone mode for the application folder and Inno Setup
6 for the signed installer-ready executable. The standalone folder contains
the CPython runtime, Qt, native scientific libraries, CUDA-enabled PyTorch,
TabPFN, AVISTA assets, and the bundled TabPFN checkpoint.

## Prerequisites

- Windows 10 or Windows 11 x64.
- Official 64-bit CPython 3.13 available through the `py` launcher.
- Visual Studio 2022 Build Tools with Desktop development with C++ and the
  English language pack.
- Inno Setup 6 from <https://jrsoftware.org/isdl.php>. Install the standard
  Windows package so `ISCC.exe` is available under Program Files.
- At least 30 GB free disk space. Torch, CUDA runtime libraries, TabPFN, and
  Nuitka intermediate C files make the build large.
- Internet access for the clean environment dependency installation.

Target computers do not need Python or the CUDA Toolkit. NVIDIA users need a
compatible NVIDIA display driver. AVISTA continues in CPU mode when CUDA or an
NVIDIA GPU is unavailable.

## Build

From the repository root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\packaging\build_nuitka.ps1 -Configuration Release -Clean
```

The script creates `build_env` and invokes its Python executable directly,
which provides the same isolation as activation without changing the caller's
PowerShell session. It installs `requirements_lock.txt` plus `Nuitka`,
`ordered-set`, and `zstandard`, builds `dist`, stages the complete application
under `release\AVISTA`, and compiles the installer into `installer`.

If `app\assets\logo.ico` is missing, the script generates it from `logo.png`
with Pillow before invoking Nuitka.

For a console-enabled troubleshooting build:

```powershell
.\packaging\build_nuitka.ps1 -Configuration Debug -Clean -SkipInstaller
```

## Outputs

- `dist\`: Nuitka output and compilation report.
- `release\AVISTA\`: installer-ready standalone application.
- `installer\AVISTA_Setup.exe`: Inno Setup installer.
- `release\AVISTA_Setup.exe`: release copy of the installer.

## Test The Standalone Build

1. Run `release\AVISTA\AVISTA.exe`.
2. Confirm the Environment page displays the completed startup check.
3. Inspect `%LOCALAPPDATA%\AVISTA\logs\environment_info.json`, or the active
   project's corresponding `logs\environment_info.json`.
4. Confirm `app_version`, `bundled_python_path`, `torch_version`,
   `cuda_available`, `gpu_name`, `xgboost_available`, `tabpfn_available`,
   `tabpfn_checkpoint_exists`, and `logo_exists` are present.
5. Test both an NVIDIA system and a CPU-only Windows VM.

To verify command-line project loading:

```powershell
.\release\AVISTA\AVISTA.exe "D:\AVISTA Projects\example.avista"
```

The named project should be loaded on the Project Setup page.

## Create And Test The Installer

The normal build reads `APP_NAME` and `__version__` from
`app\__version__.py`, then passes both values to Inno Setup. Use the build
script rather than compiling the `.iss` file directly:

```powershell
.\packaging\build_nuitka.ps1 -Configuration Release
```

To compile only the prepared installer manually after a standalone build,
pass the same name and version definitions to Inno Setup:

```powershell
& "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe" `
  "/DMyAppName=AVISTA" `
  "/DMyAppVersion=<version-from-app\__version__.py>" `
  ".\packaging\avista_installer.iss"
```

Install on a clean Windows VM, verify the Program Files installation, desktop
shortcut, Start Menu shortcut, startup environment JSON, CPU fallback, report
export, and uninstallation.

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
- If pip reports no matching TorchAudio distribution, do not continue to
  Nuitka. The build script now treats every native command failure as fatal
  and verifies required imports before compilation.
- Keep `--include-package=torch`, `torchvision`, `torchaudio`, and `tabpfn`.
- Keep package-data includes for TabPFN, Matplotlib, and QtAwesome.
- Review `dist\nuitka-compilation-report.xml` for omitted dynamic imports.
- Confirm the checkpoint exists at
  `release\AVISTA\app\assets\tabpfn-v2.5-classifier-v2.5_default.ckpt`.
- CUDA wheels bundle the CUDA runtime required by PyTorch. They do not bundle
  an NVIDIA display driver and do not require a separately installed toolkit.
- A CPU-only target should report CUDA unavailable and continue normally.
- Build and smoke-test standalone mode before changing to any one-file design.

Nuitka documentation recommends standalone builds for portable application
folders and its PySide6 plugin for Qt plugin collection. Inno Setup owns the
installation, shortcuts, file association, and clean uninstall.

## How To Build Release Using GitHub Actions

The workflow is defined in `.github\workflows\windows-release.yml`.

Manual build:

1. Push the repository to GitHub.
2. Open the repository's **Actions** tab.
3. Select **Build Windows Release**.
4. Choose **Run workflow**.
5. After completion, download the `AVISTA_Setup` artifact. It contains
   `AVISTA_Setup.exe`.

Tagged release:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

Tags matching `v*` build the installer, create a GitHub Release, generate
release notes, and attach `AVISTA_Setup.exe`.

The workflow uses `windows-latest`, Python 3.13, pip caching, Chocolatey Inno
Setup installation, the focused packaging/resource tests, and the same
`packaging\build_nuitka.ps1` used locally.

GitHub-hosted Windows runners do not provide an NVIDIA GPU. GPU training and
CUDA execution cannot be tested during the release build. CUDA-enabled
PyTorch can still be downloaded and packaged because its wheels include the
required CUDA runtime libraries. The target computer still needs a compatible
NVIDIA driver. AVISTA performs its runtime GPU check on the user's computer
after installation and continues in CPU mode when no compatible GPU exists.

The workflow verifies that `logo.png`, `logo.ico`, and the bundled TabPFN
checkpoint exist before compilation. If the checkpoint is stored with Git
LFS, ensure GitHub LFS storage and bandwidth are available; checkout enables
LFS downloads.
