# AVISTA Developer Guide

## Product Identity

AVISTA is a standalone product name, not an acronym. The canonical identity
and release constants are `APP_NAME`, `APP_DESCRIPTION`, `__version__`, and
`RELEASE_DATE` in `app/__version__.py`. Reuse them in the splash screen, UI,
reports, metadata, and packaging rather than duplicating their values. Routine
AVISTA version and release-date changes require editing only that module.

## Project Files

AVISTA uses JSON-formatted `.avista` project files. New projects only use `.avista`.

```python
config.save()
config = ProjectConfig.load(project_file)
```

Do not construct or directly read `project_config.json` paths. `ProjectConfig.load()` imports legacy `.xtab` and `project_config.json` files and writes a sibling `.avista` file.

`config.project_file` is the canonical absolute path. Generated metadata should use `config.project_metadata()` so it includes:

- `application`
- `application_description`
- `application_version`
- `application_release_date`
- `project_name`
- `project_file`
- `project_file_version`

## Dataset Ownership

Project datasets are stored under `data/`. Use `copy_dataset_into_project()` from `app.core.dataset_manager`; do not persist a newly selected external path directly.

The `.avista` `dataset` object stores its project-relative path, original source, copied path, file size, and copy timestamp. Paths inside the project directory are serialized relatively and resolved when loaded.

## Startup

`main.py` accepts an optional project:

```powershell
AVISTA.exe "D:\path\MyProject.avista"
```

The path must exist and use `.avista` or legacy `.xtab`. Legacy files are migrated before the main window is populated.

The main window schedules startup diagnostics after it is visible. Automatic
update checking follows the same pattern: it runs once in a background
`QThread`, does not block project loading, and stays silent when the installed
version is current.

The launch splash keeps its existing dimensions and timing while drawing
`APP_NAME`, `APP_DESCRIPTION`, `__version__`, and `RELEASE_DATE` from
`app/__version__.py`.

## Reproducible Development Environment

Use Python 3.12 and `requirements_lock.txt` for the canonical reproducible
development and release environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements_lock.txt
```

`requirements_base.txt` and `requirements_ml.txt` are unpinned convenience
groups. `requirements_full.txt` is an unpinned CPU-installable convenience
environment. `requirements_deep_cpu.txt` and `requirements_deep_gpu.txt`
provide mutually exclusive PyTorch installation choices; do not mix the CPU
trio with CUDA TorchVision or TorchAudio. `requirements_xai.txt` is reserved
for future optional XAI work and is not part of implemented AVISTA behavior.

## Updates

Update metadata lives in repository-root `updates.json` and is expected to be
published at:

```text
https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json
```

Required fields:

- `latest_version`: semantic version string compared with
  `app.__version__.__version__`.
- `release_date`: display date for the update dialog.
- `release_notes`: list of strings shown in the update dialog.
- `installer_url`: HTTPS URL to the GitHub Release `AVISTA_Setup.exe`.
- `sha256`: optional installer hash. Leave empty only for development.
- `mandatory`: whether the automatic checker may ignore a skipped version.

The Windows release workflow builds the final `AVISTA_Setup.exe`, calculates
its SHA256 locally, updates only the `sha256` field in `updates.json`, verifies
the tag/version, release URL, and installer bytes, uploads that same installer,
then commits only `updates.json` to the default branch. For a local check use:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_release.py `
  --verify-installer .\installer\AVISTA_Setup.exe `
  --expected-tag vX.Y.Z
```

User update preferences are app-level settings at
`%APPDATA%\AVISTA\settings.json`; do not store skipped versions or automatic
check preferences in `.avista` project files.

## Windows Packaging

The active release build uses `packaging/avista_pyinstaller.spec` through
`packaging/build_pyinstaller.ps1`. The obsolete root `AVISTA.spec` has been
removed so local and release builds use the same two-executable onedir spec.
Windows file-description and installer metadata must receive the centralized
`APP_DESCRIPTION`, `__version__`, and `RELEASE_DATE` values from
`app/__version__.py`.

## Theme Styling

Application-wide QSS belongs in `app/gui/theme.py`. Do not add global
`QLabel { background: transparent; }` or
`QWidget { background: transparent; }` rules. Scope top-level application
backgrounds to windows and dialogs so ordinary labels naturally inherit their
parent surface. Cards, panels, badges, status indicators, warnings, errors,
successes, previews, and empty states retain their existing object-name or
style-class selectors.

The installer should register `.avista` with `AVISTA.exe`:

1. Create a ProgID such as `AVISTA.Project`.
2. Associate `.avista` with that ProgID.
3. Set its display name to `AVISTA Project`.
4. Register the open command as `"C:\Program Files\AVISTA\AVISTA.exe" "%1"`.
5. Notify Windows that file associations changed.
6. Store the selected install folder in `Software\AVISTA\InstallDir`.
7. Read `Software\AVISTA\InstallDir` from HKCU or HKLM on update installs so
   the wizard defaults to the existing installation folder.

The installer may also associate legacy `.xtab` files with AVISTA for migration. New files must always use `.avista`.

## Packaged Resources

Use `get_app_resource_path(relative_path)` for bundled files. The PyInstaller
specification includes:

```text
app/assets/logo.png
```

Keep that relative destination unchanged. TabPFN 2.5 model weights must not be
added to `app/assets` or any package manifest. AVISTA resolves the checkpoint
from the official TabPFN user cache and offers setup through **Help → TabPFN
Model Status**.
