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

To publish an update, build and upload `AVISTA_Setup.exe` to a GitHub Release,
update `updates.json` on `main`, and set `installer_url` to the release asset.
Calculate the hash with:

```powershell
Get-FileHash .\installer\AVISTA_Setup.exe -Algorithm SHA256
```

User update preferences are app-level settings at
`%APPDATA%\AVISTA\settings.json`; do not store skipped versions or automatic
check preferences in `.avista` project files.

## Windows Packaging

Build with `AVISTA.spec` so the executable and distribution are named `AVISTA`.
Windows file-description and installer metadata must receive the centralized
`APP_DESCRIPTION`, `__version__`, and `RELEASE_DATE` values from
`app/__version__.py`.

## Theme Styling

Application-wide QSS belongs in `app/gui/theme.py`. Ordinary `QLabel` widgets
use transparent backgrounds so they inherit the containing page or card
surface. Intentional badges, status indicators, warnings, errors, successes,
and other filled labels must use a more specific object-name or style-class
selector.

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

Use `get_app_resource_path(relative_path)` for bundled files. `AVISTA.spec` includes:

```text
app/assets/tabpfn-v2.5-classifier-v2.5_default.ckpt
app/assets/logo.png
```

Keep that relative destination unchanged.

For Nuitka builds, include the logo with:

```powershell
--include-data-file=app/assets/logo.png=app/assets/logo.png
```
