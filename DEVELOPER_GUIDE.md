# AVISTA Developer Guide

## Project Files

AVISTA uses JSON-formatted `.avista` project files. New projects only use `.avista`.

```python
config.save()
config = ProjectConfig.load(project_file)
```

Do not construct or directly read `project_config.json` paths. `ProjectConfig.load()` imports legacy `.xtab` and `project_config.json` files and writes a sibling `.avista` file.

`config.project_file` is the canonical absolute path. Generated metadata should use `config.project_metadata()` so it includes:

- `application`
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

## Windows Packaging

Build with `AVISTA.spec` so the executable and distribution are named `AVISTA`.

The installer should register `.avista` with `AVISTA.exe`:

1. Create a ProgID such as `AVISTA.Project`.
2. Associate `.avista` with that ProgID.
3. Set its display name to `AVISTA Project`.
4. Register the open command as `"C:\Program Files\AVISTA\AVISTA.exe" "%1"`.
5. Notify Windows that file associations changed.

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
