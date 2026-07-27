# AVISTA

**An extensible desktop platform for tabular machine learning and deep learning analytics.**

AVISTA is a professional Python desktop application for generic tabular machine learning workflows. It supports portable project setup, environment inspection, tabular data import, column configuration, edge-case validation, splitting, imbalance handling, model selection, training, evaluation, and saved analytics.

The launch screen and About dialog identify the current release as **Version
1.0.6**, released **July 27, 2026**. Product name, description, version, and
release date come from `app/__version__.py`.

## Project Files

AVISTA uses JSON-formatted `.avista` project files containing:

```json
{
  "application": "AVISTA",
  "project_file_version": "1.0"
}
```

Creating `MyProject.avista` creates:

```text
MyProject/
|-- MyProject.avista
|-- data/
|-- outputs/
|-- logs/
`-- artifacts/
```

Project-relative paths keep the folder portable. The initial dataset is copied into `data/`, and opening the `.avista` file restores the managed dataset and preview.

Legacy `.xtab` and `project_config.json` files remain supported. Opening either format writes a sibling `.avista` file and continues with the AVISTA project while leaving the source file unchanged.

## Development Status

The PySide6 desktop GUI includes Project Setup, Environment, Data Import, Column Configuration, Data Split & Imbalance, Model Selection, Edge-Case Report, Training, and Report pages.

The classification registry includes sklearn, XGBoost, PyTorch tabular, and TabPFN models. Training uses six AVISTA cards with primary-blue icons, readiness tiles, an animated running-state Start button, threaded live progress, realtime deep-model accuracy/loss curves, streaming model results, aggregate CSV/JSON outputs, confirmed saved split artifacts, train-only balancing and cross-validation, decoded reports, publication-quality plots, and isolated subprocesses for torch-dependent models.

Selected categorical modeling features normalize missing, empty, and
whitespace-only values to `Unknown` before training-fitted encoding. Data
Split & Imbalance exposes separate **Run Data Split**, **Apply Imbalance
Handling**, and **Confirm Split & Imbalance** actions, persists each stage,
restores saved tables after reopening, and never balances validation or test
data.

The Report page generates one comprehensive saved-artifact report without retraining. Its primary Generate Report action is directly below Report Summary. It exports Markdown, a paginated PDF, a combined performance CSV, clean test-set ROC and precision-recall comparisons, deep-training curves, every trained model's test confusion matrix and classification report, feature importance, project metadata, and reproducibility details under `outputs/report`. Its interactive Model Diagnostic Report switches models and Train/Validation/Test splits immediately from saved artifacts.

AVISTA checks GitHub-hosted update metadata after startup when automatic
checks are enabled. Manual checks are available from **Help > Check for
Updates**. Update preferences are stored outside project files in
`%APPDATA%\AVISTA\settings.json`, and update activity is logged to
`logs\update.log`.

Installed builds inspect their bundled optional packages without relaunching
`AVISTA.exe` as a Python interpreter, so creating or loading a project does not
freeze the original window or open a second command-line error window.
Packaged deep-learning jobs run through the GUI-free
`AVISTADeepWorker.exe` installed beside the main executable; source runs
continue to use the active Python interpreter and the same structured worker
protocol.

Latest completed full-suite baseline: `207 passed`. Latest focused packaged-project
restart regression verification: `26 passed`.

Latest focused startup, branding, release-metadata, packaging, and theme/UI
verification: `31 passed`.

Latest focused deep-worker launch and packaging regression verification:
`41 passed` across launcher/diagnostic checks, all four source-mode deep-model
smoke tests, and missing-checkpoint handling. A clean installed-build smoke
test remains required on the Windows release host.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the authoritative implementation status and roadmap.

## Dependencies

- `requirements_ml.txt`: classical ML, XGBoost, imbalance handling, and analysis.
- `requirements_deep_cpu.txt`: CPU PyTorch packages.
- `requirements_deep_gpu.txt`: CUDA-specific PyTorch installation instructions.
- `requirements_full.txt`: complete CPU-installable application environment.

GPU PyTorch is installed separately using `requirements_deep_gpu.txt`.

## Run

```powershell
.venv\Scripts\python.exe main.py
```

Open a project directly:

```powershell
.venv\Scripts\python.exe main.py "D:\path\MyProject.avista"
```

Packaged Windows installers can associate `.avista` with `AVISTA.exe`. Legacy `.xtab` command-line files are accepted and migrated automatically.

## Updates

The updater reads:

```text
https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json
```

`latest_version` is compared with `app.__version__.__version__` using semantic
version ordering. `installer_url` must use HTTPS. If `sha256` is provided,
AVISTA verifies the downloaded installer before it can run.

Prepare a future release with one command:

```powershell
.venv\Scripts\python.exe scripts\prepare_release.py `
  --version 1.0.6 `
  --release-date "July 25, 2026" `
  --note "First release-note item" `
  --note "Second release-note item"
```

This updates the canonical values in `app/__version__.py` and synchronizes
`updates.json`, the installer URL, README release banner, changelog heading,
and project status. Use `--dry-run` to preview changes and `--check` to verify
the repository before committing or tagging. Git tags remain an explicit
post-commit action.

The Windows installer stores the selected install folder in
`Software\AVISTA\InstallDir` and uses that value as the default for future
updates, so custom locations such as `D:\AVISTA\` are preserved.

## License

Copyright 2026 AVISTA Developers.

AVISTA is licensed under the Apache License, Version 2.0
([SPDX: Apache-2.0](https://spdx.org/licenses/Apache-2.0.html)). See
[LICENSE.txt](LICENSE.txt) for the complete license terms. Third-party
components remain subject to the licenses listed in
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
