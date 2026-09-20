# AVISTA

**An extensible desktop platform for tabular machine learning and deep learning analytics.**

AVISTA is a professional Python desktop application for generic tabular machine learning workflows. It supports portable project setup, environment inspection, tabular data import, column configuration, edge-case validation, splitting, imbalance handling, model selection, training, evaluation, and saved analytics.

The launch screen and About dialog identify the current release as **Version
1.0.7**, released **July 28, 2026**. Product name, description, version, and
release date come from `app/__version__.py`.

## Repository and Documentation

- **Source repository:** [Xatta-Trone/avista-dl-desktop](https://github.com/Xatta-Trone/avista-dl-desktop)
- **License:** [Apache License 2.0](LICENSE.txt)
- **Developer guide:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Current implementation status and roadmap:** [PROJECT_STATUS.md](PROJECT_STATUS.md)
- **Windows packaging guide:** [packaging/README_PACKAGING.md](packaging/README_PACKAGING.md)
- **Release history:** [CHANGELOG.md](CHANGELOG.md)
- **Third-party licenses:** [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)



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

The classification registry includes sklearn, XGBoost, PyTorch tabular, and TabPFN models. Training uses six AVISTA cards with primary-blue icons, readiness tiles, an animated running-state Start button, threaded live progress, realtime deep-model accuracy/loss curves, streaming model results, aggregate CSV/JSON outputs, confirmed saved split artifacts, fold-local preprocessing and balancing during cross-validation, decoded reports, publication-quality plots, and isolated subprocesses for torch-dependent models. Deep-model CV splits raw outer fold-training first, fits preprocessing on inner training, applies balancing only to inner training, uses unchanged inner validation for early stopping, and reserves outer fold-validation for scoring. Final deep-model fitting uses external validation for checkpoint selection and reserves external test for final evaluation. TabPFN 2.5 uses a model-specific raw pandas input path with its native preprocessing, no AVISTA one-hot encoding/scaling/resampling, and a reproducible stratified cap only above its supported 50,000-row range.

AVISTA resolves the TabPFN 2.5 checkpoint only from the official TabPFN user
cache (`%APPDATA%\tabpfn` on Windows, or `TABPFN_MODEL_CACHE_DIR`). When it is
absent, startup offers user-initiated setup without blocking other AVISTA
features, and **Help → TabPFN Model Status** provides download, verification,
and cache-status actions. The download uses Prior Labs' official gated browser
flow and never runs silently. AVISTA source is Apache-2.0. AVISTA does not
redistribute TabPFN 2.5 model weights; users obtain them separately from Prior
Labs GmbH under the TabPFN-2.5 License v1.1 for
non-commercial/non-production use.

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

Latest focused cross-validation leakage regression verification:
`47 passed` across preprocessing and trainer tests. The corrected protocol
splits the original external-training rows before fitting preprocessing or
applying resampling in each fold.

Latest focused nested deep-CV verification: `14 passed`. Inner validation is
created before preprocessing/resampling, both validation levels remain
unresampled, and final deep training retains external validation for checkpoint
selection and external test for final evaluation.

Latest focused TabPFN 2.5 raw-input and supported-limit verification: `12 passed`
across trainer, edge-case, model-registry, and report tests.

Latest focused startup, branding, release-metadata, packaging, and theme/UI
verification: `31 passed`.

Latest focused deep-worker launch and packaging regression verification:
`41 passed` across launcher/diagnostic checks, all four source-mode deep-model
smoke tests, and missing-checkpoint handling. A clean installed-build smoke
test remains required on the Windows release host.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the authoritative implementation status and roadmap.

## System Requirements and Dependencies

- Windows 10 or Windows 11, 64-bit.
- Python 3.12 is recommended for source execution and required for the
  reproducible Windows release build.
- At least 8 GB RAM is recommended; deep-learning workloads may require more.
- A compatible NVIDIA GPU and driver are optional. AVISTA supports CPU
  execution when CUDA is unavailable.

- `requirements_lock.txt`: canonical reproducible Python 3.12 environment,
  including the matched CUDA 12.6 PyTorch/TorchVision/TorchAudio trio.
- `requirements_base.txt`: unpinned GUI and data-foundation convenience group.
- `requirements_ml.txt`: unpinned conventional ML and analysis group.
- `requirements_deep_cpu.txt`: optional unpinned CPU PyTorch trio.
- `requirements_deep_gpu.txt`: optional matched CUDA installation commands;
  do not combine them with the CPU PyTorch group.
- `requirements_full.txt`: unpinned CPU-installable convenience environment.
- `requirements_xai.txt`: future optional XAI dependencies; AVISTA does not yet
  implement an XAI workflow.

## Install from Source

```powershell
git clone https://github.com/Xatta-Trone/avista-dl-desktop.git
cd avista-dl-desktop

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements_lock.txt
```

The locked environment is the reproducible source and release configuration.
The other requirements files are optional unpinned convenience groups. Use
only one matched PyTorch trio; the lock uses CUDA 12.6, while explicit CPU or
alternative CUDA setup instructions live in the corresponding deep-learning
requirements files.

## Run

```powershell
.venv\Scripts\python.exe main.py
```

Open a project directly:

```powershell
.venv\Scripts\python.exe main.py "D:\path\MyProject.avista"
```

Packaged Windows installers can associate `.avista` with `AVISTA.exe`. Legacy `.xtab` command-line files are accepted and migrated automatically.

## Typical Workflow

1. Create or load an `.avista` project.
2. Inspect the active Python, CPU, memory, and optional GPU environment.
3. Import a tabular dataset and select modeling features and the target.
4. Configure categorical encoding and numerical scaling.
5. Run the data split, then apply optional training-only imbalance handling.
6. Run the Edge-Case Report and resolve blocking issues.
7. Select and train models.
8. Generate the saved Markdown, PDF, metrics, and diagnostic reports.

## Validation examples

Reproducible validation examples used in the SoftwareX evaluation are
available in [`examples/`](examples/README.md). They cover Mushroom, Obesity,
and National Poll (NPHA), with saved AVISTA projects and corresponding Jupyter
Notebooks for direct-library and AutoGluon comparison.

## Updates

The updater reads:

```text
https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json
```

`latest_version` is compared with `app.__version__.__version__` using semantic
version ordering. `installer_url` must use HTTPS. If `sha256` is provided,
AVISTA verifies the downloaded installer before it can run. The Windows
release workflow calculates this SHA256 from the final installer and updates
the public `updates.json` automatically.

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


## Support

For AVISTA support, contact Md Monzurul Islam at
[monzurul@txstate.edu](mailto:monzurul@txstate.edu).
