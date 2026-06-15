# AVISTA

**Automated Vehicle Infrastructure-Sensitive Tabular Analysis**

AVISTA is a professional Python desktop application for generic tabular machine learning workflows. It supports portable project setup, environment inspection, tabular data import, column configuration, edge-case validation, splitting, imbalance handling, model selection, training, evaluation, and saved analytics.

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

The Report page generates one comprehensive saved-artifact report without retraining. It exports Markdown, a paginated PDF, a combined performance CSV, clean test-set ROC and precision-recall comparisons, deep-training curves, every trained model's test confusion matrix and classification report, feature importance, project metadata, and reproducibility details under `outputs/report`. Its interactive Model Diagnostic Report switches models and Train/Validation/Test splits immediately from saved artifacts.

Latest verified test status: `207 passed`.

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
