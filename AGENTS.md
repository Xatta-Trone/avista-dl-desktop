# AVISTA Agent Instructions

This file is the first-read guide for every Codex session working in this
repository.

## Required First Step

Before inspecting code, planning changes, or editing files:

1. Read this `AGENTS.md`.
2. Read the project status file at `PROJECT_STATUS.md` in the repository root.
3. Use `PROJECT_STATUS.md` as the authoritative record of completed work,
   current tests, known issues, design decisions, and the next roadmap item.
4. Inspect the relevant implementation and tests before making assumptions.

Update `PROJECT_STATUS.md` when a task materially changes project capabilities,
test status, known issues, or the roadmap.

## Project Overview

AVISTA is a modular Python desktop application for generic tabular machine
learning. It uses PySide6 and supports project setup, environment and GPU
inspection, data import, column configuration, data splitting, imbalance
handling, edge-case validation, and basic sklearn-compatible model training.

The application must work with arbitrary tabular datasets. The script under
`reference/` is a reference pipeline only. Do not copy its domain-specific
SAE/crash assumptions into application logic.

## Architecture

- `main.py`: application entry point.
- `app/gui/`: PySide6 pages, Qt models, workers, and main-window coordination.
- `app/core/`: configuration, environments, data loading, validation,
  preprocessing, splitting, imbalance handling, training, and evaluation.
- `app/models/`: model factories and wrappers.
- `app/utils/`: shared utilities.
- `tests/`: backend and GUI smoke tests.
- `reference/`: reference scripts and pipeline ideas, not production modules.
- `PROJECT_STATUS.md`: authoritative implementation status and roadmap.
- `requirements_*.txt`: dependency groups for base, ML, deep learning, XAI,
  GPU instructions, and full CPU-installable environments.

Keep GUI presentation separate from reusable backend logic. Long-running GUI
operations must use `QThread` workers.

## Current Implementation Status

At the time this file was written:

- The modular backend and PySide6 GUI are operational.
- Project create/load, environment inspection, GPU validation and confirmed
  repair, paginated data import, column configuration, data split and
  imbalance configuration, edge-case reporting, and a basic Training page are
  implemented.
- A Model Selection page appears after Data Split & Imbalance. It loads all
  classification models from the registry, groups them by category, and saves
  selected models, editable parameters, cross-validation options, and random
  state to `ProjectConfig`.
- Column Configuration persists selected feature and target columns plus
  user-selected categorical/text columns for future label encoding. Actual
  label-encoding preprocessing is not implemented yet.
- Data splitting supports train/validation/test partitions and random,
  stratified, group, stratified-group, and time methods.
- Imbalance handling supports none, random over/under sampling, SMOTE,
  SMOTE-NC, and independent class-weight configuration.
- All resampling class sets, class counts, majority/minority counts, and
  sampling strategies must be derived only from `y_train`. Validation, test,
  and full-dataset classes must never influence balancing.
- Validation and test sets remain unchanged by balancing and may omit classes.
  Missing-class conditions are informational warnings, not split errors.
- Data Split & Imbalance success details use a green card. Warnings are shown
  separately in an orange card. Errors use a red card.
- Saved split artifacts are target-aware and stale results are rejected.
- The selectable model registry contains 16 classification models across
  linear, tree, boosting, kernel/distance, naive Bayes, deep tabular, and
  foundation-model categories. No regression models are registered.
- Built-in sklearn classifiers use a central factory. XGBoost, TabPFN, and
  PyTorch are imported only when the requested model requires them.
- MambaAttention, FT-Transformer, AutoInt, and TabResNet architectures are
  implemented and instantiable. Current training still skips torch models.
- Deep architecture, optimization, monitoring, checkpoint, and early-stopping
  parameters are editable and persisted by Model Selection, but do not yet
  change training behavior.
- Dependency files are explicit:
  - `requirements_ml.txt` contains classical ML and analysis packages.
  - `requirements_deep_cpu.txt` contains CPU-installable PyTorch packages.
  - `requirements_deep_gpu.txt` contains CUDA 12.6 installation instructions
    with CUDA 11.8 as fallback and must not contain a plain `torch` entry.
  - `requirements_full.txt` contains the full CPU-installable application,
    ML, XAI, and TabPFN package set. GPU PyTorch is installed separately.
- Existing basic sklearn-compatible classification and legacy regression
  training/evaluation remain operational.
- Deep training integration, XAI, robustness workflows, export/report pages,
  packaging, and real training cancellation remain unfinished.
- The latest verified test baseline is `88 passed`.

Always verify this summary against `PROJECT_STATUS.md`, which may be newer.

## Development Rules

1. Do not modify unrelated files.
2. Preserve existing behavior unless the task explicitly requires a change.
3. Do not hard-code target, feature, ID, group, date, subgroup, class, or
   domain-specific column names.
4. Keep all user-selected settings configurable and persist them through
   `ProjectConfig` where appropriate.
5. Run edge-case checks before training and block training when blocking
   issues exist.
6. Apply sampling or synthetic balancing only to training data.
   Build every resampling strategy exclusively from classes and counts in
   `y_train`; never fabricate classes absent from training.
7. Do not automatically install, uninstall, or repair dependencies without an
   explicit function call or GUI confirmation.
   For GPU PyTorch, follow `requirements_deep_gpu.txt`; do not add plain
   `torch` to that file or `requirements_full.txt`.
8. Retain support for packaged, shared CPU, shared GPU, and optional
   project-isolated environments.
9. Treat saved artifacts as configuration-sensitive. Do not reuse artifacts
   created for a different target or incompatible configuration.
10. Add or update focused tests for behavioral changes, then run the full test
    suite when feasible.
11. Use the project `.venv` for local dependency-sensitive verification:
    `.venv\Scripts\python.exe -m pytest tests`.
12. Relaunch the application after completing application changes.
13. Keep `PROJECT_STATUS.md` and the README development-status summary
    accurate when milestones or test counts change.

## Coding Conventions

- Use Python type hints and dataclasses for structured configuration and
  artifacts.
- Follow the existing module and naming patterns before adding abstractions.
- Keep functions focused, testable, and dataset-generic.
- Use `pathlib.Path` for filesystem paths.
- Use structured JSON, CSV, NumPy, pandas, and joblib APIs instead of ad hoc
  text parsing.
- Raise clear, actionable exceptions at backend boundaries and show readable
  messages in the GUI.
- Use safe fallbacks for unavailable runtimes or packages where the existing
  design supports them.
- Keep Qt widgets and state updates on the GUI thread.
- Use `QThread` and signals for long-running work.
- Render only bounded dataset previews; never populate a Qt item table with an
  entire large DataFrame.
- Add comments only where logic is not self-explanatory.
- Avoid broad refactors during focused fixes.

## Documentation Source of Truth

The current project status is:

`PROJECT_STATUS.md`

Every future Codex session must read it before making changes. When this file
and `PROJECT_STATUS.md` differ on implementation progress, tests, or roadmap,
follow the newer repository state and update the documentation as part of the
task.

