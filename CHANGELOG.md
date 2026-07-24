# Changelog

## 1.0.5 - 2026-07-24

- Added the fixed categorical missing-value policy for selected modeling
  features: null, empty, and whitespace-only values become `Unknown` before
  training-fitted categorical encoding and remain supported on validation,
  test, reload, and future inference paths.
- Split Data Split & Imbalance into explicit **Run Data Split**, **Apply
  Imbalance Handling**, and **Confirm Split & Imbalance** stages with
  independently persisted status, scoped artifact invalidation, and restored
  tables after reopening a project.
- Kept balancing restricted to training data while preserving validation and
  test artifacts exactly.
- Moved the existing **Generate Report** action directly below Report Summary
  and before the Model Performance Table without changing report generation or
  export behavior.

## 1.0.4 - 2026-07-23

- Fixed the Edge-Case Report page so empty and generated report content starts
  at the top of the viewport.
- Kept **Run Edge-Case Checks** visible without initial scrolling and removed
  horizontal overflow while preserving responsive cards and existing styling.
- Fixed packaged deep-learning launches so AVISTA uses the dedicated
  `AVISTADeepWorker.exe` beside `AVISTA.exe` instead of treating the desktop
  executable as a Python interpreter and recursively opening application
  windows.
- Kept source-mode deep training on the project Python interpreter while
  centralizing source and packaged worker command construction.
- Added worker JSON Lines diagnostics, per-model training logs, explicit
  working-directory and environment handling, GUI startup guards, and readable
  Windows native-process exit reporting, including `0xC0000409` fast-fail
  termination.
- Updated the PyInstaller, Inno Setup, and GitHub Actions release definitions
  to build, verify, and install `AVISTA.exe` and `AVISTADeepWorker.exe` side by
  side without changing `.avista` file associations.
- Completed a tag-by-tag regression audit of v1.0.1, v1.0.2, and v1.0.3. The
  unsafe packaged `sys.executable -m app.training.run_torch_model` command was
  introduced in the initial commit, not v1.0.2; no deep-launch, worker,
  Torch/CUDA lock, or PyInstaller-spec change occurred between those tags.

## 1.0.3 - 2026-07-23

- Reframed AVISTA as the standalone product name and centralized its product
  description across the UI, reports, metadata, documentation, and Windows
  packaging.
- Centralized AVISTA release-date metadata and added it to the existing
  launch splash, About dialog, reports, project/training metadata, runtime
  inventory, and Windows packaging.
- Removed the broad label-transparency regression by scoping application
  backgrounds to top-level windows and dialogs, preserving targeted card,
  panel, badge, preview, empty-state, and status surfaces in both themes.

## 1.0.0

- Added the modular AVISTA desktop workflow for project setup, environment
  inspection, data preparation, model selection, training, and reporting.
- Added automatic non-blocking startup environment verification.
- Added PyInstaller onedir and Inno Setup Windows packaging workflow.
- Added `.avista` Windows file association and bundled application assets.
