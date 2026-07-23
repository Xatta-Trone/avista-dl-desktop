# Changelog

## 1.0.4 - 2026-07-23

- Fixed the Edge-Case Report page so empty and generated report content starts
  at the top of the viewport.
- Kept **Run Edge-Case Checks** visible without initial scrolling and removed
  horizontal overflow while preserving responsive cards and existing styling.

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
- Added Nuitka standalone and Inno Setup Windows packaging workflow.
- Added `.avista` Windows file association and bundled application assets.
