# AVISTA Project Status

## 1. Project Name

AVISTA

Automated Vehicle Infrastructure-Sensitive Tabular Analysis

## 2. Project Goal

Build a modular Python desktop GUI application for generic tabular machine learning with configurable project setup, environment selection, data import, column configuration, edge-case validation, preprocessing, splitting, imbalance handling, basic model training, evaluation, and future explainability workflows.

The app must work with arbitrary tabular datasets. Do not hard-code target, ID, group, date, subgroup, or domain-specific columns.

## 3. Current Status Summary

The project has a working modular backend and PySide6 desktop GUI. Core tabular ML pipeline components are implemented and tested for basic sklearn-compatible classification and regression training. The selectable model registry is classification-only and now contains built-in sklearn classifiers, lazy optional-package classifiers, and instantiable deep tabular architectures.

The GUI can create or load projects, inspect and repair GPU environments with user confirmation, import large datasets with a paginated preview, select modeling and target columns, save label-encoding choices for categorical modeling columns, configure train/validation/test splits, apply train-class-only imbalance handling, select classification models and edit their saved parameters, restore matching saved split artifacts, run edge-case checks, and train sklearn-compatible models from confirmed saved artifacts through a `QThread` worker.

The Data Split & Imbalance page uses AVISTA-style subsection cards, compact distribution and coverage tables, a primary confirm action, and auto-dismissing compact success notifications. Warnings remain separate from successful save and load notifications, and errors remain visually distinct.

MambaAttention, FT-Transformer, AutoInt, and TabResNet have real saved-artifact training with live curves and PyTorch state-dict persistence. TabPFN 2.5 now has capped saved-artifact training, optional cross-validation, decoded evaluation exports, subprocess isolation, and safe serialization fallback. A comprehensive Report page now combines saved model outputs into Markdown, PDF, CSV, and comparison figures without retraining. A Nuitka standalone and Inno Setup Windows packaging workflow is implemented. XAI and robustness analysis are not implemented yet.

## 4. Completed Phases

- Initial project skeleton created with `app/gui`, `app/core`, `app/models`, `app/utils`, `reference`, and `tests`.
- `ProjectConfig` persists JSON-formatted AVISTA `.avista` project files with portable relative paths, project-file versioning, application identity, and legacy import.
- Project datasets are copied into each project's `data` folder and recorded in `.avista` with source, relative copy path, size, and timestamp metadata.
- AVISTA branding is centralized across the main window, splash screen, About dialog, project setup, metadata, reports, documentation, and PyInstaller specification.
- `app/__version__.py` is the canonical source for `APP_NAME` and `__version__`; About, reports, training metadata, edge-case metadata, runtime inventory, project metadata, and Windows packaging consume it.
- `.avista` files and `project_metadata.json` now store `application_version` separately from `project_file_version`, preserving the distinction between the AVISTA release and project schema versions.
- The AVISTA name expands to "Automated Vehicle Infrastructure-Sensitive Tabular Analysis."
- Generated report metadata and publication plots include AVISTA version and generation timestamps.
- The AVISTA visual refresh phase has started with a shared Font Awesome icon system powered by `qtawesome`.
- Project Setup now uses equal-width Create and Open cards, a current-project status card, reusable icon-led feedback cards, AVISTA colors, and professional spacing and shadows.
- The sidebar now uses Font Awesome page icons, hover styling, rounded navigation buttons, and a clear active-page state.
- Environment mode is no longer displayed or edited on Project Setup; environment management remains on the Environment page.
- The bundled `app/assets/logo.png` is used by the main window, application icon, sidebar branding, and About dialog through the packaged-resource resolver.
- About AVISTA shows both developers and opens each contributor's GitHub profile through clickable rich-text links.
- The Environment page now presents professional CPU, GPU, and memory summary cards with status badges, AVISTA styling, and project-drive free-space reporting.
- CPU, core, architecture, usage, RAM, and disk metrics are collected through `psutil` with a non-crashing fallback when it is unavailable.
- GPU checks run through `EnvironmentCheckWorker` on a `QThread`, keeping the main window responsive while controls and progress state reflect the active check.
- AVISTA automatically starts the read-only system/GPU environment check after the main window is shown, without delaying or blocking GUI startup.
- Startup and manual environment checks share the same worker lifecycle, cached main-window result, Environment-page running/error state, and disabled Run GPU Check control.
- Startup environment results are saved to the active project's `logs/environment_info.json`, the repository-level `logs/environment_info.json` in development, or `%LOCALAPPDATA%\AVISTA\logs\environment_info.json` in packaged mode when no project is loaded.
- Startup environment logging records check start, completion, and failure; GPU runtime repair remains an explicit user action and is never triggered by startup checks.
- Startup runtime inventory records the AVISTA version, bundled executable path, PyTorch/CUDA/GPU details, XGBoost and TabPFN availability, and bundled checkpoint existence.
- GPU diagnostics now report PyTorch/CUDA/cuDNN, CUDA device count, tensor validation, `nvidia-smi`, GPU/driver identity, and total/used/free GPU memory.
- Environment checks are report-only: the page does not install or repair dependencies, and results or errors are saved to `logs/environment_info.json`.
- The GPU card uses the Font Awesome fan icon and explicit OK, Warning, and Not available states; environment mode is no longer shown on the page.
- When NVIDIA hardware is detected but CUDA PyTorch is inactive, Environment shows a Repair GPU Runtime action backed by the existing `repair_gpu_torch()` implementation.
- GPU runtime repair runs through `EnvironmentRepairWorker` on a `QThread`, automatically reruns `check_gpu()`, and targets the active AVISTA runtime Python from `sys.executable`.
- Environment CPU, GPU, and Memory cards now match the Project Setup visual language with white surfaces, rounded borders, soft shadows, icon/title/subtitle headers, clean label/value rows, and top-right status badges.
- CPU, GPU, and Memory use independent styled card widgets and natural-height layout sizing so each white card remains visually separated like Project Setup's Create, Open, and Current Project cards.
- A 30-second `QTimer` automatically refreshes lightweight CPU, RAM, and project-drive metrics; full GPU checks and repairs remain manual and threaded.
- Data Import now follows the AVISTA card style for project-required information, missing-dataset warnings, dataset summaries, and the bounded preview table.
- Managed project datasets load automatically from the project folder; Load Project Dataset and Replace Dataset controls are no longer shown on Data Import.
- Pagination and the virtualized preview table are displayed only when a dataset is loaded, while missing datasets direct users back to Project Setup.
- With no active project, Data Import now shows a compact centered welcome card with a large icon, clear guidance, and a Go to Project Setup action instead of a full-width message and blank workspace.
- Opening a project automatically restores its managed dataset and Data Import preview; missing copies produce a replacement warning.
- Data Import can replace a project dataset with CSV, XLSX, Parquet, Feather, or FST data.
- Column Configuration now uses the AVISTA card style for modeling-column transfer, target selection, categorical encoding, and confirmation feedback.
- Available Columns and Selected Modeling Columns are both alphabetically sorted.
- The target selector is restricted to selected modeling columns and includes a full-width matplotlib class-distribution chart with counts and percentages.
- Label encoding now uses independent checkboxes plus a unique-values preview; the target is excluded and encoding metadata is saved with the project.
- Column Configuration confirmation uses compact light success rows with a Font Awesome check icon on every message, including the saved modeling-subset path.
- Column Configuration success notifications use a resettable five-second auto-dismiss timer.
- The label-encoding preview sorts unique values by frequency and shows each value's row count and percentage, including an explicit Missing/Null entry.
- Duplicate dataset names support overwrite, timestamped keep-both, and cancel, with keep-both as the default.
- Moving or zipping the complete project folder preserves dataset resolution through project-relative paths.
- Environment management implemented:
  - project `.venv` helpers remain available.
  - shared managed environment path resolution added.
  - packaged runtime mode supported.
- Dependency import validation implemented.
- GPU checker implemented with safe PyTorch fallback.
- GPU checker now validates PyTorch CUDA first.
- If PyTorch CUDA is unavailable, it checks `nvidia-smi`.
- If `nvidia-smi` detects an NVIDIA GPU, the app reports CUDA PyTorch is inactive or mismatched.
- GPU repair is not automatic.
- User must confirm repair from the GUI.
- `repair_gpu_torch()` uninstalls CPU PyTorch, installs CUDA 12.6 PyTorch, validates, then falls back to CUDA 11.8 if needed.
- All install outputs/errors go to `install_log.txt`.
- Environment page now shows the repair message and repair button.
- Data loader implemented for CSV, Excel, and Parquet.
- Dataset summary implemented.
- Data Import page shows visual summary cards.
- Data preview uses a pandas-backed `QAbstractTableModel` and `QTableView`.
- Preview pagination supports 25, 50, 100, or 200 rows per page and never renders more than 200 rows at once.
- Data Import table headers show simplified type and missingness per column.
- Common null markers such as `null`, `nan`, `N/A`, empty strings, and `""` are normalized and displayed as `null`.
- Edge-case checker implemented with warning, error, and fatal levels.
- Generic preprocessing implemented with imputation, one-hot encoding, optional scaling, target encoding, and joblib artifact persistence.
- Classification target encoding is centralized for saved split training:
  - one `LabelEncoder` is fitted or reused for the confirmed target.
  - mixed-type classification target labels are normalized to display strings before encoder fit/transform so sklearn never receives heterogeneous raw label types.
  - encoded and original train/balanced-train/validation/test targets are saved separately.
  - the encoder and encoded-to-original JSON mapping are saved under `outputs/data_split`.
  - imbalance handling and model fitting use encoded integer targets.
  - distribution tables, edge-case messages, metrics, confusion matrices, predictions, misclassifications, and probability headers use original class labels.
  - confirming a different target removes stale split and target-encoder artifacts.
- Data splitting implemented for random, stratified, group, stratified group, and time splits.
- Three-way train/validation/test splitting is available from the GUI with configurable percentages and random seed.
- Imbalance handling implemented for none, class weights, random over/under sampling, SMOTE, and SMOTE-NC, with safe fallback when `imbalanced-learn` is unavailable.
- Data Split & Imbalance page implemented:
  - validates that split percentages total exactly 100%.
  - displays full, train, validation, and test class distributions before balancing.
  - uses AVISTA-style subsection cards for Split Configuration, Before Balancing, Class Coverage After Splitting, Imbalance Handling, After Balancing, and Confirmation / Status.
  - applies improved compact table styling with clean headers, alternating rows, subtle borders, aligned numeric values, and status-badge text for class coverage.
  - uses validated Font Awesome 6 solid icons through qtawesome for split card headers, success/error/warning notifications, empty-state info, and the primary confirm action.
  - applies polished AVISTA input styling to split percentage spin boxes, split/imbalance dropdowns, balancing preset dropdowns, and the random seed spin box with compact widths, rounded borders, hover/focus states, qtawesome Font Awesome angle-up/angle-down spin controls, and qtawesome angle-down combo arrows.
  - applies sampling only to the training set.
  - derives resampling class targets and majority/minority counts only from the training target.
  - warns when full-dataset classes are absent from training or training classes are absent from validation/test.
  - uses compact auto-dismissing success notifications for confirmed saves and saved split/imbalance reloads while displaying warnings separately in an orange card.
  - displays the actual balanced training distribution while leaving validation and test unchanged.
  - keeps class weights independent from the selected sampling method.
  - supports Light, Moderate, Strong, and Custom SMOTE/SMOTE-NC targets.
  - uses dynamic SMOTE neighbor validation and categorical feature indices for SMOTE-NC.
  - saves split indices, distribution CSVs, imbalance metadata, and NumPy data artifacts.
  - stores the target column in saved split and imbalance metadata.
  - reloads saved distributions only when their target matches the latest project configuration.
  - clears stale results and recomputes pre-balancing distributions when the target changes.
- Classification-only model registry implemented with unique canonical names, display names, categories, capability metadata, defaults, optional-package requirements, enablement status, and descriptions.
- Registered classification models:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Decision Tree
  - XGBoost
  - Gradient Boosting
  - Hist Gradient Boosting
  - AdaBoost
  - Support Vector Classifier
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
  - MambaAttention
  - FT-Transformer
  - AutoInt
  - TabResNet
  - TabPFN
- Model categories implemented:
  - Linear Models
  - Tree-Based Models
  - Boosting Models
  - Kernel/Distance Models
  - Naive Bayes
  - Deep Tabular Models
  - Foundation Tabular Models
- Central `create_model()` factory implemented for classification models.
- XGBoost, TabPFN, and PyTorch are imported only when their models are requested and raise clear `ImportError` messages when unavailable.
- Requirements are grouped explicitly for classical ML, CPU PyTorch, GPU PyTorch installation instructions, and the full CPU-installable environment. GPU PyTorch uses the CUDA 12.6 index with CUDA 11.8 as fallback.
- `requirements_ml.txt` includes scikit-learn, XGBoost, imbalanced-learn, SciPy, statsmodels, Matplotlib, and seaborn.
- `requirements_deep_cpu.txt` includes torch, torchvision, and torchaudio.
- `requirements_deep_gpu.txt` contains commented pip commands for CUDA 12.6 and CUDA 11.8 and has no installable plain `torch` entry.
- `requirements_full.txt` includes the complete CPU-installable application, classical ML, XAI, and TabPFN dependencies. It documents which packages provide the registered classifiers and directs GPU users to `requirements_deep_gpu.txt`.
- Generic `MambaAttentionClassifier`, `FTTransformerClassifier`, `AutoIntClassifier`, and `TabResNet` architectures were extracted from the reference pipeline without domain-specific logic.
- MambaAttention metadata and Model Selection controls were corrected directly from `reference/Phase2_SAE_Classification_v10_ADAS__FINAL.py`:
  - architecture defaults match `hidden_dim=256` and `dropout=0.3`.
  - `input_dim` and `num_classes` are not editable and remain training-inferred.
  - training controls reflect AdamW, learning rate `1e-3`, weight decay `1e-3`, batch size `128`, `80` epochs, `5` warmup epochs, linear warmup followed by cosine annealing, and warmup start factor `0.1`.
  - focal-loss controls reflect gamma `1.0`, label smoothing `0.05`, and training class weights.
  - monitoring reflects macro-F1, early-stopping patience `30`, best-weight restoration, and final state-dict saving.
  - the duplicate generic patience field was removed from MambaAttention.
- MambaAttention is trainable from confirmed saved split artifacts:
  - uses balanced encoded training arrays plus saved validation and test arrays.
  - infers input dimension and class count from saved artifacts rather than UI values.
  - uses CUDA when available and CPU otherwise.
  - trains the reference architecture with focal loss, validation macro-F1 monitoring, best-weight restoration, and optional stratified cross-validation.
  - saves `trained_model.pt`, configuration and training metadata, epoch history, decoded split metrics/reports/predictions/probabilities, publication-quality confusion matrices, and CV summaries.
  - creates `failure_reason.json` with the exact error when training fails.
- FT-Transformer is trainable from confirmed saved split artifacts:
  - matches the reference feature-token parameters, class token, Transformer encoder, normalization, and classification head.
  - uses architecture defaults `d_token=128`, `n_heads=8`, `n_layers=3`, and `dropout=0.1`.
  - infers feature count and class count from balanced encoded training artifacts rather than UI values.
  - uses reference training-call values `learning_rate=1e-3` and `focal_gamma=1.0`.
  - supports CPU/CUDA training, validation macro-F1 monitoring, best-weight restoration, optional stratified CV, and decoded reports.
  - saves state dict, configuration, metadata, history, curves, split evaluations, and optional CV summaries under `outputs/training/FT-Transformer`.
  - creates `failure_reason.json` with the exact Python training error when training fails.
- AutoInt is trainable from confirmed saved split artifacts:
  - matches the reference `W`, `b`, multi-head attention layer, normalization, and classification-head implementation.
  - uses architecture defaults `d=64`, `n_heads=4`, `n_layers=3`, and `dropout=0.1`.
  - infers feature count and class count from balanced encoded training artifacts rather than UI values.
  - uses reference final-training values `learning_rate=1e-3` and `focal_gamma=1.0`; the separate 40-epoch learning-curve call is not used as the normal training default.
  - supports CPU/CUDA training, validation macro-F1 monitoring, best-weight restoration, optional stratified CV, decoded reports, and live curves.
  - saves state dict, configuration, metadata, history, curves, split evaluations, and optional CV summaries under `outputs/training/AutoInt`.
  - creates `failure_reason.json` with the exact Python training error when training fails.
- TabResNet is trainable from confirmed saved split artifacts:
  - matches the reference projection, residual blocks, classification head, and Xavier initialization.
  - uses architecture defaults `hidden=256`, `n_blocks=6`, and `dropout=0.2`.
  - infers input dimension and class count from balanced encoded training artifacts rather than UI values.
  - uses reference final-training values `learning_rate=1e-3` and `focal_gamma=1.0`.
  - supports CPU/CUDA training, validation macro-F1 monitoring, best-weight restoration, optional stratified CV, decoded reports, and live curves.
  - saves state dict, configuration, metadata, history, curves, split evaluations, and optional CV summaries under `outputs/training/TabResNet`.
  - creates `failure_reason.json` with the exact Python training error when training fails.
- TabPFN 2.5 is trainable from confirmed saved split artifacts:
  - exposes only `n_estimators`; checkpoint selection is not user-editable.
  - resolves the bundled `app/assets/tabpfn-v2.5-classifier-v2.5_default.ckpt` in development and packaged modes and passes it to every CV/final `TabPFNClassifier`.
  - missing bundled checkpoints create `failure_reason.json` with a clear error and do not attempt the default automatic download path.
  - Model Selection exposes only `n_estimators`, with the installed TabPFN package default of `8` and an allowed range of 1 through 100.
  - the same selected `n_estimators` value is used for both CV and final training.
  - the 3,000-row training cap and 500-row prediction batching remain fixed internal implementation details; randomness comes from the global experiment seed.
  - saved `model_config.json` contains only the selected `n_estimators`.
  - uses balanced encoded training arrays and saved validation/test arrays.
  - caps each CV and final training subset with deterministic sampling from the experiment seed.
  - predicts validation/test probabilities in bounded batches and decodes reports to original class labels.
  - saves validation/test metrics, reports, confusion matrices, predictions, probabilities, and optional CV summaries under `outputs/training/TabPFN_2_5`.
  - saves `trained_model.joblib` when serialization succeeds, otherwise records `model_not_serialized_reason.json`.
  - does not emit epoch events or display live loss curves.
  - when `tabpfn` is unavailable, saves `skip_reason.json` and returns the exact skipped status without crashing the GUI.
- Torch/deep training is isolated from PySide6 on Windows:
  - `TrainingWorker` keeps sklearn/XGBoost in the existing in-process workflow.
  - MambaAttention, FT-Transformer, AutoInt, TabResNet, and TabPFN 2.5 launch through `python -u -m app.training.run_torch_model`.
  - the child process receives project, config, model, and output paths and emits JSON-lines progress over stdout.
  - the GUI parses only text progress and compact result dictionaries; torch models, tensors, CUDA state, and large arrays never cross Qt signals.
  - nonzero/native child exits create `failure_reason.json` with the return code and stderr tail while the GUI remains stable.
  - subprocesses enable `PYTHONFAULTHANDLER` and limit OMP/MKL thread counts.
- All registered deep/foundation classification models now have training orchestration; missing optional TabPFN dependencies produce an explicit skipped result.
- Model Selection page implemented after Data Split & Imbalance:
  - loads all classification models through `get_available_models(task_type="classification")`.
  - uses AVISTA-style cards for Model Library, Model Parameters, Global Training Options, and Confirmation / Status.
  - groups model checkboxes into clean inner cards for Linear Models, Tree-Based Models, Boosting Models, Kernel/Distance Models, Naive Bayes, and Tabular Models.
  - keeps Model Library spacing compact so category cards begin directly below the description and avoid tall empty gaps.
  - top-aligns the Model Library and Model Parameters cards and lets both cards size naturally instead of stretching to fill the page.
  - keeps the Model Library list inside an internal scroll area for tall model lists.
  - displays a model's parameter panel when its checkbox or adjacent text is clicked.
  - renders metadata-driven selects, integer/float controls, checkboxes, text fields, and nullable custom-number controls.
  - styles parameter controls and global training options with polished AVISTA input styling, including qtawesome angle-down combo controls and angle-up/angle-down numeric controls.
  - uses a reusable padded, scrollable parameter-panel shell so all current and future model parameter pages inherit consistent margins, spacing, label alignment, and in-card scrolling.
  - provides expanded sklearn defaults and parameter metadata for Logistic Regression, Random Forest, Extra Trees, and Decision Tree.
  - provides typed defaults and parameter metadata for XGBoost, Gradient Boosting, Hist Gradient Boosting, and AdaBoost.
  - converts nullable, boolean, thread-count, and experiment-seed selections to estimator-compatible Python values.
  - shows non-blocking boosting warnings for target/objective compatibility, categorical handling, early stopping, and expensive settings.
  - checks optional model dependencies in the configured active environment.
  - disables unavailable optional models and provides confirmation-gated background pip install buttons.
  - hides installed optional dependency messages to reduce visual clutter while retaining missing-package install buttons.
  - logs optional dependency installation output to the project `install_log.txt` and refreshes all models sharing the installed package.
  - converts nullable selections and `n_jobs` choices to their estimator-compatible Python types.
  - conditionally enables dependent parameters and shows non-blocking warnings for risky parameter combinations.
  - saves selected model canonical names and per-model parameters.
  - saves shared cross-validation enablement, CV folds, and random state.
  - uses compact auto-dismissing success notifications after model selection is saved.
  - exposes future deep-model architecture, optimization, monitoring, checkpoint, and early-stopping settings.
  - persists deep-model settings without changing the current training implementation.
- Basic sklearn-compatible classification wrappers implemented. Existing legacy regression training remains available outside the classification registry.
- Trainer and evaluator implemented for basic sklearn-compatible classification and regression models.
- Initial PySide6 GUI skeleton implemented.
- Project Setup provides Create New Project and Open Existing Project workflows.
- New projects create `<name>/<name>.avista` plus `data`, `outputs`, `logs`, and `artifacts` folders.
- Existing projects are opened by selecting their `.avista` file; legacy `.xtab` and `project_config.json` files are imported to sibling `.avista` files.
- `main.py` accepts `.avista` and legacy `.xtab` command-line arguments for installer/file-association startup.
- Column Configuration supports selecting categorical/text modeling columns for future label encoding and persists the selection without changing preprocessing behavior.
- GUI pages implemented:
  - Project Setup
  - Environment
  - Data Import
  - Column Configuration
  - Data Split & Imbalance
  - Model Selection
  - Edge-Case Report
  - Training
  - Report
- Background training worker implemented with `QThread` for stable sklearn orchestration and subprocess monitoring.
- Training results stream to the Model Results table per model:
  - a row is created with `Running` when each model starts.
  - trained, failed, and skipped results update that row immediately.
  - sklearn models use the unchanged backend one model at a time so results do not wait for the complete sklearn batch.
  - deep subprocess results update as soon as the child process returns its final result.
  - workflow completion reconciles missing rows without rebuilding or duplicating the table.
- PyTorch/CUDA training no longer runs inside the PySide6 process, avoiding the Windows `0xc0000374` native heap-corruption failure during CUDA/Qt cleanup.
- Training page redesigned into six vertically stacked AVISTA cards for Training Readiness, Training Controls, Live Training Progress, Deep Learning Training Curves, Model Results, and Training Outputs.
- Training readiness uses compact status tiles for target, feature count, train/validation/test rows, selected-model count, cross-validation, and overall readiness instead of the previous large fieldset.
- Training controls use AVISTA primary, danger, and secondary buttons for start, stop, and output-folder actions.
- Training card headers and readiness-tile icons use AVISTA primary blue; semantic green, orange, and red are reserved for readiness state, notifications, and warning/error messaging.
- Training button icons follow their text colors, including gray disabled icons, white primary/danger icons, and primary-blue secondary icons.
- While training runs, Start Training becomes a disabled `Training...` button with an animated white Font Awesome spinner, Stop Training becomes enabled, and existing output-folder actions remain available.
- Successful completion, failure, cancellation, subprocess failure, and bare worker-thread exit all stop the spinner and restore the normal Start Training play icon and readiness-based enabled state.
- Live progress uses status badges for model, fold, epoch, and step; a large AVISTA progress bar; and a bounded timestamped log with blue, green, orange, and red event levels.
- Training notifications use the AVISTA feedback system: success messages dismiss after five seconds, warnings after eight seconds, and errors remain visible.
- Training results retain immediate per-model row creation and updates with the shared AVISTA table style.
- Training outputs show the output folder, saved-model count, generated-report count, latest timestamp, and actions for the folder, aggregate results CSV, and JSON report.
- `TrainingWorker` writes both `training_results.json` and a flattened `training_results.csv` aggregate when output saving is enabled.
- Training page includes a Matplotlib `FigureCanvasQTAgg` live deep-learning curve panel:
  - resets when any supported torch model starts and shows a centered empty state for sklearn, XGBoost, and TabPFN.
  - uses horizontal Accuracy and Loss subplots.
  - plots real train/validation accuracy and train/validation loss from the shared epoch event.
  - updates synchronously after every final-training `epoch_progress` JSON event without GUI throttling.
  - retains the last received curve when the torch subprocess fails.
- All four supported torch models emit subprocess progress with epoch, total epochs, training loss, training accuracy, validation loss, validation macro-F1, and validation accuracy.
- Torch subprocesses use Python unbuffered mode, flush each JSON line before per-epoch artifact I/O, and are parsed line-by-line before process completion.
- All trainable torch models save `training_history.csv`, `training_curves.png`, and `training_curves.pdf`; history is flushed after each final-training epoch so partial history survives many subprocess failures.
- Training consumes confirmed balanced-training, validation, test, split metadata, model settings, and preprocessing artifacts without recomputing splits.
- Per-model and per-fold progress reports the current model, fold, step, percentage, and timestamped log messages.
- Cooperative cancellation stops between folds, models, and evaluation/saving steps.
- Optional stratified cross-validation runs only on balanced training data, validates minimum class counts, saves fold metrics and mean/std summaries, and excludes validation/test data.
- Final models train on the full balanced training set and are evaluated independently on train, validation, and test data.
- Per-model output folders save trained models, preprocessing artifacts, configuration snapshots, training metadata, metrics, reports, confusion matrices, predictions, probabilities, ROC/PR curves, and misclassified records.
- Tree models save feature importance outputs; Logistic Regression saves coefficient and odds-ratio outputs.
- Saved confusion matrices, ROC curves, precision-recall curves, and feature-importance plots use publication-oriented Matplotlib styling and are exported as 300-DPI PNG and PDF files.
- Report page added after Training with AVISTA cards for Report Summary, Model Performance Table, ROC Curve Comparison, Precision-Recall Curve Comparison, Training/Loss Curves, Confusion Matrix Summary, Feature Importance Summary, and Export Report.
- Report generation runs in a `QThread` and reads only saved project/training artifacts; it never retrains models.
- Comprehensive report generation added under `outputs/report`:
  - `AVISTA_Report.md`
  - `AVISTA_Report.pdf`
  - `model_performance_summary.csv`
  - combined ROC, precision-recall, and deep-training curve PNG/PDF files when source data is available.
- Combined ROC and precision-recall figures use clean macro one-vs-rest lines from saved test-set outputs with no standard-deviation shading, larger high-resolution canvases, and legends placed below the axes to avoid cropping.
- Combined deep-learning figures use saved MambaAttention, FT-Transformer, AutoInt, and TabResNet histories with clean loss and validation accuracy/macro-F1 lines and no standard-deviation shading.
- Report figure previews preserve full source aspect ratio, use approximately 50% of the card width, remain horizontally centered, shrink responsively on narrow screens, and are capped at 900 pixels so wide screens do not make them oversized.
- The Model Diagnostic Report uses a two-column layout on wide screens with the confusion matrix constrained to the left half and the classification report on the right; narrow screens stack the two sections vertically.
- A Model Diagnostic Report card provides saved-model and Train/Validation/Test selectors, defaults to Test, and immediately updates the selected confusion matrix and classification report without regenerating the full report.
- Diagnostic confusion matrices load saved PNG/CSV artifacts and fall back to regeneration from saved `predictions.csv` using original class labels.
- Missing selected confusion matrices or classification reports show a compact warning without crashing the Report page.
- Markdown and paginated PDF exports include project/dataset/split metadata, combined model performance, available figures, every trained model's test confusion matrix and classification report, feature importance, output-file listings, reproducibility metadata, AVISTA version, timestamps, footers, and PDF page numbers.
- Reports state that model comparison figures and diagnostic tables are test-set based unless otherwise noted.
- Missing model metrics, histories, curves, confusion matrices, and feature importance outputs are represented as `Not available` without aborting report generation.
- SHAP, tree visualization/rule extraction, forest summaries, and other advanced XAI outputs remain deferred.
- Windows packaging workflow added:
  - `requirements_lock.txt` pins the release build stack, including CUDA 12.6 PyTorch, TabPFN, Qt, scientific packages, and Nuitka.
  - `packaging/build_nuitka.ps1` creates a dedicated `build_env`, explicitly installs Nuitka, ordered-set, and zstandard, builds a console-free release or console-enabled debug standalone folder, includes assets and dynamic ML packages, generates compilation/license reports, and stages `dist`, `installer`, and `release` outputs.
  - Missing `logo.ico` files are generated from the bundled PNG with Pillow before compilation.
  - `packaging/avista_installer.iss` installs under Program Files, creates Start Menu and optional desktop shortcuts, registers `.avista` project files, and removes AVISTA registry entries on uninstall.
  - Packaging documentation covers prerequisites, builds, installer testing, command-line and double-click project opening, CUDA driver expectations, CPU fallback, and Torch/TabPFN troubleshooting.
  - Release documents now include `LICENSE.txt`, `THIRD_PARTY_NOTICES.txt`, and `CHANGELOG.md`.
  - Nuitka and Inno Setup build versions are parsed from `app/__version__.py`; packaging scripts no longer contain an independent release version.
  - `.github/workflows/windows-release.yml` builds the Windows installer on `windows-latest` for manual dispatches and `v*` tags, caches pip downloads, installs Inno Setup, runs only packaging/resource/version tests, uploads `AVISTA_Setup.exe`, and publishes it to tagged GitHub Releases.
  - The Inno Setup output and release artifact use the stable filename `installer/AVISTA_Setup.exe`.
  - GitHub Actions packaging uses Python 3.12, NumPy 1.26.4, Captum 0.8.0, and a matched CUDA 12.6 trio: PyTorch 2.9.1, TorchVision 0.24.1, and TorchAudio 2.9.1. Python 3.12 and NumPy 1.26.4 avoid Captum's NumPy-below-2.0 resolver conflict. The build script fails immediately on native command errors and verifies PySide6/Torch/TabPFN imports before invoking Nuitka.
  - Nuitka explicitly receives `--module-parameter=torch-disable-jit=no` so standalone builds retain normal Torch/TabPFN JIT-capable runtime behavior and avoid the options-nanny ambiguity warning.
- The app was launched successfully from the project `.venv`.

## 5. Implemented Files

Core backend:

- `app/core/project_config.py`
- `app/core/environment_manager.py`
- `app/core/dependency_manager.py`
- `app/core/gpu_checker.py`
- `app/core/data_loader.py`
- `app/core/error_handler.py`
- `app/core/edge_case_checker.py`
- `app/core/preprocessing.py`
- `app/core/splitter.py`
- `app/core/imbalance.py`
- `app/core/model_registry.py`
- `app/core/evaluator.py`
- `app/core/trainer.py`
- `app/core/report_generator.py`
- `app/training/run_torch_model.py`

Models:

- `app/models/sklearn_models.py`
- `app/models/torch_tabular_models.py`
- `app/models/model_factory.py`

GUI:

- `app/gui/main_window.py`
- `app/gui/project_setup_page.py`
- `app/gui/environment_page.py`
- `app/gui/data_import_page.py`
- `app/gui/dataframe_model.py`
- `app/gui/column_config_page.py`
- `app/gui/data_split_imbalance_page.py`
- `app/gui/model_selection_page.py`
- `app/gui/edge_case_report_page.py`
- `app/gui/training_page.py`
- `app/gui/report_page.py`
- `app/gui/workers.py`

Entry point and requirements:

- `main.py`
- `requirements_base.txt`
- `requirements_ml.txt`
- `requirements_deep_cpu.txt`
- `requirements_deep_gpu.txt`
- `requirements_xai.txt`
- `requirements_full.txt`

Tests:

- `tests/test_project_config.py`
- `tests/test_environment_manager.py`
- `tests/test_gpu_checker.py`
- `tests/test_data_loader.py`
- `tests/test_edge_case_checker.py`
- `tests/test_preprocessing.py`
- `tests/test_splitter_imbalance.py`
- `tests/test_model_registry_sklearn.py`
- `tests/test_trainer_evaluator.py`
- `tests/test_gui_smoke.py`
- `tests/test_report_page.py`

## 6. Current Test Status

Latest verified run in the project `.venv`:

```powershell
.venv\Scripts\python.exe -m pytest tests
```

Result:

```text
207 passed
```

This includes backend tests and PySide6 GUI smoke tests for AVISTA branding, Font Awesome icon loading, redesigned Project Setup cards, hidden Project Setup environment controls, sidebar icons, About content, `.avista` project creation/loading, legacy `.xtab` migration, command-line project loading, data preview, split validation, centralized string and mixed-type target encoding, XGBoost encoded-target training, decoded reports and probability columns, target-change invalidation, saved-artifact edge checks, typed Model Selection parameters including the single TabPFN estimator control, active-environment optional dependency checks and installation status, saved-split Training readiness, cross-validation, cancellation, evaluation reports, model-specific outputs, and publication-quality PNG/PDF plotting exports.

Latest focused Environment verification:

```text
25 passed
```

This focused run covered environment collection, psutil fallback, GPU diagnostics, fan-icon/card rendering, GPU-state repair visibility, active-runtime Python selection, background checking and repair, existing repair-function reuse, JSON persistence, and failure handling.

Latest focused Environment card/timer verification:

```text
6 passed
```

This run covered Project Setup-style card rendering, hidden environment mode, the active 30-second timer, manual CPU/Memory refresh, threaded GPU checking, and repair-button visibility.

Latest focused startup/environment verification:

```text
24 passed, 78 deselected
```

This run covered startup scheduling after the main window is shown, non-blocking worker execution, cached Environment-page results, disabled manual checks while startup diagnostics run, app/project JSON persistence, GPU status handling, and confirmation that startup never triggers GPU repair.

Latest focused packaging verification:

```text
13 passed, 82 deselected
```

This run covered development, Nuitka, and PyInstaller resource resolution; packaged asset existence; `.avista` command-line loading; runtime package/checkpoint inventory; dedicated build-environment and Nuitka options; installer association/uninstall declarations; and required release documents. PowerShell syntax parsing also passed.

Latest packaging/resource-only verification:

```text
12 passed
```

This run covered development and Nuitka resource resolution, command-line `.avista` loading, required logo/icon/checkpoint assets, runtime logo/checkpoint/package inventory, clean build-environment configuration, explicit Nuitka build dependencies, ICO fallback generation, and Inno Setup file association declarations. PowerShell syntax parsing passed.

Latest GitHub Actions packaging verification:

```text
17 passed
```

This run covered the packaging/resource tests, `.avista` command-line loading, centralized version metadata, compatible CUDA package pins, fail-fast build commands, required GitHub Actions triggers and Windows runner, pip caching, Inno Setup installation, focused test selection, standalone build invocation, exact installer artifact path, and tagged release publication. Workflow YAML and PowerShell syntax parsing passed.

Latest focused centralized-version verification:

```text
5 passed, 95 deselected
```

This run covered the About dialog, report and edge-case metadata, project and training metadata, and enforcement that release consumers do not hard-code the current version. The five focused project configuration tests also passed after adding `application_version`.

Latest focused Column Configuration verification:

```text
5 passed
```

This run covered alphabetically sorted available and selected columns, target-only dropdown options, target chart updates, target exclusion from label encoding, independent label preview and checkbox behavior, count-sorted value frequencies with percentages and missing values, encoding metadata persistence, resettable five-second success dismissal, styled subset-path confirmation, empty-state visibility, and target-change artifact invalidation.

Latest focused Data Split & Imbalance page verification:

```text
13 passed
```

This run covered saved split artifacts, string-target encoding, saved split reloads, redesigned subsection cards, no-target empty state behavior, compact success notification dismissal, improved table styling, primary confirm button styling, invalid split totals, balanced training distribution display, train-only balancing class sets, target-change refresh, and stale saved-target rejection.

Latest focused Data Split target-encoding verification:

```text
2 passed, 82 deselected
```

This run covered string and mixed-type classification target encoding through the Data Split page, including saved label mappings and balanced encoded artifacts.

Latest focused Data Split icon/rendering verification:

```text
8 passed
```

This run covered Data Split page loading, safe Font Awesome icon names, error and success notification icons, Split Configuration card icon rendering, primary confirm button icon rendering, and default Qt arrow rendering for percentage spin boxes and combo boxes.

Latest focused Data Split UI verification:

```text
19 passed
```

This run covered saved split artifacts, redesigned cards, no-target state, compact notifications, table styling, safe icons, polished combo/spin input styling with Font Awesome angle spin and combo controls, invalid split totals, string and mixed-type target encoding, train-only balancing behavior, target-change refresh, and stale saved-target rejection.

Latest focused Model Selection verification:

```text
14 passed
```

This run covered registry model population, hidden installed dependency messages, missing-package install buttons, dependency install refresh, redesigned Model Selection cards, top-aligned/natural-height card behavior, compact Model Library spacing, internal model-list scrolling, padded scrollable parameter panels, parameter combo/spin arrow icons, no-config empty state, primary confirm styling, compact success notification dismissal, metadata-driven parameter widgets, boosting warnings, dependent parameter enablement, styled restore-default behavior, and saved model selection configuration.

Latest focused Training page verification:

```text
13 passed
```

This run covered the six-card AVISTA redesign, primary-blue card/tile icons, button icon color modes, animated running-state spinner, completion/failure/cancellation/thread-exit resets, output-folder availability while running, readiness tiles, immediate result-row updates, all supported deep-model live curves, non-deep empty states, notification timing and persistence, and output summaries/actions.

Latest focused Training backend verification:

```text
2 passed
```

This run covered saved-artifact MambaAttention training with the updated epoch metrics and cooperative training cancellation.

Latest focused Report page verification:

```text
13 passed
```

This run covered Report page/sidebar loading, clean ROC/PR/deep figures without shaded bands, large uncropped high-resolution figure output, aspect-preserving responsive previews, Model Diagnostic Report controls, Test default selection, immediate split updates, classification-report rendering, safe missing diagnostics, Markdown/PDF creation, combined performance metrics, report-folder opening, and five-second success notification dismissal. The generated eight-page PDF was rendered to PNG and visually verified for outside legends, full axes, diagnostic tables, footers, and page-number quality.

Latest focused Report preview-sizing verification:

```text
5 passed
```

This run covered centered half-width ROC/PR/training previews, the 900-pixel cap, preserved source aspect ratio without cropping, half-width diagnostic confusion matrices on wide layouts, vertical stacking on narrow layouts, and unchanged Markdown/PDF export generation.

## 7. Important Design Decisions

- The codebase is modular: GUI, core pipeline logic, model wrappers, and utilities are separated.
- The app is dataset-generic. No SAE/crash-specific columns are hard-coded.
- `ProjectConfig` stores all user-selected settings in the current `.avista` project file; developers use `ProjectConfig.save()` and `ProjectConfig.load()` rather than constructing config paths.
- Application identity and release metadata must import `APP_NAME` and `__version__` from `app/__version__.py`; `PROJECT_FILE_VERSION` remains independent schema metadata.
- Project-local `.venv` support remains available, but it is optional.
- Environment modes are:
  - `packaged_runtime`
  - `shared_cpu_env`
  - `shared_gpu_env`
  - `project_isolated_env`
- GUI pages call only completed backend modules.
- Long-running training runs in a `QThread` via `TrainingWorker` with structured progress and cooperative cancellation.
- Training is blocked unless column configuration, saved split artifacts, and a current passing edge-case report are present.
- Cross-validation uses only the balanced training target and never includes validation or test data.
- The Data Import page stores the full DataFrame in application state but renders only the current paginated slice.
- Column Configuration saves alphabetically sorted feature columns, the target, selected label-encoding columns, and label-encoding metadata only after explicit confirmation.
- Sampling and synthetic balancing are applied only to the training partition.
- Resampling strategies use only classes and counts present in the training target; validation and test class sets never influence balancing.
- Classes absent from training are never fabricated. Missing training or evaluation classes produce warnings shown separately from successful save details.
- Class weights are configured independently and may be combined with supported sampling methods.
- Saved split/imbalance artifacts are target-aware; results for a previous target are never restored into the current view.
- Saved classification training arrays use integer target ids from the persisted target encoder; original labels remain available in parallel arrays for validation, display, and export.
- The selectable registry contains classification models only; existing regression training compatibility is not exposed as registered models.
- Model Selection stores canonical registry names, per-model parameters, and global CV options in `ProjectConfig`; all supported torch-model settings drive the shared training path.
- Deep tabular architectures and TabPFN 2.5 are implemented and trainable when their optional dependencies are available.
- Registry and central-factory imports do not eagerly import XGBoost, TabPFN, or PyTorch model implementations.
- GPU PyTorch is intentionally excluded from `requirements_full.txt` and must be installed using the CUDA-specific instructions in `requirements_deep_gpu.txt`.
- Artifacts are saved with joblib where appropriate.

## 8. Current Known Issues

- Dependencies may be missing from a newly created environment until the appropriate requirements files are installed.
- On Windows, the latest PySide6 wheel may hit long-path installation issues under long OneDrive paths. The project `.venv` was successfully verified with `PySide6==6.8.0.2`, which satisfies `PySide6>=6.7`.
- Cancellation is cooperative and cannot interrupt an estimator while its current `fit()` call is executing.
- MambaAttention, FT-Transformer, AutoInt, TabResNet, and TabPFN 2.5 training are implemented.
- No XAI page or XAI computation is implemented yet.
- The packaging workflow is implemented, but a full standalone build and installer smoke test remain pending on a Windows build host with Visual Studio 2022 C++ Build Tools and Inno Setup 6.
- Inno Setup 6 is not installed on the current development machine.

## 9. Next Immediate Task

Add the first XAI workflow while keeping it separate from model training:

- SHAP support for compatible trained models
- feature importance summaries
- per-row explanation view
- persisted XAI artifacts and report integration

## 10. Remaining Roadmap

- Add XAI:
  - SHAP support
  - LIME support
  - feature importance summaries
  - per-row explanation view
- Add robustness and validation workflows:
  - missingness stress tests
  - subgroup checks
  - drift checks
  - cross-validation summaries
- Expand report integration:
  - optional edge-case and environment appendices
  - model card summaries
  - prediction-level appendices
- Produce and smoke-test the first signed Windows release on a clean build host.
- Expand tests:
  - GUI interaction tests
  - training worker tests
  - optional package behavior tests
  - end-to-end small dataset workflow tests

## 11. Rules for Future Codex Work

- Do not hard-code dataset-specific target, ID, group, date, subgroup, or domain columns.
- Do not use the `reference/` script as application code; treat it as a reference pipeline only.
- Keep the architecture modular and testable.
- Do not modify unrelated files when implementing a focused task.
- Preserve existing backend behavior unless the task explicitly asks to change it.
- Run tests after changes whenever possible.
- Relaunch the project after completing application changes.
- Use the project `.venv` for dependency-sensitive verification.
- Use `QThread` for long-running GUI tasks.
- Keep training blocked when edge-case report contains errors or fatals.
- Do not implement deep learning or XAI inside sklearn/basic ML tasks.
- Prefer safe fallbacks when optional packages are missing.
- Keep all user-selected settings configurable through `ProjectConfig`.

