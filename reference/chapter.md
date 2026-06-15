# AVISTA Chapter 7 Planning and Structure Guide

## Chapter 7

# AVISTA: An Automated Tabular Machine Learning and Analytics Platform

This chapter should present AVISTA as a dissertation software-system
contribution: a research-to-software platform that operationalizes reproducible
tabular machine learning workflows for practitioners. It should not read like a
software manual. The emphasis should be on system design, reproducibility,
analytical capability, engineering decisions, and the contribution made by
turning research pipelines into deployable software.

The implemented AVISTA platform is a general-purpose PySide6 desktop
application for tabular machine learning. The chapter should therefore avoid
the earlier transportation-specific framing around ODD evaluation, road segment
screening, and policy analytics dashboards. Transportation safety may still be
mentioned as the research origin and motivating domain, but the system described
in this chapter is the implemented generic AVISTA platform.

## 7.1 Introduction

Key content:

- Motivation for AVISTA as a bridge between empirical machine learning research
  and usable practitioner software.
- Gap between research pipelines, notebooks, and repeatable deployment-ready
  workflows.
- Need for reproducible tabular ML workflows across arbitrary structured
  datasets.
- Research-to-software transition from experimental modeling code to a
  maintained desktop platform.
- Chapter overview, with a clear statement that Chapter 7 evaluates AVISTA as a
  software-system contribution rather than a domain dashboard.

Writing direction:

Open with the problem that research-grade tabular ML pipelines often remain
fragmented across scripts, notebooks, environment assumptions, undocumented data
preparation steps, and manually repeated evaluation routines. Position AVISTA as
the dissertation artifact that formalizes those steps into a reproducible,
auditable, practitioner-facing platform.

Avoid presenting AVISTA as a how-to tool. Instead, frame the platform as an
answer to a systems research question: what infrastructure is required to make
advanced tabular ML reproducible, inspectable, configurable, and deployable for
non-developer analysts?

## 7.2 AVISTA as a Dissertation Contribution

Key content:

- Software artifact contribution.
- Analytical contribution.
- Reproducibility contribution.
- Practitioner usability contribution.
- Deployment contribution.

The purpose of this section is to explicitly justify AVISTA as a dissertation
contribution. The chapter should argue that AVISTA is not merely an interface
around a model; it is an integrated software artifact that encodes decisions
about project persistence, runtime validation, data handling, edge-case
screening, model configuration, training monitoring, reporting, export, and
deployment packaging.

### Table 7.1

**AVISTA Contribution Summary**

| Contribution Type | AVISTA Contribution | Evidence in Implemented Platform | Dissertation Significance |
| --- | --- | --- | --- |
| Software artifact | A modular desktop platform for generic tabular ML workflows | PySide6 GUI, `app/core`, `app/gui`, `app/models`, project files, training and report pages | Demonstrates translation from research code to usable software |
| Analytical workflow | Configurable data preparation, splitting, imbalance handling, model selection, training, evaluation, and reporting | Dataset ingestion, feature/target configuration, edge-case validation, model registry, trainer, evaluator, report generator | Turns fragmented analysis into an end-to-end analytical system |
| Reproducibility | Persistent `.avista` projects, saved split artifacts, saved model outputs, version metadata, and generated reports | `ProjectConfig`, target-aware artifacts, output folders, Markdown/PDF reports | Supports repeatable experiments and auditable modeling decisions |
| Practitioner usability | Desktop workflow with guided pages, warnings, progress indicators, and visual feedback | Project Setup, Environment, Data Import, Column Configuration, Data Split, Edge-Case Report, Model Selection, Training, Report pages | Lowers the barrier between advanced ML methods and applied analysis |
| Deployment readiness | Runtime diagnostics, GPU validation/repair, packaging scripts, installer workflow, and release documentation | Environment checks, GPU repair workflow, Nuitka build script, Inno Setup installer, `.avista` file association | Shows that the artifact is engineered beyond notebook-level execution |

## 7.3 System Requirements and Design Goals

Key content:

- Reproducibility.
- Usability.
- Explainability readiness.
- Scalability.
- Extensibility.
- Hardware awareness.
- Deployment readiness.

Recommended structure:

1. Reproducibility: project files, saved artifacts, version metadata, explicit
   dependency groups, persistent user choices.
2. Usability: page-based desktop workflow, bounded previews, readable warnings,
   guided confirmations, output navigation.
3. Explainability readiness: feature and target metadata, saved model outputs,
   probability exports, reports, and future XAI extension points.
4. Scalability: paginated dataset preview, cached datasets, threaded long-running
   tasks, artifact-based training.
5. Extensibility: model registry, central model factory, modular backend, lazy
   optional imports.
6. Hardware awareness: CPU/GPU diagnostics, CUDA validation, repair workflow,
   runtime inventory.
7. Deployment readiness: packaged-resource handling, Nuitka build workflow,
   installer generation, version and release metadata.

Writing direction:

Present these as design goals derived from the observed limitations of research
ML pipelines. Each goal should connect to an implemented platform behavior and a
dissertation-level rationale.

## 7.4 System Architecture

Key content:

- Overall architecture.
- Layered design.
- Module interactions.
- Data flow.

The implemented architecture should be described as a layered desktop software
system:

1. User Interface Layer: PySide6 pages coordinate workflow state and user
   interaction.
2. Project and Runtime Layer: project persistence, metadata, environment checks,
   dependency diagnostics, GPU validation, and packaged-resource handling.
3. Data and Validation Layer: dataset import, caching, feature/target
   configuration, preprocessing metadata, splitting, imbalance handling, and
   edge-case validation.
4. Modeling and Training Layer: model registry, model factory, sklearn models,
   deep tabular models, TabPFN, saved-artifact training, cross-validation, live
   progress, and training outputs.
5. Reporting and Export Layer: saved metrics, plots, confusion matrices,
   classification reports, Markdown export, PDF export, CSV summaries, and
   publication-quality figures.
6. Deployment Layer: Nuitka standalone build, Inno Setup installer, release
   documents, `.avista` file association, runtime inventory.

### Figure 7.1

**AVISTA System Architecture**

Recommended figure content:

- Show the nine user-facing pages as the workflow spine:
  Project Setup -> Environment -> Data Import -> Column Configuration -> Data
  Split -> Edge-Case Report -> Model Selection -> Training -> Report.
- Beneath the workflow spine, show backend modules grouped by layer:
  `ProjectConfig`, environment/gpu/dependency managers, data loader and dataset
  manager, preprocessing/splitting/imbalance/edge-case checker, model registry
  and factory, trainer/evaluator, report generator.
- Show artifact outputs:
  `.avista` project file, managed dataset copy, split artifacts, trained model
  folders, metrics, plots, Markdown report, PDF report.
- Show deployment outputs:
  standalone application folder, Windows installer, `.avista` file association.

Writing direction:

Explain the architecture as a separation of concerns: GUI pages orchestrate
state and interaction; backend modules implement reusable ML operations; saved
artifacts make later stages configuration-sensitive and reproducible.

## 7.5 Core Platform Components

This section should describe the platform foundations that make AVISTA a
reproducible tabular ML system.

### 7.5.1 Project Management Module

Key content:

- `.avista` project files.
- Project persistence.
- Project metadata.

Writing roadmap:

Explain that AVISTA treats an analysis as a portable project rather than a loose
set of scripts and files. Discuss project-local data folders, output folders,
portable relative paths, application version metadata, and the role of
`ProjectConfig` as the persistent record of user decisions.

### 7.5.2 Environment Management Module

Key content:

- CPU/GPU detection.
- Startup validation.
- Dependency verification.
- Runtime diagnostics.

Writing roadmap:

Describe runtime awareness as part of reproducibility. AVISTA records CPU, RAM,
disk, PyTorch/CUDA, XGBoost, TabPFN, and bundled checkpoint status. Emphasize
that GPU repair is explicit and user-confirmed, not automatic.

### 7.5.3 Data Management Module

Key content:

- Dataset ingestion.
- Dataset caching.
- Supported formats.

Writing roadmap:

Explain that AVISTA supports arbitrary tabular datasets through managed project
copies and bounded previews. Mention CSV, Excel, Parquet, Feather, and FST as
implemented import targets where applicable, and stress that the system is not
hard-coded to a specific target, feature set, or domain schema.

### 7.5.4 Feature Configuration Module

Key content:

- Feature selection.
- Target selection.
- Categorical encoding configuration.

Writing roadmap:

Discuss the shift from implicit notebook column assumptions to explicit user
configuration. The user selects modeling columns, target column, and categorical
or text columns for label encoding metadata. Explain why this matters for
generic tabular ML reproducibility.

### 7.5.5 Data Split and Imbalance Module

Key content:

- Train/validation/test management.
- Stratified splitting.
- Imbalance handling.

Writing roadmap:

Describe random, stratified, group, stratified-group, and time-based splitting.
Explain that balancing applies only to the training partition. Emphasize the
design decision that validation and test sets remain unchanged and that
resampling strategies are derived only from `y_train`.

### 7.5.6 Edge-Case Validation Module

Key content:

- Target integrity.
- Missing value checks.
- Class coverage checks.
- Training readiness assessment.

Writing roadmap:

Frame edge-case validation as a formal gate between data preparation and model
training. Discuss warning, error, and fatal levels; class coverage diagnostics;
and the role of validation in preventing misleading model runs.

## 7.6 Machine Learning Engine

This section should describe AVISTA's model layer as an extensible engine rather
than a fixed one-model pipeline.

### 7.6.1 Traditional Machine Learning Models

Implemented models to describe:

- Logistic Regression.
- Random Forest.
- Extra Trees.
- Decision Tree.
- Gradient Boosting.
- HistGradientBoosting.
- AdaBoost.
- XGBoost.

### Table 7.2

**Traditional ML Models Implemented**

| Model | Category | Role in AVISTA | Implementation Notes |
| --- | --- | --- | --- |
| Logistic Regression | Linear Models | Interpretable classical baseline | Created through central sklearn-compatible factory |
| Random Forest | Tree-Based Models | Nonlinear ensemble baseline | Supports feature importance outputs where available |
| Extra Trees | Tree-Based Models | High-variance randomized tree ensemble | Registered as selectable classifier |
| Decision Tree | Tree-Based Models | Simple interpretable tree baseline | Registered as selectable classifier |
| Gradient Boosting | Boosting Models | Classical boosting baseline | Built-in sklearn classifier |
| HistGradientBoosting | Boosting Models | Efficient histogram-based boosting | Built-in sklearn classifier |
| AdaBoost | Boosting Models | Adaptive boosting baseline | Built-in sklearn classifier |
| XGBoost | Boosting Models | Optional high-performance gradient boosting | Imported lazily only when requested |

Writing roadmap:

Explain why AVISTA includes traditional models even when deep models are
available: baselines, interpretability, computational efficiency, and practical
robustness. Note that the registry contains additional categories such as
Support Vector Classifier, K-Nearest Neighbors, and Gaussian Naive Bayes, but
Table 7.2 should follow the requested traditional-model list.

### 7.6.2 Deep Tabular Learning Models

Implemented models to describe:

- MambaAttention.
- FT-Transformer.
- AutoInt.
- TabResNet.
- TabPFN.

### Table 7.3

**Deep Learning Models Implemented**

| Model | Category | Role in AVISTA | Implementation Notes |
| --- | --- | --- | --- |
| MambaAttention | Deep Tabular Models | Sequence-inspired deep tabular classifier | Trainable from saved split artifacts with live curves and state-dict persistence |
| FT-Transformer | Deep Tabular Models | Transformer-based feature-token classifier | Trainable with validation monitoring and decoded reports |
| AutoInt | Deep Tabular Models | Attention-based feature interaction model | Trainable with saved histories and split evaluations |
| TabResNet | Deep Tabular Models | Residual deep tabular network | Trainable with CPU/CUDA support and live metrics |
| TabPFN | Foundation Tabular Models | Foundation-model tabular classifier | Optional dependency, capped training path, subprocess isolation, safe serialization fallback |

Writing roadmap:

Present these models as evidence that AVISTA is not only a wrapper around
classical ML. The emphasis should be on the platform's ability to configure,
train, monitor, serialize, and report deep tabular models in the same workflow
as classical estimators.

### 7.6.3 Hyperparameter Configuration Framework

Key content:

- Parameter management.
- Default parameter strategy.
- Model extensibility.

Writing roadmap:

Describe the registry-driven parameter framework: model metadata drives
available controls, default values, editable parameters, optional dependency
status, and saved project configuration. Explain lazy imports for XGBoost,
PyTorch, and TabPFN as a software engineering decision that keeps startup and
basic workflows robust.

## 7.7 Training and Monitoring Framework

### 7.7.1 Training Workflow

Writing roadmap:

Describe the artifact-based training workflow: confirmed project configuration,
saved split artifacts, edge-case validation, selected models, threaded training,
saved outputs, and report-ready artifacts. Explain that training is blocked
when required configuration or validation artifacts are missing.

### 7.7.2 Cross Validation Framework

Writing roadmap:

Explain cross-validation as an optional user-selected framework managed through
Model Selection and Training. Note that cross-validation uses training data and
does not leak validation or test sets into model selection.

### 7.7.3 Real-Time Progress Monitoring

Writing roadmap:

Describe the training worker, progress signals, running-state indicators, logs,
status badges, cancellation behavior, and per-model result updates. Emphasize
that long-running training runs off the GUI thread through `QThread`.

### 7.7.4 Real-Time Learning Curves

Writing roadmap:

Describe live curves for supported deep tabular models, including loss and
validation metrics. Frame these as monitoring evidence rather than merely UI
decoration: they support diagnosis of convergence, overfitting, and training
failure.

### Figure 7.2

**Training Workflow**

Recommended figure content:

Project configuration -> dataset import -> feature/target configuration ->
split and imbalance artifacts -> edge-case validation -> model selection ->
training worker -> saved model artifacts -> report generation.

### Figure 7.3

**Real-Time Training Interface**

Recommended screenshot content:

Show the Training page with readiness tiles, training controls, live progress
bar/logs, deep learning curves, model results table, and training output actions.

## 7.8 Reporting Framework

### 7.8.1 Model Performance Reporting

Describe the combined model performance table and CSV summary as the platform's
primary comparative output across trained models.

### 7.8.2 Confusion Matrix Reporting

Describe saved confusion matrices for train, validation, and test splits,
including diagnostic selection controls on the Report page.

### 7.8.3 Classification Report Generation

Describe precision, recall, F1-score, support, macro averages, and weighted
averages as persisted report outputs using original class labels where
available.

### 7.8.4 ROC and Precision-Recall Analysis

Describe combined ROC and precision-recall figures generated from saved
test-set outputs. Emphasize that these are report artifacts, not retraining
steps.

### 7.8.5 Markdown and PDF Report Export

Describe automated Markdown and paginated PDF export, including metadata,
project information, dataset/split information, figures, diagnostics, output
file listings, reproducibility metadata, AVISTA version, timestamps, footers,
and page numbers.

### Figure 7.4

**AVISTA Report Interface**

Recommended screenshot content:

Show the Report page with model performance summary, ROC/precision-recall
previews, diagnostic confusion matrix/classification report controls, and export
buttons for Markdown and PDF.

## 7.9 User Interface Design

This section should analyze UI design as evidence of practitioner usability,
not list button instructions. Each subsection should explain the user's task,
the design problem, and how the page supports reproducible analysis.

### 7.9.1 Project Setup Page

Focus on portable project creation, `.avista` project loading, project metadata,
and entry into a managed workflow.

### 7.9.2 Environment Page

Focus on runtime transparency: CPU, GPU, memory, disk, dependency status, CUDA
readiness, and explicit repair action.

### 7.9.3 Data Import Page

Focus on managed dataset import, bounded previews, dataset summaries, supported
formats, and large-data usability.

### 7.9.4 Column Configuration Page

Focus on explicit modeling-column selection, target selection, categorical
encoding choices, target distribution preview, and persistence.

### 7.9.5 Data Split Page

Focus on split method, train/validation/test percentages, imbalance handling,
class coverage, warnings, and artifact confirmation.

### 7.9.6 Edge-Case Report Page

Focus on readiness checks, warning/error/fatal levels, class and missingness
issues, and training gatekeeping.

### 7.9.7 Model Selection Page

Focus on registry-driven model selection, grouped model library, parameter
editing, optional dependency status, cross-validation options, and saved
configuration.

### 7.9.8 Training Page

Focus on readiness, training controls, live progress, deep-learning curves,
result rows, output folders, and non-blocking execution.

### 7.9.9 Report Page

Focus on saved-artifact reporting, model comparison, diagnostic controls,
figures, Markdown/PDF export, and reproducibility metadata.

### Figure 7.5 through Figure 7.13

**Interface Screens**

Recommended numbering:

- Figure 7.5: Project Setup Page.
- Figure 7.6: Environment Page.
- Figure 7.7: Data Import Page.
- Figure 7.8: Column Configuration Page.
- Figure 7.9: Data Split and Imbalance Page.
- Figure 7.10: Edge-Case Report Page.
- Figure 7.11: Model Selection Page.
- Figure 7.12: Training Page.
- Figure 7.13: Report Page.

## 7.10 Deployment and Software Engineering Considerations

Key content:

- Packaging architecture.
- Nuitka deployment.
- Installer generation.
- Resource management.
- Version management.
- Future plugin architecture.

Writing roadmap:

Present deployment readiness as a research artifact maturity marker. Discuss
packaged-resource resolution, bundled assets, TabPFN checkpoint handling,
centralized version metadata, release documents, Nuitka standalone builds, Inno
Setup installer generation, Start Menu and optional desktop shortcuts, `.avista`
file association, and GitHub Actions release workflow.

Do not overstate the deployment status. State that the packaging workflow is
implemented and verified through focused tests, while full standalone and
installer smoke testing on a clean Windows build host remains a future release
step if applicable.

## 7.11 Limitations and Future Extensions

Key content:

- Explainability integration.
- Additional deep models.
- AutoML support.
- Distributed training.
- Cloud deployment.
- Plugin ecosystem.

Writing roadmap:

Frame limitations honestly as the boundary between the implemented platform and
the future research/software roadmap. Mention that XAI workflows are not yet
implemented as a dedicated page, even though the report infrastructure is ready
to include additional artifacts. Discuss future SHAP/LIME workflows, per-row
explanations, robustness checks, drift checks, subgroup analysis, AutoML,
distributed/cloud training, and plugin-style extension.

## 7.12 Chapter Summary

Key content:

- Summary of AVISTA.
- Contributions.
- Research impact.
- Transition to next chapter.

Writing roadmap:

Conclude by restating AVISTA as a dissertation contribution that converts
research-grade tabular ML into a reproducible, configurable, hardware-aware,
report-generating, deployment-ready desktop platform. The summary should connect
the chapter back to the dissertation's larger argument: methodological research
has greater impact when it is packaged into software systems that preserve
assumptions, artifacts, diagnostics, and outputs.

## Revised Table List

| Table | Title | Purpose |
| --- | --- | --- |
| Table 7.1 | AVISTA Contribution Summary | Frames AVISTA as a dissertation software-system contribution |
| Table 7.2 | Traditional ML Models Implemented | Summarizes classical models available through the platform |
| Table 7.3 | Deep Learning Models Implemented | Summarizes deep tabular and foundation models implemented |
| Table 7.4 | System Requirements and Design Goals | Maps design goals to implemented platform mechanisms |
| Table 7.5 | Core Platform Component Summary | Summarizes project, environment, data, feature, split, and validation modules |
| Table 7.6 | Training and Reporting Artifact Summary | Lists major saved outputs produced by training and reporting |
| Table 7.7 | Deployment Readiness Summary | Maps packaging, resources, versioning, and installer components to release goals |
| Table 7.8 | Limitations and Future Extensions | Separates implemented capabilities from planned extensions |

## Revised Figure List

| Figure | Title | Recommended Content |
| --- | --- | --- |
| Figure 7.1 | AVISTA System Architecture | Layered architecture, module interactions, workflow spine, artifact outputs |
| Figure 7.2 | Training Workflow | Configuration-to-report training pipeline |
| Figure 7.3 | Real-Time Training Interface | Training page with progress, logs, curves, and outputs |
| Figure 7.4 | AVISTA Report Interface | Report page with performance summaries, diagnostics, figures, and export controls |
| Figure 7.5 | Project Setup Page | Create/open project workflow and active project metadata |
| Figure 7.6 | Environment Page | CPU/GPU/memory diagnostics and GPU repair visibility |
| Figure 7.7 | Data Import Page | Dataset summary, managed import, and paginated preview |
| Figure 7.8 | Column Configuration Page | Feature/target selection and label encoding configuration |
| Figure 7.9 | Data Split and Imbalance Page | Split configuration, class distributions, balancing, warnings |
| Figure 7.10 | Edge-Case Report Page | Readiness checks and warning/error/fatal diagnostics |
| Figure 7.11 | Model Selection Page | Model registry, grouped models, parameters, CV options |
| Figure 7.12 | Training Page | Readiness, live progress, learning curves, result table |
| Figure 7.13 | Report Page | Model diagnostics, ROC/PR plots, Markdown/PDF export |

## Estimated Page Allocation

Target chapter length: approximately 28-36 pages including tables and figures.

| Section | Suggested Pages | Notes |
| --- | ---: | --- |
| 7.1 Introduction | 2 | Motivation and chapter framing |
| 7.2 AVISTA as a Dissertation Contribution | 2-3 | Include Table 7.1 |
| 7.3 System Requirements and Design Goals | 2-3 | Include a compact requirements table if needed |
| 7.4 System Architecture | 3-4 | Include Figure 7.1 |
| 7.5 Core Platform Components | 5-6 | Six subsections; one core paragraph cluster each |
| 7.6 Machine Learning Engine | 4-5 | Include Tables 7.2 and 7.3 |
| 7.7 Training and Monitoring Framework | 3-4 | Include Figures 7.2 and 7.3 |
| 7.8 Reporting Framework | 3-4 | Include Figure 7.4 |
| 7.9 User Interface Design | 5-6 | Include Figures 7.5 through 7.13; keep prose analytical |
| 7.10 Deployment and Software Engineering Considerations | 2-3 | Packaging, versioning, resources, installer |
| 7.11 Limitations and Future Extensions | 2 | Honest future work boundary |
| 7.12 Chapter Summary | 1 | Synthesis and transition |

## Writing Roadmap by Section

| Section | Main Claim | Evidence to Use | Writing Caution |
| --- | --- | --- | --- |
| 7.1 | AVISTA addresses the deployment gap in tabular ML research | Need for repeatable projects, runtime checks, guided workflow | Do not begin with domain-specific dashboard language |
| 7.2 | AVISTA is a dissertation contribution as software artifact | Table 7.1, implemented modules, packaging work | Avoid claiming XAI page functionality that is not implemented |
| 7.3 | The design goals explain the platform shape | Reproducibility, usability, explainability readiness, hardware awareness | Tie every goal to an implemented mechanism |
| 7.4 | AVISTA uses a layered, modular architecture | GUI/core/model/report/deployment modules | Avoid implementation minutiae better suited to an appendix |
| 7.5 | Core components transform raw data into training-ready artifacts | Project files, data import, split artifacts, validation reports | Emphasize generic tabular data, not fixed schemas |
| 7.6 | The ML engine supports classical, deep, and foundation tabular models | Registry, factory, sklearn models, torch models, TabPFN | Distinguish implemented trainable models from future models |
| 7.7 | Training is monitored, artifact-based, and GUI-safe | `QThread` workers, live logs, curves, cancellation, CV | Mention cooperative cancellation limitations accurately |
| 7.8 | Reporting converts saved outputs into reproducible documents | Markdown/PDF exports, plots, CSV summaries, diagnostics | Explain reports are generated from saved artifacts, not retraining |
| 7.9 | The UI guides the practitioner through reproducible analysis | Nine interface screenshots | Analyze design decisions; do not write click-by-click instructions |
| 7.10 | Deployment work moves AVISTA beyond a notebook prototype | Nuitka, Inno Setup, resources, version metadata, release docs | Be honest about any remaining clean-machine smoke test |
| 7.11 | Future work extends an already functional core platform | XAI, AutoML, distributed/cloud training, plugins | Separate limitations from defects |
| 7.12 | AVISTA completes the research-to-software transition | Summarize artifact, impact, and next chapter bridge | Keep concise and synthetic |

## Screenshot Recommendations for UI Pages

General screenshot guidance:

- Use a small, clean demonstration dataset that is not domain-specific unless
  the dissertation explicitly requires a domain example.
- Ensure the active project name, dataset summary, selected target, split
  distributions, selected models, and report outputs are internally consistent
  across screenshots.
- Capture the application at a readable desktop resolution with no overlapping
  UI elements.
- Prefer screens that show completed states rather than empty first-run states,
  except where the empty state itself demonstrates a design goal.
- Captions should explain the system purpose of the page, not merely identify
  visible controls.

### Figure 7.5: Project Setup Page

Show:

- Create project and open project cards.
- Current project status.
- AVISTA branding and sidebar.

Caption emphasis:

Project-level persistence and the decision to treat analysis as a managed,
portable software artifact.

### Figure 7.6: Environment Page

Show:

- CPU, GPU, and memory cards.
- GPU status badge.
- Dependency or runtime diagnostic details.
- Repair action visible only if the state supports it.

Caption emphasis:

Runtime transparency, hardware awareness, and explicit user control over GPU
repair.

### Figure 7.7: Data Import Page

Show:

- Loaded dataset summary cards.
- Paginated preview table.
- Column headers with types or missingness.

Caption emphasis:

Generic dataset ingestion, managed project data, and bounded previews for
large-tabular usability.

### Figure 7.8: Column Configuration Page

Show:

- Available and selected modeling columns.
- Target selector.
- Target distribution chart.
- Label encoding configuration area.

Caption emphasis:

Explicit feature/target declaration and reproducibility of modeling assumptions.

### Figure 7.9: Data Split and Imbalance Page

Show:

- Split settings.
- Class distribution before and after balancing.
- Class coverage table.
- Confirmation/status area with warnings if appropriate.

Caption emphasis:

Training-only balancing, target-aware split artifacts, and transparent class
coverage diagnostics.

### Figure 7.10: Edge-Case Report Page

Show:

- A completed edge-case report.
- Warning/error/fatal categories if available.
- Training readiness assessment.

Caption emphasis:

Automated validation as a safeguard against misleading training runs.

### Figure 7.11: Model Selection Page

Show:

- Grouped model library.
- Selected traditional and deep models.
- Parameter editor.
- Cross-validation settings.

Caption emphasis:

Registry-driven extensibility and configurable model experimentation without
hard-coded pipeline assumptions.

### Figure 7.12: Training Page

Show:

- Readiness tiles.
- Active or recently completed progress logs.
- Deep learning curve panel if a deep model has been run.
- Model results table.
- Output folder action.

Caption emphasis:

Non-blocking training, live monitoring, and artifact-generating model execution.

### Figure 7.13: Report Page

Show:

- Model performance summary.
- ROC and precision-recall previews.
- Diagnostic confusion matrix and classification report.
- Markdown/PDF export controls.

Caption emphasis:

Automated conversion of saved modeling outputs into reproducible dissertation
and practitioner-facing reports.

## Suggested Chapter Framing Language

Use language like:

"AVISTA is presented in this chapter as a software-system contribution: a
general-purpose desktop platform that translates research-grade tabular machine
learning workflows into reproducible, configurable, hardware-aware, and
report-generating software."

"The contribution is not only the inclusion of particular models, but the
integration of project persistence, environment diagnostics, dataset handling,
validation, model configuration, training monitoring, reporting, and deployment
packaging into one coherent analytical workflow."

"Although AVISTA originated from a transportation-safety research context, the
implemented platform is intentionally dataset-generic. It does not assume
specific target names, feature names, road-segment identifiers, or policy
analytics modules."

## Academic Honesty Notes

- Do not describe AVISTA as a web dashboard; it is an implemented desktop
  application.
- Do not claim implemented ODD evaluation, road segment screening, or policy
  analytics dashboard modules.
- Do not claim a completed XAI page; XAI remains a planned extension.
- It is accurate to describe explainability readiness through saved artifacts,
  feature metadata, reports, model outputs, and future XAI integration points.
- It is accurate to describe deep tabular training for MambaAttention,
  FT-Transformer, AutoInt, TabResNet, and TabPFN as implemented.
- It is accurate to describe Markdown/PDF report generation and Windows
  packaging workflow as implemented.
- Keep the chapter focused on AVISTA as a dissertation software artifact, not a
  user manual.

**End of AVISTA Chapter 7 Planning Document**
