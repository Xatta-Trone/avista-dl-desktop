# AVISTA Validation Examples

This directory contains the public validation experiments used to compare
AVISTA with direct Python implementations and AutoGluon. These examples
support the external validation reported in the SoftwareX manuscript. AVISTA
results are compared with closely aligned direct Python implementations,
while AutoGluon is used as a platform-level benchmark.

| Dataset | Task | Split | Models | CV |
|---|---|---|---|---|
| Mushroom | Binary classification | 70/10/20 | Logistic Regression, Random Forest, XGBoost | 5-fold |
| Obesity | 7-class classification | 70/10/20 | Logistic Regression, Random Forest, XGBoost | 5-fold |
| National Poll (NPHA) | 3-class classification | 70/10/20 | Logistic Regression, Random Forest, XGBoost | 5-fold |

## Contents

Each dataset directory contains a primary Jupyter Notebook with the direct
Python validation and AutoGluon comparison, plus an `avista_project/`
directory containing the saved AVISTA project, its managed dataset, split
artifacts, trained models, and evaluation outputs. Detailed metrics remain in
the notebooks and saved AVISTA outputs rather than being duplicated here.

The National Poll (NPHA) directory also contains
`national-poll-data.ipynb`, which records the conversion from the UCI numeric
codes to the labeled categorical dataset used by its saved AVISTA project.
It is a data-preparation notebook, not a second validation run. AutoGluon
model directories are intentionally excluded; the executed notebook outputs
retain the comparison results.

## Validation protocol

The common target allocation is 70% training, 10% validation, and 20% test,
with five-fold cross-validation and random seed 42. The direct Python
experiments were designed to align with AVISTA in data preparation, model
hyperparameters, imbalance handling, and evaluation metrics where the saved
experiments support the same procedure. AutoGluon is treated as a
platform-level comparison because it retains its own preprocessing, bagging,
and training procedures.

The National Poll direct notebook uses group-aware splitting to keep repeated
feature patterns together. Its saved AVISTA project records the same target
proportions and seed with AVISTA's random split, so those row partitions
should not be interpreted as identical. The Mushroom and Obesity materials
likewise preserve their recorded split artifacts; consult each notebook and
project metadata for the exact procedure.

## Mushroom

- Task: binary classification.
- Notebook: [`mushroom.ipynb`](mushroom/mushroom.ipynb).
- Saved AVISTA project: [`mushroom/avista_project/`](mushroom/avista_project/).
- Source: UCI Machine Learning Repository, Mushroom dataset, accessed in the
  notebook through `ucimlrepo` dataset ID 73.

## Obesity

- Task: multiclass classification with seven target classes.
- Notebook: [`obesity.ipynb`](obesity/obesity.ipynb).
- Saved AVISTA project: [`obesity/avista_project/`](obesity/avista_project/).
- Source: UCI Machine Learning Repository, Estimation of Obesity Levels Based
  on Eating Habits and Physical Condition, accessed through `ucimlrepo`
  dataset ID 544.

## National Poll (NPHA)

- Task: multiclass classification with three target classes.
- Notebook: [`national-poll.ipynb`](national-poll/national-poll.ipynb).
- Data-preparation notebook:
  [`national-poll-data.ipynb`](national-poll/national-poll-data.ipynb).
- Saved AVISTA project:
  [`national-poll/avista_project/`](national-poll/avista_project/).
- Source: UCI Machine Learning Repository, National Poll on Healthy Aging,
  accessed through `ucimlrepo` dataset ID 936.

The managed CSV used by each AVISTA project is included under that project's
`data/` directory. The notebooks also show how the public UCI source data was
obtained and prepared.

## Reproducibility

The experiments use fixed data partitions or recorded split procedures and
fixed random seeds. Minor numerical variation may occur across package
versions, hardware, and implementation-specific behavior. Open `.avista`
files only from trusted sources; these examples include serialized trained
model and preprocessing artifacts under their project directories.
