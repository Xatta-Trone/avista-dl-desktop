# Project File Structure

```text
mushroom/
├── artifacts/
│   └── preprocessing/                  # Preprocessing configuration
├── data/                               # Dataset files
├── logs/                               # Execution and worker logs
├── outputs/
│   ├── data_split/                     # Generated data splits and split configuration
│   ├── edge_cases/                     # Edge-case report
│   ├── report/                         # AVISTA reports, summaries, and plots
│   └── training/                       # Trained models, configurations, metrics, and results
│       ├── FT-Transformer/
│       ├── LogisticRegression/
│       ├── RandomForest/
│       ├── TabPFN_2_5/
│       ├── XGBoost/
│       ├── training_results.csv
│       └── training_results.json       # Consolidated AVISTA training results
├── reproduciablity_python/             # Independent Python results for LR, RF, XGBoost, FT-Transformer, and TabPFN
├── reproducibility_orange/             # Orange results for LR, RF, and XGBoost
├── mushroom.avista                     # AVISTA project file
├── orange_implementation.ipynb         # Orange Python implementation notebook
└── raw_python_implementation.ipynb     # Independent Python implementation notebook
```

`artifacts/` contains AVISTA project artifacts; `data/` contains the modeling and reduced UCI mushroom datasets; and `logs/` contains environment, project, and training execution logs. `outputs/` contains AVISTA-generated data splits, trained models, configurations, and results. The `reproduciablity_python/` and `reproducibility_orange/` folders contain the independent Python and Orange reproducibility results, respectively. `orange_implementation.ipynb` implements the Orange comparison, while `raw_python_implementation.ipynb` contains the independent Python implementation.

## Consolidated Results

Metrics are shown to six decimal places. `—` indicates an unavailable saved result; Orange results for FT-Transformer and TabPFN are not applicable. FT-Transformer's five-fold CV row is labeled diagnostic because its saved CSV specifies an exported-feature CV diagnostic, not fold-local pipeline parity.

| Model | Split | AVISTA Accuracy | Python Accuracy | Orange Accuracy | AVISTA Macro-F1 | Python Macro-F1 | Orange Macro-F1 | AVISTA ROC-AUC | Python ROC-AUC | Orange ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 0.985051 | 0.985051 | 0.985051 | 0.985040 | 0.985040 | 0.985040 | 0.999489 | — | 0.999489 |
| Logistic Regression | Validation | 0.986470 | 0.986470 | 0.986470 | 0.986449 | 0.986449 | 0.986449 | 0.999321 | — | 0.999321 |
| Logistic Regression | Test | 0.976000 | 0.976000 | 0.976000 | 0.975988 | 0.975988 | 0.975988 | 0.999273 | — | 0.999273 |
| Logistic Regression | 5-fold CV | 0.982765 | 0.983117 | — | 0.982754 | 0.983106 | — | 0.999203 | 0.999212 | — |
| Random Forest | Train | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Random Forest | Validation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Random Forest | Test | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Random Forest | 5-fold CV | 1.000000 | 1.000000 | — | 1.000000 | 1.000000 | — | 1.000000 | 1.000000 | — |
| XGBoost | Train | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | Validation | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | Test | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | 5-fold CV | 1.000000 | 1.000000 | — | 1.000000 | 1.000000 | — | 1.000000 | 1.000000 | — |
| FT-Transformer | Train | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
| FT-Transformer | Validation | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
| FT-Transformer | Test | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
| FT-Transformer | 5-fold CV (diagnostic; different protocol) | 0.999297 | 0.999297 | N/A | 0.999296 | 0.999296 | N/A | 0.999989 | 0.999986 | N/A |
| TabPFN 2.5 | Train | — | — | N/A | — | — | N/A | — | — | N/A |
| TabPFN 2.5 | Validation | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
| TabPFN 2.5 | Test | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
| TabPFN 2.5 | 5-fold CV | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A | 1.000000 | 1.000000 | N/A |
