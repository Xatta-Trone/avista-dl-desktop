## Project File Structure

```text
national-poll2/
├── artifacts/ — AVISTA project artifacts
│   └── preprocessing/
│       └── numerical_scaling_config.json
├── data/ — Dataset files
│   ├── modeling_subset.csv
│   └── uci_npha_avista_categorical.csv
├── logs/ — Execution logs
│   ├── environment_info.json
│   ├── project_metadata.json
│   ├── training_log.txt
│   └── training/ — Model worker logs
├── outputs/ — AVISTA-generated data splits, trained models, configurations, and results
│   ├── data_split/ — Processed splits, labels, and split/preprocessing artifacts
│   ├── edge_cases/ — Edge-case report
│   ├── report/ — AVISTA report, performance summary, and plots
│   └── training/ — Model outputs and consolidated training results
│       ├── FT-Transformer/
│       ├── LogisticRegression/
│       ├── RandomForest/
│       ├── TabPFN_2_5/
│       ├── XGBoost/
│       ├── training_results.csv
│       └── training_results.json
├── reproduciablity_python/ — Independent Python reproducibility results for LR, RF, XGBoost, FT-Transformer, and TabPFN
│   ├── FT_cv_comparison_DIAGNOSTIC.csv
│   ├── FT_cv_fold_results_DIAGNOSTIC.csv
│   ├── FT_reproducibility_results.csv
│   ├── LR_cv_comparison.csv
│   ├── LR_cv_fold_results.csv
│   ├── LR_reproducibility_diagnostics.csv
│   ├── LR_reproducibility_results.csv
│   ├── RF_cv_comparison.csv
│   ├── RF_cv_fold_results.csv
│   ├── RF_reproducibility_results.csv
│   ├── TabPFN_cv_comparison.csv
│   ├── TabPFN_cv_fold_results.csv
│   ├── TabPFN_reproducibility_results.csv
│   ├── XGB_cv_comparison.csv
│   ├── XGB_cv_fold_results.csv
│   └── XGB_reproducibility_results.csv
├── reproducibility_orange/ — Orange implementation results for LR, RF, and XGBoost
│   ├── LR_orange_reproducibility_results.csv
│   ├── Orange_all_models_comparison.csv
│   ├── Orange_test_summary.csv
│   ├── RF_orange_reproducibility_results.csv
│   ├── XGB_orange_configuration_audit.json
│   └── XGB_orange_reproducibility_results.csv
├── national-poll2.avista — AVISTA project file
├── orange_implementation.ipynb — Orange Python implementation notebook
└── raw_python_implementation.ipynb — Independent Python implementation notebook
```

## Consolidated Results

| Model | Split | AVISTA Accuracy | Python Accuracy | Orange Accuracy | AVISTA Macro-F1 | Python Macro-F1 | Orange Macro-F1 | AVISTA ROC-AUC | Python ROC-AUC | Orange ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 0.517560 | 0.517560 | 0.517560 | 0.516819 | 0.516819 | 0.516819 | 0.699686 | — | 0.699686 |
| Logistic Regression | Validation | 0.361111 | 0.361111 | 0.361111 | 0.367735 | 0.367735 | 0.367735 | 0.575776 | — | 0.575776 |
| Logistic Regression | Test | 0.398601 | 0.398601 | 0.398601 | 0.376359 | 0.376359 | 0.376359 | 0.600150 | — | 0.600150 |
| Logistic Regression | 5-fold CV (diagnostic; separate CV protocol) | 0.374687 | 0.474929 | — | 0.363321 | 0.298228 | — | 0.534646 | 0.539535 | — |
| Random Forest | Train | 0.950092 | 0.951941 | 0.951941 | 0.946386 | 0.948553 | 0.947985 | 0.995412 | 0.996479 | 0.994555 |
| Random Forest | Validation | 0.402778 | 0.402778 | 0.416667 | 0.337217 | 0.235897 | 0.264665 | 0.595291 | 0.582196 | 0.564462 |
| Random Forest | Test | 0.433566 | 0.482517 | 0.447552 | 0.361977 | 0.348416 | 0.313535 | 0.545571 | 0.523578 | 0.518760 |
| Random Forest | 5-fold CV (diagnostic; separate CV protocol) | 0.374727 | 0.440788 | — | 0.305729 | 0.296331 | — | 0.483269 | 0.478067 | — |
| XGBoost | Train | 0.905730 | 0.903882 | 0.903882 | 0.899450 | 0.898242 | 0.898242 | 0.983431 | 0.983302 | 0.983302 |
| XGBoost | Validation | 0.416667 | 0.388889 | 0.388889 | 0.281661 | 0.268111 | 0.268111 | 0.539835 | 0.550907 | 0.550907 |
| XGBoost | Test | 0.475524 | 0.475524 | 0.475524 | 0.382068 | 0.390165 | 0.390165 | 0.558880 | 0.558966 | 0.558966 |
| XGBoost | 5-fold CV (diagnostic; separate CV protocol) | 0.412788 | 0.424848 | — | 0.306509 | 0.315898 | — | 0.485602 | 0.487102 | — |
| FT-Transformer | Train | 0.495379 | 0.395564 | — | 0.496033 | 0.376591 | — | 0.767658 | 0.676278 | — |
| FT-Transformer | Validation | 0.375000 | 0.361111 | — | 0.372173 | 0.359712 | — | 0.540302 | 0.557632 | — |
| FT-Transformer | Test | 0.335664 | 0.272727 | — | 0.332862 | 0.220179 | — | 0.553623 | 0.523680 | — |
| FT-Transformer | 5-fold CV (diagnostic; exported-feature protocol, not fold-local pipeline parity) | 0.336667 | 0.332646 | — | 0.319175 | 0.296379 | — | 0.529781 | 0.521291 | — |
| TabPFN 2.5 | Train | — | — | — | — | — | — | — | — | — |
| TabPFN 2.5 | Validation | 0.527778 | 0.527778 | — | 0.230303 | 0.230303 | — | 0.592084 | 0.591284 | — |
| TabPFN 2.5 | Test | 0.517483 | 0.517483 | — | 0.227343 | 0.227343 | — | 0.548308 | 0.546165 | — |
| TabPFN 2.5 | 5-fold CV (diagnostic; separate CV protocol) | 0.521051 | 0.521051 | — | 0.228372 | 0.228372 | — | 0.509765 | 0.510477 | — |
