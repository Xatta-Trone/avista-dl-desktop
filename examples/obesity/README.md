# Project File Structure

```text
obesity/
├── artifacts/                         # AVISTA project artifacts
│   └── preprocessing/
├── data/                              # Dataset files
├── logs/                              # Execution and model-worker logs
├── outputs/                           # AVISTA-generated data splits, models, configurations, and results
│   ├── data_split/
│   ├── edge_cases/
│   └── training/
│       ├── FT-Transformer/
│       ├── LogisticRegression/
│       ├── RandomForest/
│       ├── TabPFN_2_5/
│       └── XGBoost/
├── reproduciablity_python/            # Independent Python results for LR, RF, XGBoost, FT-Transformer, and TabPFN
├── reproducibility_orange/            # Orange results for LR, RF, and XGBoost
├── obesity.avista                     # AVISTA project file
├── orange_implementation.ipynb        # Orange Python implementation notebook
└── raw_python_implementation.ipynb    # Independent Python implementation notebook
```

# Consolidated Results

| Model | Split | AVISTA Accuracy | Python Accuracy | Orange Accuracy | AVISTA Macro-F1 | Python Macro-F1 | Orange Macro-F1 | AVISTA ROC-AUC | Python ROC-AUC | Orange ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 0.897945 | 0.897945 | 0.897945 | 0.893220 | 0.893220 | 0.893220 | 0.989751 | — | 0.989751 |
| Logistic Regression | Validation | 0.866029 | 0.866029 | 0.866029 | 0.864122 | 0.864122 | 0.864122 | 0.983762 | — | 0.983762 |
| Logistic Regression | Test | 0.861244 | 0.861244 | 0.861244 | 0.853330 | 0.853330 | 0.853330 | 0.985191 | — | 0.985191 |
| Logistic Regression | 5-fold CV | 0.869178 ± 0.030814 | 0.869178 ± 0.027001 | — | 0.861528 ± 0.035129 | 0.861213 ± 0.031393 | — | 0.980716 ± 0.006278 | 0.980605 ± 0.006202 | — |
| Random Forest | Train | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Random Forest | Validation | 0.918660 | 0.937799 | 0.937799 | 0.920916 | 0.939152 | 0.939952 | 0.995583 | 0.996380 | 0.996198 |
| Random Forest | Test | 0.942584 | 0.933014 | 0.928230 | 0.941010 | 0.931492 | 0.925905 | 0.993385 | 0.992978 | 0.993838 |
| Random Forest | 5-fold CV | 0.928767 ± 0.007929 | 0.932192 ± 0.012704 | — | 0.925874 ± 0.008808 | 0.929852 ± 0.013795 | — | 0.991950 ± 0.002485 | 0.992283 ± 0.002507 | — |
| XGBoost | Train | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| XGBoost | Validation | 0.966507 | 0.961722 | 0.961722 | 0.965196 | 0.961424 | 0.961424 | 0.999579 | 0.999602 | 0.999602 |
| XGBoost | Test | 0.961722 | 0.956938 | 0.956938 | 0.961212 | 0.956234 | 0.956234 | 0.998808 | 0.998754 | 0.998754 |
| XGBoost | 5-fold CV | 0.963699 ± 0.007052 | 0.964384 ± 0.007688 | — | 0.961954 ± 0.006945 | 0.962619 ± 0.007608 | — | 0.997961 ± 0.001147 | 0.997920 ± 0.001143 | — |
| FT-Transformer | Train | 0.994521 | 0.999315 | — | 0.994030 | 0.999256 | — | 0.999968 | 1.000000 | — |
| FT-Transformer | Validation | 0.966507 | 0.976077 | — | 0.966113 | 0.974270 | — | 0.997755 | 0.997703 | — |
| FT-Transformer | Test | 0.942584 | 0.944976 | — | 0.940684 | 0.942457 | — | 0.996044 | 0.994626 | — |
| FT-Transformer | 5-fold CV (diagnostic; exported-feature protocol) | 0.943151 ± 0.010522 | 0.945205 ± 0.006849 | — | 0.939728 ± 0.011153 | 0.941531 ± 0.008100 | — | 0.995414 ± 0.002231 | 0.996591 ± 0.001407 | — |
| TabPFN 2.5 | Train | — | — | — | — | — | — | — | — | — |
| TabPFN 2.5 | Validation | 0.980861 | 0.980861 | — | 0.979268 | 0.979268 | — | 0.999864 | 0.999864 | — |
| TabPFN 2.5 | Test | 0.990431 | 0.990431 | — | 0.990103 | 0.990103 | — | 0.999818 | 0.999818 | — |
| TabPFN 2.5 | 5-fold CV | 0.991096 ± 0.004110 | 0.991096 ± 0.004110 | — | 0.990840 ± 0.004469 | 0.990840 ± 0.004469 | — | 0.999928 ± 0.000065 | 0.999928 ± 0.000065 | — |
