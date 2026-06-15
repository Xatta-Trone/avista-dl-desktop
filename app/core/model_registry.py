"""Classification model registry for AVISTA."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a selectable classification model."""

    name: str
    display_name: str
    model_family: str
    category: str
    supports_proba: bool
    supports_class_weight: bool
    requires_gpu: bool
    requires_torch: bool
    requires_optional_package: str | None
    default_params: dict[str, Any] = field(default_factory=dict)
    parameter_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    enabled: bool = True
    description: str = ""
    task_type: str = field(default="classification", init=False)

    @property
    def estimator_type(self) -> str:
        """Compatibility label used by the current training orchestration."""

        if self.model_family == "xgboost":
            return "xgboost"
        if self.model_family == "sklearn":
            return "sklearn"
        if self.model_family == "tabpfn":
            return "tabpfn"
        return "deep"


def _spec(
    name: str,
    display_name: str,
    model_family: str,
    category: str,
    *,
    supports_proba: bool = True,
    supports_class_weight: bool = False,
    requires_gpu: bool = False,
    requires_torch: bool = False,
    requires_optional_package: str | None = None,
    default_params: dict[str, Any] | None = None,
    parameter_metadata: dict[str, dict[str, Any]] | None = None,
    enabled: bool = True,
    description: str,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        display_name=display_name,
        model_family=model_family,
        category=category,
        supports_proba=supports_proba,
        supports_class_weight=supports_class_weight,
        requires_gpu=requires_gpu,
        requires_torch=requires_torch,
        requires_optional_package=requires_optional_package,
        default_params=default_params or {},
        parameter_metadata=parameter_metadata or {},
        enabled=enabled,
        description=description,
    )


def _tree_ensemble_defaults(*, bootstrap: bool) -> dict[str, Any]:
    return {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "min_impurity_decrease": 0.0,
        "bootstrap": bootstrap,
        "oob_score": False,
        "n_jobs": None,
        "verbose": 0,
        "warm_start": False,
        "class_weight": None,
        "ccp_alpha": 0.0,
        "max_samples": None,
        "monotonic_cst": None,
    }


def _tree_ensemble_metadata(*, bootstrap: bool) -> dict[str, dict[str, Any]]:
    return {
        "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 10_000},
        "criterion": {
            "type": "select",
            "default": "gini",
            "options": ["gini", "entropy", "log_loss"],
        },
        "max_depth": {
            "type": "select_or_int",
            "default": "none",
            "options": ["none", "custom"],
            "min": 1,
            "max": 10_000,
        },
        "min_samples_split": {
            "type": "int",
            "default": 2,
            "min": 2,
            "max": 100_000,
        },
        "min_samples_leaf": {
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 100_000,
        },
        "min_weight_fraction_leaf": {
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 0.5,
            "step": 0.01,
        },
        "max_features": {
            "type": "select",
            "default": "sqrt",
            "options": ["sqrt", "log2", "none"],
        },
        "max_leaf_nodes": {
            "type": "select_or_int",
            "default": "none",
            "options": ["none", "custom"],
            "min": 2,
            "max": 100_000,
        },
        "min_impurity_decrease": {
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
        },
        "bootstrap": {"type": "bool", "default": bootstrap},
        "oob_score": {"type": "bool", "default": False},
        "n_jobs": {
            "type": "select",
            "default": "none",
            "options": ["none", "-1", "1", "2", "4", "8"],
        },
        "verbose": {"type": "int", "default": 0, "min": 0, "max": 10},
        "warm_start": {"type": "bool", "default": False},
        "class_weight": {
            "type": "select",
            "default": "none",
            "options": ["none", "balanced", "balanced_subsample"],
        },
        "ccp_alpha": {
            "type": "float",
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.0001,
        },
        "max_samples": {
            "type": "select_or_float",
            "default": "none",
            "options": ["none", "custom"],
            "min": 0.01,
            "max": 1.0,
            "step": 0.01,
            "enabled_when": {"parameter": "bootstrap", "equals": True},
        },
        "monotonic_cst": {
            "type": "text",
            "default": None,
            "help": "Optional advanced parameter; leave blank unless needed.",
        },
    }


def _xgboost_defaults() -> dict[str, Any]:
    return {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "tree_method": "auto",
        "booster": "gbtree",
        "gamma": 0.0,
        "min_child_weight": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "random_state": None,
        "n_jobs": None,
        "verbosity": 0,
        "enable_categorical": False,
    }


def _xgboost_metadata() -> dict[str, dict[str, Any]]:
    return {
        "n_estimators": {"type": "int", "default": 300, "min": 1, "max": 10_000},
        "learning_rate": {
            "type": "float", "default": 0.05, "min": 0.0001, "max": 1.0, "step": 0.01
        },
        "max_depth": {"type": "int", "default": 5, "min": 1, "max": 100},
        "subsample": {
            "type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05
        },
        "colsample_bytree": {
            "type": "float", "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05
        },
        "objective": {
            "type": "select",
            "default": "multi:softprob",
            "options": ["multi:softprob", "binary:logistic"],
        },
        "eval_metric": {
            "type": "select",
            "default": "mlogloss",
            "options": ["mlogloss", "logloss", "auc", "aucpr", "error"],
        },
        "tree_method": {
            "type": "select",
            "default": "auto",
            "options": ["auto", "hist", "approx", "exact"],
        },
        "booster": {
            "type": "select", "default": "gbtree", "options": ["gbtree", "dart"]
        },
        "gamma": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1
        },
        "min_child_weight": {
            "type": "float", "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1
        },
        "reg_alpha": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1
        },
        "reg_lambda": {
            "type": "float", "default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.1
        },
        "scale_pos_weight": {
            "type": "float",
            "default": 1.0,
            "min": 0.0001,
            "max": 100_000.0,
            "step": 0.1,
        },
        "random_state": {
            "type": "select",
            "default": "use_experiment_seed",
            "options": ["none", "use_experiment_seed"],
        },
        "n_jobs": {
            "type": "select",
            "default": "none",
            "options": ["none", "-1", "1", "2", "4", "8"],
        },
        "verbosity": {"type": "int", "default": 0, "min": 0, "max": 3},
        "enable_categorical": {"type": "bool", "default": False},
    }


def _gradient_boosting_defaults() -> dict[str, Any]:
    return {
        "loss": "log_loss",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "criterion": "friedman_mse",
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "min_weight_fraction_leaf": 0.0,
        "max_depth": 3,
        "min_impurity_decrease": 0.0,
        "init": None,
        "random_state": None,
        "max_features": None,
        "verbose": 0,
        "max_leaf_nodes": None,
        "warm_start": False,
        "validation_fraction": 0.1,
        "n_iter_no_change": None,
        "tol": 0.0001,
        "ccp_alpha": 0.0,
    }


def _gradient_boosting_metadata() -> dict[str, dict[str, Any]]:
    return {
        "loss": {
            "type": "select", "default": "log_loss", "options": ["log_loss", "exponential"]
        },
        "learning_rate": {
            "type": "float", "default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.01
        },
        "n_estimators": {"type": "int", "default": 100, "min": 1, "max": 10_000},
        "subsample": {
            "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05
        },
        "criterion": {
            "type": "select",
            "default": "friedman_mse",
            "options": ["friedman_mse", "squared_error"],
        },
        "min_samples_split": {"type": "int", "default": 2, "min": 2, "max": 100_000},
        "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 100_000},
        "min_weight_fraction_leaf": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01
        },
        "max_depth": {
            "type": "select_or_int",
            "default": "custom",
            "options": ["none", "custom"],
            "custom_default": 3,
            "min": 1,
            "max": 10_000,
        },
        "min_impurity_decrease": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001
        },
        "init": {
            "type": "text",
            "default": None,
            "help": "Advanced estimator object; leave blank unless needed.",
        },
        "random_state": {
            "type": "select",
            "default": "use_experiment_seed",
            "options": ["none", "use_experiment_seed"],
        },
        "max_features": {
            "type": "select", "default": "none", "options": ["none", "sqrt", "log2"]
        },
        "verbose": {"type": "int", "default": 0, "min": 0, "max": 10},
        "max_leaf_nodes": {
            "type": "select_or_int",
            "default": "none",
            "options": ["none", "custom"],
            "min": 2,
            "max": 100_000,
        },
        "warm_start": {"type": "bool", "default": False},
        "validation_fraction": {
            "type": "float", "default": 0.1, "min": 0.01, "max": 0.9, "step": 0.01
        },
        "n_iter_no_change": {
            "type": "select_or_int",
            "default": "none",
            "options": ["none", "custom"],
            "min": 1,
            "max": 10_000,
        },
        "tol": {
            "type": "float",
            "default": 0.0001,
            "min": 0.00000001,
            "max": 1.0,
            "step": 0.0001,
        },
        "ccp_alpha": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001
        },
    }


def _hist_gradient_boosting_defaults() -> dict[str, Any]:
    return {
        "loss": "log_loss",
        "learning_rate": 0.1,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "max_depth": None,
        "min_samples_leaf": 20,
        "l2_regularization": 0.0,
        "max_features": 1.0,
        "max_bins": 255,
        "categorical_features": "from_dtype",
        "monotonic_cst": None,
        "interaction_cst": None,
        "warm_start": False,
        "early_stopping": "auto",
        "scoring": "loss",
        "validation_fraction": 0.1,
        "n_iter_no_change": 10,
        "tol": 0.0000001,
        "verbose": 0,
        "random_state": None,
        "class_weight": None,
    }


def _hist_gradient_boosting_metadata() -> dict[str, dict[str, Any]]:
    return {
        "loss": {"type": "select", "default": "log_loss", "options": ["log_loss"]},
        "learning_rate": {
            "type": "float", "default": 0.1, "min": 0.0001, "max": 1.0, "step": 0.01
        },
        "max_iter": {"type": "int", "default": 100, "min": 1, "max": 10_000},
        "max_leaf_nodes": {
            "type": "select_or_int",
            "default": "custom",
            "options": ["none", "custom"],
            "custom_default": 31,
            "min": 2,
            "max": 100_000,
        },
        "max_depth": {
            "type": "select_or_int",
            "default": "none",
            "options": ["none", "custom"],
            "min": 1,
            "max": 10_000,
        },
        "min_samples_leaf": {"type": "int", "default": 20, "min": 1, "max": 100_000},
        "l2_regularization": {
            "type": "float", "default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1
        },
        "max_features": {
            "type": "float", "default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01
        },
        "max_bins": {"type": "int", "default": 255, "min": 2, "max": 255},
        "categorical_features": {
            "type": "select",
            "default": "from_dtype",
            "options": ["from_dtype", "none"],
        },
        "monotonic_cst": {
            "type": "text",
            "default": None,
            "help": "Advanced parameter; leave blank unless needed.",
        },
        "interaction_cst": {
            "type": "text",
            "default": None,
            "help": "Advanced parameter; leave blank unless needed.",
        },
        "warm_start": {"type": "bool", "default": False},
        "early_stopping": {
            "type": "select", "default": "auto", "options": ["auto", "true", "false"]
        },
        "scoring": {
            "type": "select",
            "default": "loss",
            "options": ["loss", "accuracy", "balanced_accuracy", "f1_macro"],
        },
        "validation_fraction": {
            "type": "float", "default": 0.1, "min": 0.01, "max": 0.9, "step": 0.01
        },
        "n_iter_no_change": {"type": "int", "default": 10, "min": 1, "max": 10_000},
        "tol": {
            "type": "float",
            "default": 0.0000001,
            "min": 0.000000001,
            "max": 1.0,
            "step": 0.0000001,
        },
        "verbose": {"type": "int", "default": 0, "min": 0, "max": 10},
        "random_state": {
            "type": "select",
            "default": "use_experiment_seed",
            "options": ["none", "use_experiment_seed"],
        },
        "class_weight": {
            "type": "select", "default": "none", "options": ["none", "balanced"]
        },
    }


def _adaboost_defaults() -> dict[str, Any]:
    return {
        "estimator": None,
        "n_estimators": 50,
        "learning_rate": 1.0,
        "random_state": None,
    }


def _adaboost_metadata() -> dict[str, dict[str, Any]]:
    return {
        "estimator": {
            "type": "select",
            "default": "none",
            "options": ["none"],
            "help": "Base estimator customization is not implemented yet.",
        },
        "n_estimators": {"type": "int", "default": 50, "min": 1, "max": 10_000},
        "learning_rate": {
            "type": "float", "default": 1.0, "min": 0.0001, "max": 10.0, "step": 0.1
        },
        "random_state": {
            "type": "select",
            "default": "use_experiment_seed",
            "options": ["none", "use_experiment_seed"],
        },
    }


def _mamba_attention_defaults() -> dict[str, Any]:
    return {
        # from reference MambaAttentionClassifier(...): hidden_dim=256
        "hidden_dim": 256,
        # from reference MambaAttentionClassifier(...): dropout=0.3
        "dropout": 0.3,
    }


def _mamba_attention_metadata() -> dict[str, dict[str, Any]]:
    return {
        "hidden_dim": {
            "type": "int",
            # from reference MambaAttentionClassifier(...): hidden_dim=256
            "default": 256,
            "label": "Hidden Dimension",
            "section": "Architecture Parameters",
        },
        "dropout": {
            "type": "float",
            # from reference MambaAttentionClassifier(...): dropout=0.3
            "default": 0.3,
            "label": "Dropout Rate",
            "section": "Architecture Parameters",
        },
    }


def _ft_transformer_defaults() -> dict[str, Any]:
    return {
        # from reference FTTransformerClassifier(...): d_token=128
        "d_token": 128,
        # from reference FTTransformerClassifier(...): n_heads=8
        "n_heads": 8,
        # from reference FTTransformerClassifier(...): n_layers=3
        "n_layers": 3,
        # from reference FTTransformerClassifier(...): dropout=0.1
        "dropout": 0.1,
    }


def _ft_transformer_metadata() -> dict[str, dict[str, Any]]:
    return {
        "d_token": {
            "type": "int",
            # from reference FTTransformerClassifier(...): d_token=128
            "default": 128,
            "label": "Token Dimension",
            "section": "Architecture Parameters",
        },
        "n_heads": {
            "type": "int",
            # from reference FTTransformerClassifier(...): n_heads=8
            "default": 8,
            "label": "Attention Heads",
            "section": "Architecture Parameters",
        },
        "n_layers": {
            "type": "int",
            # from reference FTTransformerClassifier(...): n_layers=3
            "default": 3,
            "label": "Transformer Layers",
            "section": "Architecture Parameters",
        },
        "dropout": {
            "type": "float",
            # from reference FTTransformerClassifier(...): dropout=0.1
            "default": 0.1,
            "label": "Dropout Rate",
            "section": "Architecture Parameters",
        },
    }


def _autoint_defaults() -> dict[str, Any]:
    return {
        # from reference AutoIntClassifier(...): d=64
        "d": 64,
        # from reference AutoIntClassifier(...): n_heads=4
        "n_heads": 4,
        # from reference AutoIntClassifier(...): n_layers=3
        "n_layers": 3,
        # from reference AutoIntClassifier(...): dropout=0.1
        "dropout": 0.1,
    }


def _autoint_metadata() -> dict[str, dict[str, Any]]:
    return {
        "d": {
            "type": "int",
            # from reference AutoIntClassifier(...): d=64
            "default": 64,
            "label": "Interaction Dimension",
            "section": "Architecture Parameters",
        },
        "n_heads": {
            "type": "int",
            # from reference AutoIntClassifier(...): n_heads=4
            "default": 4,
            "label": "Attention Heads",
            "section": "Architecture Parameters",
        },
        "n_layers": {
            "type": "int",
            # from reference AutoIntClassifier(...): n_layers=3
            "default": 3,
            "label": "Interaction Layers",
            "section": "Architecture Parameters",
        },
        "dropout": {
            "type": "float",
            # from reference AutoIntClassifier(...): dropout=0.1
            "default": 0.1,
            "label": "Dropout Rate",
            "section": "Architecture Parameters",
        },
    }


def _tab_resnet_defaults() -> dict[str, Any]:
    return {
        # from reference TabResNet(...): hidden=256
        "hidden": 256,
        # from reference TabResNet(...): n_blocks=6
        "n_blocks": 6,
        # from reference TabResNet(...): dropout=0.2
        "dropout": 0.2,
    }


def _tab_resnet_metadata() -> dict[str, dict[str, Any]]:
    return {
        "hidden": {
            "type": "int",
            # from reference TabResNet(...): hidden=256
            "default": 256,
            "label": "Hidden Dimension",
            "section": "Architecture Parameters",
        },
        "n_blocks": {
            "type": "int",
            # from reference TabResNet(...): n_blocks=6
            "default": 6,
            "label": "Residual Blocks",
            "section": "Architecture Parameters",
        },
        "dropout": {
            "type": "float",
            # from reference TabResNet(...): dropout=0.2
            "default": 0.2,
            "label": "Dropout Rate",
            "section": "Architecture Parameters",
        },
    }


def _tabpfn_defaults() -> dict[str, Any]:
    return {
        # from installed TabPFNClassifier signature: n_estimators=8
        "n_estimators": 8,
    }


def _tabpfn_metadata() -> dict[str, dict[str, Any]]:
    return {
        "n_estimators": {
            "type": "int",
            "default": 8,
            "min": 1,
            "max": 100,
            "label": "Estimators",
        },
    }


MODEL_SPECS: tuple[ModelSpec, ...] = (
    _spec(
        "logistic_regression",
        "Logistic Regression",
        "sklearn",
        "Linear Models",
        supports_class_weight=True,
        default_params={
            "penalty": "l2",
            "C": 1.0,
            "l1_ratio": None,
            "dual": False,
            "tol": 0.0001,
            "fit_intercept": True,
            "intercept_scaling": 1.0,
            "class_weight": None,
            "solver": "lbfgs",
            "max_iter": 100,
            "verbose": 0,
            "warm_start": False,
            "n_jobs": None,
        },
        parameter_metadata={
            "penalty": {
                "type": "select",
                "default": "l2",
                "options": ["l2", "l1", "elasticnet", "none"],
            },
            "C": {
                "type": "float",
                "default": 1.0,
                "min": 0.000001,
                "max": 1_000_000.0,
                "step": 0.1,
            },
            "l1_ratio": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "enabled_when": {"parameter": "penalty", "equals": "elasticnet"},
            },
            "dual": {"type": "bool", "default": False},
            "tol": {
                "type": "float",
                "default": 0.0001,
                "min": 0.00000001,
                "max": 1.0,
                "step": 0.0001,
            },
            "fit_intercept": {"type": "bool", "default": True},
            "intercept_scaling": {
                "type": "float",
                "default": 1.0,
                "min": 0.000001,
                "max": 1_000_000.0,
                "step": 0.1,
            },
            "class_weight": {
                "type": "select",
                "default": "none",
                "options": ["none", "balanced"],
            },
            "solver": {
                "type": "select",
                "default": "lbfgs",
                "options": [
                    "lbfgs",
                    "liblinear",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ],
            },
            "max_iter": {"type": "int", "default": 100, "min": 1, "max": 100_000},
            "verbose": {"type": "int", "default": 0, "min": 0, "max": 10},
            "warm_start": {"type": "bool", "default": False},
            "n_jobs": {
                "type": "select",
                "default": "none",
                "options": ["none", "-1", "1", "2", "4", "8"],
            },
        },
        description="Regularized linear classifier with probability estimates.",
    ),
    _spec(
        "random_forest",
        "Random Forest",
        "sklearn",
        "Tree-Based Models",
        supports_class_weight=True,
        default_params=_tree_ensemble_defaults(bootstrap=True),
        parameter_metadata=_tree_ensemble_metadata(bootstrap=True),
        description="Bagged decision-tree ensemble with robust nonlinear behavior.",
    ),
    _spec(
        "extra_trees",
        "Extra Trees",
        "sklearn",
        "Tree-Based Models",
        supports_class_weight=True,
        default_params=_tree_ensemble_defaults(bootstrap=False),
        parameter_metadata=_tree_ensemble_metadata(bootstrap=False),
        description="Highly randomized tree ensemble for fast nonlinear classification.",
    ),
    _spec(
        "decision_tree",
        "Decision Tree",
        "sklearn",
        "Tree-Based Models",
        supports_class_weight=True,
        default_params={
            "criterion": "gini",
            "splitter": "best",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": None,
            "max_leaf_nodes": None,
            "min_impurity_decrease": 0.0,
            "class_weight": None,
            "ccp_alpha": 0.0,
            "monotonic_cst": None,
        },
        parameter_metadata={
            "criterion": {
                "type": "select",
                "default": "gini",
                "options": ["gini", "entropy", "log_loss"],
            },
            "splitter": {
                "type": "select",
                "default": "best",
                "options": ["best", "random"],
            },
            "max_depth": {
                "type": "select_or_int",
                "default": "none",
                "options": ["none", "custom"],
                "min": 1,
                "max": 10_000,
            },
            "min_samples_split": {
                "type": "int",
                "default": 2,
                "min": 2,
                "max": 100_000,
            },
            "min_samples_leaf": {
                "type": "int",
                "default": 1,
                "min": 1,
                "max": 100_000,
            },
            "min_weight_fraction_leaf": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 0.5,
                "step": 0.01,
            },
            "max_features": {
                "type": "select",
                "default": "none",
                "options": ["none", "sqrt", "log2"],
            },
            "max_leaf_nodes": {
                "type": "select_or_int",
                "default": "none",
                "options": ["none", "custom"],
                "min": 2,
                "max": 100_000,
            },
            "min_impurity_decrease": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.0001,
            },
            "class_weight": {
                "type": "select",
                "default": "none",
                "options": ["none", "balanced"],
            },
            "ccp_alpha": {
                "type": "float",
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.0001,
            },
            "monotonic_cst": {
                "type": "text",
                "default": None,
                "help": "Optional advanced parameter; leave blank unless needed.",
            },
        },
        description="Single interpretable nonlinear classification tree.",
    ),
    _spec(
        "xgboost",
        "XGBoost",
        "xgboost",
        "Boosting Models",
        requires_optional_package="xgboost",
        default_params=_xgboost_defaults(),
        parameter_metadata=_xgboost_metadata(),
        description="Gradient-boosted decision trees provided by XGBoost.",
    ),
    _spec(
        "gradient_boosting",
        "Gradient Boosting",
        "sklearn",
        "Boosting Models",
        default_params=_gradient_boosting_defaults(),
        parameter_metadata=_gradient_boosting_metadata(),
        description="Classic stage-wise gradient-boosted tree classifier.",
    ),
    _spec(
        "hist_gradient_boosting",
        "Hist Gradient Boosting",
        "sklearn",
        "Boosting Models",
        supports_class_weight=True,
        default_params=_hist_gradient_boosting_defaults(),
        parameter_metadata=_hist_gradient_boosting_metadata(),
        description="Histogram-based gradient boosting for larger tabular datasets.",
    ),
    _spec(
        "adaboost",
        "AdaBoost",
        "sklearn",
        "Boosting Models",
        default_params=_adaboost_defaults(),
        parameter_metadata=_adaboost_metadata(),
        description="Adaptive boosting classifier using sequential weak learners.",
    ),
    _spec(
        "svc",
        "Support Vector Classifier",
        "sklearn",
        "Kernel/Distance Models",
        supports_class_weight=True,
        default_params={"probability": True, "random_state": 42},
        description="Kernel support-vector classifier with calibrated probabilities.",
    ),
    _spec(
        "knn",
        "K-Nearest Neighbors",
        "sklearn",
        "Kernel/Distance Models",
        default_params={"n_neighbors": 5},
        description="Distance-based classifier using neighboring training samples.",
    ),
    _spec(
        "gaussian_nb",
        "Gaussian Naive Bayes",
        "sklearn",
        "Naive Bayes",
        description="Probabilistic classifier with Gaussian feature assumptions.",
    ),
    _spec(
        "mamba_attention",
        "MambaAttention",
        "torch",
        "Deep Tabular Models",
        requires_torch=True,
        requires_optional_package="torch",
        default_params=_mamba_attention_defaults(),
        parameter_metadata=_mamba_attention_metadata(),
        enabled=True,
        description=(
            "Reference-backed attention tabular network. Input dimension and class "
            "count are inferred from confirmed saved training artifacts."
        ),
    ),
    _spec(
        "ft_transformer",
        "FT-Transformer",
        "torch",
        "Deep Tabular Models",
        requires_torch=True,
        requires_optional_package="torch",
        default_params=_ft_transformer_defaults(),
        parameter_metadata=_ft_transformer_metadata(),
        enabled=True,
        description=(
            "Reference-backed feature-tokenizing Transformer. Feature count and "
            "class count are inferred from confirmed saved training artifacts."
        ),
    ),
    _spec(
        "autoint",
        "AutoInt",
        "torch",
        "Deep Tabular Models",
        requires_torch=True,
        requires_optional_package="torch",
        default_params=_autoint_defaults(),
        parameter_metadata=_autoint_metadata(),
        enabled=True,
        description=(
            "Reference-backed self-attention network for tabular feature "
            "interactions. Feature count and class count are training-inferred."
        ),
    ),
    _spec(
        "tab_resnet",
        "TabResNet",
        "torch",
        "Deep Tabular Models",
        requires_torch=True,
        requires_optional_package="torch",
        default_params=_tab_resnet_defaults(),
        parameter_metadata=_tab_resnet_metadata(),
        enabled=True,
        description=(
            "Reference-backed residual multilayer network. Input dimension and "
            "class count are inferred from confirmed saved training artifacts."
        ),
    ),
    _spec(
        "tabpfn",
        "TabPFN 2.5",
        "tabpfn",
        "Foundation Tabular Models",
        requires_torch=True,
        requires_optional_package="tabpfn",
        default_params=_tabpfn_defaults(),
        parameter_metadata=_tabpfn_metadata(),
        enabled=True,
        description="Reference-backed TabPFN 2.5 classifier for capped tabular datasets.",
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}
_MODEL_ALIASES: dict[str, str] = {
    alias.casefold(): spec.name
    for spec in MODEL_SPECS
    for alias in (spec.name, spec.display_name)
}


def get_available_models(task_type: str | None = None) -> list[ModelSpec]:
    """Return registered classification models."""

    if task_type is not None and task_type.strip().lower() != "classification":
        return []
    return list(MODEL_SPECS)


def get_model_spec(model_name: str) -> ModelSpec:
    """Return model metadata by canonical name or display name."""

    canonical_name = _MODEL_ALIASES.get(model_name.strip().casefold())
    if canonical_name is None:
        raise ValueError(f"Unknown model '{model_name}'.")
    return MODEL_REGISTRY[canonical_name]
