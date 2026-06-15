import builtins
import importlib.util
import inspect
import sys

import pytest
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.core.model_registry import get_available_models, get_model_spec
from app.models.model_factory import create_model


EXPECTED_MODEL_NAMES = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "mamba_attention",
    "ft_transformer",
    "autoint",
    "tab_resnet",
    "tabpfn",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "svc",
    "knn",
    "gaussian_nb",
    "decision_tree",
    "adaboost",
}


def test_registry_returns_classification_models_only():
    specs = get_available_models()

    assert {spec.name for spec in specs} == EXPECTED_MODEL_NAMES
    assert all(spec.task_type == "classification" for spec in specs)
    assert get_available_models("regression") == []


def test_model_names_and_display_names_are_unique():
    specs = get_available_models("classification")

    assert len({spec.name for spec in specs}) == len(specs)
    assert len({spec.display_name for spec in specs}) == len(specs)


def test_registry_has_required_categories_and_metadata():
    specs = get_available_models()
    categories = {spec.category for spec in specs}

    assert categories == {
        "Linear Models",
        "Tree-Based Models",
        "Boosting Models",
        "Kernel/Distance Models",
        "Naive Bayes",
        "Deep Tabular Models",
        "Foundation Tabular Models",
    }
    logistic = get_model_spec("Logistic Regression")
    assert logistic.name == "logistic_regression"
    assert logistic.supports_proba is True
    assert logistic.supports_class_weight is True
    assert logistic.description


def test_mamba_attention_defaults_match_reference_architecture():
    spec = get_model_spec("mamba_attention")

    assert spec.default_params == {"hidden_dim": 256, "dropout": 0.3}
    assert set(spec.parameter_metadata) == {"hidden_dim", "dropout"}
    assert "input_dim" not in spec.default_params
    assert "num_classes" not in spec.default_params
    assert spec.enabled is True


def test_ft_transformer_defaults_match_reference_architecture():
    spec = get_model_spec("ft_transformer")

    assert spec.default_params == {
        "d_token": 128,
        "n_heads": 8,
        "n_layers": 3,
        "dropout": 0.1,
    }
    assert set(spec.parameter_metadata) == {
        "d_token",
        "n_heads",
        "n_layers",
        "dropout",
    }
    assert "n_features" not in spec.default_params
    assert "n_classes" not in spec.default_params
    assert spec.enabled is True


def test_autoint_defaults_match_reference_architecture():
    spec = get_model_spec("autoint")

    assert spec.default_params == {
        "d": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
    }
    assert set(spec.parameter_metadata) == {
        "d",
        "n_heads",
        "n_layers",
        "dropout",
    }
    assert "n_features" not in spec.default_params
    assert "n_classes" not in spec.default_params
    assert spec.enabled is True


def test_tab_resnet_defaults_match_reference_architecture():
    spec = get_model_spec("tab_resnet")

    assert spec.default_params == {
        "hidden": 256,
        "n_blocks": 6,
        "dropout": 0.2,
    }
    assert set(spec.parameter_metadata) == {
        "hidden",
        "n_blocks",
        "dropout",
    }
    assert "input_dim" not in spec.default_params
    assert "n_classes" not in spec.default_params
    assert spec.enabled is True


def test_tabpfn_metadata_exposes_only_estimators():
    spec = get_model_spec("tabpfn")

    assert spec.display_name == "TabPFN 2.5"
    assert spec.model_family == "tabpfn"
    assert spec.category == "Foundation Tabular Models"
    assert spec.requires_optional_package == "tabpfn"
    assert spec.requires_torch is True
    assert spec.supports_proba is True
    assert spec.default_params == {"n_estimators": 8}
    assert set(spec.parameter_metadata) == {"n_estimators"}
    assert spec.parameter_metadata["n_estimators"] == {
        "type": "int",
        "default": 8,
        "min": 1,
        "max": 100,
        "label": "Estimators",
    }
    assert spec.enabled is True


@pytest.mark.parametrize(
    ("model_name", "expected_defaults", "metadata_checks"),
    [
        (
            "logistic_regression",
            {
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
            {
                "penalty": ("select", ["l2", "l1", "elasticnet", "none"]),
                "C": ("float", None),
                "max_iter": ("int", None),
            },
        ),
        (
            "random_forest",
            {"n_estimators": 100, "bootstrap": True, "max_depth": None, "n_jobs": None},
            {
                "criterion": ("select", ["gini", "entropy", "log_loss"]),
                "max_depth": ("select_or_int", ["none", "custom"]),
                "max_samples": ("select_or_float", ["none", "custom"]),
            },
        ),
        (
            "extra_trees",
            {"n_estimators": 100, "bootstrap": False, "max_depth": None, "n_jobs": None},
            {
                "criterion": ("select", ["gini", "entropy", "log_loss"]),
                "max_depth": ("select_or_int", ["none", "custom"]),
                "max_samples": ("select_or_float", ["none", "custom"]),
            },
        ),
        (
            "decision_tree",
            {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": None,
                "max_features": None,
            },
            {
                "splitter": ("select", ["best", "random"]),
                "max_depth": ("select_or_int", ["none", "custom"]),
                "max_features": ("select", ["none", "sqrt", "log2"]),
            },
        ),
        (
            "xgboost",
            {
                "n_estimators": 300,
                "objective": "multi:softprob",
                "tree_method": "auto",
                "random_state": None,
                "n_jobs": None,
                "enable_categorical": False,
            },
            {
                "objective": (
                    "select",
                    ["multi:softprob", "binary:logistic"],
                ),
                "learning_rate": ("float", None),
                "n_estimators": ("int", None),
            },
        ),
        (
            "gradient_boosting",
            {
                "loss": "log_loss",
                "criterion": "friedman_mse",
                "max_depth": 3,
                "n_iter_no_change": None,
                "random_state": None,
            },
            {
                "loss": ("select", ["log_loss", "exponential"]),
                "max_depth": ("select_or_int", ["none", "custom"]),
                "n_iter_no_change": ("select_or_int", ["none", "custom"]),
            },
        ),
        (
            "hist_gradient_boosting",
            {
                "max_iter": 100,
                "max_leaf_nodes": 31,
                "max_depth": None,
                "categorical_features": "from_dtype",
                "early_stopping": "auto",
                "class_weight": None,
            },
            {
                "max_leaf_nodes": ("select_or_int", ["none", "custom"]),
                "early_stopping": ("select", ["auto", "true", "false"]),
                "class_weight": ("select", ["none", "balanced"]),
            },
        ),
        (
            "adaboost",
            {
                "estimator": None,
                "n_estimators": 50,
                "learning_rate": 1.0,
                "random_state": None,
            },
            {
                "estimator": ("select", ["none"]),
                "n_estimators": ("int", None),
                "learning_rate": ("float", None),
            },
        ),
    ],
)
def test_requested_model_defaults_and_parameter_metadata(
    model_name, expected_defaults, metadata_checks
):
    spec = get_model_spec(model_name)

    assert set(spec.parameter_metadata) == set(spec.default_params)
    for name, value in expected_defaults.items():
        assert spec.default_params[name] == value
    for name, (parameter_type, options) in metadata_checks.items():
        metadata = spec.parameter_metadata[name]
        assert metadata["type"] == parameter_type
        assert "default" in metadata
        if options is not None:
            assert metadata["options"] == options


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("Logistic Regression", LogisticRegression),
        ("random_forest", RandomForestClassifier),
        ("Extra Trees", ExtraTreesClassifier),
        ("Gradient Boosting", GradientBoostingClassifier),
        ("Hist Gradient Boosting", HistGradientBoostingClassifier),
        ("Support Vector Classifier", SVC),
        ("K-Nearest Neighbors", KNeighborsClassifier),
        ("Gaussian Naive Bayes", GaussianNB),
        ("Decision Tree", DecisionTreeClassifier),
        ("AdaBoost", AdaBoostClassifier),
    ],
)
def test_create_model_works_for_builtin_classifiers(model_name, expected_type):
    model = create_model(model_name, params={"random_state": 7} if model_name == "Decision Tree" else None)

    assert isinstance(model, expected_type)


def test_create_model_applies_supported_class_weights():
    model = create_model(
        "Logistic Regression",
        params={"max_iter": 200},
        class_weights={0: 1.0, 1: 2.0},
    )

    assert model.max_iter == 200
    assert model.class_weight == {0: 1.0, 1: 2.0}


def test_optional_package_missing_behavior_is_clear(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("No module named xgboost")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Optional package 'xgboost' is required"):
        create_model("XGBoost")


def test_tabpfn_missing_behavior_is_clear(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tabpfn":
            raise ImportError("No module named tabpfn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="Optional package 'tabpfn' is required"):
        create_model("TabPFN")


def test_optional_packages_are_not_imported_by_registry_or_factory_import():
    assert "tabpfn" not in sys.modules
    assert "app.models.torch_tabular_models" not in sys.modules


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed.")
def test_torch_tabular_models_import_and_instantiate_when_torch_is_installed():
    import torch

    from app.models.torch_tabular_models import (
        AutoIntClassifier,
        FTTransformerClassifier,
        MambaAttentionClassifier,
        TabResNet,
    )

    models = [
        MambaAttentionClassifier(input_dim=4, hidden_dim=32, num_classes=3),
        FTTransformerClassifier(
            n_features=4,
            n_classes=3,
            d_token=16,
            n_heads=4,
            n_layers=1,
        ),
        AutoIntClassifier(
            n_features=4,
            n_classes=3,
            d=16,
            n_heads=4,
            n_layers=1,
        ),
        TabResNet(input_dim=4, n_classes=3, hidden=16, n_blocks=2),
    ]

    inputs = torch.randn(2, 4)
    assert all(model(inputs).shape == (2, 3) for model in models)
    factory_model = create_model(
        "tab_resnet",
        params={"input_dim": 4, "n_classes": 3, "hidden": 16, "n_blocks": 1},
        device="cpu",
    )
    assert factory_model(inputs).shape == (2, 3)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed.")
def test_mamba_attention_signature_matches_reference():
    from app.models.torch_tabular_models import MambaAttentionClassifier

    signature = inspect.signature(MambaAttentionClassifier.__init__)

    assert list(signature.parameters) == [
        "self",
        "input_dim",
        "hidden_dim",
        "num_classes",
        "dropout",
    ]
    assert signature.parameters["input_dim"].default is inspect.Parameter.empty
    assert signature.parameters["hidden_dim"].default == 256
    assert signature.parameters["num_classes"].default == 3
    assert signature.parameters["dropout"].default == 0.3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed.")
def test_ft_transformer_signature_matches_reference():
    from app.models.torch_tabular_models import FTTransformerClassifier

    signature = inspect.signature(FTTransformerClassifier.__init__)

    assert list(signature.parameters) == [
        "self",
        "n_features",
        "n_classes",
        "d_token",
        "n_heads",
        "n_layers",
        "dropout",
    ]
    assert signature.parameters["n_features"].default is inspect.Parameter.empty
    assert signature.parameters["n_classes"].default == 3
    assert signature.parameters["d_token"].default == 128
    assert signature.parameters["n_heads"].default == 8
    assert signature.parameters["n_layers"].default == 3
    assert signature.parameters["dropout"].default == 0.1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed.")
def test_autoint_signature_matches_reference():
    from app.models.torch_tabular_models import AutoIntClassifier

    signature = inspect.signature(AutoIntClassifier.__init__)

    assert list(signature.parameters) == [
        "self",
        "n_features",
        "n_classes",
        "d",
        "n_heads",
        "n_layers",
        "dropout",
    ]
    assert signature.parameters["n_features"].default is inspect.Parameter.empty
    assert signature.parameters["n_classes"].default == 3
    assert signature.parameters["d"].default == 64
    assert signature.parameters["n_heads"].default == 4
    assert signature.parameters["n_layers"].default == 3
    assert signature.parameters["dropout"].default == 0.1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed.")
def test_tab_resnet_signature_matches_reference():
    from app.models.torch_tabular_models import TabResNet

    signature = inspect.signature(TabResNet.__init__)

    assert list(signature.parameters) == [
        "self",
        "input_dim",
        "n_classes",
        "hidden",
        "n_blocks",
        "dropout",
    ]
    assert signature.parameters["input_dim"].default is inspect.Parameter.empty
    assert signature.parameters["n_classes"].default == 3
    assert signature.parameters["hidden"].default == 256
    assert signature.parameters["n_blocks"].default == 6
    assert signature.parameters["dropout"].default == 0.2


def test_no_regression_models_are_registered():
    for spec in get_available_models():
        assert spec.task_type == "classification"
        assert "regressor" not in spec.display_name.lower()
