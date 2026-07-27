"""Factories for sklearn-compatible classification models."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.core.model_registry import get_model_spec
from app.utils.resources import (
    expected_packaged_xgboost_dll,
    is_packaged_application,
)


_SKLEARN_CLASSIFIERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "extra_trees": ExtraTreesClassifier,
    "decision_tree": DecisionTreeClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "adaboost": AdaBoostClassifier,
    "svc": SVC,
    "knn": KNeighborsClassifier,
    "gaussian_nb": GaussianNB,
}


def create_sklearn_model(
    model_name: str,
    task_type: str,
    params: dict[str, Any] | None = None,
    class_weights: dict[Any, float] | str | None = None,
) -> Any:
    """Instantiate a supported sklearn-compatible model."""

    normalized_task = task_type.strip().lower()
    if normalized_task == "regression":
        return _create_legacy_regression_model(model_name, params)
    if normalized_task != "classification":
        raise ValueError(f"Unsupported task type '{task_type}'.")

    spec = get_model_spec(model_name)
    model_params = dict(spec.default_params)
    if params:
        model_params.update(params)
    if class_weights is not None and spec.supports_class_weight:
        model_params["class_weight"] = class_weights

    if spec.name in _SKLEARN_CLASSIFIERS:
        return _SKLEARN_CLASSIFIERS[spec.name](**model_params)
    if spec.name == "xgboost":
        return _create_xgboost_classifier(model_params)
    raise ValueError(f"Model '{model_name}' is not an sklearn-compatible classifier.")


def _create_xgboost_classifier(params: dict[str, Any]) -> Any:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        if is_packaged_application():
            expected_dll = expected_packaged_xgboost_dll()
            raise RuntimeError(
                "XGBoost could not start because its native library was not "
                "included in the AVISTA installation.\n\n"
                f"Expected file:\n{expected_dll}\n\n"
                f"Package import error: {type(exc).__name__}: {exc}\n\n"
                "This is an AVISTA packaging error, not a dataset or "
                "model-parameter error. Review the AVISTA training log."
            ) from exc
        if not isinstance(exc, ImportError):
            raise
        raise ImportError(
            "Optional package 'xgboost' is required for XGBoost. Install xgboost to use this model."
        ) from exc
    return XGBClassifier(**params)


def _create_legacy_regression_model(
    model_name: str,
    params: dict[str, Any] | None,
) -> Any:
    """Preserve existing regression training without registering regression models."""

    model_params = dict(params or {})
    if model_name == "Linear Regression":
        return LinearRegression(**model_params)
    if model_name == "Random Forest Regressor":
        defaults = {"n_estimators": 200, "random_state": 42}
        defaults.update(model_params)
        return RandomForestRegressor(**defaults)
    raise ValueError(f"Model '{model_name}' is not registered for task type 'regression'.")
