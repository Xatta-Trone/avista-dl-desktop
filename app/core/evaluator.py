"""Model evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    root_mean_squared_error = None


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    task_type: str = "classification",
    class_labels: Any = None,
) -> dict[str, Any]:
    """Evaluate predictions and return JSON-serializable metrics."""

    normalized_task = str(task_type or "classification").strip().lower()
    if normalized_task == "classification":
        return _evaluate_classification(y_true, y_pred, y_proba, class_labels)
    if normalized_task == "regression":
        return _evaluate_regression(y_true, y_pred)
    raise ValueError(f"Unsupported task_type '{task_type}'.")


def _evaluate_classification(
    y_true: Any,
    y_pred: Any,
    y_proba: Any = None,
    class_labels: Any = None,
) -> dict[str, Any]:
    labels = list(class_labels) if class_labels is not None else None
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    roc_auc = _safe_roc_auc(y_true, y_proba)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc

    return _to_jsonable(metrics)


def _evaluate_regression(y_true: Any, y_pred: Any) -> dict[str, Any]:
    return _to_jsonable(
        {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": _rmse(y_true, y_pred),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        }
    )


def _safe_roc_auc(y_true: Any, y_proba: Any) -> float | None:
    if y_proba is None:
        return None

    try:
        probabilities = np.asarray(y_proba)
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return None
        if probabilities.ndim == 1:
            return float(roc_auc_score(y_true, probabilities))
        if probabilities.shape[1] == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def _rmse(y_true: Any, y_pred: Any) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y_true, y_pred))
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
