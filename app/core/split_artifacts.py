"""Signatures and scoped invalidation for saved split workflow artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SPLIT_ARTIFACT_FILES = (
    "split_indices.json",
    "class_distribution_before.csv",
    "class_coverage_report.csv",
    "preprocessing_artifact.joblib",
    "X_train.npy",
    "y_train.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
    "y_train_encoded.npy",
    "y_val_encoded.npy",
    "y_test_encoded.npy",
    "y_train_original.npy",
    "y_val_original.npy",
    "y_test_original.npy",
    "target_label_encoder.joblib",
    "target_label_mapping.json",
)

IMBALANCE_ARTIFACT_FILES = (
    "imbalance_config.json",
    "class_distribution_after.csv",
    "X_train_balanced.npy",
    "y_train_balanced.npy",
    "y_train_balanced_encoded.npy",
    "y_train_balanced_original.npy",
)


def modeling_signature(config: Any) -> dict[str, Any]:
    """Return configuration fields that determine preprocessing artifacts."""

    options = dict(getattr(config, "preprocessing_options", {}) or {})
    return {
        "target_column": getattr(config, "target_column", None),
        "feature_columns": list(getattr(config, "feature_columns", []) or []),
        "label_encoding_columns": list(
            getattr(config, "label_encoding_columns", []) or []
        ),
        "task_type": getattr(config, "task_type", None),
        "numerical_scaling_method": options.get(
            "numerical_scaling_method",
            "none",
        ),
        "scaled_numerical_columns": list(
            options.get("scaled_numerical_columns", []) or []
        ),
        "categorical_missing_value": options.get(
            "categorical_missing_value",
            "Unknown",
        ),
        "categorical_missing_value_strategy": options.get(
            "categorical_missing_value_strategy",
            "replace_missing_and_blank_before_encoding",
        ),
    }


def split_signature(config: Any) -> dict[str, Any]:
    """Return all configuration fields that determine a three-way split."""

    return {
        "schema_version": 1,
        "modeling": modeling_signature(config),
        "split_method": getattr(config, "split_method", None) or "random",
        "train_percent": float(getattr(config, "train_percent", 70.0)),
        "validation_percent": float(
            getattr(config, "validation_percent", 10.0)
        ),
        "test_percent": float(getattr(config, "test_percent", 20.0)),
        "random_seed": int(getattr(config, "random_seed", 42)),
        "group_column": getattr(config, "group_column", None),
        "date_column": getattr(config, "date_column", None),
    }


def imbalance_signature(config: Any) -> dict[str, Any]:
    """Return configuration fields that determine training-only balancing."""

    options = dict(getattr(config, "preprocessing_options", {}) or {})
    imbalance_options = dict(options.get("imbalance", {}) or {})
    return {
        "schema_version": 1,
        "split_signature": split_signature(config),
        "imbalance_method": getattr(config, "imbalance_method", None) or "none",
        "use_class_weights": bool(
            getattr(config, "use_class_weights", False)
        ),
        "smote_ratio_preset": getattr(
            config,
            "smote_ratio_preset",
            "baseline",
        ),
        "custom_ratio": imbalance_options.get("custom_ratio", ""),
        "sampling_strategy": imbalance_options.get(
            "sampling_strategy",
            "auto",
        ),
    }


def signatures_match(saved: Any, current: Any) -> bool:
    """Compare JSON-compatible signatures with stable normalization."""

    return _normalized_json(saved) == _normalized_json(current)


def invalidate_split_artifacts(project_dir: str | Path) -> None:
    """Invalidate split and downstream balancing artifacts."""

    output_dir = Path(project_dir) / "outputs" / "data_split"
    _remove_files(
        output_dir,
        (*SPLIT_ARTIFACT_FILES, *IMBALANCE_ARTIFACT_FILES),
    )


def invalidate_imbalance_artifacts(project_dir: str | Path) -> None:
    """Invalidate balancing outputs while retaining the current split."""

    output_dir = Path(project_dir) / "outputs" / "data_split"
    _remove_files(output_dir, IMBALANCE_ARTIFACT_FILES)


def _remove_files(output_dir: Path, filenames: tuple[str, ...]) -> None:
    if not output_dir.is_dir():
        return
    for filename in filenames:
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def _normalized_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
