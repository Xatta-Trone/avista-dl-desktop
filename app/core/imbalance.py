"""Class imbalance handling utilities."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


def apply_imbalance_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts: Any,
    config: Any,
) -> dict[str, Any]:
    """Apply the configured imbalance strategy to training data."""

    strategy = _normalized(getattr(config, "imbalance_method", None)) or "none"
    if strategy in {"", "none", "no", "disabled"}:
        return _result(
            X_train,
            y_train,
            "none",
            "No imbalance strategy applied.",
            None,
            before_counts=_class_counts(y_train),
            after_counts=_class_counts(y_train),
        )

    if strategy == "class_weight":
        class_weights = _compute_class_weights(y_train)
        return _result(X_train, y_train, "class_weight", "Computed class weights.", class_weights)

    if strategy in {"random_oversample", "random_undersample", "smote", "smote_nc", "smote-nc"}:
        if importlib.util.find_spec("imblearn") is None:
            return _result(
                X_train,
                y_train,
                strategy,
                "imbalanced-learn is not installed; data was not resampled.",
                None,
                success=False,
                error="Install imbalanced-learn to use resampling strategies.",
            )
        return _apply_imblearn_strategy(X_train, y_train, artifacts, config, strategy)

    raise ValueError(f"Unsupported imbalance_method '{strategy}'.")


def _apply_imblearn_strategy(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts: Any,
    config: Any,
    strategy: str,
) -> dict[str, Any]:
    random_seed = _get_option(config, "random_seed", 42)
    warnings: list[str] = []
    sampling_strategy = _training_only_sampling_strategy(
        y_train,
        _get_option(config, "sampling_strategy", "auto"),
    )

    try:
        if isinstance(sampling_strategy, dict) and not sampling_strategy:
            return _result(
                X_train,
                y_train,
                strategy,
                "No training classes require resampling.",
                None,
                before_counts=_class_counts(y_train),
                after_counts=_class_counts(y_train),
                warnings=warnings,
            )
        if strategy == "random_oversample":
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(
                random_state=random_seed,
                sampling_strategy=sampling_strategy,
            )
        elif strategy == "random_undersample":
            from imblearn.under_sampling import RandomUnderSampler

            sampler = RandomUnderSampler(
                random_state=random_seed,
                sampling_strategy=sampling_strategy,
            )
        elif strategy == "smote":
            from imblearn.over_sampling import SMOTE

            if getattr(artifacts, "categorical_columns", []):
                warnings.append(
                    "Categorical features are present. SMOTE-NC is recommended instead of regular SMOTE."
                )
            k_neighbors = _validated_k_neighbors(y_train, config, sampling_strategy)
            sampler = SMOTE(
                random_state=random_seed,
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy,
            )
        elif strategy in {"smote_nc", "smote-nc"}:
            from imblearn.over_sampling import SMOTENC

            categorical_indices = _categorical_output_indices(artifacts)
            if not categorical_indices:
                return _result(
                    X_train,
                    y_train,
                    strategy,
                    "SMOTE-NC requires categorical features; data was not resampled.",
                    None,
                    success=False,
                    error="No categorical feature indices are available for SMOTE-NC.",
                )
            k_neighbors = _validated_k_neighbors(y_train, config, sampling_strategy)
            sampler = SMOTENC(
                categorical_features=categorical_indices,
                random_state=random_seed,
                k_neighbors=k_neighbors,
                sampling_strategy=sampling_strategy,
            )
        else:
            raise ValueError(f"Unsupported imbalance_method '{strategy}'.")

        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        return _result(
            _as_dataframe(X_resampled, X_train.columns),
            _as_series(y_resampled, y_train.name),
            strategy,
            "Resampling completed.",
            None,
            before_counts=_class_counts(y_train),
            after_counts=_class_counts(pd.Series(y_resampled)),
            warnings=warnings,
        )
    except (ValueError, RuntimeError) as exc:
        return _result(
            X_train,
            y_train,
            strategy,
            "Resampling failed validation; original data was returned.",
            None,
            success=False,
            error=str(exc),
            before_counts=_class_counts(y_train),
            after_counts=_class_counts(y_train),
            warnings=warnings,
        )


def _validated_k_neighbors(
    y_train: pd.Series,
    config: Any,
    sampling_strategy: Any = "auto",
) -> int:
    counts = y_train.value_counts()
    if counts.empty:
        raise ValueError("Cannot apply SMOTE because y_train is empty.")

    target_classes = list(sampling_strategy) if isinstance(sampling_strategy, dict) else list(counts.index)
    eligible_counts = [int(counts[class_name]) for class_name in target_classes if class_name in counts]
    if not eligible_counts:
        raise ValueError("SMOTE sampling strategy does not contain any classes present in y_train.")

    minority_count = min(eligible_counts)
    if minority_count < 2:
        affected = [str(class_name) for class_name in target_classes if class_name in counts and counts[class_name] < 2]
        raise ValueError(
            "SMOTE requires at least 2 samples in every class being oversampled. "
            f"Too few samples for: {affected}."
        )

    requested = int(_get_option(config, "smote_k_neighbors", 5))
    return max(1, min(requested, minority_count - 1))


def _training_only_sampling_strategy(
    y_train: pd.Series,
    sampling_strategy: Any,
) -> Any:
    if not isinstance(sampling_strategy, dict):
        return sampling_strategy

    training_classes = set(y_train.dropna().unique())
    return {
        class_name: target
        for class_name, target in sampling_strategy.items()
        if class_name in training_classes
    }


def _compute_class_weights(y_train: pd.Series) -> dict[Any, float]:
    classes = np.array(sorted(y_train.dropna().unique()))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.dropna())
    return {cls.item() if hasattr(cls, "item") else cls: float(weight) for cls, weight in zip(classes, weights)}


def _categorical_output_indices(artifacts: Any) -> list[int]:
    output_names = list(getattr(artifacts, "output_feature_names", []) or [])
    categorical_columns = list(getattr(artifacts, "categorical_columns", []) or [])
    indices = []
    for index, feature_name in enumerate(output_names):
        if any(feature_name == column or feature_name.startswith(f"{column}_") for column in categorical_columns):
            indices.append(index)
    return indices


def _result(
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series,
    strategy: str,
    message: str,
    class_weights: dict[Any, float] | None,
    success: bool = True,
    error: str | None = None,
    before_counts: dict[Any, int] | None = None,
    after_counts: dict[Any, int] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "X_resampled": X_resampled,
        "y_resampled": y_resampled,
        "imbalance_info": {
            "strategy": strategy,
            "success": success,
            "message": message,
            "error": error,
            "before_counts": before_counts,
            "after_counts": after_counts,
            "warnings": warnings or [],
        },
        "class_weights": class_weights,
    }


def _as_dataframe(values: Any, columns: pd.Index) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.reset_index(drop=True)
    return pd.DataFrame(values, columns=columns)


def _as_series(values: Any, name: str | None) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, name=name)


def _class_counts(y: pd.Series) -> dict[Any, int]:
    return {
        key.item() if hasattr(key, "item") else key: int(value)
        for key, value in y.value_counts().to_dict().items()
    }


def _get_option(config: Any, key: str, default: Any) -> Any:
    options = getattr(config, "preprocessing_options", {}) or {}
    imbalance_options = options.get("imbalance", {}) if isinstance(options.get("imbalance", {}), dict) else {}
    return imbalance_options.get(key, options.get(key, default))


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()
