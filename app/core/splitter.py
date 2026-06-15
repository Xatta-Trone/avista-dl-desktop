"""Train/test splitting utilities."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split


DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
CLASS_COVERAGE_FIX = (
    "Increase dataset size for this class. Reduce validation/test percentages. "
    "Use group-aware/manual split if needed. Merge rare classes if scientifically valid."
)


def split_data(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Split preprocessed data according to the project configuration."""

    split_method = _normalized(getattr(config, "split_method", None)) or "random"
    test_size = _get_option(config, "test_size", DEFAULT_TEST_SIZE)
    random_seed = _get_option(config, "random_seed", DEFAULT_RANDOM_SEED)

    if split_method == "random":
        train_index, test_index = _random_indices(X, y, test_size, random_seed, stratify=False)
    elif split_method == "stratified":
        train_index, test_index = _random_indices(X, y, test_size, random_seed, stratify=True)
    elif split_method == "group":
        train_index, test_index = _group_indices(X, df, config, test_size, random_seed)
    elif split_method == "stratified_group":
        train_index, test_index = _stratified_group_indices(X, y, df, config, test_size, random_seed)
    elif split_method == "time":
        train_index, test_index = _time_indices(X, df, config, test_size)
    else:
        raise ValueError(f"Unsupported split_method '{split_method}'.")

    return {
        "X_train": X.loc[train_index],
        "X_test": X.loc[test_index],
        "y_train": y.loc[train_index],
        "y_test": y.loc[test_index],
        "train_index": list(train_index),
        "test_index": list(test_index),
        "split_info": {
            "split_method": split_method,
            "test_size": test_size,
            "random_seed": random_seed,
            "train_rows": len(train_index),
            "test_rows": len(test_index),
        },
    }


def split_data_three_way(X: pd.DataFrame, y: pd.Series, df: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Split data into train, validation, and test partitions."""

    train_percent = float(getattr(config, "train_percent", 70.0))
    validation_percent = float(getattr(config, "validation_percent", 10.0))
    test_percent = float(getattr(config, "test_percent", 20.0))
    total = train_percent + validation_percent + test_percent
    if any(value < 0 for value in (train_percent, validation_percent, test_percent)):
        raise ValueError("Split percentages cannot be negative.")
    if abs(total - 100.0) > 1e-6:
        raise ValueError("Train, validation, and test percentages must total 100.")
    if train_percent <= 0 or test_percent <= 0:
        raise ValueError("Train and test percentages must be greater than zero.")

    split_method = _normalized(getattr(config, "split_method", None)) or "random"
    random_seed = _get_option(config, "random_seed", DEFAULT_RANDOM_SEED)
    test_size = test_percent / 100.0
    remaining_index, test_index = _split_indices(
        X, y, df, config, split_method, test_size, random_seed
    )

    if validation_percent > 0:
        validation_fraction = validation_percent / (train_percent + validation_percent)
        remaining_X = X.loc[remaining_index]
        remaining_y = y.loc[remaining_index]
        train_index, validation_index = _split_indices(
            remaining_X,
            remaining_y,
            df,
            config,
            split_method,
            validation_fraction,
            random_seed + 1,
        )
    else:
        train_index = remaining_index
        validation_index = pd.Index([])

    return {
        "X_train": X.loc[train_index],
        "X_val": X.loc[validation_index],
        "X_test": X.loc[test_index],
        "y_train": y.loc[train_index],
        "y_val": y.loc[validation_index],
        "y_test": y.loc[test_index],
        "train_index": list(train_index),
        "validation_index": list(validation_index),
        "test_index": list(test_index),
        "split_info": {
            "split_method": split_method,
            "train_percent": train_percent,
            "validation_percent": validation_percent,
            "test_percent": test_percent,
            "random_seed": random_seed,
            "train_rows": len(train_index),
            "validation_rows": len(validation_index),
            "test_rows": len(test_index),
        },
    }


def build_class_coverage_report(
    full_y: pd.Series,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Summarize class presence across a three-way classification split."""

    full_counts = pd.Series(full_y).dropna().value_counts()
    train_counts = pd.Series(y_train).dropna().value_counts()
    validation_counts = pd.Series(y_validation).dropna().value_counts()
    test_counts = pd.Series(y_test).dropna().value_counts()
    classes = sorted(
        set(full_counts.index)
        | set(train_counts.index)
        | set(validation_counts.index)
        | set(test_counts.index),
        key=str,
    )

    rows = []
    for class_name in classes:
        train_count = int(train_counts.get(class_name, 0))
        validation_count = int(validation_counts.get(class_name, 0))
        test_count = int(test_counts.get(class_name, 0))
        statuses = []
        if train_count == 0:
            statuses.append("Missing in train - blocking")
        if train_count > 0 and validation_count == 0:
            statuses.append("Missing in validation - warning")
        if train_count > 0 and test_count == 0:
            statuses.append("Missing in test - warning")
        rows.append(
            {
                "Class": _json_scalar(class_name),
                "Full count": int(full_counts.get(class_name, 0)),
                "Train count": train_count,
                "Validation count": validation_count,
                "Test count": test_count,
                "Status": "; ".join(statuses) if statuses else "OK",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Class",
            "Full count",
            "Train count",
            "Validation count",
            "Test count",
            "Status",
        ],
    )


def class_coverage_issues(coverage: pd.DataFrame) -> list[dict[str, str]]:
    """Return blocking and warning messages represented by a coverage report."""

    issues = []
    for _, row in coverage.iterrows():
        class_name = row["Class"]
        train_count = int(row["Train count"])
        validation_count = int(row["Validation count"])
        test_count = int(row["Test count"])
        if train_count == 0:
            if validation_count > 0 or test_count > 0:
                message = (
                    f"Class '{class_name}' appears in validation/test but not training. "
                    "The model cannot predict this class."
                )
            else:
                message = (
                    f"Class '{class_name}' is absent from the training set. "
                    "The model cannot learn this class."
                )
            issues.append(
                {
                    "level": "fatal",
                    "message": message,
                    "suggestion": CLASS_COVERAGE_FIX,
                }
            )
            continue
        if validation_count == 0:
            issues.append(
                {
                    "level": "warning",
                    "message": (
                        f"Class '{class_name}' is absent from the validation set. "
                        "Validation metrics for this class may be unavailable."
                    ),
                    "suggestion": CLASS_COVERAGE_FIX,
                }
            )
        if test_count == 0:
            issues.append(
                {
                    "level": "warning",
                    "message": (
                        f"Class '{class_name}' is absent from the test set. "
                        "Test metrics for this class may be unavailable."
                    ),
                    "suggestion": CLASS_COVERAGE_FIX,
                }
            )
    return issues


def _split_indices(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    config: Any,
    split_method: str,
    holdout_size: float,
    random_seed: int,
) -> tuple[pd.Index, pd.Index]:
    if split_method == "random":
        return _random_indices(X, y, holdout_size, random_seed, stratify=False)
    if split_method == "stratified":
        return _random_indices(X, y, holdout_size, random_seed, stratify=True)
    if split_method == "group":
        return _group_indices(X, df, config, holdout_size, random_seed)
    if split_method == "stratified_group":
        return _stratified_group_indices(X, y, df, config, holdout_size, random_seed)
    if split_method == "time":
        return _time_indices(X, df, config, holdout_size)
    raise ValueError(f"Unsupported split_method '{split_method}'.")


def _random_indices(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_seed: int,
    stratify: bool,
) -> tuple[pd.Index, pd.Index]:
    stratify_values = y if stratify else None
    try:
        train_index, test_index = train_test_split(
            X.index,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify_values,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to create {'stratified' if stratify else 'random'} split: {exc}") from exc
    return pd.Index(train_index), pd.Index(test_index)


def _group_indices(
    X: pd.DataFrame,
    df: pd.DataFrame,
    config: Any,
    test_size: float,
    random_seed: int,
) -> tuple[pd.Index, pd.Index]:
    group_column = _required_column(df, getattr(config, "group_column", None), "group split", "group_column")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_pos, test_pos = next(splitter.split(X, groups=df.loc[X.index, group_column]))
    return X.index[train_pos], X.index[test_pos]


def _stratified_group_indices(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    config: Any,
    test_size: float,
    random_seed: int,
) -> tuple[pd.Index, pd.Index]:
    group_column = _required_column(df, getattr(config, "group_column", None), "stratified group split", "group_column")
    n_splits = max(2, round(1 / test_size))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    try:
        train_pos, test_pos = next(splitter.split(X, y, groups=df.loc[X.index, group_column]))
    except ValueError as exc:
        raise ValueError(f"Unable to create stratified group split: {exc}") from exc
    return X.index[train_pos], X.index[test_pos]


def _time_indices(
    X: pd.DataFrame,
    df: pd.DataFrame,
    config: Any,
    test_size: float,
) -> tuple[pd.Index, pd.Index]:
    date_column = _required_column(df, getattr(config, "date_column", None), "time split", "date_column")
    dates = pd.to_datetime(df.loc[X.index, date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Date column '{date_column}' contains invalid date values.")

    sorted_index = dates.sort_values().index
    test_count = max(1, int(round(len(sorted_index) * test_size)))
    if test_count >= len(sorted_index):
        raise ValueError("Time split test size leaves no training rows.")

    return sorted_index[:-test_count], sorted_index[-test_count:]


def _required_column(df: pd.DataFrame, column: str | None, split_name: str, config_name: str) -> str:
    if not column:
        raise ValueError(f"{split_name} requires config.{config_name}.")
    if column not in df.columns:
        raise ValueError(f"{split_name} requires existing column '{column}'.")
    return column


def _get_option(config: Any, key: str, default: Any) -> Any:
    options = getattr(config, "preprocessing_options", {}) or {}
    split_options = options.get("split", {}) if isinstance(options.get("split", {}), dict) else {}
    return split_options.get(key, options.get(key, default))


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
