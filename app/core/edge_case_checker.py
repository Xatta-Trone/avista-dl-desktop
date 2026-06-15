"""General tabular dataset edge-case checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.error_handler import ERROR, FATAL, WARNING, EdgeCaseReport
from app.core.splitter import (
    CLASS_COVERAGE_FIX,
    build_class_coverage_report,
    class_coverage_issues,
    split_data_three_way,
)


HIGH_CARDINALITY_THRESHOLD = 100
RARE_LEVEL_PERCENT_THRESHOLD = 1.0
SEVERE_IMBALANCE_RATIO = 0.05
MISSING_TEXT_VALUES = {"", "na", "n/a", "null", "none"}
SAVED_SPLIT_ARRAYS = {
    "balanced training": ("X_train_balanced.npy", "y_train_balanced.npy"),
    "validation": ("X_val.npy", "y_val.npy"),
    "test": ("X_test.npy", "y_test.npy"),
}


def run_edge_case_checks(
    df: pd.DataFrame,
    config: Any,
    environment_info: dict[str, Any] | None = None,
) -> EdgeCaseReport:
    """Run general safety checks before model training."""

    report = EdgeCaseReport()
    environment_info = environment_info or {}

    _check_dataset(df, report)
    _check_target(df, config, report)
    _check_features(df, config, report)
    _check_split(df, config, report)
    _check_split_class_coverage(df, config, report)
    _check_imbalance(df, config, report)
    _check_model_environment(df, config, environment_info, report)

    return report


def run_saved_edge_case_checks(
    df: pd.DataFrame,
    config: Any,
    environment_info: dict[str, Any] | None = None,
) -> EdgeCaseReport:
    """Validate confirmed modeling columns and saved split/balancing artifacts."""

    report = EdgeCaseReport()
    project_dir = Path(getattr(config, "project_dir", ""))
    output_dir = project_dir / "outputs" / "data_split"
    feature_columns = list(getattr(config, "feature_columns", []) or [])
    target_column = getattr(config, "target_column", None)
    imbalance_method = getattr(config, "imbalance_method", None) or "none"
    modeling_subset_path = project_dir / "data" / "modeling_subset.csv"
    split_indices_path = output_dir / "split_indices.json"
    imbalance_path = output_dir / "imbalance_config.json"

    column_confirmed = _column_configuration_confirmed(
        modeling_subset_path,
        feature_columns,
        target_column,
    )
    split_confirmed = split_indices_path.exists() and imbalance_path.exists()
    report.context = {
        **(
            config.project_metadata()
            if callable(getattr(config, "project_metadata", None))
            else {}
        ),
        "target_column": target_column,
        "feature_count": len(feature_columns),
        "column_configuration_confirmed": column_confirmed,
        "split_confirmed": split_confirmed,
        "imbalance_method": imbalance_method,
    }

    if not column_confirmed:
        report.add(
            FATAL,
            "configuration",
            "Please confirm Column Configuration before running edge-case checks.",
            "Confirm the selected modeling columns and target on the Column Configuration page.",
        )
        return report

    _check_confirmed_modeling_columns(df, config, report)
    if not split_confirmed:
        report.add(
            FATAL,
            "split",
            "Please confirm Data Split & Imbalance before running edge-case checks.",
            "Confirm and save the Data Split & Imbalance configuration.",
        )
        return report

    split_metadata = _load_json_artifact(split_indices_path, "split metadata", report)
    imbalance_metadata = _load_json_artifact(imbalance_path, "imbalance metadata", report)
    if split_metadata is not None:
        _check_saved_target(split_metadata, target_column, "split metadata", report)
    if imbalance_metadata is not None:
        _check_saved_target(imbalance_metadata, target_column, "imbalance metadata", report)

    arrays = _load_split_arrays(output_dir, report)
    if arrays:
        _check_saved_class_coverage(df[target_column], arrays, report)
        _check_balanced_distribution(output_dir, arrays["balanced training"][1], report)
    _check_coverage_report(output_dir, report)
    _check_model_environment(
        df[feature_columns + [target_column]],
        config,
        environment_info or {},
        report,
    )
    return report


def _column_configuration_confirmed(
    modeling_subset_path: Path,
    feature_columns: list[str],
    target_column: str | None,
) -> bool:
    if not feature_columns or not target_column or not modeling_subset_path.exists():
        return False
    try:
        saved_columns = list(pd.read_csv(modeling_subset_path, nrows=0).columns)
    except (OSError, ValueError, pd.errors.ParserError):
        return False
    return saved_columns == feature_columns + [target_column]


def _check_confirmed_modeling_columns(
    df: pd.DataFrame,
    config: Any,
    report: EdgeCaseReport,
) -> None:
    feature_columns = list(getattr(config, "feature_columns", []) or [])
    target_column = getattr(config, "target_column", None)
    if target_column in feature_columns:
        report.add(
            ERROR,
            "features",
            f"Target column '{target_column}' is included in feature_columns.",
            "Remove the target from the selected feature columns and confirm Column Configuration again.",
        )
    missing_columns = [
        column for column in feature_columns + [target_column] if column not in df.columns
    ]
    if missing_columns:
        report.add(
            FATAL,
            "configuration",
            f"Confirmed modeling columns are missing from the dataset: {missing_columns}.",
            "Reload the matching dataset or confirm Column Configuration again.",
        )
        return
    for column in feature_columns:
        _add_missing_column_issue(df[column], column, "features", report)
    _add_missing_column_issue(df[target_column], target_column, "target", report)


def _load_json_artifact(
    path: Path,
    label: str,
    report: EdgeCaseReport,
) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        report.add(
            FATAL,
            "artifacts",
            f"Could not read saved {label}: {exc}",
            "Confirm Data Split & Imbalance again to recreate the saved artifacts.",
        )
        return None


def _check_saved_target(
    metadata: dict[str, Any],
    target_column: str,
    label: str,
    report: EdgeCaseReport,
) -> None:
    saved_target = metadata.get("target_column")
    if saved_target != target_column:
        report.add(
            FATAL,
            "artifacts",
            f"Saved {label} target '{saved_target}' does not match current target '{target_column}'.",
            "Confirm Data Split & Imbalance again for the current target column.",
        )


def _load_split_arrays(
    output_dir: Path,
    report: EdgeCaseReport,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    arrays = {}
    for split_name, (x_name, y_name) in SAVED_SPLIT_ARRAYS.items():
        x_path = output_dir / x_name
        y_path = output_dir / y_name
        missing = [path.name for path in (x_path, y_path) if not path.exists()]
        if missing:
            report.add(
                FATAL,
                "artifacts",
                f"Saved {split_name} artifact(s) are missing: {missing}.",
                "Confirm Data Split & Imbalance again to recreate all saved arrays.",
            )
            continue
        try:
            X = np.load(x_path, allow_pickle=False)
            y = np.load(y_path, allow_pickle=True)
        except (OSError, ValueError) as exc:
            report.add(
                FATAL,
                "artifacts",
                f"Could not load saved {split_name} arrays: {exc}",
                "Confirm Data Split & Imbalance again to recreate valid arrays.",
            )
            continue
        if len(X) != len(y):
            report.add(
                FATAL,
                "artifacts",
                f"Saved {split_name} row counts do not match: X has {len(X)} rows and y has {len(y)} rows.",
                "Confirm Data Split & Imbalance again to recreate aligned arrays.",
            )
        _check_finite_array(X, f"X_{split_name}", report, numeric_required=True)
        _check_finite_array(y, f"y_{split_name}", report, numeric_required=False)
        arrays[split_name] = (X, y)
    return arrays


def _check_finite_array(
    values: np.ndarray,
    label: str,
    report: EdgeCaseReport,
    *,
    numeric_required: bool,
) -> None:
    if not numeric_required:
        series = pd.Series(values.reshape(-1))
        if missing_value_mask(series).any():
            report.add(
                FATAL,
                "artifacts",
                f"Saved array '{label}' contains empty or missing target values.",
                "Clean the target column and confirm Data Split & Imbalance again.",
            )
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_values = numeric[numeric.notna()]
        if not numeric_values.empty and not np.isfinite(numeric_values.to_numpy()).all():
            report.add(
                FATAL,
                "artifacts",
                f"Saved array '{label}' contains NaN or infinite values.",
                "Clean the target column and confirm Data Split & Imbalance again.",
            )
        return
    try:
        finite = np.isfinite(values.astype(float))
    except (TypeError, ValueError):
        report.add(
            FATAL,
            "artifacts",
            f"Saved array '{label}' contains non-numeric values that cannot be validated.",
            "Confirm Data Split & Imbalance again to recreate numeric modeling arrays.",
        )
        return
    if not finite.all():
        report.add(
            FATAL,
            "artifacts",
            f"Saved array '{label}' contains NaN or infinite values.",
            "Clean the selected modeling columns and confirm Data Split & Imbalance again.",
        )


def _check_saved_class_coverage(
    full_target: pd.Series,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    report: EdgeCaseReport,
) -> None:
    required = {"balanced training", "validation", "test"}
    if not required.issubset(arrays):
        return
    coverage = build_class_coverage_report(
        full_target,
        pd.Series(arrays["balanced training"][1]),
        pd.Series(arrays["validation"][1]),
        pd.Series(arrays["test"][1]),
    )
    for issue in class_coverage_issues(coverage):
        report.add(issue["level"], "split", issue["message"], issue["suggestion"])


def _check_coverage_report(output_dir: Path, report: EdgeCaseReport) -> None:
    path = output_dir / "class_coverage_report.csv"
    if not path.exists():
        report.add(
            FATAL,
            "artifacts",
            "Saved class coverage report is missing.",
            "Confirm Data Split & Imbalance again to create class_coverage_report.csv.",
        )
        return
    try:
        coverage = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        report.add(
            FATAL,
            "artifacts",
            f"Could not read saved class coverage report: {exc}",
            "Confirm Data Split & Imbalance again to recreate the coverage report.",
        )
        return
    required_columns = {
        "Class",
        "Full count",
        "Train count",
        "Validation count",
        "Test count",
        "Status",
    }
    if not required_columns.issubset(coverage.columns):
        report.add(
            FATAL,
            "artifacts",
            "Saved class coverage report has an invalid format.",
            "Confirm Data Split & Imbalance again to recreate the coverage report.",
        )


def _check_balanced_distribution(
    output_dir: Path,
    y_train_balanced: np.ndarray,
    report: EdgeCaseReport,
) -> None:
    path = output_dir / "class_distribution_after.csv"
    if not path.exists():
        return
    try:
        distribution = pd.read_csv(path)
        saved = distribution[distribution["split"] == "Train Set (Balanced)"]
        saved_counts = {
            str(row["class"]): int(row["count"]) for _, row in saved.iterrows()
        }
    except (OSError, KeyError, ValueError, pd.errors.ParserError) as exc:
        report.add(
            ERROR,
            "artifacts",
            f"Could not validate saved balanced class distribution: {exc}",
            "Confirm Data Split & Imbalance again to recreate distribution artifacts.",
        )
        return
    actual_counts = {
        str(key.item() if hasattr(key, "item") else key): int(value)
        for key, value in pd.Series(y_train_balanced).value_counts().items()
    }
    if saved_counts != actual_counts:
        report.add(
            FATAL,
            "artifacts",
            "Balanced training distribution does not match class_distribution_after.csv.",
            "Confirm Data Split & Imbalance again to recreate consistent artifacts.",
        )


def _check_dataset(df: pd.DataFrame, report: EdgeCaseReport) -> None:
    if df.empty:
        report.add(FATAL, "dataset", "The dataset is empty.", "Load a dataset with at least one row and one column.")
        return

    duplicate_column_names = _duplicate_names(df.columns)
    if duplicate_column_names:
        report.add(
            ERROR,
            "dataset",
            f"Duplicate column names found: {duplicate_column_names}.",
            "Rename duplicate columns before training.",
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        report.add(
            WARNING,
            "dataset",
            f"The dataset contains {duplicate_rows} duplicate row(s).",
            "Review duplicates and remove them if they are not intentional repeated observations.",
        )

    if len(df) < 30:
        report.add(
            WARNING,
            "dataset",
            f"The dataset has fewer than 30 rows ({len(df)} rows).",
            "Use more rows when possible; small samples can produce unstable models and explanations.",
        )

    if len(df.columns) > 5000:
        report.add(
            ERROR,
            "dataset",
            f"The dataset has more than 5000 columns ({len(df.columns)} columns).",
            "Reduce dimensionality or select a smaller feature set before training.",
        )

    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
    if constant_columns:
        report.add(
            WARNING,
            "dataset",
            f"Constant columns found: {constant_columns}.",
            "Exclude constant columns because they do not add predictive signal.",
        )

    numeric_df = df.select_dtypes(include=["number"])
    infinite_columns = [
        column for column in numeric_df.columns if np.isinf(numeric_df[column].to_numpy(dtype=float)).any()
    ]
    if infinite_columns:
        report.add(
            ERROR,
            "dataset",
            f"Infinite numeric values found in columns: {infinite_columns}.",
            "Replace infinite values with missing values or finite numeric values before training.",
        )


def _check_target(df: pd.DataFrame, config: Any, report: EdgeCaseReport) -> None:
    target_column = getattr(config, "target_column", None)
    task_type = _normalized(getattr(config, "task_type", None))

    if not target_column:
        report.add(ERROR, "target", "No target column selected.", "Select a target column before training.")
        return

    if target_column not in df.columns:
        report.add(
            ERROR,
            "target",
            f"Target column '{target_column}' is missing from the dataset.",
            "Select an existing target column.",
        )
        return

    target = df[target_column]
    _add_missing_column_issue(target, target_column, "target", report)

    if task_type == "classification":
        class_counts = target.dropna().value_counts()
        if len(class_counts) <= 1:
            report.add(
                ERROR,
                "target",
                "Classification target has only one class.",
                "Choose a target with at least two classes.",
            )
        elif class_counts.min() / class_counts.sum() < SEVERE_IMBALANCE_RATIO:
            report.add(
                WARNING,
                "target",
                "Classification target has severe class imbalance.",
                "Consider class weighting, resampling, or collecting more minority-class examples.",
            )

    if task_type == "regression" and not pd.api.types.is_numeric_dtype(target):
        report.add(
            ERROR,
            "target",
            f"Regression target '{target_column}' is not numeric.",
            "Choose a numeric target column or change the task type.",
        )


def _check_features(df: pd.DataFrame, config: Any, report: EdgeCaseReport) -> None:
    feature_columns = list(getattr(config, "feature_columns", []) or [])
    id_columns = set(getattr(config, "id_columns", []) or [])

    if not feature_columns:
        report.add(ERROR, "features", "No feature columns selected.", "Select at least one feature column.")
        return

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        report.add(
            ERROR,
            "features",
            f"Selected feature columns are missing: {missing_features}.",
            "Remove missing features from the project configuration.",
        )

    existing_features = [column for column in feature_columns if column in df.columns]
    for column in existing_features:
        _add_missing_column_issue(df[column], column, "features", report)

    id_like_features = [column for column in existing_features if column in id_columns or _looks_id_like(column, df[column])]
    if id_like_features:
        report.add(
            WARNING,
            "features",
            f"ID-like columns are selected as features: {id_like_features}.",
            "Exclude identifiers unless they are intentionally predictive and valid at inference time.",
        )

    categorical_features = _categorical_features(df, existing_features)
    high_cardinality = [
        column for column in categorical_features if df[column].nunique(dropna=True) > HIGH_CARDINALITY_THRESHOLD
    ]
    if high_cardinality:
        report.add(
            WARNING,
            "features",
            f"High-cardinality categorical features found: {high_cardinality}.",
            "Consider grouping rare values, target encoding, hashing, or excluding these columns.",
        )

    numeric_looking = [
        column
        for column in categorical_features
        if _is_numeric_looking_text(df[column])
    ]
    if numeric_looking:
        report.add(
            WARNING,
            "features",
            f"Numeric-looking columns are stored as text: {numeric_looking}.",
            "Convert these columns to numeric values during preprocessing.",
        )

    rare_level_columns = [
        column for column in categorical_features if _has_rare_levels(df[column])
    ]
    if rare_level_columns:
        report.add(
            WARNING,
            "features",
            f"Categorical columns with rare levels found: {rare_level_columns}.",
            "Consider grouping rare categories before training.",
        )


def _check_split(df: pd.DataFrame, config: Any, report: EdgeCaseReport) -> None:
    split_method = _normalized(getattr(config, "split_method", None))
    group_column = getattr(config, "group_column", None)
    date_column = getattr(config, "date_column", None)
    target_column = getattr(config, "target_column", None)
    task_type = _normalized(getattr(config, "task_type", None))

    if "group" in split_method and (not group_column or group_column not in df.columns):
        report.add(
            ERROR,
            "split",
            "Group split selected but the group column is missing.",
            "Select an existing group column or choose a different split method.",
        )

    if "time" in split_method:
        if not date_column or date_column not in df.columns:
            report.add(
                ERROR,
                "split",
                "Time split selected but the date column is missing.",
                "Select an existing date column or choose a different split method.",
            )
        else:
            invalid_dates = int(pd.to_datetime(df[date_column], errors="coerce").isna().sum())
            if invalid_dates:
                report.add(
                    ERROR,
                    "split",
                    f"Date column '{date_column}' has {invalid_dates} invalid date value(s).",
                    "Fix invalid dates before using a time-based split.",
                )

    if "strat" in split_method and task_type == "classification" and target_column in df.columns:
        class_counts = df[target_column][~missing_value_mask(df[target_column])].value_counts()
        for class_name, count in class_counts[class_counts < 3].items():
            report.add(
                ERROR,
                "split",
                f"Class '{class_name}' has only {int(count)} samples. "
                "It may not appear in all train/validation/test subsets.",
                CLASS_COVERAGE_FIX,
            )


def _check_split_class_coverage(
    df: pd.DataFrame,
    config: Any,
    report: EdgeCaseReport,
) -> None:
    if _normalized(getattr(config, "task_type", None)) != "classification":
        return
    target_column = getattr(config, "target_column", None)
    feature_columns = list(getattr(config, "feature_columns", []) or [])
    if (
        not target_column
        or target_column not in df.columns
        or not feature_columns
        or any(column not in df.columns for column in feature_columns)
        or any(missing_value_mask(df[column]).any() for column in feature_columns + [target_column])
    ):
        return

    try:
        X = df[feature_columns]
        y = df[target_column]
        split = split_data_three_way(X, y, df, config)
    except ValueError:
        return

    coverage = build_class_coverage_report(
        y,
        split["y_train"],
        split["y_val"],
        split["y_test"],
    )
    for issue in class_coverage_issues(coverage):
        report.add(issue["level"], "split", issue["message"], issue["suggestion"])


def _check_imbalance(df: pd.DataFrame, config: Any, report: EdgeCaseReport) -> None:
    imbalance_method = _normalized(getattr(config, "imbalance_method", None))
    target_column = getattr(config, "target_column", None)
    feature_columns = [column for column in (getattr(config, "feature_columns", []) or []) if column in df.columns]

    if "smote" not in imbalance_method:
        return

    if target_column in df.columns:
        class_counts = df[target_column].dropna().value_counts()
        if not class_counts.empty and class_counts.min() < 2:
            report.add(
                ERROR,
                "imbalance",
                "SMOTE selected but the minority class has fewer than 2 samples.",
                "Use class weighting, random oversampling, or add more minority-class samples.",
            )

    categorical_features = _categorical_features(df, feature_columns)
    if imbalance_method == "smote" and categorical_features:
        report.add(
            WARNING,
            "imbalance",
            f"SMOTE selected while categorical features exist: {categorical_features}.",
            "Use SMOTE-NC or encode categorical features carefully before oversampling.",
        )

    if "smote_nc" in imbalance_method or "smote-nc" in imbalance_method:
        if not categorical_features:
            report.add(
                WARNING,
                "imbalance",
                "SMOTE-NC selected but no categorical features exist.",
                "Use regular SMOTE for all-numeric features.",
            )


def _check_model_environment(
    df: pd.DataFrame,
    config: Any,
    environment_info: dict[str, Any],
    report: EdgeCaseReport,
) -> None:
    selected_models = [_normalized(model) for model in (getattr(config, "selected_models", []) or [])]
    environment_mode = _normalized(getattr(config, "environment_mode", None))

    deep_model_selected = any(_is_deep_model(model) for model in selected_models)
    if deep_model_selected and importlib.util.find_spec("torch") is None:
        report.add(
            ERROR,
            "environment",
            "A deep model is selected but PyTorch is unavailable.",
            "Install the deep learning requirements or remove deep models from the selection.",
        )

    if "gpu" in environment_mode and environment_info.get("cuda_available") is False:
        report.add(
            ERROR,
            "environment",
            "GPU mode is selected but CUDA is unavailable.",
            "Use CPU mode or install a CUDA-compatible PyTorch environment.",
        )

    if any("tabpfn" in model for model in selected_models) and len(df) > 3000:
        report.add(
            WARNING,
            "model",
            f"TabPFN is selected with more than 3000 rows ({len(df)} rows).",
            "Subsample the data or choose another model for larger datasets.",
        )


def _duplicate_names(columns: pd.Index) -> list[str]:
    seen = set()
    duplicates = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return duplicates


def _normalized(value: Any) -> str:
    return str(value or "").strip().lower()


def _looks_id_like(column: str, series: pd.Series) -> bool:
    normalized = column.lower()
    if normalized in {"id", "uuid", "guid"} or normalized.endswith("_id") or normalized.endswith("id"):
        return True
    non_missing = series.dropna()
    return len(non_missing) > 0 and non_missing.nunique(dropna=True) == len(non_missing) and len(non_missing) >= 30


def _categorical_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(df[column])
    ]


def _is_numeric_looking_text(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    parsed = pd.to_numeric(non_missing, errors="coerce")
    return bool(parsed.notna().mean() >= 0.9)


def _has_rare_levels(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty:
        return False
    level_percentages = non_missing.value_counts(normalize=True) * 100
    return bool((level_percentages < RARE_LEVEL_PERCENT_THRESHOLD).any())


def _is_deep_model(model: str) -> bool:
    deep_terms = ("deep", "neural", "mlp", "tabnet", "torch", "transformer", "cnn", "rnn")
    return any(term in model for term in deep_terms)


def missing_value_mask(series: pd.Series) -> pd.Series:
    """Return values treated as empty for confirmed modeling columns."""

    missing = series.isna()
    text = series.astype("string").str.strip().str.casefold()
    return missing | text.isin(MISSING_TEXT_VALUES).fillna(False)


def selected_column_missing_counts(
    df: pd.DataFrame,
    config: Any,
) -> dict[str, tuple[int, float]]:
    """Return missing counts and percentages for selected features and target only."""

    columns = list(getattr(config, "feature_columns", []) or [])
    target_column = getattr(config, "target_column", None)
    if target_column:
        columns.append(target_column)
    results = {}
    for column in dict.fromkeys(columns):
        if column not in df.columns:
            continue
        count = int(missing_value_mask(df[column]).sum())
        if count:
            results[column] = (count, count / len(df) * 100 if len(df) else 0.0)
    return results


def _add_missing_column_issue(
    series: pd.Series,
    column: str,
    category: str,
    report: EdgeCaseReport,
) -> None:
    missing_count = int(missing_value_mask(series).sum())
    if not missing_count:
        return
    percentage = missing_count / len(series) * 100 if len(series) else 0.0
    report.add(
        ERROR,
        category,
        f"Column '{column}' contains {missing_count} empty values ({percentage:.1f}%). "
        "Please clean this column before training.",
        "Fill or correct every empty value in this selected modeling column. "
        "Rows will not be imputed or dropped automatically.",
    )
