"""Dataset loading and summary helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".fst"}
CSV_ENCODINGS = ("utf-8", "utf-8-sig", "latin1")
NULL_MARKERS = {"", '""', "null", "nan", "n/a", "na", "none"}


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a supported tabular dataset into a pandas DataFrame."""

    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported dataset format '{extension}'. Supported formats: {supported}.")

    if not path.exists():
        raise ValueError(f"Dataset file does not exist: {path}")

    if extension == ".csv":
        return normalize_missing_values(_load_csv_with_fallbacks(path))

    try:
        if extension in {".xlsx", ".xls"}:
            return normalize_missing_values(pd.read_excel(path))
        if extension == ".parquet":
            return normalize_missing_values(pd.read_parquet(path))
        if extension == ".feather":
            return normalize_missing_values(pd.read_feather(path))
        if extension == ".fst":
            return normalize_missing_values(_load_fst(path))
    except Exception as exc:
        raise ValueError(f"Failed to load dataset '{path}': {exc}") from exc

    raise ValueError(f"Unsupported dataset format '{extension}'.")


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert common textual missing markers to pandas missing values."""

    normalized = df.copy()
    for column in normalized.select_dtypes(include=["object", "string"]).columns:
        normalized[column] = normalized[column].map(_normalize_missing_value)
    return normalized


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact summary for a pandas DataFrame."""

    missing_counts = df.isna().sum()
    missing_percent = (missing_counts / len(df) * 100) if len(df) else missing_counts.astype(float)
    total_cells = int(df.shape[0] * df.shape[1])
    total_missing_values = int(missing_counts.sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_like_columns = _detect_datetime_like_columns(df)
    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns and column not in datetime_like_columns
    ]
    simplified_column_types = {
        column: _simplified_column_type(df[column], datetime_like_columns)
        for column in df.columns
    }

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_counts": {column: int(count) for column, count in missing_counts.items()},
        "missing_percent": {column: float(percent) for column, percent in missing_percent.items()},
        "column_missing_percent": {column: float(percent) for column, percent in missing_percent.items()},
        "total_missing_values": total_missing_values,
        "total_missing_percent": float((total_missing_values / total_cells * 100) if total_cells else 0.0),
        "duplicate_rows": duplicate_rows,
        "duplicate_row_percent": float((duplicate_rows / len(df) * 100) if len(df) else 0.0),
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "datetime_like_columns": datetime_like_columns,
        "numeric_percent": float((len(numeric_columns) / len(df.columns) * 100) if len(df.columns) else 0.0),
        "categorical_percent": float((len(categorical_columns) / len(df.columns) * 100) if len(df.columns) else 0.0),
        "simplified_column_types": simplified_column_types,
    }


def _load_csv_with_fallbacks(path: Path) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None

    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise ValueError(f"Failed to read CSV '{path}' with supported encodings: {CSV_ENCODINGS}.") from last_error


def _load_fst(path: Path) -> pd.DataFrame:
    try:
        import fst
    except ImportError as exc:
        raise ValueError(
            "FST datasets require the optional 'fst' Python package."
        ) from exc
    reader = getattr(fst, "read_fst", None) or getattr(fst, "read", None)
    if not callable(reader):
        raise ValueError("The installed 'fst' package does not provide an FST reader.")
    result = reader(str(path))
    return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)


def _normalize_missing_value(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, str) and value.strip().lower() in NULL_MARKERS:
        return pd.NA
    return value


def _detect_datetime_like_columns(df: pd.DataFrame) -> list[str]:
    datetime_columns = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    for column in df.select_dtypes(include=["object", "string"]).columns:
        if column in datetime_columns:
            continue

        values = df[column].dropna()
        if values.empty:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(values, errors="coerce")
        if parsed.notna().all():
            datetime_columns.append(column)

    return datetime_columns


def _simplified_column_type(series: pd.Series, datetime_like_columns: list[str]) -> str:
    if series.name in datetime_like_columns:
        return "Date/Time"
    if pd.api.types.is_bool_dtype(series):
        return "Boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric"
    if isinstance(series.dtype, pd.CategoricalDtype):
        return "Categorical"
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        non_missing = series.dropna()
        if non_missing.empty:
            return "Unknown"
        unique_ratio = non_missing.nunique(dropna=True) / len(non_missing)
        return "Text" if unique_ratio > 0.5 and non_missing.nunique(dropna=True) > 20 else "Categorical"
    return "Unknown"
