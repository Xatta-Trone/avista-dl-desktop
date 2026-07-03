"""Generic preprocessing for tabular AVISTA datasets."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


ScalerType = MinMaxScaler | StandardScaler | None
SCALER_CONFIG_FILENAME = "numerical_scaling_config.json"
SCALER_ARTIFACT_FILENAME = "numerical_scaler.joblib"


@dataclass
class PreprocessingArtifacts:
    """Fitted preprocessing objects needed for training and inference."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_columns: list[str]
    target_column: str | None
    task_type: str | None
    numeric_imputer: SimpleImputer | None
    categorical_imputer: SimpleImputer | None
    encoder: OneHotEncoder | None
    scaler: ScalerType
    target_encoder: LabelEncoder | None
    output_feature_names: list[str]
    numeric_scaling_method: str = "none"
    scaled_numeric_columns: list[str] = field(default_factory=list)
    preprocessing_metadata: dict[str, Any] = field(default_factory=dict)

    def save_artifacts(self, path: str | Path) -> Path:
        return save_artifacts(self, path)

    @classmethod
    def load_artifacts(cls, path: str | Path) -> "PreprocessingArtifacts":
        return load_artifacts(path)


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    config: Any,
    *,
    encode_classification_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series, PreprocessingArtifacts]:
    """Fit preprocessing on a dataset and return transformed X, y, and artifacts."""

    target_column = getattr(config, "target_column", None)
    task_type = _normalized(getattr(config, "task_type", None))
    feature_columns = _resolve_feature_columns(df, config)
    _validate_columns(df, feature_columns, target_column)
    spec = resolve_preprocessing_spec(df, config, feature_columns=feature_columns)

    numeric_imputer = SimpleImputer(
        strategy=spec["numeric_impute_strategy"],
    )
    categorical_imputer = SimpleImputer(
        strategy="constant",
        fill_value=spec["categorical_fill_value"],
    )
    encoder = _make_one_hot_encoder() if spec["categorical_columns"] else None
    scaler = (
        _make_scaler(spec["numeric_scaling_method"])
        if spec["scaled_numeric_columns"]
        else None
    )

    numeric_array = _fit_transform_numeric(
        df,
        spec["numeric_columns"],
        spec["scaled_numeric_columns"],
        numeric_imputer,
        scaler,
    )
    categorical_array = _fit_transform_categorical(
        df,
        spec["categorical_columns"],
        categorical_imputer,
        encoder,
    )

    output_feature_names = _output_feature_names(
        spec["numeric_columns"],
        spec["categorical_columns"],
        encoder,
    )
    X = pd.DataFrame(
        _combine_arrays(numeric_array, categorical_array),
        columns=output_feature_names,
        index=df.index,
    )

    if task_type == "classification" and not encode_classification_target:
        y = pd.Series(df[target_column], index=df.index, name=target_column)
        target_encoder = None
    else:
        y, target_encoder = _prepare_target(df[target_column], task_type)

    metadata = build_preprocessing_metadata(spec, feature_columns=feature_columns, target_column=target_column)
    artifacts = PreprocessingArtifacts(
        numeric_columns=spec["numeric_columns"],
        categorical_columns=spec["categorical_columns"],
        feature_columns=feature_columns,
        target_column=target_column,
        task_type=task_type,
        numeric_imputer=numeric_imputer if spec["numeric_columns"] else None,
        categorical_imputer=categorical_imputer if spec["categorical_columns"] else None,
        encoder=encoder,
        scaler=scaler,
        target_encoder=target_encoder,
        output_feature_names=output_feature_names,
        numeric_scaling_method=spec["numeric_scaling_method"],
        scaled_numeric_columns=spec["scaled_numeric_columns"],
        preprocessing_metadata=metadata,
    )
    return X, y, artifacts


def fit_split_preprocessing(
    train_df: pd.DataFrame,
    config: Any,
) -> PreprocessingArtifacts:
    """Fit preprocessing artifacts on the training subset only."""

    _, _, artifacts = build_preprocessing_pipeline(
        train_df,
        config,
        encode_classification_target=False,
    )
    artifacts.target_encoder = None
    return artifacts


def transform_split_features(
    df: pd.DataFrame,
    artifacts: PreprocessingArtifacts,
) -> pd.DataFrame:
    """Transform split-specific features using training-fitted artifacts."""

    return transform_new_data(df, artifacts)


def transform_new_data(df: pd.DataFrame, artifacts: PreprocessingArtifacts) -> pd.DataFrame:
    """Transform inference data using fitted preprocessing artifacts."""

    missing_columns = [column for column in artifacts.feature_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"New data is missing required feature columns: {missing_columns}.")

    numeric_array = _transform_numeric(
        df,
        artifacts.numeric_columns,
        artifacts.scaled_numeric_columns,
        artifacts.numeric_imputer,
        artifacts.scaler,
    )
    categorical_array = _transform_categorical(
        df,
        artifacts.categorical_columns,
        artifacts.categorical_imputer,
        artifacts.encoder,
    )
    return pd.DataFrame(
        _combine_arrays(numeric_array, categorical_array),
        columns=artifacts.output_feature_names,
        index=df.index,
    )


def save_artifacts(artifacts: PreprocessingArtifacts, path: str | Path) -> Path:
    """Persist fitted preprocessing artifacts with joblib."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)
    return output_path


def load_artifacts(path: str | Path) -> PreprocessingArtifacts:
    """Load fitted preprocessing artifacts from joblib."""

    artifacts = joblib.load(Path(path))
    if not isinstance(artifacts, PreprocessingArtifacts):
        raise ValueError("Loaded object is not PreprocessingArtifacts.")
    return artifacts


def save_numerical_scaling_artifacts(
    project_dir: str | Path,
    artifacts: PreprocessingArtifacts,
) -> tuple[Path | None, Path]:
    """Save numerical scaling metadata and optional scaler artifact."""

    output_dir = Path(project_dir) / "artifacts" / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / SCALER_CONFIG_FILENAME
    config_path.write_text(
        json.dumps(artifacts.preprocessing_metadata, indent=2),
        encoding="utf-8",
    )
    scaler_path = output_dir / SCALER_ARTIFACT_FILENAME
    if artifacts.scaler is None:
        if scaler_path.exists():
            scaler_path.unlink()
        return None, config_path
    joblib.dump(artifacts.scaler, scaler_path)
    return scaler_path, config_path


def invalidate_preprocessing_artifacts(project_dir: str | Path) -> None:
    """Remove persisted preprocessing artifacts that no longer match the project state."""

    output_dir = Path(project_dir) / "artifacts" / "preprocessing"
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)


def resolve_preprocessing_spec(
    df: pd.DataFrame,
    config: Any,
    *,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Resolve numeric, categorical, and scaling configuration from data and config."""

    feature_columns = list(feature_columns or _resolve_feature_columns(df, config))
    preprocessing_options = getattr(config, "preprocessing_options", {}) or {}
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    numeric_scaling_method = _normalize_scaling_method(
        preprocessing_options.get("numerical_scaling_method"),
        preprocessing_options,
    )
    has_configured_columns = "scaled_numerical_columns" in preprocessing_options
    configured_columns = list(preprocessing_options.get("scaled_numerical_columns", []) or [])
    candidate_scaled_columns = configured_columns if has_configured_columns else numeric_columns
    scaled_numeric_columns = [
        column
        for column in candidate_scaled_columns
        if column in numeric_columns and column not in set(getattr(config, "label_encoding_columns", []) or [])
    ]
    return {
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "scaled_numeric_columns": scaled_numeric_columns,
        "numeric_scaling_method": numeric_scaling_method,
        "numeric_impute_strategy": preprocessing_options.get("numeric_impute_strategy", "median"),
        "categorical_fill_value": preprocessing_options.get("categorical_fill_value", "Unknown"),
    }


def build_preprocessing_metadata(
    spec: dict[str, Any],
    *,
    feature_columns: list[str],
    target_column: str | None,
) -> dict[str, Any]:
    """Build JSON-serializable preprocessing metadata."""

    return {
        "target_column": target_column,
        "feature_columns": list(feature_columns),
        "numeric_columns": list(spec["numeric_columns"]),
        "categorical_columns": list(spec["categorical_columns"]),
        "numerical_scaling_method": spec["numeric_scaling_method"],
        "scaled_numerical_columns": list(spec["scaled_numeric_columns"]),
        "numeric_impute_strategy": spec["numeric_impute_strategy"],
        "categorical_fill_value": spec["categorical_fill_value"],
    }


def _resolve_feature_columns(df: pd.DataFrame, config: Any) -> list[str]:
    configured_features = list(getattr(config, "feature_columns", []) or [])
    if configured_features:
        return configured_features

    excluded = set(getattr(config, "id_columns", []) or [])
    excluded.update(getattr(config, "excluded_columns", []) or [])
    excluded.update(getattr(config, "subgroup_columns", []) or [])

    target_column = getattr(config, "target_column", None)
    group_column = getattr(config, "group_column", None)
    date_column = getattr(config, "date_column", None)
    for column in (target_column, group_column, date_column):
        if column:
            excluded.add(column)

    return [column for column in df.columns if column not in excluded]


def _validate_columns(df: pd.DataFrame, feature_columns: list[str], target_column: str | None) -> None:
    if not target_column:
        raise ValueError("target_column is required for preprocessing.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")
    if not feature_columns:
        raise ValueError("At least one feature column is required for preprocessing.")

    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        raise ValueError(f"Feature columns are missing from the dataset: {missing_features}.")


def _fit_transform_numeric(
    df: pd.DataFrame,
    numeric_columns: list[str],
    scaled_numeric_columns: list[str],
    imputer: SimpleImputer,
    scaler: ScalerType,
) -> np.ndarray:
    if not numeric_columns:
        return np.empty((len(df), 0))

    numeric_frame = pd.DataFrame(
        imputer.fit_transform(df[numeric_columns]),
        columns=numeric_columns,
        index=df.index,
    )
    if scaler is not None and scaled_numeric_columns:
        numeric_frame.loc[:, scaled_numeric_columns] = scaler.fit_transform(
            numeric_frame[scaled_numeric_columns]
        )
    return numeric_frame.to_numpy(dtype=float)


def _transform_numeric(
    df: pd.DataFrame,
    numeric_columns: list[str],
    scaled_numeric_columns: list[str],
    imputer: SimpleImputer | None,
    scaler: ScalerType,
) -> np.ndarray:
    if not numeric_columns:
        return np.empty((len(df), 0))
    if imputer is None:
        raise ValueError("Numeric imputer is missing from preprocessing artifacts.")

    numeric_frame = pd.DataFrame(
        imputer.transform(df[numeric_columns]),
        columns=numeric_columns,
        index=df.index,
    )
    if scaler is not None and scaled_numeric_columns:
        numeric_frame.loc[:, scaled_numeric_columns] = scaler.transform(
            numeric_frame[scaled_numeric_columns]
        )
    return numeric_frame.to_numpy(dtype=float)


def _fit_transform_categorical(
    df: pd.DataFrame,
    categorical_columns: list[str],
    imputer: SimpleImputer,
    encoder: OneHotEncoder | None,
) -> np.ndarray:
    if not categorical_columns:
        return np.empty((len(df), 0))
    if encoder is None:
        raise ValueError("Categorical encoder is required when categorical columns exist.")

    categorical_array = imputer.fit_transform(_categorical_frame(df, categorical_columns))
    return encoder.fit_transform(categorical_array)


def _transform_categorical(
    df: pd.DataFrame,
    categorical_columns: list[str],
    imputer: SimpleImputer | None,
    encoder: OneHotEncoder | None,
) -> np.ndarray:
    if not categorical_columns:
        return np.empty((len(df), 0))
    if imputer is None or encoder is None:
        raise ValueError("Categorical preprocessing artifacts are missing.")

    categorical_array = imputer.transform(_categorical_frame(df, categorical_columns))
    return encoder.transform(categorical_array)


def _prepare_target(series: pd.Series, task_type: str | None) -> tuple[pd.Series, LabelEncoder | None]:
    if task_type == "classification":
        target_encoder = LabelEncoder()
        encoded = target_encoder.fit_transform(series.astype(str))
        return pd.Series(encoded, index=series.index, name=series.name), target_encoder

    if task_type == "regression":
        numeric_target = pd.to_numeric(series, errors="coerce")
        if numeric_target.isna().any():
            raise ValueError("Regression target must be numeric.")
        return pd.Series(numeric_target, index=series.index, name=series.name), None

    return series.copy(), None


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_scaler(method: str) -> ScalerType:
    if method == "minmax":
        return MinMaxScaler()
    if method == "standard":
        return StandardScaler()
    return None


def _output_feature_names(
    numeric_columns: list[str],
    categorical_columns: list[str],
    encoder: OneHotEncoder | None,
) -> list[str]:
    names = list(numeric_columns)
    if categorical_columns and encoder is not None:
        names.extend(encoder.get_feature_names_out(categorical_columns).tolist())
    return names


def _combine_arrays(numeric_array: np.ndarray, categorical_array: np.ndarray) -> np.ndarray:
    if numeric_array.size and categorical_array.size:
        return np.hstack([numeric_array, categorical_array])
    if numeric_array.size:
        return numeric_array
    if categorical_array.size:
        return categorical_array
    return np.empty((numeric_array.shape[0] or categorical_array.shape[0], 0))


def _categorical_frame(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    return df[categorical_columns].astype(object).where(pd.notna(df[categorical_columns]), np.nan)


def _normalize_scaling_method(value: Any, preprocessing_options: dict[str, Any]) -> str:
    normalized = _normalized(value) or ""
    if normalized in {"minmax", "min_max", "min-max"}:
        return "minmax"
    if normalized in {"standard", "standardization", "standardize", "zscore"}:
        return "standard"
    if normalized in {"none", ""}:
        if bool(preprocessing_options.get("standard_scale", preprocessing_options.get("scale_numeric", False))):
            return "standard"
        return "none"
    return "none"


def _normalized(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()
