"""Training orchestration for basic ML models."""

from __future__ import annotations

import copy
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from app.__version__ import (
    APP_DESCRIPTION,
    APP_NAME,
    RELEASE_DATE,
    __version__,
)
from app.branding import report_footer
from app.core.data_loader import load_dataset
from app.core.dataset_manager import project_dataset_path
from app.core.evaluator import evaluate_predictions
from app.core.imbalance import apply_imbalance_strategy
from app.core.model_registry import get_model_spec
from app.core.preprocessing import (
    build_preprocessing_pipeline,
    categorical_feature_columns,
    fit_split_preprocessing,
    save_artifacts,
    transform_split_features,
)
from app.core.splitter import split_data
from app.core.target_encoding import decode_target, encode_target
from app.core.tabpfn_model_manager import (
    checkpoint_sha256,
    get_tabpfn_model_status,
)
from app.utils.plotting import (
    plot_confusion_matrix_publication,
    plot_feature_importance_publication,
    plot_pr_curve_publication,
    plot_roc_curve_publication,
)
from app.utils.resources import (
    is_packaged_application,
    resolve_tabpfn_checkpoint,
)
from app.models.sklearn_models import create_sklearn_model


ProgressCallback = Callable[[dict[str, Any]], None]
CancelCallback = Callable[[], bool]
TABPFN_MAX_SAMPLES = 50_000
TABPFN_MAX_FEATURES = 2_000
TABPFN_MAX_CLASSES = 10
TABPFN_PREDICTION_BATCH_SIZE = 500
CV_PROTOCOL_METADATA = {
    "cv_preprocessing_scope": "fold_training_only",
    "cv_imbalance_scope": "fold_training_only",
    "cv_source": "original_external_training_partition",
    "cv_resampling_before_split": False,
}


def train_selected_models(
    df: pd.DataFrame,
    config: Any,
    environment_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train through the legacy two-way API with training-only preprocessing.

    The GUI and workers use :func:`train_saved_models`; this compatibility API
    remains for callers that still supply an in-memory dataframe.
    """

    environment_info = environment_info or {}
    task_type = str(getattr(config, "task_type", "") or "").strip().lower()
    if task_type not in {"classification", "regression"}:
        raise ValueError("config.task_type must be 'classification' or 'regression'.")

    target_column = getattr(config, "target_column", None)
    if not target_column or target_column not in df.columns:
        raise ValueError("A valid target_column is required for training.")

    raw_features = df.drop(columns=[target_column])
    raw_target = df[target_column]
    raw_split = split_data(raw_features, raw_target, df, config)
    train_df = df.loc[raw_split["train_index"]]
    test_df = df.loc[raw_split["test_index"]]
    X_train, y_train, preprocessing_artifacts = build_preprocessing_pipeline(
        train_df,
        config,
    )
    X_test = transform_split_features(test_df, preprocessing_artifacts)
    if task_type == "classification":
        if preprocessing_artifacts.target_encoder is None:
            raise ValueError("Classification target encoder was not fitted.")
        y_test = encode_target(
            preprocessing_artifacts.target_encoder,
            test_df[target_column],
        )
    else:
        y_test = pd.to_numeric(test_df[target_column], errors="coerce")
        if y_test.isna().any():
            raise ValueError("Regression target must be numeric.")
    split = {
        **raw_split,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    imbalance = apply_imbalance_strategy(
        split["X_train"],
        split["y_train"],
        preprocessing_artifacts,
        config,
    )

    selected_models = list(getattr(config, "selected_models", []) or [])
    if not selected_models:
        raise ValueError("No models selected for training.")

    artifact_dir = Path(getattr(config, "output_dir")) / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_path = save_artifacts(preprocessing_artifacts, artifact_dir / "preprocessing.joblib")

    results = []
    for model_name in selected_models:
        results.append(
            _train_one_model(
                model_name=model_name,
                task_type=task_type,
                split=split,
                imbalance=imbalance,
                config=config,
                artifact_dir=artifact_dir,
                preprocessing_path=preprocessing_path,
            )
        )

    return {
        "task_type": task_type,
        "environment_info": environment_info,
        "split_info": split["split_info"],
        "imbalance_info": imbalance["imbalance_info"],
        "results": results,
    }


def train_saved_models(
    config: Any,
    *,
    save_outputs: bool = True,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> dict[str, Any]:
    """Train classification models from confirmed saved split artifacts."""

    selected_models = list(getattr(config, "selected_models", []) or [])
    if not selected_models:
        raise ValueError("At least one model must be selected before training.")

    split_dir = Path(config.project_dir) / "outputs" / "data_split"
    data = _load_saved_training_data(split_dir, config.target_column)
    target_diagnostics = _target_type_diagnostics(config, data)
    _write_target_diagnostics_log(config, target_diagnostics)
    _emit(
        progress_callback,
        model="",
        fold=0,
        total_folds=0,
        step="target validation",
        percent=0,
        message=_format_target_diagnostics(target_diagnostics),
    )
    if target_diagnostics["detected_target_type"] != "classification":
        raise ValueError(
            "Training blocked.\n"
            f"Current task_type={target_diagnostics['current_task_type']}\n"
            f"Target={target_diagnostics['current_target_column']}\n"
            f"Saved target={target_diagnostics['saved_target_column']}\n"
            f"Detected target type={target_diagnostics['detected_target_type']}"
        )
    cv_folds = int(getattr(config, "cv_folds", 5))
    cv_enabled = bool(getattr(config, "enable_cross_validation", False))
    tabpfn_selected = any(
        get_model_spec(model_name).name == "tabpfn" for model_name in selected_models
    )
    if cv_enabled or tabpfn_selected:
        _load_cv_training_source(
            config,
            data,
            include_holdouts=tabpfn_selected,
        )
    if cv_enabled:
        _validate_cv_counts(data["cv_y_train"], cv_folds)

    output_root = Path(config.project_dir) / "outputs" / "training"
    if save_outputs:
        output_root.mkdir(parents=True, exist_ok=True)

    results = []
    supported_models = []
    for model_name in selected_models:
        spec = get_model_spec(model_name)
        if (
            spec.estimator_type in {"sklearn", "xgboost"}
            or spec.name
            in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
                "tabpfn",
            }
        ):
            supported_models.append((model_name, spec))
        else:
            results.append(
                {
                    "model_name": spec.display_name,
                    "status": "skipped",
                    "reason": "Deep/foundation model training is not implemented yet.",
                    "saved": False,
                }
            )

    total_units = max(1, len(supported_models) * (cv_folds + 4 if cv_enabled else 4))
    completed_units = 0
    for model_name, spec in supported_models:
        _raise_if_cancelled(should_cancel)
        model_output = output_root / _model_output_name(spec.display_name)
        _emit(
            progress_callback,
            model=spec.display_name,
            fold=0,
            total_folds=cv_folds if cv_enabled else 0,
            step="started",
            percent=int(completed_units / total_units * 100),
            message=f"{spec.display_name} started",
        )
        try:
            if spec.name in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
            }:
                model_result, used_units = _train_saved_torch_classifier(
                    spec.name,
                    spec.display_name,
                    config,
                    data,
                    model_output,
                    save_outputs,
                    progress_callback,
                    should_cancel,
                    completed_units,
                    total_units,
                )
            elif spec.name == "tabpfn":
                model_result, used_units = _train_saved_tabpfn(
                    config,
                    data,
                    model_output,
                    save_outputs,
                    progress_callback,
                    should_cancel,
                    completed_units,
                    total_units,
                )
            else:
                model_result, used_units = _train_saved_model(
                    model_name,
                    spec.display_name,
                    config,
                    data,
                    model_output,
                    save_outputs,
                    progress_callback,
                    should_cancel,
                    completed_units,
                    total_units,
                )
            completed_units += used_units
            results.append(model_result)
        except TrainingCancelled:
            raise
        except Exception as exc:
            completed_units += cv_folds + 4 if cv_enabled else 4
            if spec.name in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
            }:
                model_output.mkdir(parents=True, exist_ok=True)
                _write_json(
                    model_output / "failure_reason.json",
                    {
                        "model_name": spec.display_name,
                        "error": str(exc),
                        "failure_timestamp": datetime.now().isoformat(timespec="seconds"),
                        **_project_metadata(config),
                    },
                )
            results.append(
                {
                    "model_name": spec.display_name,
                    "status": "failed",
                    "error": str(exc),
                    "saved": False,
                }
            )
            _emit(
                progress_callback,
                model=spec.display_name,
                fold=0,
                total_folds=cv_folds if cv_enabled else 0,
                step="failed",
                percent=min(100, int(completed_units / total_units * 100)),
                message=f"{spec.display_name} failed: {exc}",
            )

    summary = {
        "status": "completed",
        "training_timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        **_project_metadata(config),
        "report_footer": report_footer(),
        "results": results,
    }
    if save_outputs:
        (output_root / "training_results.json").write_text(
            json.dumps(summary, indent=2, default=_json_scalar),
            encoding="utf-8",
        )
    _emit(
        progress_callback,
        model="",
        fold=0,
        total_folds=0,
        step="complete",
        percent=100,
        message="Training workflow complete",
    )
    return summary


class TrainingCancelled(RuntimeError):
    """Raised when the worker requests cancellation."""


def _load_saved_training_data(
    split_dir: Path,
    target_column: str,
) -> dict[str, Any]:
    encoded_required = {
        "X_train": "X_train_balanced.npy",
        "y_train": "y_train_balanced_encoded.npy",
        "X_val": "X_val.npy",
        "y_val": "y_val_encoded.npy",
        "X_test": "X_test.npy",
        "y_test": "y_test_encoded.npy",
    }
    legacy_required = {
        **encoded_required,
        "y_train": "y_train_balanced.npy",
        "y_val": "y_val.npy",
        "y_test": "y_test.npy",
    }
    required = (
        encoded_required
        if all((split_dir / filename).exists() for filename in encoded_required.values())
        else legacy_required
    )
    missing = [filename for filename in required.values() if not (split_dir / filename).exists()]
    if missing:
        raise ValueError(f"Saved split artifact(s) are missing: {missing}.")

    data = {}
    for key, filename in required.items():
        data[key] = np.load(
            split_dir / filename,
            allow_pickle=key.startswith("y_"),
        )
    for split_name in ("train", "val", "test"):
        if len(data[f"X_{split_name}"]) != len(data[f"y_{split_name}"]):
            raise ValueError(f"Saved {split_name} X/y row counts do not match.")

    indices_path = split_dir / "split_indices.json"
    data["split_metadata"] = json.loads(indices_path.read_text(encoding="utf-8"))
    if data["split_metadata"].get("target_column") != target_column:
        raise ValueError(
            "Saved split target does not match the current configured target. "
            "Confirm Data Split & Imbalance again."
        )
    original_training_rows = len(data["split_metadata"].get("train_index") or [])
    if original_training_rows == 0:
        original_target_path = split_dir / "y_train.npy"
        original_training_rows = (
            int(len(np.load(original_target_path, allow_pickle=True)))
            if original_target_path.is_file()
            else int(len(data["y_train"]))
        )
    data["original_training_rows"] = int(original_training_rows)
    data["effective_training_rows"] = int(len(data["y_train"]))
    data["original_dataset_rows"] = int(
        original_training_rows + len(data["y_val"]) + len(data["y_test"])
    )
    imbalance_path = split_dir / "imbalance_config.json"
    if imbalance_path.exists():
        imbalance_metadata = json.loads(imbalance_path.read_text(encoding="utf-8"))
        if imbalance_metadata.get("target_column") != target_column:
            raise ValueError(
                "Saved imbalance target does not match the current configured target. "
                "Confirm Data Split & Imbalance again."
            )
    encoder_path = split_dir / "target_label_encoder.joblib"
    data["target_encoder"] = joblib.load(encoder_path) if encoder_path.exists() else None
    if data["target_encoder"] is not None:
        data["class_labels"] = list(data["target_encoder"].classes_)
        original_files = {
            "y_train_original": "y_train_balanced_original.npy",
            "y_val_original": "y_val_original.npy",
            "y_test_original": "y_test_original.npy",
        }
        for key, filename in original_files.items():
            path = split_dir / filename
            data[key] = (
                np.load(path, allow_pickle=True)
                if path.exists()
                else decode_target(data["target_encoder"], data[key.replace("_original", "")])
            )
    else:
        data["class_labels"] = list(np.unique(data["y_train"]))
    data["preprocessing_path"] = split_dir / "preprocessing_artifact.joblib"
    if data["preprocessing_path"].exists():
        artifacts = joblib.load(data["preprocessing_path"])
        data["feature_names"] = list(getattr(artifacts, "output_feature_names", []) or [])
        data["saved_task_type"] = getattr(artifacts, "task_type", None)
        data["saved_target_column"] = (
            getattr(artifacts, "target_column", None)
            or data["split_metadata"].get("target_column")
        )
    else:
        data["feature_names"] = [
            f"feature_{index}" for index in range(data["X_train"].shape[1])
        ]
        data["saved_task_type"] = data["split_metadata"].get("task_type")
        data["saved_target_column"] = data["split_metadata"].get("target_column")
    return data


def _load_cv_training_source(
    config: Any,
    data: dict[str, Any],
    *,
    include_holdouts: bool = False,
) -> None:
    """Load raw saved-split rows for leakage-safe CV and model-specific input."""

    dataset_path = project_dataset_path(config)
    if dataset_path is not None and dataset_path.is_file():
        raw_dataset = load_dataset(dataset_path)
    else:
        subset_path = Path(config.project_dir) / "data" / "modeling_subset.csv"
        if not subset_path.is_file():
            raise ValueError(
                "Training requires the project dataset or data/modeling_subset.csv "
                "to reconstruct the original saved data split."
            )
        raw_dataset = load_dataset(subset_path)

    feature_columns = list(getattr(config, "feature_columns", []) or [])
    target_column = getattr(config, "target_column", None)
    required_columns = [*feature_columns, target_column]
    missing_columns = [
        column for column in required_columns if column and column not in raw_dataset.columns
    ]
    if not target_column or missing_columns:
        raise ValueError(
            "Raw source data does not match the saved modeling configuration. "
            f"Missing columns: {missing_columns}."
        )

    split_metadata = data["split_metadata"]
    train_index = list(split_metadata.get("train_index") or [])
    validation_index = list(split_metadata.get("validation_index") or [])
    test_index = list(split_metadata.get("test_index") or [])
    split_indices = {"train": train_index}
    if include_holdouts:
        split_indices.update(
            {"validation": validation_index, "test": test_index}
        )
    if not train_index:
        raise ValueError("Saved split metadata contains no external training rows.")
    duplicate_splits = [
        name for name, indices in split_indices.items() if len(indices) != len(set(indices))
    ]
    if duplicate_splits:
        raise ValueError(f"Saved split indices contain duplicates: {duplicate_splits}.")
    if set(train_index) & (set(validation_index) | set(test_index)):
        raise ValueError(
            "Saved split metadata overlaps external training with validation/test rows."
        )
    if include_holdouts and (
        set(validation_index) & set(test_index)
    ):
        raise ValueError(
            "Saved split metadata contains overlapping train/validation/test rows."
        )
    all_indices = [index for indices in split_indices.values() for index in indices]
    missing_indices = [index for index in all_indices if index not in raw_dataset.index]
    if missing_indices:
        raise ValueError(
            "Saved split indices are not present in the project dataset. "
            "Confirm Data Split & Imbalance again."
        )

    raw_splits = {
        name: raw_dataset.loc[indices, required_columns].copy()
        for name, indices in split_indices.items()
    }

    def encode_raw_target(frame: pd.DataFrame) -> np.ndarray:
        if data["target_encoder"] is not None:
            return np.asarray(
                encode_target(data["target_encoder"], frame[target_column])
            )
        return frame[target_column].to_numpy()

    encoded_targets = {
        name: encode_raw_target(frame) for name, frame in raw_splits.items()
    }
    split_dir = Path(config.project_dir) / "outputs" / "data_split"
    saved_target_path = split_dir / (
        "y_train_encoded.npy"
        if data["target_encoder"] is not None
        else "y_train.npy"
    )
    if saved_target_path.is_file():
        saved_target = np.load(saved_target_path, allow_pickle=True)
        if len(saved_target) != len(encoded_targets["train"]) or not np.array_equal(
            np.asarray(saved_target),
            encoded_targets["train"],
        ):
            raise ValueError(
                "The raw external training rows do not match the saved split target. "
                "Confirm Data Split & Imbalance again."
            )
    if include_holdouts:
        for split_name, saved_key in (("validation", "y_val"), ("test", "y_test")):
            if len(data[saved_key]) != len(encoded_targets[split_name]) or not np.array_equal(
                np.asarray(data[saved_key]),
                encoded_targets[split_name],
            ):
                raise ValueError(
                    f"The raw external {split_name} rows do not match the saved split target. "
                    "Confirm Data Split & Imbalance again."
                )
    data["raw_dataset_rows"] = int(len(raw_dataset))
    data["cv_raw_train"] = raw_splits["train"]
    data["cv_y_train"] = encoded_targets["train"]
    if include_holdouts:
        data["raw_validation"] = raw_splits["validation"]
        data["raw_test"] = raw_splits["test"]
        data["raw_y_validation"] = encoded_targets["validation"]
        data["raw_y_test"] = encoded_targets["test"]


def _prepare_cv_fold(
    data: dict[str, Any],
    config: Any,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
) -> dict[str, Any]:
    """Fit preprocessing and imbalance handling using one CV training fold."""

    raw_train = data["cv_raw_train"].iloc[train_pos].copy()
    raw_validation = data["cv_raw_train"].iloc[eval_pos].copy()
    y_train = pd.Series(
        np.asarray(data["cv_y_train"])[train_pos],
        name=getattr(config, "target_column", None),
    )
    y_validation = np.asarray(data["cv_y_train"])[eval_pos]

    artifacts = fit_split_preprocessing(raw_train, config)
    X_train = transform_split_features(raw_train, artifacts).reset_index(drop=True)
    X_validation = transform_split_features(
        raw_validation,
        artifacts,
    ).reset_index(drop=True)

    imbalance = _apply_cv_training_imbalance(
        X_train,
        y_train,
        artifacts,
        config,
        data.get("target_encoder"),
    )
    if not imbalance["imbalance_info"]["success"]:
        reason = (
            imbalance["imbalance_info"].get("error")
            or imbalance["imbalance_info"].get("message")
        )
        raise ValueError(f"Fold-local imbalance handling failed: {reason}")
    return {
        "X_train": np.asarray(imbalance["X_resampled"]),
        "y_train": np.asarray(imbalance["y_resampled"]),
        "X_validation": np.asarray(X_validation),
        "y_validation": np.asarray(y_validation),
        "artifacts": artifacts,
        "imbalance_info": imbalance["imbalance_info"],
        "train_row_indices": raw_train.index.tolist(),
        "validation_row_indices": raw_validation.index.tolist(),
    }


def _prepare_deep_cv_fold(
    data: dict[str, Any],
    config: Any,
    train_pos: np.ndarray,
    eval_pos: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> dict[str, Any]:
    """Prepare nested deep-CV data without fitting on either validation set."""

    raw_outer_train = data["cv_raw_train"].iloc[train_pos].copy()
    raw_outer_validation = data["cv_raw_train"].iloc[eval_pos].copy()
    y_outer_train = np.asarray(data["cv_y_train"])[train_pos]
    y_outer_validation = np.asarray(data["cv_y_train"])[eval_pos]
    inner_split = _deep_cv_inner_split(
        y_outer_train,
        validation_fraction,
        random_state,
    )
    inner_train_pos = inner_split["train_positions"]
    inner_validation_pos = inner_split["validation_positions"]
    early_stopping_enabled = bool(inner_split["early_stopping_enabled"])

    raw_inner_train = raw_outer_train.iloc[inner_train_pos].copy()
    raw_inner_validation = raw_outer_train.iloc[inner_validation_pos].copy()
    y_inner_train = pd.Series(
        y_outer_train[inner_train_pos],
        name=getattr(config, "target_column", None),
    )
    y_inner_validation = y_outer_train[inner_validation_pos]

    artifacts = fit_split_preprocessing(raw_inner_train, config)
    X_inner_train = transform_split_features(
        raw_inner_train,
        artifacts,
    ).reset_index(drop=True)
    X_inner_validation = (
        transform_split_features(
            raw_inner_validation,
            artifacts,
        ).reset_index(drop=True)
        if early_stopping_enabled
        else pd.DataFrame(columns=artifacts.output_feature_names)
    )
    X_outer_validation = transform_split_features(
        raw_outer_validation,
        artifacts,
    ).reset_index(drop=True)

    imbalance = _apply_cv_training_imbalance(
        X_inner_train,
        y_inner_train,
        artifacts,
        config,
        data.get("target_encoder"),
    )
    if not imbalance["imbalance_info"]["success"]:
        reason = (
            imbalance["imbalance_info"].get("error")
            or imbalance["imbalance_info"].get("message")
        )
        raise ValueError(f"Fold-local imbalance handling failed: {reason}")
    return {
        "X_train": np.asarray(imbalance["X_resampled"]),
        "y_train": np.asarray(imbalance["y_resampled"]),
        "X_inner_validation": np.asarray(X_inner_validation),
        "y_inner_validation": np.asarray(y_inner_validation),
        "X_outer_validation": np.asarray(X_outer_validation),
        "y_outer_validation": np.asarray(y_outer_validation),
        "artifacts": artifacts,
        "imbalance_info": imbalance["imbalance_info"],
        "inner_split": inner_split,
        "early_stopping_enabled": early_stopping_enabled,
        "inner_train_row_indices": raw_inner_train.index.tolist(),
        "inner_validation_row_indices": raw_inner_validation.index.tolist(),
        "outer_validation_row_indices": raw_outer_validation.index.tolist(),
    }


def _apply_cv_training_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    artifacts: Any,
    config: Any,
    target_encoder: Any = None,
) -> dict[str, Any]:
    """Apply configured balancing to one CV training subset only."""

    fold_config = copy.copy(config)
    options = dict(getattr(config, "preprocessing_options", {}) or {})
    imbalance_options = dict(options.get("imbalance", {}) or {})
    imbalance_options["sampling_strategy"] = _fold_sampling_strategy(
        y_train,
        config,
        target_encoder,
    )
    options["imbalance"] = imbalance_options
    fold_config.preprocessing_options = options
    return apply_imbalance_strategy(
        X_train,
        y_train,
        artifacts,
        fold_config,
    )


def _fold_sampling_strategy(
    y_train: pd.Series,
    config: Any,
    target_encoder: Any = None,
) -> Any:
    """Recompute the configured balancing targets from fold-training labels."""

    method = str(getattr(config, "imbalance_method", None) or "none").strip().lower()
    if method not in {
        "random_oversample",
        "random_undersample",
        "smote",
        "smote_nc",
        "smote-nc",
    }:
        return "auto"
    counts = y_train.value_counts()
    if counts.empty or len(counts) < 2:
        return "auto"

    majority_count = int(counts.max())
    minority_count = int(counts.min())
    if method == "random_oversample":
        return {
            _json_scalar(class_name): majority_count
            for class_name, count in counts.items()
            if int(count) < majority_count
        }
    if method == "random_undersample":
        return {
            _json_scalar(class_name): minority_count
            for class_name, count in counts.items()
            if int(count) > minority_count
        }

    preset = str(getattr(config, "smote_ratio_preset", "baseline") or "baseline")
    if preset.strip().lower() == "custom":
        options = dict(getattr(config, "preprocessing_options", {}) or {})
        imbalance_options = dict(options.get("imbalance", {}) or {})
        custom_text = str(imbalance_options.get("custom_ratio") or "").strip()
        try:
            custom_values = json.loads(custom_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid custom CV balancing ratio JSON: {exc}") from exc
        if not isinstance(custom_values, dict) or not custom_values:
            raise ValueError("Custom CV balancing ratio must be a non-empty JSON object.")
        class_lookup = {}
        for class_name in counts.index:
            display_name = class_name
            if target_encoder is not None:
                display_name = decode_target(target_encoder, [class_name])[0]
            class_lookup[str(_json_scalar(display_name))] = class_name
        strategy = {}
        for class_label, target in custom_values.items():
            if str(class_label) not in class_lookup:
                raise ValueError(
                    f"Custom ratio class '{class_label}' is not present in the CV training fold."
                )
            numeric_target = float(target)
            target_count = (
                int(majority_count * numeric_target)
                if numeric_target <= 1
                else int(numeric_target)
            )
            class_name = class_lookup[str(class_label)]
            current_count = int(counts[class_name])
            if target_count > current_count:
                strategy[_json_scalar(class_name)] = target_count
        return strategy

    ratios = {
        "light": 0.40,
        "moderate": 0.50,
        "baseline": 0.50,
        "strong": 0.70,
    }
    ratio = ratios.get(preset.strip().lower(), 0.50)
    majority_class = counts.idxmax()
    return {
        _json_scalar(class_name): max(int(count), int(majority_count * ratio))
        for class_name, count in counts.items()
        if class_name != majority_class
        and max(int(count), int(majority_count * ratio)) > int(count)
    }


def _deep_cv_inner_validation_fraction(config: Any) -> float:
    """Mirror the configured train/validation ratio within outer CV training."""

    train_percent = float(getattr(config, "train_percent", 0.0) or 0.0)
    validation_percent = float(
        getattr(config, "validation_percent", 0.0) or 0.0
    )
    development_percent = train_percent + validation_percent
    if development_percent <= 0 or validation_percent <= 0:
        return 0.0
    return validation_percent / development_percent


def _deep_cv_inner_split(
    y_train: np.ndarray,
    validation_fraction: float,
    random_state: int,
) -> dict[str, Any]:
    """Create a stratified early-stopping holdout within outer fold-training."""

    targets = np.asarray(y_train).reshape(-1)
    positions = np.arange(len(targets))
    disabled_reason = ""
    if not 0 < validation_fraction < 1:
        disabled_reason = "Configured validation fraction is not between 0 and 1."
    elif len(targets) < 2:
        disabled_reason = "Outer fold-training has too few rows for an inner split."
    elif pd.Series(targets).value_counts().min() < 2:
        disabled_reason = (
            "At least one outer fold-training class has fewer than two rows."
        )
    else:
        try:
            train_positions, validation_positions = train_test_split(
                positions,
                test_size=validation_fraction,
                random_state=random_state,
                stratify=targets,
            )
            return {
                "train_positions": np.asarray(train_positions),
                "validation_positions": np.asarray(validation_positions),
                "early_stopping_enabled": True,
                "disabled_reason": "",
            }
        except ValueError as exc:
            disabled_reason = str(exc)
    return {
        "train_positions": positions,
        "validation_positions": np.array([], dtype=int),
        "early_stopping_enabled": False,
        "disabled_reason": disabled_reason,
    }


def _target_type_diagnostics(config: Any, data: dict[str, Any]) -> dict[str, Any]:
    y_train = np.asarray(data["y_train"]).reshape(-1)
    current_task_type = str(getattr(config, "task_type", None) or "auto").strip().lower()
    saved_task_type = str(data.get("saved_task_type") or "unknown").strip().lower()
    detected_target_type = _detect_target_type(
        y_train,
        current_task_type=current_task_type,
        saved_task_type=saved_task_type,
    )
    return {
        "current_target_column": getattr(config, "target_column", None),
        "current_task_type": current_task_type,
        "saved_target_column": data.get("saved_target_column"),
        "saved_task_type": saved_task_type,
        "y_train_dtype": str(y_train.dtype),
        "unique_target_classes": int(pd.Series(y_train).nunique(dropna=True)),
        "detected_target_type": detected_target_type,
    }


def _detect_target_type(
    y_train: np.ndarray,
    *,
    current_task_type: str,
    saved_task_type: str,
) -> str:
    if current_task_type == "classification" or saved_task_type == "classification":
        return "classification"

    series = pd.Series(np.asarray(y_train).reshape(-1)).dropna()
    if series.empty:
        return "unknown"
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(series)
    ):
        return "classification"

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        return "classification"

    unique_count = int(numeric.nunique())
    sample_count = int(len(numeric))
    integer_like = bool(np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy())))
    reasonable_limit = min(100, max(20, int(sample_count**0.5)))
    discrete_numeric = integer_like and (
        unique_count <= reasonable_limit or unique_count / sample_count <= 0.2
    )
    if discrete_numeric:
        return "classification"
    return "regression"


def _format_target_diagnostics(diagnostics: dict[str, Any]) -> str:
    return (
        "Training target diagnostics:\n"
        f"current target column={diagnostics['current_target_column']}\n"
        f"current task_type={diagnostics['current_task_type']}\n"
        f"saved target column={diagnostics['saved_target_column']}\n"
        f"saved task_type={diagnostics['saved_task_type']}\n"
        f"y_train dtype={diagnostics['y_train_dtype']}\n"
        f"unique target classes={diagnostics['unique_target_classes']}\n"
        f"detected target type={diagnostics['detected_target_type']}"
    )


def _write_target_diagnostics_log(
    config: Any,
    diagnostics: dict[str, Any],
) -> None:
    log_dir = Path(getattr(config, "project_dir")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with (log_dir / "training_log.txt").open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {_format_target_diagnostics(diagnostics)}\n")


def _validate_cv_counts(y_train: np.ndarray, cv_folds: int) -> None:
    counts = pd.Series(y_train).value_counts()
    insufficient = counts[counts < cv_folds]
    if insufficient.empty:
        return
    class_name = insufficient.index[0]
    count = int(insufficient.iloc[0])
    raise ValueError(
        f"Class '{class_name}' has only {count} samples but CV folds = {cv_folds}. "
        "Reduce CV folds or increase data for this class."
    )


def _train_saved_model(
    model_name: str,
    display_name: str,
    config: Any,
    data: dict[str, Any],
    output_dir: Path,
    save_outputs: bool,
    progress_callback: ProgressCallback | None,
    should_cancel: CancelCallback | None,
    completed_units: int,
    total_units: int,
) -> tuple[dict[str, Any], int]:
    cv_enabled = bool(getattr(config, "enable_cross_validation", False))
    cv_folds = int(getattr(config, "cv_folds", 5))
    params = dict((getattr(config, "model_params", {}) or {}).get(model_name, {}))
    if "random_state" in get_model_spec(model_name).default_params:
        params.setdefault("random_state", getattr(config, "random_state", 42))
    class_weights = _saved_class_weights(config, data["y_train"])
    cv_rows = []
    units = 0

    if cv_enabled:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=int(getattr(config, "random_state", 42)),
        )
        cv_y_train = np.asarray(data["cv_y_train"])
        for fold, (train_pos, eval_pos) in enumerate(
            splitter.split(np.zeros(len(cv_y_train)), cv_y_train),
            start=1,
        ):
            _raise_if_cancelled(should_cancel)
            fold_data = _prepare_cv_fold(data, config, train_pos, eval_pos)
            model = create_sklearn_model(
                model_name,
                "classification",
                params=params,
                class_weights=_saved_class_weights(config, fold_data["y_train"]),
            )
            model.fit(fold_data["X_train"], fold_data["y_train"])
            predictions = model.predict(fold_data["X_validation"])
            probabilities = _predict_probabilities(
                model,
                fold_data["X_validation"],
                "classification",
            )
            metrics = evaluate_predictions(
                fold_data["y_validation"],
                predictions,
                probabilities,
            )
            cv_rows.append(
                {
                    "fold": fold,
                    **{
                        key: metrics.get(key)
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "weighted_f1",
                            "balanced_accuracy",
                            "roc_auc",
                        )
                    },
                }
            )
            units += 1
            _emit(
                progress_callback,
                model=display_name,
                fold=fold,
                total_folds=cv_folds,
                step="evaluating",
                percent=int((completed_units + units) / total_units * 100),
                message=f"Fold {fold}/{cv_folds} completed",
            )

    _raise_if_cancelled(should_cancel)
    model = create_sklearn_model(
        model_name,
        "classification",
        params=params,
        class_weights=class_weights,
    )
    _emit_step(
        progress_callback,
        display_name,
        "fitting",
        completed_units + units,
        total_units,
    )
    model.fit(data["X_train"], data["y_train"])
    units += 1

    evaluations = {}
    for split_name, display_split in (
        ("train", "train"),
        ("val", "validation"),
        ("test", "test"),
    ):
        _raise_if_cancelled(should_cancel)
        X = data[f"X_{split_name}"]
        y = data[f"y_{split_name}"]
        predictions = model.predict(X)
        probabilities = _predict_probabilities(model, X, "classification")
        if data["target_encoder"] is not None:
            y_for_reporting = decode_target(data["target_encoder"], y)
            predictions_for_reporting = decode_target(
                data["target_encoder"],
                predictions,
            )
        else:
            y_for_reporting = y
            predictions_for_reporting = predictions
        metrics = evaluate_predictions(
            y_for_reporting,
            predictions_for_reporting,
            probabilities,
            class_labels=data["class_labels"],
        )
        evaluations[display_split] = {
            "metrics": metrics,
            "predictions": predictions_for_reporting,
            "probabilities": probabilities,
            "actual": y_for_reporting,
        }
        units += 1
        _emit_step(
            progress_callback,
            display_name,
            f"evaluating {display_split}",
            completed_units + units,
            total_units,
        )

    cv_summary = _cv_summary(cv_rows)
    saved = False
    if save_outputs:
        _raise_if_cancelled(should_cancel)
        _save_model_outputs(
            output_dir,
            model,
            model_name,
            display_name,
            params,
            config,
            data,
            evaluations,
            cv_rows,
            cv_summary,
        )
        saved = True
        _emit_step(
            progress_callback,
            display_name,
            "saving",
            completed_units + units,
            total_units,
        )

    return (
        {
            "model_name": display_name,
            "status": "trained",
            "train_metrics": evaluations["train"]["metrics"],
            "validation_metrics": evaluations["validation"]["metrics"],
            "test_metrics": evaluations["test"]["metrics"],
            "cv_summary": cv_summary,
            "saved": saved,
            "output_dir": str(output_dir) if saved else "",
        },
        units,
    )


def _train_saved_torch_classifier(
    model_key: str,
    display_name: str,
    config: Any,
    data: dict[str, Any],
    output_dir: Path,
    save_outputs: bool,
    progress_callback: ProgressCallback | None,
    should_cancel: CancelCallback | None,
    completed_units: int,
    total_units: int,
) -> tuple[dict[str, Any], int]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset

        from app.models.torch_tabular_models import (
            AutoIntClassifier,
            FTTransformerClassifier,
            MambaAttentionClassifier,
            TabResNet,
        )
    except ImportError as exc:
        raise ImportError(
            f"Optional package 'torch' is required for {display_name} training."
        ) from exc

    class FocalLoss(nn.Module):
        def __init__(
            self,
            alpha: Any = None,
            gamma: float = 1.0,
            label_smoothing: float = 0.05,
        ) -> None:
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.label_smoothing = label_smoothing

        def forward(self, inputs: Any, targets: Any) -> Any:
            cross_entropy = F.cross_entropy(
                inputs,
                targets,
                weight=self.alpha,
                label_smoothing=self.label_smoothing,
                reduction="none",
            )
            return (
                (1 - torch.exp(-cross_entropy)) ** self.gamma * cross_entropy
            ).mean()

    raw_params = dict(
        (getattr(config, "model_params", {}) or {}).get(
            model_key,
            (getattr(config, "model_params", {}) or {}).get(display_name, {}),
        )
    )
    architecture_params: dict[str, Any]
    if model_key == "mamba_attention":
        architecture_params = {
            "hidden_dim": int(raw_params.get("hidden_dim", 256)),
            "dropout": float(raw_params.get("dropout", 0.3)),
        }
    elif model_key == "ft_transformer":
        architecture_params = {
            "d_token": int(raw_params.get("d_token", 128)),
            "n_heads": int(raw_params.get("n_heads", 8)),
            "n_layers": int(raw_params.get("n_layers", 3)),
            "dropout": float(raw_params.get("dropout", 0.1)),
        }
    elif model_key == "autoint":
        architecture_params = {
            "d": int(raw_params.get("d", 64)),
            "n_heads": int(raw_params.get("n_heads", 4)),
            "n_layers": int(raw_params.get("n_layers", 3)),
            "dropout": float(raw_params.get("dropout", 0.1)),
        }
    elif model_key == "tab_resnet":
        architecture_params = {
            "hidden": int(raw_params.get("hidden", 256)),
            "n_blocks": int(raw_params.get("n_blocks", 6)),
            "dropout": float(raw_params.get("dropout", 0.2)),
        }
    else:
        raise ValueError(f"Unsupported torch model '{model_key}'.")
    learning_rate = float(raw_params.get("learning_rate", 1e-3))
    focal_gamma = float(
        raw_params.get("focal_gamma", raw_params.get("focal_loss_gamma", 1.0))
    )
    batch_size = int(raw_params.get("batch_size", 128))
    epochs = int(raw_params.get("epochs", 80))
    warmup_epochs = int(raw_params.get("warmup_epochs", 5))
    patience = int(raw_params.get("early_stopping_patience", 30))
    weight_decay = float(raw_params.get("weight_decay", 1e-3))
    warmup_start_factor = float(raw_params.get("warmup_start_factor", 0.1))
    label_smoothing = float(raw_params.get("label_smoothing", 0.05))
    restore_best_weights = bool(raw_params.get("restore_best_weights", True))
    use_class_weights = bool(raw_params.get("use_class_weights", True))

    X_train = np.asarray(data["X_train"], dtype=np.float32)
    y_train = np.asarray(data["y_train"], dtype=np.int64).reshape(-1)
    X_val = np.asarray(data["X_val"], dtype=np.float32)
    y_val = np.asarray(data["y_val"], dtype=np.int64).reshape(-1)
    X_test = np.asarray(data["X_test"], dtype=np.float32)
    y_test = np.asarray(data["y_test"], dtype=np.int64).reshape(-1)
    input_dim = int(X_train.shape[1])
    num_classes = int(len(data["class_labels"]))
    if num_classes < 2:
        raise ValueError(f"{display_name} requires at least two target classes.")
    if any(
        target.size and (target.min() < 0 or target.max() >= num_classes)
        for target in (y_train, y_val, y_test)
    ):
        raise ValueError(
            "Saved encoded targets are incompatible with the inferred target classes."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture_text = "; ".join(
        f"{name}={value}" for name, value in architecture_params.items()
    )
    dimension_text = (
        f"n_features={input_dim}; n_classes={num_classes}"
        if model_key in {"ft_transformer", "autoint"}
        else f"input_dim={input_dim}; num_classes={num_classes}"
    )
    settings = (
        f"device={device}; {dimension_text}; "
        f"{architecture_text}; "
        f"learning_rate={learning_rate}; focal_gamma={focal_gamma}"
    )
    _emit(
        progress_callback,
        model=display_name,
        fold=0,
        total_folds=int(config.cv_folds) if config.enable_cross_validation else 0,
        step="configuration",
        percent=int(completed_units / max(1, total_units) * 100),
        message=f"{display_name} configuration: {settings}",
    )

    def make_loader(
        features: np.ndarray,
        targets: np.ndarray,
        *,
        shuffle: bool,
    ) -> Any:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.long),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def class_weight_tensor(targets: np.ndarray) -> Any:
        if not use_class_weights:
            return None
        present_classes = np.unique(targets)
        weights = compute_class_weight(
            "balanced",
            classes=present_classes,
            y=targets,
        )
        tensor = torch.ones(num_classes, dtype=torch.float32, device=device)
        for class_name, weight in zip(present_classes, weights):
            tensor[int(class_name)] = float(weight)
        return tensor

    def fit_model(
        train_features: np.ndarray,
        train_targets: np.ndarray,
        validation_features: np.ndarray | None,
        validation_targets: np.ndarray | None,
        *,
        fold: int = 0,
        total_folds: int = 0,
        early_stopping_enabled: bool = True,
        validation_role: str = "checkpoint_selection",
    ) -> tuple[Any, list[dict[str, Any]], float | None, int | None]:
        _raise_if_cancelled(should_cancel)
        early_stopping_enabled = bool(
            early_stopping_enabled
            and validation_features is not None
            and validation_targets is not None
            and len(validation_targets) > 0
        )
        if model_key == "mamba_attention":
            model = MambaAttentionClassifier(
                input_dim=int(train_features.shape[1]),
                num_classes=num_classes,
                **architecture_params,
            ).to(device)
        elif model_key == "ft_transformer":
            model = FTTransformerClassifier(
                n_features=int(train_features.shape[1]),
                n_classes=num_classes,
                **architecture_params,
            ).to(device)
        elif model_key == "autoint":
            model = AutoIntClassifier(
                n_features=int(train_features.shape[1]),
                n_classes=num_classes,
                **architecture_params,
            ).to(device)
        else:
            model = TabResNet(
                input_dim=int(train_features.shape[1]),
                n_classes=num_classes,
                **architecture_params,
            ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=max(1, warmup_epochs),
        )
        criterion = FocalLoss(
            alpha=class_weight_tensor(train_targets),
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
        train_loader = make_loader(train_features, train_targets, shuffle=True)
        validation_loader = (
            make_loader(
                validation_features,
                validation_targets,
                shuffle=False,
            )
            if early_stopping_enabled
            else None
        )
        best_metric: float | None = -1.0 if early_stopping_enabled else None
        best_epoch: int | None = 0 if early_stopping_enabled else None
        patience_count = 0
        best_state = None
        history: list[dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            _raise_if_cancelled(should_cancel)
            model.train()
            total_loss = 0.0
            total_rows = 0
            training_predictions = []
            training_actual = []
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_targets)
                total_rows += len(batch_targets)
                training_predictions.extend(
                    outputs.argmax(dim=1).detach().cpu().numpy().tolist()
                )
                training_actual.extend(batch_targets.detach().cpu().numpy().tolist())

            training_accuracy = float(
                accuracy_score(training_actual, training_predictions)
            )
            validation_loss_value = None
            validation_metric = None
            validation_accuracy = None
            if validation_loader is not None:
                model.eval()
                validation_loss = 0.0
                validation_rows = 0
                validation_predictions = []
                validation_actual = []
                with torch.no_grad():
                    for batch_features, batch_targets in validation_loader:
                        batch_features = batch_features.to(device)
                        batch_targets = batch_targets.to(device)
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_targets)
                        validation_loss += float(loss.item()) * len(batch_targets)
                        validation_rows += len(batch_targets)
                        validation_predictions.extend(
                            outputs.argmax(dim=1).cpu().numpy().tolist()
                        )
                        validation_actual.extend(
                            batch_targets.cpu().numpy().tolist()
                        )
                validation_loss_value = validation_loss / max(1, validation_rows)
                validation_metric = float(
                    f1_score(
                        validation_actual,
                        validation_predictions,
                        average="macro",
                        zero_division=0,
                    )
                )
                validation_accuracy = float(
                    accuracy_score(validation_actual, validation_predictions)
                )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, total_rows),
                    "train_accuracy": training_accuracy,
                    "validation_loss": validation_loss_value,
                    "validation_macro_f1": validation_metric,
                    "validation_accuracy": validation_accuracy,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            if (
                early_stopping_enabled
                and validation_metric is not None
                and best_metric is not None
                and validation_metric > best_metric
            ):
                best_metric = validation_metric
                best_epoch = epoch
                patience_count = 0
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            elif early_stopping_enabled:
                patience_count += 1

            if epoch <= warmup_epochs:
                warmup.step()
            else:
                cosine.step()
            _emit(
                progress_callback,
                model=display_name,
                fold=fold,
                total_folds=total_folds,
                step="epoch",
                epoch=epoch,
                train_loss=total_loss / max(1, total_rows),
                train_accuracy=training_accuracy,
                validation_loss=validation_loss_value,
                val_macro_f1=validation_metric,
                validation_accuracy=validation_accuracy,
                validation_role=validation_role,
                percent=min(
                    99,
                    int(
                        (
                            completed_units
                            + (fold if fold else 0)
                            + epoch / max(1, epochs)
                        )
                        / max(1, total_units)
                        * 100
                    ),
                ),
                message=(
                    f"{display_name}"
                    f"{f' fold {fold}/{total_folds}' if fold else ''} "
                    f"epoch {epoch}/{epochs}: "
                    + (
                        f"validation macro-F1={validation_metric:.4f}"
                        if validation_metric is not None
                        else "early stopping disabled"
                    )
                ),
            )
            if early_stopping_enabled and patience_count >= patience:
                break

        if early_stopping_enabled and restore_best_weights and best_state is not None:
            model.load_state_dict(best_state)
        if early_stopping_enabled:
            _emit(
                progress_callback,
                model=display_name,
                fold=fold,
                total_folds=total_folds,
                step="best validation metric",
                percent=min(
                    99,
                    int((completed_units + 1) / max(1, total_units) * 100),
                ),
                validation_role=validation_role,
                message=(
                    f"{display_name} best validation macro-F1={best_metric:.4f} "
                    f"at epoch {best_epoch}"
                ),
            )
        else:
            _emit(
                progress_callback,
                model=display_name,
                fold=fold,
                total_folds=total_folds,
                step="early stopping disabled",
                percent=min(
                    99,
                    int((completed_units + 1) / max(1, total_units) * 100),
                ),
                validation_role="none",
                message=(
                    f"{display_name} completed all {epochs} configured epochs; "
                    "early stopping was disabled for this fold."
                ),
            )
        return model, history, best_metric, best_epoch

    def predict(model: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        loader = make_loader(
            features,
            np.zeros(len(features), dtype=np.int64),
            shuffle=False,
        )
        predictions = []
        probabilities = []
        model.eval()
        with torch.no_grad():
            for batch_features, _ in loader:
                outputs = model(batch_features.to(device))
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
                predictions.append(outputs.argmax(dim=1).cpu().numpy())
        return np.concatenate(predictions), np.vstack(probabilities)

    cv_enabled = bool(getattr(config, "enable_cross_validation", False))
    cv_folds = int(getattr(config, "cv_folds", 5))
    cv_rows: list[dict[str, Any]] = []
    units = 0
    if cv_enabled:
        inner_validation_fraction = _deep_cv_inner_validation_fraction(config)
        inner_random_state = int(getattr(config, "random_state", 42))
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=int(getattr(config, "random_state", 42)),
        )
        cv_y_train = np.asarray(data["cv_y_train"])
        for fold, (train_pos, eval_pos) in enumerate(
            splitter.split(np.zeros(len(cv_y_train)), cv_y_train),
            start=1,
        ):
            fold_data = _prepare_deep_cv_fold(
                data,
                config,
                train_pos,
                eval_pos,
                inner_validation_fraction,
                inner_random_state,
            )
            inner_split = fold_data["inner_split"]
            inner_train_pos = inner_split["train_positions"]
            inner_validation_pos = inner_split["validation_positions"]
            early_stopping_enabled = fold_data["early_stopping_enabled"]
            if not early_stopping_enabled:
                _emit(
                    progress_callback,
                    model=display_name,
                    fold=fold,
                    total_folds=cv_folds,
                    step="early stopping disabled",
                    percent=int((completed_units + units) / total_units * 100),
                    message=(
                        f"{display_name} fold {fold}/{cv_folds}: inner "
                        "early-stopping split was infeasible; training for all "
                        f"configured epochs. {inner_split['disabled_reason']}"
                    ),
                )
            fold_model, fold_history, _, fold_best_epoch = fit_model(
                fold_data["X_train"],
                fold_data["y_train"],
                (
                    fold_data["X_inner_validation"]
                    if early_stopping_enabled
                    else None
                ),
                (
                    fold_data["y_inner_validation"]
                    if early_stopping_enabled
                    else None
                ),
                fold=fold,
                total_folds=cv_folds,
                early_stopping_enabled=early_stopping_enabled,
                validation_role=(
                    "cv_inner_early_stopping"
                    if early_stopping_enabled
                    else "none"
                ),
            )
            fold_predictions, fold_probabilities = predict(
                fold_model,
                fold_data["X_outer_validation"],
            )
            fold_metrics = evaluate_predictions(
                fold_data["y_outer_validation"],
                fold_predictions,
                fold_probabilities,
                class_labels=list(range(num_classes)),
            )
            cv_rows.append(
                {
                    "fold": fold,
                    "cv_outer_validation_role": "fold_scoring_only",
                    "cv_inner_validation_role": (
                        "early_stopping"
                        if early_stopping_enabled
                        else "disabled"
                    ),
                    "cv_inner_validation_fraction": inner_validation_fraction,
                    "cv_inner_random_seed": inner_random_state,
                    "cv_inner_preprocessing_scope": "inner_training_only",
                    "cv_inner_imbalance_scope": "inner_training_only",
                    "cv_inner_validation_resampled": False,
                    "cv_outer_validation_resampled": False,
                    "cv_early_stopping_enabled": early_stopping_enabled,
                    "cv_early_stopping_disabled_reason": inner_split[
                        "disabled_reason"
                    ],
                    "cv_inner_train_size": int(len(fold_data["y_train"])),
                    "cv_inner_train_original_size": int(len(inner_train_pos)),
                    "cv_inner_validation_size": int(len(inner_validation_pos)),
                    "cv_epochs_trained": int(len(fold_history)),
                    "cv_best_epoch": fold_best_epoch,
                    **{
                        key: fold_metrics.get(key)
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "weighted_f1",
                            "balanced_accuracy",
                            "roc_auc",
                        )
                    },
                }
            )
            units += 1
            del fold_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    final_model, training_history, best_metric, best_epoch = fit_model(
        X_train,
        y_train,
        X_val,
        y_val,
        validation_role="external_validation_checkpoint_selection",
    )
    units += 1
    evaluations = {}
    for split_name, display_split, features, targets in (
        ("train", "train", X_train, y_train),
        ("val", "validation", X_val, y_val),
        ("test", "test", X_test, y_test),
    ):
        _raise_if_cancelled(should_cancel)
        encoded_predictions, probabilities = predict(final_model, features)
        if data["target_encoder"] is not None:
            actual = decode_target(data["target_encoder"], targets)
            predictions = decode_target(
                data["target_encoder"],
                encoded_predictions,
            )
        else:
            actual = targets
            predictions = encoded_predictions
        metrics = evaluate_predictions(
            actual,
            predictions,
            probabilities,
            class_labels=data["class_labels"],
        )
        evaluations[display_split] = {
            "metrics": metrics,
            "predictions": predictions,
            "probabilities": probabilities,
            "actual": actual,
        }
        units += 1
        _emit_step(
            progress_callback,
            display_name,
            f"evaluating {display_split}",
            completed_units + units,
            total_units,
        )

    cv_summary = _cv_summary(cv_rows)
    inferred_config = (
        {"n_features": input_dim, "n_classes": num_classes}
        if model_key in {"ft_transformer", "autoint"}
        else ({"n_classes": num_classes} if model_key == "tab_resnet" else {})
    )
    saved = False
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(final_model.state_dict(), output_dir / "trained_model.pt")
        pd.DataFrame(training_history).to_csv(
            output_dir / "training_history.csv",
            index=False,
        )
        for split_name, evaluation in evaluations.items():
            split_output = output_dir / split_name
            split_output.mkdir(parents=True, exist_ok=True)
            _save_evaluation_outputs(
                split_output,
                display_name,
                split_name,
                data["class_labels"],
                evaluation["actual"],
                evaluation["predictions"],
                evaluation["probabilities"],
                evaluation["metrics"],
                data["split_metadata"].get(
                    "validation_index"
                    if split_name == "validation"
                    else f"{split_name}_index",
                    [],
                ),
            )
        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
            _write_json(output_dir / "cv_summary.json", cv_summary)
        _write_json(
            output_dir / "model_config.json",
            {
                "model_name": model_key,
                "display_name": display_name,
                "input_dim": input_dim,
                "num_classes": num_classes,
                **inferred_config,
                **architecture_params,
                "learning_rate": learning_rate,
                "focal_gamma": focal_gamma,
                **_cv_protocol_metadata(config),
                **_deep_validation_role_metadata(config),
            },
        )
        _write_json(
            output_dir / "training_metadata.json",
            {
                **_project_metadata(config),
                "report_footer": report_footer(),
                "device": str(device),
                "input_dim": input_dim,
                "num_classes": num_classes,
                **inferred_config,
                "target_column": config.target_column,
                "train_size": int(len(y_train)),
                "original_dataset_rows": int(data["original_dataset_rows"]),
                "original_training_rows": int(data["original_training_rows"]),
                "effective_training_rows": int(data["effective_training_rows"]),
                "validation_size": int(len(y_val)),
                "test_size": int(len(y_test)),
                "best_validation_macro_f1": best_metric,
                "best_epoch": best_epoch,
                "training_timestamp": datetime.now().isoformat(timespec="seconds"),
                **_cv_protocol_metadata(config),
                **_deep_validation_role_metadata(config),
            },
        )
        saved = True
        _emit(
            progress_callback,
            model=display_name,
            fold=0,
            total_folds=cv_folds if cv_enabled else 0,
            step="saving",
            percent=min(100, int((completed_units + units) / total_units * 100)),
            message=f"{display_name} saved output path: {output_dir}",
        )

    return (
        {
            "model_name": display_name,
            "status": "trained",
            "train_metrics": evaluations["train"]["metrics"],
            "validation_metrics": evaluations["validation"]["metrics"],
            "test_metrics": evaluations["test"]["metrics"],
            "cv_summary": cv_summary,
            "saved": saved,
            "output_dir": str(output_dir) if saved else "",
            "device": str(device),
            "input_dim": input_dim,
            "num_classes": num_classes,
            **inferred_config,
        },
        units,
    )


def _save_model_outputs(
    output_dir: Path,
    model: Any,
    model_name: str,
    display_name: str,
    params: dict[str, Any],
    config: Any,
    data: dict[str, Any],
    evaluations: dict[str, dict[str, Any]],
    cv_rows: list[dict[str, Any]],
    cv_summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "trained_model.joblib")
    if data["preprocessing_path"].exists():
        shutil.copy2(
            data["preprocessing_path"],
            output_dir / "preprocessing_artifact.joblib",
        )

    for split_name, evaluation in evaluations.items():
        split_output = output_dir / split_name
        split_output.mkdir(parents=True, exist_ok=True)
        _save_evaluation_outputs(
            split_output,
            display_name,
            split_name,
            data["class_labels"],
            evaluation["actual"],
            evaluation["predictions"],
            evaluation["probabilities"],
            evaluation["metrics"],
            data["split_metadata"].get(
                "validation_index" if split_name == "validation" else f"{split_name}_index",
                [],
            ),
        )

    if cv_rows:
        pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
        _write_json(output_dir / "cv_summary.json", cv_summary)
    _write_json(
        output_dir / "model_config.json",
        {
            "model_name": model_name,
            "display_name": display_name,
            "parameters": params,
            "cross_validation_enabled": bool(config.enable_cross_validation),
            "cv_folds": int(config.cv_folds),
            "random_state": int(config.random_state),
            "numerical_scaling_method": (
                (config.preprocessing_options or {}).get("numerical_scaling_method") or "none"
            ),
            **_cv_protocol_metadata(config),
        },
    )
    _write_json(
        output_dir / "training_metadata.json",
        {
            **_project_metadata(config),
            "report_footer": report_footer(),
            "dataset_size": int(
                data["original_dataset_rows"]
            ),
            "original_dataset_rows": int(data["original_dataset_rows"]),
            "feature_count": int(data["X_train"].shape[1]),
            "target_column": config.target_column,
            "train_size": int(len(data["y_train"])),
            "original_training_rows": int(data["original_training_rows"]),
            "effective_training_rows": int(data["effective_training_rows"]),
            "validation_size": int(len(data["y_val"])),
            "test_size": int(len(data["y_test"])),
            "imbalance_method": config.imbalance_method,
            "numerical_scaling_method": (
                (config.preprocessing_options or {}).get("numerical_scaling_method") or "none"
            ),
            "random_seed": int(config.random_seed),
            "split_method": config.split_method,
            "model_name": display_name,
            "training_timestamp": datetime.now().isoformat(timespec="seconds"),
            **_cv_protocol_metadata(config),
        },
    )
    _save_model_specific_outputs(
        output_dir,
        model,
        model_name,
        data["feature_names"],
        data.get("class_labels"),
    )


def prepare_tabpfn_training_rows(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_rows: int = TABPFN_MAX_SAMPLES,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Apply TabPFN's supported row cap with deterministic stratified sampling."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("TabPFN training features must be a pandas DataFrame.")
    y_array = np.asarray(y).reshape(-1)
    if len(X) != len(y_array):
        raise ValueError("TabPFN training X/y row counts do not match.")
    if max_rows < 1:
        raise ValueError("TabPFN maximum row count must be positive.")

    available_rows = len(X)
    metadata = {
        "available_training_rows": int(available_rows),
        "effective_training_rows": int(min(available_rows, max_rows)),
        "training_subsampled": available_rows > max_rows,
        "training_subsampling_strategy": (
            "stratified" if available_rows > max_rows else "none"
        ),
        "training_subsampling_seed": int(random_state),
        "training_row_limit": int(max_rows),
    }
    if available_rows <= max_rows:
        return X, y_array, metadata

    classes, inverse, counts = np.unique(
        y_array,
        return_inverse=True,
        return_counts=True,
    )
    if len(classes) > max_rows:
        raise ValueError(
            "TabPFN stratified row limiting cannot retain every class because the "
            "class count exceeds the row limit."
        )

    ideal = counts.astype(float) * (max_rows / available_rows)
    quotas = np.minimum(counts, np.floor(ideal).astype(int))
    quotas = np.maximum(quotas, 1)
    while int(quotas.sum()) > max_rows:
        candidates = np.flatnonzero(quotas > 1)
        remove_index = candidates[np.argmax(quotas[candidates] - ideal[candidates])]
        quotas[remove_index] -= 1
    remaining = max_rows - int(quotas.sum())
    remainders = ideal - np.floor(ideal)
    while remaining > 0:
        candidates = np.flatnonzero(quotas < counts)
        if not len(candidates):
            raise ValueError("Unable to construct the requested TabPFN row sample.")
        order = sorted(
            candidates.tolist(),
            key=lambda index: (remainders[index], counts[index] - quotas[index]),
            reverse=True,
        )
        for index in order:
            if remaining == 0:
                break
            if quotas[index] < counts[index]:
                quotas[index] += 1
                remaining -= 1

    rng = np.random.RandomState(random_state)
    selected_parts = [
        rng.choice(np.flatnonzero(inverse == class_index), size=quota, replace=False)
        for class_index, quota in enumerate(quotas)
    ]
    selected_positions = np.sort(np.concatenate(selected_parts).astype(int))
    return (
        X.iloc[selected_positions].copy(),
        y_array[selected_positions],
        metadata,
    )


def _validate_tabpfn_input(X: pd.DataFrame, y: np.ndarray, *, context: str) -> None:
    feature_count = int(X.shape[1])
    class_count = int(len(np.unique(np.asarray(y))))
    if feature_count > TABPFN_MAX_FEATURES:
        raise ValueError(
            f"TabPFN 2.5 supports at most {TABPFN_MAX_FEATURES:,} input features; "
            f"{context} has {feature_count:,}."
        )
    if class_count > TABPFN_MAX_CLASSES:
        raise ValueError(
            f"TabPFN 2.5 supports at most {TABPFN_MAX_CLASSES} target classes; "
            f"{context} has {class_count}."
        )
    if class_count < 2:
        raise ValueError(f"TabPFN 2.5 requires at least two classes in {context}.")


def _tabpfn_categorical_indices(X: pd.DataFrame, config: Any) -> list[int]:
    categorical = set(categorical_feature_columns(X, config))
    categorical.update(
        column
        for column in list(getattr(config, "label_encoding_columns", []) or [])
        if column in X.columns
    )
    return [index for index, column in enumerate(X.columns) if column in categorical]


def _train_saved_tabpfn(
    config: Any,
    data: dict[str, Any],
    output_dir: Path,
    save_outputs: bool,
    progress_callback: ProgressCallback | None,
    should_cancel: CancelCallback | None,
    completed_units: int,
    total_units: int,
) -> tuple[dict[str, Any], int]:
    display_name = "TabPFN 2.5"
    raw_params = dict(
        (getattr(config, "model_params", {}) or {}).get(
            "tabpfn",
            (getattr(config, "model_params", {}) or {}).get(display_name, {}),
        )
    )
    n_estimators = int(raw_params.get("n_estimators", 8))
    skip_reason = "TabPFN skipped: package tabpfn is not installed."
    try:
        import tabpfn
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        packaged = is_packaged_application()
        checkpoint_status = get_tabpfn_model_status()
        checkpoint = checkpoint_status.active_checkpoint_path or checkpoint_status.cache_path
        if packaged:
            reason = (
                "TabPFN 2.5 could not start because the TabPFN Python "
                "package is missing from AVISTADeepWorker.exe.\n\n"
                f"Checkpoint found: {'Yes' if checkpoint.is_file() else 'No'}\n"
                f"Checkpoint path: {checkpoint}\n"
                "TabPFN package import: Failed\n"
                f"Import error: {type(exc).__name__}: {exc}\n"
                f"Worker: {Path(sys.executable).resolve()}\n\n"
                "This is an AVISTA packaging error, not a dataset or "
                "model-parameter error."
            )
            status = "failed"
            artifact_name = "failure_reason.json"
            result_key = "error"
            step = "failed"
        else:
            reason = skip_reason
            status = "skipped"
            artifact_name = "skip_reason.json"
            result_key = "reason"
            step = "skipped"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / artifact_name,
            {
                "model_name": display_name,
                result_key: reason,
                "packaged": packaged,
                "tabpfn_import_succeeded": False,
                "import_error": f"{type(exc).__name__}: {exc}",
                "checkpoint_found": checkpoint.is_file(),
                "checkpoint_path": str(checkpoint),
                "worker": str(Path(sys.executable).resolve()),
                **_project_metadata(config),
            },
        )
        _emit(
            progress_callback,
            model=display_name,
            fold=0,
            total_folds=0,
            step=step,
            percent=min(100, int((completed_units + 1) / max(1, total_units) * 100)),
            message=reason,
        )
        return (
            {
                "model_name": display_name,
                "status": status,
                result_key: reason,
                "saved": False,
                "output_dir": str(output_dir),
            },
            1,
        )

    try:
        model_path = resolve_tabpfn_checkpoint()
        checkpoint_status = get_tabpfn_model_status()
        checkpoint_source = checkpoint_status.active_checkpoint_source
    except FileNotFoundError:
        checkpoint_status = get_tabpfn_model_status()
        model_path = checkpoint_status.cache_path
        checkpoint_source = "unavailable"
        checkpoint_reason = (
            "TabPFN 2.5 is not currently available. Use Model Selection or "
            "Help > TabPFN Model Status to set it up."
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            output_dir / "failure_reason.json",
            {
                "model_name": display_name,
                "error": checkpoint_reason,
                "model_path": str(model_path),
                **_project_metadata(config),
            },
        )
        _emit(
            progress_callback,
            model=display_name,
            fold=0,
            total_folds=0,
            step="failed",
            percent=min(100, int((completed_units + 1) / max(1, total_units) * 100)),
            message=checkpoint_reason,
        )
        return (
            {
                "model_name": display_name,
                "status": "failed",
                "error": checkpoint_reason,
                "saved": False,
                "output_dir": str(output_dir),
            },
            1,
        )
    import torch

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    tabpfn_version = str(getattr(tabpfn, "__version__", "unknown"))
    tabpfn_file = getattr(tabpfn, "__file__", None)
    tabpfn_path = (
        str(Path(tabpfn_file).resolve())
        if tabpfn_file
        else "unknown"
    )
    random_state = int(getattr(config, "random_state", 42))
    feature_columns = list(getattr(config, "feature_columns", []) or [])
    X_train = data["cv_raw_train"].loc[:, feature_columns].copy()
    y_train = np.asarray(data["cv_y_train"], dtype=np.int64).reshape(-1)
    X_val = data["raw_validation"].loc[:, feature_columns].copy()
    y_val = np.asarray(data["raw_y_validation"], dtype=np.int64).reshape(-1)
    X_test = data["raw_test"].loc[:, feature_columns].copy()
    y_test = np.asarray(data["raw_y_test"], dtype=np.int64).reshape(-1)
    num_classes = int(len(data["class_labels"]))
    _validate_tabpfn_input(X_train, y_train, context="external training data")
    if num_classes > TABPFN_MAX_CLASSES:
        raise ValueError(
            f"TabPFN 2.5 supports at most {TABPFN_MAX_CLASSES} target classes; "
            f"the saved target encoder contains {num_classes}."
        )
    if not 1 <= n_estimators <= 100:
        raise ValueError("TabPFN n_estimators must be between 1 and 100.")

    categorical_indices = _tabpfn_categorical_indices(X_train, config)
    project_imbalance_method = str(
        getattr(config, "imbalance_method", "none") or "none"
    )
    prepared_train, prepared_y_train, final_row_metadata = (
        prepare_tabpfn_training_rows(
            X_train,
            y_train,
            random_state=random_state,
        )
    )

    _emit(
        progress_callback,
        model=display_name,
        fold=0,
        total_folds=int(config.cv_folds) if config.enable_cross_validation else 0,
        step="configuration",
        percent=int(completed_units / max(1, total_units) * 100),
        message=(
            f"{display_name} package available; "
            f"package version={tabpfn_version}; "
            f"package path={tabpfn_path}; "
            f"packaged={is_packaged_application()}; "
            f"selected device={selected_device}; "
            f"n_estimators={n_estimators}; "
            f"checkpoint exists={model_path.is_file()}; "
            f"checkpoint size={model_path.stat().st_size}; "
            f"user-cache checkpoint={model_path}; "
            f"supported maximum training samples={TABPFN_MAX_SAMPLES}; "
            "input representation=raw pandas DataFrame; "
            "AVISTA external preprocessing/resampling=disabled; "
            f"internal prediction batch size={TABPFN_PREDICTION_BATCH_SIZE}"
        ),
    )

    def create_classifier() -> Any:
        return TabPFNClassifier(
            n_estimators=n_estimators,
            categorical_features_indices=categorical_indices,
            model_path=str(model_path),
            device=selected_device,
            random_state=random_state,
        )

    def predict_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
        batches = [
            np.asarray(
                model.predict_proba(
                    features.iloc[start : start + TABPFN_PREDICTION_BATCH_SIZE]
                )
            )
            for start in range(0, len(features), TABPFN_PREDICTION_BATCH_SIZE)
        ]
        probabilities = np.vstack(batches)
        classes = np.asarray(getattr(model, "classes_", np.arange(probabilities.shape[1])))
        aligned = np.zeros((len(features), num_classes), dtype=float)
        for source_column, class_id in enumerate(classes):
            aligned[:, int(class_id)] = probabilities[:, source_column]
        return aligned

    cv_enabled = bool(getattr(config, "enable_cross_validation", False))
    cv_folds = int(getattr(config, "cv_folds", 5))
    cv_rows: list[dict[str, Any]] = []
    units = 0
    if cv_enabled:
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        cv_y_train = np.asarray(data["cv_y_train"])
        for fold, (train_pos, eval_pos) in enumerate(
            splitter.split(np.zeros(len(cv_y_train)), cv_y_train),
            start=1,
        ):
            _raise_if_cancelled(should_cancel)
            fold_X_train = X_train.iloc[train_pos].copy()
            fold_y_train = y_train[train_pos]
            fold_X_validation = X_train.iloc[eval_pos].copy()
            fold_y_validation = y_train[eval_pos]
            _validate_tabpfn_input(
                fold_X_train,
                fold_y_train,
                context=f"CV fold {fold} training data",
            )
            prepared_fold_X, prepared_fold_y, fold_row_metadata = (
                prepare_tabpfn_training_rows(
                    fold_X_train,
                    fold_y_train,
                    random_state=random_state,
                )
            )
            fold_model = create_classifier()
            fold_model.fit(prepared_fold_X, prepared_fold_y)
            fold_probabilities = predict_probabilities(
                fold_model,
                fold_X_validation,
            )
            fold_predictions = fold_probabilities.argmax(axis=1)
            fold_metrics = evaluate_predictions(
                fold_y_validation,
                fold_predictions,
                fold_probabilities,
                class_labels=list(range(num_classes)),
            )
            cv_rows.append(
                {
                    "fold": fold,
                    **fold_row_metadata,
                    "validation_rows": int(len(fold_X_validation)),
                    **{
                        key: fold_metrics.get(key)
                        for key in (
                            "accuracy",
                            "macro_f1",
                            "weighted_f1",
                            "balanced_accuracy",
                            "roc_auc",
                        )
                    },
                }
            )
            units += 1
            _emit(
                progress_callback,
                model=display_name,
                fold=fold,
                total_folds=cv_folds,
                step="cross validation",
                percent=min(
                    99,
                    int((completed_units + units) / max(1, total_units) * 100),
                ),
                message=(
                    f"{display_name} fold {fold}/{cv_folds}: "
                    f"available/effective training rows="
                    f"{fold_row_metadata['available_training_rows']}/"
                    f"{fold_row_metadata['effective_training_rows']}"
                ),
            )

    _raise_if_cancelled(should_cancel)
    final_model = create_classifier()
    final_model.fit(prepared_train, prepared_y_train)
    units += 1
    _emit(
        progress_callback,
        model=display_name,
        fold=0,
        total_folds=cv_folds if cv_enabled else 0,
        step="training",
        percent=min(99, int((completed_units + units) / max(1, total_units) * 100)),
        message=(
            f"{display_name} available/effective external-training rows="
            f"{final_row_metadata['available_training_rows']}/"
            f"{final_row_metadata['effective_training_rows']}"
        ),
    )

    evaluations: dict[str, dict[str, Any]] = {}
    for artifact_name, display_split, features, targets in (
        ("validation", "validation", X_val, y_val),
        ("test", "test", X_test, y_test),
    ):
        _raise_if_cancelled(should_cancel)
        probabilities = predict_probabilities(final_model, features)
        encoded_predictions = probabilities.argmax(axis=1)
        if data["target_encoder"] is not None:
            actual = decode_target(data["target_encoder"], targets)
            predictions = decode_target(
                data["target_encoder"],
                encoded_predictions,
            )
        else:
            actual = targets
            predictions = encoded_predictions
        metrics = evaluate_predictions(
            actual,
            predictions,
            probabilities,
            class_labels=data["class_labels"],
        )
        evaluations[display_split] = {
            "artifact_name": artifact_name,
            "actual": actual,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": metrics,
        }
        units += 1

    cv_summary = _cv_summary(cv_rows)
    saved = False
    if save_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split_name, evaluation in evaluations.items():
            split_output = output_dir / evaluation["artifact_name"]
            split_output.mkdir(parents=True, exist_ok=True)
            _save_evaluation_outputs(
                split_output,
                display_name,
                split_name,
                data["class_labels"],
                evaluation["actual"],
                evaluation["predictions"],
                evaluation["probabilities"],
                evaluation["metrics"],
                data["split_metadata"].get(
                    "validation_index"
                    if split_name == "validation"
                    else "test_index",
                    [],
                ),
            )
        if cv_rows:
            pd.DataFrame(cv_rows).to_csv(output_dir / "cv_results.csv", index=False)
            _write_json(output_dir / "cv_summary.json", cv_summary)
        _write_json(
            output_dir / "model_config.json",
            {
                "n_estimators": n_estimators,
                "model_path": str(model_path),
                "categorical_features_indices": categorical_indices,
                "tabpfn_input_representation": "raw_dataframe",
                "tabpfn_external_one_hot_encoding": False,
                "tabpfn_external_scaling": False,
                "tabpfn_external_resampling_applied": False,
                **_cv_protocol_metadata(config),
            },
        )
        _write_json(
            output_dir / "training_metadata.json",
            {
                **_project_metadata(config),
                "report_footer": report_footer(),
                "package_available": True,
                "tabpfn_version": tabpfn_version,
                "tabpfn_package_path": tabpfn_path,
                "tabpfn_checkpoint_source": checkpoint_source,
                "tabpfn_checkpoint_path": str(model_path),
                "tabpfn_checkpoint_size": model_path.stat().st_size,
                "tabpfn_checkpoint_sha256": checkpoint_sha256(model_path),
                "packaged_mode": is_packaged_application(),
                "selected_device": selected_device,
                "n_estimators": n_estimators,
                "feature_count": int(X_train.shape[1]),
                "num_classes": num_classes,
                "original_dataset_rows": int(
                    data.get(
                        "raw_dataset_rows",
                        len(X_train) + len(X_val) + len(X_test),
                    )
                ),
                "modeling_split_rows": int(len(X_train) + len(X_val) + len(X_test)),
                "external_training_rows": int(len(y_train)),
                "original_training_rows": int(len(y_train)),
                "effective_training_rows": int(len(prepared_train)),
                "train_size": int(len(y_train)),
                "validation_size": int(len(y_val)),
                "test_size": int(len(y_test)),
                "final_subset_size": int(len(prepared_train)),
                **final_row_metadata,
                "tabpfn_model_version": "2.5",
                "tabpfn_input_representation": "raw_dataframe",
                "tabpfn_preprocessing_owner": "tabpfn_internal",
                "tabpfn_external_one_hot_encoding": False,
                "tabpfn_external_scaling": False,
                "tabpfn_external_imputation": False,
                "project_imbalance_method": project_imbalance_method,
                "tabpfn_external_resampling_applied": False,
                "tabpfn_resampling_note": (
                    "Model-specific raw-input protocol; AVISTA resampling was not applied."
                ),
                "categorical_feature_columns": [
                    feature_columns[index] for index in categorical_indices
                ],
                "categorical_features_indices": categorical_indices,
                "tabpfn_internal_row_subsampling": False,
                "target_column": config.target_column,
                "training_timestamp": datetime.now().isoformat(timespec="seconds"),
                **_cv_protocol_metadata(config),
            },
        )
        try:
            joblib.dump(final_model, output_dir / "trained_model.joblib")
        except Exception as exc:
            _write_json(
                output_dir / "model_not_serialized_reason.json",
                {"reason": str(exc)},
            )
        saved = True
        _emit(
            progress_callback,
            model=display_name,
            fold=0,
            total_folds=cv_folds if cv_enabled else 0,
            step="saving",
            percent=min(100, int((completed_units + units) / total_units * 100)),
            message=f"{display_name} saved output path: {output_dir}",
        )

    return (
        {
            "model_name": display_name,
            "status": "trained",
            "train_metrics": {},
            "validation_metrics": evaluations["validation"]["metrics"],
            "test_metrics": evaluations["test"]["metrics"],
            "cv_summary": cv_summary,
            "saved": saved,
            "output_dir": str(output_dir) if saved else "",
            "subset_size": int(len(prepared_train)),
            **final_row_metadata,
        },
        units,
    )


def _save_evaluation_outputs(
    output_dir: Path,
    model_name: str,
    split_name: str,
    class_labels: list[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Any,
    metrics: dict[str, Any],
    row_indices: list[Any],
) -> None:
    report = pd.DataFrame(metrics["classification_report"]).transpose()
    footer = report_footer()
    report["generated_by"] = footer["generated_by"]
    report["version"] = footer["version"]
    report["generated_on"] = footer["generated_on"]
    report.to_csv(output_dir / "classification_report.csv")
    matrix = np.asarray(metrics["confusion_matrix"])
    pd.DataFrame(
        matrix,
        index=[str(label) for label in class_labels],
        columns=[str(label) for label in class_labels],
    ).to_csv(output_dir / "confusion_matrix.csv")
    plot_confusion_matrix_publication(
        matrix,
        class_labels,
        output_dir,
        model_name,
        split_name,
    )
    indices = row_indices if len(row_indices) == len(y_true) else list(range(len(y_true)))
    predictions = pd.DataFrame(
        {"row_identifier": indices, "actual_class": y_true, "predicted_class": y_pred}
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    if probabilities is not None:
        proba = np.asarray(probabilities)
        probability_frame = pd.DataFrame(
            proba,
            columns=[
                f"prob_{_probability_label(class_labels[index])}"
                for index in range(proba.shape[1])
            ],
        )
        probability_frame.insert(0, "row_identifier", indices)
        probability_frame.to_csv(output_dir / "probabilities.csv", index=False)
        confidence = proba.max(axis=1)
    else:
        confidence = np.full(len(y_true), np.nan)
    _write_json(output_dir / "metrics.json", metrics)
    misclassified = predictions[predictions["actual_class"] != predictions["predicted_class"]].copy()
    misclassified["probability_confidence"] = confidence[
        predictions["actual_class"] != predictions["predicted_class"]
    ]
    misclassified.to_csv(output_dir / "misclassified_records.csv", index=False)
    if probabilities is not None:
        probability_array = np.asarray(probabilities)
        plot_roc_curve_publication(
            y_true,
            probability_array,
            class_labels,
            output_dir,
            model_name,
            split_name,
        )
        plot_pr_curve_publication(
            y_true,
            probability_array,
            class_labels,
            output_dir,
            model_name,
            split_name,
        )


def _save_model_specific_outputs(
    output_dir: Path,
    model: Any,
    model_name: str,
    feature_names: list[str],
    class_labels: list[Any] | None = None,
) -> None:
    names = feature_names or [f"feature_{index}" for index in range(model.n_features_in_)]
    if model_name in {"random_forest", "extra_trees", "decision_tree"} and hasattr(
        model, "feature_importances_"
    ):
        plot_feature_importance_publication(
            names,
            model.feature_importances_,
            output_dir,
            get_model_spec(model_name).display_name,
        )
    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        coefficients = np.asarray(model.coef_)
        rows = []
        for class_index, values in enumerate(coefficients):
            for feature, value in zip(names, values):
                rows.append(
                    {
                        "class": _json_scalar(model.classes_[class_index])
                        if class_labels is None
                        else _json_scalar(
                            class_labels[
                                class_index if len(model.classes_) > 2 else int(model.classes_[-1])
                            ]
                        ),
                        "feature": feature,
                        "coefficient": float(value),
                    }
                )
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "coefficients.csv", index=False)
        odds = frame.copy()
        odds["odds_ratio"] = np.exp(odds["coefficient"])
        odds.to_csv(output_dir / "odds_ratios.csv", index=False)


def _probability_label(value: Any) -> str:
    return str(_json_scalar(value)).replace("\n", " ").strip()


def _cv_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary = {}
    for column in frame.columns:
        if (
            column == "fold"
            or frame[column].dropna().empty
            or not pd.api.types.is_numeric_dtype(frame[column])
            or pd.api.types.is_bool_dtype(frame[column])
        ):
            continue
        summary[column] = {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std(ddof=0)),
        }
    return summary


def _cv_protocol_metadata(config: Any) -> dict[str, Any]:
    if not bool(getattr(config, "enable_cross_validation", False)):
        return {}
    return dict(CV_PROTOCOL_METADATA)


def _deep_validation_role_metadata(config: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "validation_role": "checkpoint_selection",
        "test_role": "final_evaluation",
    }
    if bool(getattr(config, "enable_cross_validation", False)):
        metadata.update(
            {
                "cv_outer_validation_role": "fold_scoring_only",
                "cv_inner_validation_role": "early_stopping_when_feasible",
                "cv_inner_validation_fraction": (
                    _deep_cv_inner_validation_fraction(config)
                ),
                "cv_inner_random_seed": int(
                    getattr(config, "random_state", 42)
                ),
                "cv_inner_preprocessing_scope": "inner_training_only",
                "cv_inner_imbalance_scope": "inner_training_only",
                "cv_inner_validation_resampled": False,
                "cv_outer_validation_resampled": False,
            }
        )
    return metadata


def _saved_class_weights(config: Any, y_train: np.ndarray) -> dict[Any, float] | None:
    if not bool(getattr(config, "use_class_weights", False)):
        return None
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {
        _json_scalar(class_name): float(weight)
        for class_name, weight in zip(classes, weights)
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=_json_scalar),
        encoding="utf-8",
    )


def _project_metadata(config: Any) -> dict[str, str]:
    metadata = getattr(config, "project_metadata", None)
    if callable(metadata):
        return metadata()
    project_dir = Path(getattr(config, "project_dir", "")).resolve()
    project_name = str(getattr(config, "project_name", project_dir.name))
    project_file = getattr(config, "project_file_path", "")
    if not project_file:
        project_file = project_dir / f"{project_name}.avista"
    return {
        "project_file": str(Path(project_file).resolve()),
        "project_name": project_name,
        "project_file_version": str(
            getattr(config, "project_file_version", "1.0")
        ),
        "application": APP_NAME,
        "application_description": APP_DESCRIPTION,
        "application_version": __version__,
        "application_release_date": RELEASE_DATE,
    }


def _emit(
    callback: ProgressCallback | None,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(payload)


def _emit_step(
    callback: ProgressCallback | None,
    model: str,
    step: str,
    completed: int,
    total: int,
) -> None:
    _emit(
        callback,
        model=model,
        fold=0,
        total_folds=0,
        step=step,
        percent=min(100, int(completed / max(1, total) * 100)),
        message=f"{model}: {step}",
    )


def _raise_if_cancelled(callback: CancelCallback | None) -> None:
    if callback is not None and callback():
        raise TrainingCancelled("Training cancelled by user.")


def _model_output_name(display_name: str) -> str:
    if display_name == "FT-Transformer":
        return display_name
    if display_name == "TabPFN 2.5":
        return "TabPFN_2_5"
    return "".join(character for character in display_name if character.isalnum()) or "Model"


def _train_one_model(
    model_name: str,
    task_type: str,
    split: dict[str, Any],
    imbalance: dict[str, Any],
    config: Any,
    artifact_dir: Path,
    preprocessing_path: Path,
) -> dict[str, Any]:
    try:
        if task_type == "classification":
            spec = get_model_spec(model_name)
            if spec.estimator_type not in {"sklearn", "xgboost"}:
                return _skipped_result(
                    model_name,
                    f"Model '{model_name}' is not a sklearn-compatible model yet.",
                )

        model = create_sklearn_model(
            model_name=model_name,
            task_type=task_type,
            class_weights=imbalance["class_weights"],
        )
        model.fit(imbalance["X_resampled"], imbalance["y_resampled"])

        predictions = model.predict(split["X_test"])
        probabilities = _predict_probabilities(model, split["X_test"], task_type)
        metrics = evaluate_predictions(split["y_test"], predictions, probabilities, task_type=task_type)

        model_path = artifact_dir / f"{_safe_filename(model_name)}.joblib"
        joblib.dump(model, model_path)

        result = {
            "model_name": model_name,
            "status": "trained",
            "metrics": metrics,
            "predictions": _series_payload(split["test_index"], predictions),
            "probabilities": _probability_payload(split["test_index"], probabilities),
            "artifact_paths": {
                "model": str(model_path),
                "preprocessing": str(preprocessing_path),
            },
        }
        return result
    except Exception as exc:
        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(exc),
            "metrics": {},
            "predictions": [],
            "probabilities": None,
            "artifact_paths": {"preprocessing": str(preprocessing_path)},
        }


def _predict_probabilities(model: Any, X_test: pd.DataFrame, task_type: str) -> Any:
    if task_type != "classification" or not hasattr(model, "predict_proba"):
        return None
    try:
        return model.predict_proba(X_test)
    except Exception:
        return None


def _series_payload(index: list[Any], values: Any) -> list[dict[str, Any]]:
    return [
        {"index": idx.item() if hasattr(idx, "item") else idx, "prediction": _json_scalar(value)}
        for idx, value in zip(index, values)
    ]


def _probability_payload(index: list[Any], probabilities: Any) -> list[dict[str, Any]] | None:
    if probabilities is None:
        return None

    payload = []
    for idx, row in zip(index, probabilities):
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        payload.append(
            {
                "index": idx.item() if hasattr(idx, "item") else idx,
                "probabilities": [_json_scalar(value) for value in values],
            }
        )
    return payload


def _skipped_result(model_name: str, reason: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "status": "skipped",
        "reason": reason,
        "metrics": {},
        "predictions": [],
        "probabilities": None,
        "artifact_paths": {},
    }


def _safe_filename(model_name: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in model_name).strip("_")


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value
