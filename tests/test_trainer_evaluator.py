from pathlib import Path

import builtins
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import sys
import types
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from app.core import trainer as trainer_module
from app.core.evaluator import evaluate_predictions
from app.core.preprocessing import (
    build_preprocessing_pipeline,
    save_artifacts,
    transform_split_features,
)
from app.core.project_config import ProjectConfig
from app.core.trainer import (
    TrainingCancelled,
    _deep_cv_inner_split,
    _deep_cv_inner_validation_fraction,
    _prepare_cv_fold,
    _prepare_deep_cv_fold,
    _validate_tabpfn_input,
    prepare_tabpfn_training_rows,
    train_saved_models,
    train_selected_models,
)


def make_config(tmp_path, **overrides):
    values = {
        "project_name": "trainer-demo",
        "project_dir": str(tmp_path),
        "input_file": str(tmp_path / "data.csv"),
        "output_dir": str(tmp_path / "outputs"),
        "target_column": "target",
        "feature_columns": ["x1", "x2", "cat"],
        "task_type": "classification",
        "split_method": "stratified",
        "imbalance_method": "none",
        "selected_models": ["Logistic Regression"],
    }
    values.update(overrides)
    return ProjectConfig(**values)


def save_training_bundle(tmp_path, config):
    split_dir = tmp_path / "outputs" / "data_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "x1": range(20),
            "x2": [value % 4 for value in range(20)],
            "cat": ["a", "b"] * 10,
            "target": [0, 1] * 10,
        }
    )
    df.to_csv(config.input_file, index=False)
    X, y, artifacts = build_preprocessing_pipeline(df, config)
    train_index = list(range(12))
    validation_index = list(range(12, 16))
    test_index = list(range(16, 20))
    for filename, values in {
        "X_train_balanced.npy": X.loc[train_index].to_numpy(),
        "y_train_balanced.npy": y.loc[train_index].to_numpy(),
        "X_val.npy": X.loc[validation_index].to_numpy(),
        "y_val.npy": y.loc[validation_index].to_numpy(),
        "X_test.npy": X.loc[test_index].to_numpy(),
        "y_test.npy": y.loc[test_index].to_numpy(),
    }.items():
        np.save(split_dir / filename, values)
    (split_dir / "split_indices.json").write_text(
        json.dumps(
            {
                "target_column": "target",
                "train_index": train_index,
                "validation_index": validation_index,
                "test_index": test_index,
            }
        ),
        encoding="utf-8",
    )
    save_artifacts(artifacts, split_dir / "preprocessing_artifact.joblib")
    return split_dir


def save_encoded_string_training_bundle(tmp_path, config):
    split_dir = save_training_bundle(tmp_path, config)
    classes = np.array(
        ["Advanced_Automation", "Assisted_Driving", "Partial_Automation"]
    )
    encoder = LabelEncoder().fit(classes)
    targets = {
        "train": np.tile(classes, 4),
        "val": np.array(
            ["Advanced_Automation", "Assisted_Driving", "Partial_Automation", "Advanced_Automation"]
        ),
        "test": np.array(
            ["Partial_Automation", "Assisted_Driving", "Advanced_Automation", "Partial_Automation"]
        ),
    }
    for split_name, original in targets.items():
        encoded = encoder.transform(original)
        encoded_name = (
            "y_train_balanced_encoded.npy"
            if split_name == "train"
            else f"y_{split_name}_encoded.npy"
        )
        original_name = (
            "y_train_balanced_original.npy"
            if split_name == "train"
            else f"y_{split_name}_original.npy"
        )
        np.save(split_dir / encoded_name, encoded)
        np.save(split_dir / original_name, original)
    source = pd.read_csv(config.input_file)
    source[config.target_column] = np.concatenate(
        [targets["train"], targets["val"], targets["test"]]
    )
    source.to_csv(config.input_file, index=False)
    joblib.dump(encoder, split_dir / "target_label_encoder.joblib")
    (split_dir / "target_label_mapping.json").write_text(
        json.dumps(
            {str(index): value for index, value in enumerate(encoder.classes_)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return split_dir


def test_evaluate_classification_predictions():
    metrics = evaluate_predictions(
        [0, 1, 0, 1],
        [0, 1, 1, 1],
        y_proba=[[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]],
        task_type="classification",
    )

    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics
    assert "roc_auc" in metrics


def test_evaluate_regression_predictions():
    metrics = evaluate_predictions([1.0, 2.0, 3.0], [1.1, 1.9, 3.2], task_type="regression")

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    assert "r2" in metrics
    assert "mape" in metrics


def test_train_selected_classification_model(tmp_path):
    df = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [value % 5 for value in range(40)],
            "cat": ["a" if value % 2 == 0 else "b" for value in range(40)],
            "target": [0] * 20 + [1] * 20,
        }
    )
    config = make_config(tmp_path)

    result = train_selected_models(df, config)
    model_result = result["results"][0]

    assert model_result["model_name"] == "Logistic Regression"
    assert model_result["status"] == "trained"
    assert "accuracy" in model_result["metrics"]
    assert model_result["predictions"]
    assert model_result["probabilities"]
    assert Path(model_result["artifact_paths"]["model"]).exists()
    assert Path(model_result["artifact_paths"]["preprocessing"]).exists()


def test_train_selected_regression_model(tmp_path):
    df = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [value % 5 for value in range(40)],
            "cat": ["a" if value % 2 == 0 else "b" for value in range(40)],
            "target": [float(value * 2 + 1) for value in range(40)],
        }
    )
    config = make_config(
        tmp_path,
        task_type="regression",
        split_method="random",
        selected_models=["Linear Regression"],
    )

    result = train_selected_models(df, config)
    model_result = result["results"][0]

    assert model_result["model_name"] == "Linear Regression"
    assert model_result["status"] == "trained"
    assert "mae" in model_result["metrics"]
    assert model_result["probabilities"] is None
    assert Path(model_result["artifact_paths"]["model"]).exists()


def test_train_skips_deep_model_for_now(tmp_path):
    df = pd.DataFrame(
        {
            "x1": list(range(40)),
            "x2": [value % 5 for value in range(40)],
            "cat": ["a" if value % 2 == 0 else "b" for value in range(40)],
            "target": [0] * 20 + [1] * 20,
        }
    )
    config = make_config(tmp_path, selected_models=["FT-Transformer"])

    result = train_selected_models(df, config)

    assert result["results"][0]["status"] == "skipped"


def test_train_saved_models_runs_cv_and_saves_requested_outputs(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["logistic_regression"],
        enable_cross_validation=True,
        cv_folds=3,
        model_params={"logistic_regression": {"max_iter": 200}},
    )
    save_training_bundle(tmp_path, config)
    progress = []

    result = train_saved_models(config, progress_callback=progress.append)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "LogisticRegression"
    assert model_result["status"] == "trained"
    assert model_result["saved"] is True
    assert len([item for item in progress if item.get("fold")]) == 3
    assert (output_dir / "trained_model.joblib").exists()
    assert (output_dir / "preprocessing_artifact.joblib").exists()
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert (output_dir / "model_config.json").exists()
    assert (output_dir / "training_metadata.json").exists()
    training_metadata = json.loads(
        (output_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert training_metadata["project_name"] == "trainer-demo"
    assert training_metadata["project_file"].endswith("trainer-demo.avista")
    assert training_metadata["project_file_version"] == "1.0"
    from app.__version__ import (
        APP_DESCRIPTION,
        APP_NAME,
        RELEASE_DATE,
        __version__,
    )

    assert training_metadata["application"] == APP_NAME
    assert training_metadata["application_description"] == APP_DESCRIPTION
    assert training_metadata["application_version"] == __version__
    assert training_metadata["application_release_date"] == RELEASE_DATE
    assert training_metadata["report_footer"]["generated_by"] == APP_NAME
    assert training_metadata["report_footer"]["description"] == APP_DESCRIPTION
    assert training_metadata["report_footer"]["version"] == __version__
    assert training_metadata["report_footer"]["release_date"] == RELEASE_DATE
    assert training_metadata["report_footer"]["generated_on"]
    assert training_metadata["cv_preprocessing_scope"] == "fold_training_only"
    assert training_metadata["cv_imbalance_scope"] == "fold_training_only"
    assert training_metadata["cv_source"] == "original_external_training_partition"
    assert training_metadata["cv_resampling_before_split"] is False
    assert (output_dir / "coefficients.csv").exists()
    assert (output_dir / "odds_ratios.csv").exists()
    for split_name in ("train", "validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
            "metrics.json",
            "misclassified_records.csv",
            "roc_curve.csv",
            "roc_curve.png",
            "roc_curve.pdf",
            "pr_curve.csv",
            "pr_curve.png",
            "pr_curve.pdf",
        ):
            assert (split_output / filename).exists()


def test_saved_training_uses_encoded_targets_and_decodes_exports(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["decision_tree"],
        enable_cross_validation=False,
    )
    save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    assert result["results"][0]["status"] == "trained"
    output_dir = tmp_path / "outputs" / "training" / "DecisionTree" / "test"
    predictions = pd.read_csv(output_dir / "predictions.csv")
    probabilities = pd.read_csv(output_dir / "probabilities.csv")
    confusion = pd.read_csv(output_dir / "confusion_matrix.csv", index_col=0)

    expected_labels = {
        "Advanced_Automation",
        "Assisted_Driving",
        "Partial_Automation",
    }
    assert set(predictions["actual_class"]) == expected_labels
    assert set(predictions["predicted_class"]).issubset(expected_labels)
    assert set(probabilities.columns[1:]) == {
        "prob_Advanced_Automation",
        "prob_Assisted_Driving",
        "prob_Partial_Automation",
    }
    assert set(confusion.columns) == expected_labels
    assert set(confusion.index) == expected_labels


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_mamba_attention_trains_from_saved_encoded_artifacts(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["mamba_attention"],
        enable_cross_validation=False,
        model_params={
            "mamba_attention": {
                "hidden_dim": 8,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "focal_gamma": 1.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "input_dim": 999,
                "num_classes": 999,
            }
        },
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "MambaAttention"
    assert model_result["status"] == "trained"
    assert model_result["saved"] is True
    assert model_result["input_dim"] == np.load(
        split_dir / "X_train_balanced.npy"
    ).shape[1]
    assert model_result["num_classes"] == 3
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "model_config.json").exists()
    assert (output_dir / "training_metadata.json").exists()
    model_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert model_config["input_dim"] == model_result["input_dim"]
    assert model_config["num_classes"] == 3
    assert model_config["input_dim"] != 999
    assert model_config["num_classes"] != 999
    for split_name in ("train", "validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
        ):
            assert (split_output / filename).exists()
    predictions = pd.read_csv(output_dir / "test" / "predictions.csv")
    assert set(predictions["actual_class"]) == {
        "Advanced_Automation",
        "Assisted_Driving",
        "Partial_Automation",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_mamba_attention_saves_cv_outputs_without_fold_images(
    tmp_path,
    monkeypatch,
):
    import torch

    config = make_config(
        tmp_path,
        selected_models=["mamba_attention"],
        enable_cross_validation=True,
        cv_folds=2,
        train_percent=60.0,
        validation_percent=20.0,
        test_percent=20.0,
        model_params={
            "mamba_attention": {
                "hidden_dim": 8,
                "dropout": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
            }
        },
    )
    split_dir = save_training_bundle(tmp_path, config)
    dataset_calls = []
    original_tensor_dataset = torch.utils.data.TensorDataset

    def recording_tensor_dataset(features, targets):
        dataset_calls.append(features.detach().cpu().numpy().copy())
        return original_tensor_dataset(features, targets)

    monkeypatch.setattr(
        torch.utils.data,
        "TensorDataset",
        recording_tensor_dataset,
    )
    progress = []

    result = train_saved_models(config, progress_callback=progress.append)

    output_dir = tmp_path / "outputs" / "training" / "MambaAttention"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert not list(output_dir.glob("fold*/confusion_matrix.*"))
    cv_results = pd.read_csv(output_dir / "cv_results.csv")
    assert set(cv_results["cv_outer_validation_role"]) == {"fold_scoring_only"}
    assert set(cv_results["cv_inner_validation_role"]) == {"early_stopping"}
    assert set(cv_results["cv_inner_preprocessing_scope"]) == {
        "inner_training_only"
    }
    assert set(cv_results["cv_inner_imbalance_scope"]) == {
        "inner_training_only"
    }
    assert not cv_results["cv_inner_validation_resampled"].any()
    assert not cv_results["cv_outer_validation_resampled"].any()
    assert cv_results["cv_early_stopping_enabled"].all()
    assert (cv_results["cv_inner_validation_size"] > 0).all()
    assert (cv_results["cv_epochs_trained"] == 1).all()
    assert (cv_results["cv_best_epoch"] == 1).all()
    metadata = json.loads(
        (output_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["validation_role"] == "checkpoint_selection"
    assert metadata["test_role"] == "final_evaluation"
    assert metadata["cv_outer_validation_role"] == "fold_scoring_only"
    assert metadata["cv_inner_validation_role"] == "early_stopping_when_feasible"
    assert metadata["cv_inner_preprocessing_scope"] == "inner_training_only"
    assert metadata["cv_inner_imbalance_scope"] == "inner_training_only"
    assert metadata["cv_inner_validation_resampled"] is False
    assert metadata["cv_outer_validation_resampled"] is False
    fold_epoch_events = [
        event
        for event in progress
        if event.get("step") == "epoch" and event.get("fold")
    ]
    assert fold_epoch_events
    assert {
        event["validation_role"] for event in fold_epoch_events
    } == {"cv_inner_early_stopping"}
    final_epoch_events = [
        event
        for event in progress
        if event.get("step") == "epoch" and not event.get("fold")
    ]
    assert final_epoch_events
    assert {
        event["validation_role"] for event in final_epoch_events
    } == {"external_validation_checkpoint_selection"}

    assert len(dataset_calls) == 11
    outer_scoring_calls = [dataset_calls[2], dataset_calls[5]]
    inner_early_stopping_calls = [dataset_calls[1], dataset_calls[4]]
    assert all(
        not np.array_equal(outer, inner)
        for outer, inner in zip(outer_scoring_calls, inner_early_stopping_calls)
    )
    np.testing.assert_array_equal(dataset_calls[7], np.load(split_dir / "X_val.npy"))
    np.testing.assert_array_equal(dataset_calls[10], np.load(split_dir / "X_test.npy"))
    assert all(
        not np.array_equal(dataset_calls[10], dataset_calls[index])
        for index in (0, 1, 3, 4, 6, 7)
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_mamba_attention_failure_saves_exact_reason(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["mamba_attention"],
        model_params={
            "mamba_attention": {
                "hidden_dim": 7,
                "epochs": 1,
                "batch_size": 4,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "MambaAttention"
        / "failure_reason.json"
    )
    assert model_result["status"] == "failed"
    assert model_result["status"] != "skipped"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == model_result["error"]
    assert failure["error"]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_ft_transformer_trains_from_saved_encoded_artifacts(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["ft_transformer"],
        enable_cross_validation=False,
        model_params={
            "ft_transformer": {
                "d_token": 8,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "focal_gamma": 1.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "n_features": 999,
                "n_classes": 999,
            }
        },
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "FT-Transformer"
    assert model_result["status"] == "trained"
    assert model_result["saved"] is True
    assert model_result["input_dim"] == np.load(
        split_dir / "X_train_balanced.npy"
    ).shape[1]
    assert model_result["num_classes"] == 3
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "model_config.json").exists()
    assert (output_dir / "training_metadata.json").exists()
    model_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert model_config["input_dim"] != 999
    assert model_config["num_classes"] != 999
    assert model_config["n_features"] == model_result["input_dim"]
    assert model_config["n_classes"] == model_result["num_classes"]
    assert model_config["d_token"] == 8
    for split_name in ("train", "validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
        ):
            assert (split_output / filename).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_ft_transformer_saves_cv_outputs_without_fold_images(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["ft_transformer"],
        enable_cross_validation=True,
        cv_folds=2,
        model_params={
            "ft_transformer": {
                "d_token": 8,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    output_dir = tmp_path / "outputs" / "training" / "FT-Transformer"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert not list(output_dir.glob("fold*/confusion_matrix.*"))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_ft_transformer_failure_saves_exact_reason(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["ft_transformer"],
        model_params={
            "ft_transformer": {
                "d_token": 7,
                "n_heads": 8,
                "epochs": 1,
                "batch_size": 4,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "FT-Transformer"
        / "failure_reason.json"
    )
    assert model_result["status"] == "failed"
    assert model_result["status"] != "skipped"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == model_result["error"]
    assert failure["error"]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_autoint_trains_from_saved_encoded_artifacts(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["autoint"],
        enable_cross_validation=False,
        model_params={
            "autoint": {
                "d": 8,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "focal_gamma": 1.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "n_features": 999,
                "n_classes": 999,
            }
        },
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "AutoInt"
    assert model_result["status"] == "trained"
    assert model_result["saved"] is True
    assert model_result["input_dim"] == np.load(
        split_dir / "X_train_balanced.npy"
    ).shape[1]
    assert model_result["num_classes"] == 3
    assert model_result["n_features"] == model_result["input_dim"]
    assert model_result["n_classes"] == model_result["num_classes"]
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "model_config.json").exists()
    assert (output_dir / "training_metadata.json").exists()
    model_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert model_config["n_features"] != 999
    assert model_config["n_classes"] != 999
    assert model_config["d"] == 8
    for split_name in ("train", "validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
        ):
            assert (split_output / filename).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_autoint_saves_cv_outputs_without_fold_images(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["autoint"],
        enable_cross_validation=True,
        cv_folds=2,
        model_params={
            "autoint": {
                "d": 8,
                "n_heads": 2,
                "n_layers": 1,
                "dropout": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    output_dir = tmp_path / "outputs" / "training" / "AutoInt"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert not list(output_dir.glob("fold*/confusion_matrix.*"))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_autoint_failure_saves_exact_reason(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["autoint"],
        model_params={
            "autoint": {
                "d": 7,
                "n_heads": 4,
                "epochs": 1,
                "batch_size": 4,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "AutoInt"
        / "failure_reason.json"
    )
    assert model_result["status"] == "failed"
    assert model_result["status"] != "skipped"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == model_result["error"]
    assert failure["error"]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_tab_resnet_trains_from_saved_encoded_artifacts(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["tab_resnet"],
        enable_cross_validation=False,
        model_params={
            "tab_resnet": {
                "hidden": 8,
                "n_blocks": 1,
                "dropout": 0.0,
                "learning_rate": 1e-3,
                "focal_gamma": 1.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
                "input_dim": 999,
                "n_classes": 999,
            }
        },
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabResNet"
    assert model_result["status"] == "trained"
    assert model_result["saved"] is True
    assert model_result["input_dim"] == np.load(
        split_dir / "X_train_balanced.npy"
    ).shape[1]
    assert model_result["num_classes"] == 3
    assert model_result["n_classes"] == model_result["num_classes"]
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "model_config.json").exists()
    assert (output_dir / "training_metadata.json").exists()
    model_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert model_config["input_dim"] != 999
    assert model_config["n_classes"] != 999
    assert model_config["hidden"] == 8
    for split_name in ("train", "validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
        ):
            assert (split_output / filename).exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_tab_resnet_saves_cv_outputs_without_fold_images(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["tab_resnet"],
        enable_cross_validation=True,
        cv_folds=2,
        model_params={
            "tab_resnet": {
                "hidden": 8,
                "n_blocks": 1,
                "dropout": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    output_dir = tmp_path / "outputs" / "training" / "TabResNet"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert not list(output_dir.glob("fold*/confusion_matrix.*"))


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_tab_resnet_failure_saves_exact_reason(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["tab_resnet"],
        model_params={
            "tab_resnet": {
                "hidden": 8,
                "n_blocks": 1,
                "dropout": 1.5,
                "epochs": 1,
                "batch_size": 4,
            }
        },
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    model_result = result["results"][0]
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "TabResNet"
        / "failure_reason.json"
    )
    assert model_result["status"] == "failed"
    assert model_result["status"] != "skipped"
    assert failure_path.exists()
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == model_result["error"]
    assert failure["error"]


def test_tabpfn_missing_dependency_creates_skip_reason(tmp_path, monkeypatch):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["tabpfn"],
    )
    save_encoded_string_training_bundle(tmp_path, config)
    real_import = builtins.__import__

    def missing_tabpfn(name, *args, **kwargs):
        if name == "tabpfn":
            raise ImportError("No module named tabpfn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_tabpfn)

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabPFN_2_5"
    expected_reason = "TabPFN skipped: package tabpfn is not installed."
    assert model_result["status"] == "skipped"
    assert model_result["reason"] == expected_reason
    skip = json.loads(
        (output_dir / "skip_reason.json").read_text(encoding="utf-8")
    )
    assert skip["reason"] == expected_reason


def test_packaged_tabpfn_missing_dependency_is_a_failure(
    tmp_path,
    monkeypatch,
):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["tabpfn"],
    )
    save_encoded_string_training_bundle(tmp_path, config)
    real_import = builtins.__import__

    def missing_tabpfn(name, *args, **kwargs):
        if name == "tabpfn":
            raise ImportError("No module named tabpfn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_tabpfn)
    monkeypatch.setattr(
        "app.core.trainer.is_packaged_application",
        lambda: True,
    )

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabPFN_2_5"
    assert model_result["status"] == "failed"
    assert "missing from AVISTADeepWorker.exe" in model_result["error"]
    assert "This is an AVISTA packaging error" in model_result["error"]
    failure = json.loads(
        (output_dir / "failure_reason.json").read_text(encoding="utf-8")
    )
    assert failure["packaged"] is True
    assert failure["tabpfn_import_succeeded"] is False


def test_tabpfn_uses_one_estimator_value_and_internal_batching(tmp_path, monkeypatch):
    n_estimators = 12
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["tabpfn"],
        enable_cross_validation=True,
        cv_folds=2,
        random_state=17,
        imbalance_method="smote",
        model_params={"tabpfn": {"n_estimators": n_estimators}},
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)
    np.save(split_dir / "X_train_balanced.npy", np.full((12, 9), -999.0))

    class FakeTabPFNClassifier:
        instances = []
        prediction_batch_sizes = []
        fit_frames = []
        prediction_frames = []

        def __init__(
            self,
            n_estimators,
            model_path,
            device,
            categorical_features_indices,
            random_state,
        ):
            self.n_estimators = n_estimators
            self.model_path = model_path
            self.device = device
            self.categorical_features_indices = categorical_features_indices
            self.random_state = random_state
            self.fit_size = 0
            self.__class__.instances.append(self)

        def fit(self, features, targets):
            assert isinstance(features, pd.DataFrame)
            self.__class__.fit_frames.append(features.copy())
            self.fit_size = len(features)
            self.classes_ = np.unique(targets)
            return self

        def predict_proba(self, features):
            assert isinstance(features, pd.DataFrame)
            self.__class__.prediction_frames.append(features.copy())
            self.__class__.prediction_batch_sizes.append(len(features))
            probabilities = np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )
            return probabilities

    fake_module = types.ModuleType("tabpfn")
    fake_module.__version__ = "8.0.8"
    fake_module.TabPFNClassifier = FakeTabPFNClassifier
    monkeypatch.setitem(sys.modules, "tabpfn", fake_module)
    checkpoint = (
        tmp_path
        / "cache"
        / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"mock checkpoint")
    monkeypatch.setattr(
        trainer_module,
        "resolve_tabpfn_checkpoint",
        lambda: checkpoint.resolve(),
    )
    monkeypatch.setattr(
        trainer_module,
        "get_tabpfn_model_status",
        lambda: types.SimpleNamespace(active_checkpoint_source="user_cache"),
    )
    monkeypatch.setattr(
        trainer_module,
        "_prepare_cv_fold",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("TabPFN must not use generic preprocessing/resampling")
        ),
    )

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabPFN_2_5"
    assert model_result["status"] == "trained"
    assert model_result["subset_size"] == 12
    assert [instance.n_estimators for instance in FakeTabPFNClassifier.instances] == [
        n_estimators,
        n_estimators,
        n_estimators,
    ]
    assert {
        instance.model_path for instance in FakeTabPFNClassifier.instances
    } == {str(checkpoint.resolve())}
    assert {
        instance.device for instance in FakeTabPFNClassifier.instances
    }.issubset({"cpu", "cuda"})
    assert all(size <= 500 for size in FakeTabPFNClassifier.prediction_batch_sizes)
    assert all(list(frame.columns) == ["x1", "x2", "cat"] for frame in FakeTabPFNClassifier.fit_frames)
    assert all("cat_a" not in frame.columns for frame in FakeTabPFNClassifier.fit_frames)
    assert all(frame["cat"].isin(["a", "b"]).all() for frame in FakeTabPFNClassifier.fit_frames)
    assert [instance.categorical_features_indices for instance in FakeTabPFNClassifier.instances] == [[2], [2], [2]]
    assert [instance.random_state for instance in FakeTabPFNClassifier.instances] == [17, 17, 17]
    saved_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert saved_config["n_estimators"] == n_estimators
    assert saved_config["model_path"] == str(checkpoint.resolve())
    assert saved_config["cv_preprocessing_scope"] == "fold_training_only"
    assert saved_config["cv_imbalance_scope"] == "fold_training_only"
    assert saved_config["cv_source"] == "original_external_training_partition"
    assert saved_config["cv_resampling_before_split"] is False
    training_metadata = json.loads(
        (output_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert training_metadata["tabpfn_checkpoint_source"] == "user_cache"
    assert training_metadata["tabpfn_checkpoint_path"] == str(checkpoint.resolve())
    assert training_metadata["tabpfn_input_representation"] == "raw_dataframe"
    assert training_metadata["tabpfn_external_one_hot_encoding"] is False
    assert training_metadata["tabpfn_external_scaling"] is False
    assert training_metadata["tabpfn_external_resampling_applied"] is False
    assert training_metadata["project_imbalance_method"] == "smote"
    assert training_metadata["available_training_rows"] == 12
    assert training_metadata["effective_training_rows"] == 12
    assert training_metadata["training_row_limit"] == 50_000
    assert training_metadata["training_subsampled"] is False
    assert [len(frame) for frame in FakeTabPFNClassifier.fit_frames] == [6, 6, 12]
    source = pd.read_csv(config.input_file)
    for frame in FakeTabPFNClassifier.fit_frames:
        assert set(frame.index).issubset(set(source.index[:12]))
    raw_training_labels = LabelEncoder().fit_transform(source.loc[:11, "target"])
    expected_folds = list(
        StratifiedKFold(n_splits=2, shuffle=True, random_state=17).split(
            np.zeros(12),
            raw_training_labels,
        )
    )
    for frame, validation_frame, (train_positions, validation_positions) in zip(
        FakeTabPFNClassifier.fit_frames[:2],
        FakeTabPFNClassifier.prediction_frames[:2],
        expected_folds,
    ):
        assert set(frame.index) == set(train_positions)
        assert set(validation_frame.index) == set(validation_positions)
        assert set(frame.index).isdisjoint(validation_frame.index)
    cv_rows = pd.read_csv(output_dir / "cv_results.csv")
    assert list(cv_rows["available_training_rows"]) == [6, 6]
    assert list(cv_rows["effective_training_rows"]) == [6, 6]
    assert not cv_rows["training_subsampled"].any()
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert (
        (output_dir / "trained_model.joblib").exists()
        or (output_dir / "model_not_serialized_reason.json").exists()
    )
    for split_name in ("validation", "test"):
        split_output = output_dir / split_name
        for filename in (
            "metrics.json",
            "classification_report.csv",
            "confusion_matrix.csv",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
            "predictions.csv",
            "probabilities.csv",
        ):
            assert (split_output / filename).exists()
    predictions = pd.read_csv(output_dir / "test" / "predictions.csv")
    assert set(predictions["actual_class"]) == {
        "Advanced_Automation",
        "Assisted_Driving",
        "Partial_Automation",
    }


def test_tabpfn_row_limit_keeps_all_rows_between_3000_and_50000():
    X = pd.DataFrame({"value": np.arange(5_000), "category": ["a", "b"] * 2_500})
    y = np.array([0, 1] * 2_500)

    selected_X, selected_y, metadata = prepare_tabpfn_training_rows(
        X,
        y,
        random_state=17,
    )

    assert selected_X is X
    assert np.array_equal(selected_y, y)
    assert metadata == {
        "available_training_rows": 5_000,
        "effective_training_rows": 5_000,
        "training_subsampled": False,
        "training_subsampling_strategy": "none",
        "training_subsampling_seed": 17,
        "training_row_limit": 50_000,
    }


def test_tabpfn_fit_receives_all_5000_raw_rows(tmp_path, monkeypatch):
    fit_frames = []

    class FakeTabPFNClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, features, targets):
            fit_frames.append(features.copy())
            self.classes_ = np.unique(targets)
            return self

        def predict_proba(self, features):
            return np.full((len(features), 2), 0.5)

    fake_module = types.ModuleType("tabpfn")
    fake_module.__version__ = "8.0.8"
    fake_module.TabPFNClassifier = FakeTabPFNClassifier
    monkeypatch.setitem(sys.modules, "tabpfn", fake_module)

    config = make_config(
        tmp_path,
        selected_models=["tabpfn"],
        random_state=19,
    )
    raw_categories = ["a", "b"] * 2_500
    raw_categories[0] = None
    raw_train = pd.DataFrame(
        {
            "x1": np.arange(5_000),
            "x2": np.arange(5_000) % 7,
            "cat": raw_categories,
            "target": [0, 1] * 2_500,
        }
    )
    raw_validation = raw_train.iloc[:4].copy()
    raw_test = raw_train.iloc[4:8].copy()
    data = {
        "cv_raw_train": raw_train,
        "cv_y_train": raw_train["target"].to_numpy(),
        "raw_validation": raw_validation,
        "raw_y_validation": raw_validation["target"].to_numpy(),
        "raw_test": raw_test,
        "raw_y_test": raw_test["target"].to_numpy(),
        "class_labels": [0, 1],
        "target_encoder": None,
        "split_metadata": {
            "validation_index": raw_validation.index.tolist(),
            "test_index": raw_test.index.tolist(),
        },
    }

    result, _ = trainer_module._train_saved_tabpfn(
        config,
        data,
        tmp_path / "unused",
        False,
        None,
        None,
        0,
        4,
    )

    assert result["status"] == "trained"
    assert result["available_training_rows"] == 5_000
    assert result["effective_training_rows"] == 5_000
    assert result["training_subsampled"] is False
    assert len(fit_frames) == 1
    assert len(fit_frames[0]) == 5_000
    assert list(fit_frames[0].columns) == ["x1", "x2", "cat"]
    assert fit_frames[0]["cat"].isna().sum() == 1


def test_tabpfn_row_limit_stratifies_exactly_50000_reproducibly():
    row_count = 60_013
    y = np.concatenate(
        [
            np.zeros(42_000, dtype=int),
            np.ones(18_012, dtype=int),
            np.array([2]),
        ]
    )
    X = pd.DataFrame({"row_id": np.arange(row_count)})

    first_X, first_y, metadata = prepare_tabpfn_training_rows(
        X,
        y,
        random_state=42,
    )
    repeated_X, repeated_y, _ = prepare_tabpfn_training_rows(
        X,
        y,
        random_state=42,
    )
    different_X, _, _ = prepare_tabpfn_training_rows(
        X,
        y,
        random_state=43,
    )

    assert len(first_X) == len(first_y) == 50_000
    assert np.array_equal(first_X.index, repeated_X.index)
    assert np.array_equal(first_y, repeated_y)
    assert not np.array_equal(first_X.index, different_X.index)
    assert set(np.unique(first_y)) == {0, 1, 2}
    original_share = np.mean(y == 1)
    sampled_share = np.mean(first_y == 1)
    assert abs(original_share - sampled_share) < 0.001
    assert metadata["training_subsampled"] is True
    assert metadata["training_subsampling_strategy"] == "stratified"
    assert metadata["effective_training_rows"] == 50_000


def test_tabpfn_limits_report_clear_errors():
    with pytest.raises(ValueError, match="at most 2,000 input features"):
        _validate_tabpfn_input(
            pd.DataFrame(np.zeros((20, 2_001))),
            np.array([0, 1] * 10),
            context="test data",
        )
    with pytest.raises(ValueError, match="at most 10 target classes"):
        _validate_tabpfn_input(
            pd.DataFrame({"x": range(22)}),
            np.tile(np.arange(11), 2),
            context="test data",
        )


def test_tabpfn_missing_user_cache_checkpoint_saves_failure_reason(
    tmp_path,
    monkeypatch,
):
    missing_checkpoint = tmp_path / "missing.ckpt"
    config = make_config(
        tmp_path,
        task_type="classification",
        selected_models=["tabpfn"],
        model_params={"tabpfn": {"n_estimators": 8}},
    )
    save_encoded_string_training_bundle(tmp_path, config)
    fake_module = types.ModuleType("tabpfn")
    fake_module.TabPFNClassifier = object
    monkeypatch.setitem(sys.modules, "tabpfn", fake_module)
    monkeypatch.setattr(
        "app.core.trainer.resolve_tabpfn_checkpoint",
        lambda: (_ for _ in ()).throw(
            FileNotFoundError("missing checkpoint")
        ),
    )
    monkeypatch.setattr(
        "app.core.trainer.get_tabpfn_model_status",
        lambda: types.SimpleNamespace(
            cache_path=missing_checkpoint,
            active_checkpoint_path=None,
            active_checkpoint_source="unavailable",
        ),
    )

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabPFN_2_5"
    expected = (
        "TabPFN 2.5 is not currently available. Use Model Selection or "
        "Help > TabPFN Model Status to set it up."
    )
    assert model_result["status"] == "failed"
    assert model_result["error"] == expected
    failure = json.loads(
        (output_dir / "failure_reason.json").read_text(encoding="utf-8")
    )
    assert failure["error"] == expected
    assert failure["model_path"] == str(missing_checkpoint)


@pytest.mark.skipif(
    importlib.util.find_spec("xgboost") is None,
    reason="xgboost is not installed",
)
def test_xgboost_trains_with_central_encoded_string_target(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["xgboost"],
        enable_cross_validation=False,
        model_params={
            "xgboost": {
                "n_estimators": 5,
                "max_depth": 2,
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_jobs": 1,
            }
        },
    )
    save_encoded_string_training_bundle(tmp_path, config)

    result = train_saved_models(config, save_outputs=False)

    assert result["results"][0]["status"] == "trained"


def test_train_saved_models_blocks_cv_when_class_count_is_too_small(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["decision_tree"],
        enable_cross_validation=True,
        cv_folds=7,
    )
    save_training_bundle(tmp_path, config)

    try:
        train_saved_models(config)
    except ValueError as exc:
        assert "has only 6 samples but CV folds = 7" in str(exc)
    else:
        raise AssertionError("Expected invalid CV folds to block training.")


def test_cv_smote_runs_independently_on_each_fold_training_partition(
    tmp_path,
    monkeypatch,
):
    imblearn = pytest.importorskip("imblearn.over_sampling")
    config = make_config(
        tmp_path,
        imbalance_method="smote",
        smote_ratio_preset="moderate",
        preprocessing_options={"imbalance": {"smote_k_neighbors": 5}},
    )
    target = np.array([0] * 15 + [1] * 6)
    raw = pd.DataFrame(
        {
            "x1": np.arange(len(target)),
            "x2": np.arange(len(target)) % 4,
            "cat": ["a", "b", "c"] * 7,
            "target": target,
        }
    )
    data = {"cv_raw_train": raw, "cv_y_train": target, "target_encoder": None}
    calls = []
    original_fit_resample = imblearn.SMOTE.fit_resample

    def record_fit_resample(self, features, labels):
        calls.append((len(features), np.asarray(labels).copy()))
        return original_fit_resample(self, features, labels)

    monkeypatch.setattr(imblearn.SMOTE, "fit_resample", record_fit_resample)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_results = []
    for train_pos, validation_pos in folds.split(np.zeros(len(target)), target):
        fold_results.append(_prepare_cv_fold(data, config, train_pos, validation_pos))

    assert len(calls) == 3
    assert all(call_size == 14 for call_size, _ in calls)
    assert all(len(result["y_validation"]) == 7 for result in fold_results)
    assert all(len(result["y_train"]) > 14 for result in fold_results)
    assert all(
        set(result["train_row_indices"]).isdisjoint(
            result["validation_row_indices"]
        )
        for result in fold_results
    )


def test_cv_preprocessing_does_not_learn_validation_only_category(tmp_path):
    config = make_config(tmp_path)
    raw = pd.DataFrame(
        {
            "x1": [0, 1, 2, 3, 4, 100],
            "x2": [0, 1, 0, 1, 0, 1],
            "cat": ["a", "b", "a", "b", "a", "validation-only"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    data = {
        "cv_raw_train": raw,
        "cv_y_train": raw["target"].to_numpy(),
        "target_encoder": None,
    }

    fold = _prepare_cv_fold(data, config, np.arange(5), np.array([5]))

    learned_categories = set(fold["artifacts"].encoder.categories_[0])
    assert "validation-only" not in learned_categories
    categorical_indices = [
        index
        for index, name in enumerate(fold["artifacts"].output_feature_names)
        if name.startswith("cat_")
    ]
    assert categorical_indices
    assert np.all(fold["X_validation"][0, categorical_indices] == 0)


def test_cv_uses_raw_training_rows_without_balancing_and_preserves_holdouts(
    tmp_path,
    monkeypatch,
):
    config = make_config(
        tmp_path,
        selected_models=["decision_tree"],
        enable_cross_validation=True,
        cv_folds=3,
        imbalance_method="none",
    )
    split_dir = save_training_bundle(tmp_path, config)
    sentinel = 123456.0
    balanced = np.full_like(np.load(split_dir / "X_train_balanced.npy"), sentinel)
    np.save(split_dir / "X_train_balanced.npy", balanced)
    validation_before = np.load(split_dir / "X_val.npy").copy()
    test_before = np.load(split_dir / "X_test.npy").copy()
    fit_inputs = []
    preprocessing_calls = []
    original_preprocessing = trainer_module.fit_split_preprocessing

    class RecordingClassifier:
        def fit(self, features, labels):
            fit_inputs.append(np.asarray(features).copy())
            self.classes_ = np.unique(labels)
            return self

        def predict(self, features):
            return np.full(len(features), self.classes_[0])

        def predict_proba(self, features):
            probabilities = np.zeros((len(features), len(self.classes_)))
            probabilities[:, 0] = 1.0
            return probabilities

    def record_preprocessing(frame, fold_config):
        preprocessing_calls.append(frame.index.tolist())
        return original_preprocessing(frame, fold_config)

    monkeypatch.setattr(trainer_module, "fit_split_preprocessing", record_preprocessing)
    monkeypatch.setattr(
        trainer_module,
        "create_sklearn_model",
        lambda *args, **kwargs: RecordingClassifier(),
    )

    train_saved_models(config, save_outputs=False)

    assert len(preprocessing_calls) == 3
    assert all(set(indices).issubset(set(range(12))) for indices in preprocessing_calls)
    assert all(sentinel not in features for features in fit_inputs[:3])
    assert np.all(fit_inputs[-1] == sentinel)
    np.testing.assert_array_equal(np.load(split_dir / "X_val.npy"), validation_before)
    np.testing.assert_array_equal(np.load(split_dir / "X_test.npy"), test_before)


def test_cv_feasibility_uses_original_labels_before_resampling(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["decision_tree"],
        enable_cross_validation=True,
        cv_folds=4,
        imbalance_method="smote",
    )
    split_dir = save_training_bundle(tmp_path, config)
    source = pd.read_csv(config.input_file)
    source.loc[:11, "target"] = [0] * 9 + [1] * 3
    source.to_csv(config.input_file, index=False)
    np.save(split_dir / "y_train.npy", np.array([0] * 9 + [1] * 3))
    balanced_y = np.array([0] * 9 + [1] * 9)
    original_X = np.load(split_dir / "X_train_balanced.npy")
    np.save(split_dir / "X_train_balanced.npy", np.tile(original_X, (2, 1))[:18])
    np.save(split_dir / "y_train_balanced.npy", balanced_y)

    with pytest.raises(ValueError, match="only 3 samples but CV folds = 4"):
        train_saved_models(config, save_outputs=False)


def test_deep_cv_inner_split_is_disjoint_outer_safe_and_reproducible(tmp_path):
    config = make_config(tmp_path, train_percent=70.0, validation_percent=10.0)
    fraction = _deep_cv_inner_validation_fraction(config)
    targets = np.array([0, 1] * 10)
    outer_training_rows = np.arange(100, 120)
    outer_validation_rows = set(range(200, 210))

    first = _deep_cv_inner_split(targets, fraction, random_state=37)
    second = _deep_cv_inner_split(targets, fraction, random_state=37)

    assert fraction == pytest.approx(0.125)
    assert first["early_stopping_enabled"] is True
    np.testing.assert_array_equal(
        first["train_positions"],
        second["train_positions"],
    )
    np.testing.assert_array_equal(
        first["validation_positions"],
        second["validation_positions"],
    )
    inner_training_rows = set(
        outer_training_rows[first["train_positions"]]
    )
    inner_validation_rows = set(
        outer_training_rows[first["validation_positions"]]
    )
    assert inner_training_rows.isdisjoint(inner_validation_rows)
    assert inner_training_rows.isdisjoint(outer_validation_rows)
    assert inner_validation_rows.isdisjoint(outer_validation_rows)
    assert inner_training_rows | inner_validation_rows == set(outer_training_rows)
    assert set(targets[first["train_positions"]]) == {0, 1}
    assert set(targets[first["validation_positions"]]) == {0, 1}


def test_deep_cv_inner_split_disables_early_stopping_when_infeasible():
    targets = np.array([0, 0, 0, 1])

    result = _deep_cv_inner_split(targets, 0.25, random_state=42)

    assert result["early_stopping_enabled"] is False
    np.testing.assert_array_equal(result["train_positions"], np.arange(4))
    assert result["validation_positions"].size == 0
    assert "fewer than two rows" in result["disabled_reason"]


def test_deep_cv_inner_split_precedes_preprocessing_and_resampling(
    tmp_path,
    monkeypatch,
):
    config = make_config(
        tmp_path,
        imbalance_method="smote",
        smote_ratio_preset="moderate",
        preprocessing_options={"imbalance": {"smote_k_neighbors": 5}},
    )
    targets = np.array([0] * 20 + [1] * 8 + [0, 1] * 4)
    raw = pd.DataFrame(
        {
            "x1": np.arange(len(targets), dtype=float),
            "x2": np.arange(len(targets)) % 3,
            "cat": ["a", "b"] * 18,
            "target": targets,
        }
    )
    data = {
        "cv_raw_train": raw,
        "cv_y_train": targets,
        "target_encoder": None,
    }
    train_pos = np.arange(28)
    outer_validation_pos = np.arange(28, 36)
    expected_inner = _deep_cv_inner_split(targets[train_pos], 0.25, 19)
    calls = []
    original_apply = trainer_module.apply_imbalance_strategy

    def record_apply(features, labels, artifacts, fold_config):
        calls.append((features.copy(), np.asarray(labels).copy()))
        return original_apply(features, labels, artifacts, fold_config)

    monkeypatch.setattr(trainer_module, "apply_imbalance_strategy", record_apply)

    fold = _prepare_deep_cv_fold(
        data,
        config,
        train_pos,
        outer_validation_pos,
        validation_fraction=0.25,
        random_state=19,
    )

    expected_train_rows = set(train_pos[expected_inner["train_positions"]])
    expected_inner_validation_rows = set(
        train_pos[expected_inner["validation_positions"]]
    )
    assert set(fold["inner_train_row_indices"]) == expected_train_rows
    assert set(fold["inner_validation_row_indices"]) == expected_inner_validation_rows
    assert set(fold["outer_validation_row_indices"]) == set(outer_validation_pos)
    assert expected_train_rows.isdisjoint(expected_inner_validation_rows)
    assert expected_train_rows.isdisjoint(set(outer_validation_pos))
    assert expected_inner_validation_rows.isdisjoint(set(outer_validation_pos))
    assert len(calls) == 1
    assert len(calls[0][1]) == len(expected_train_rows)
    np.testing.assert_array_equal(
        calls[0][1],
        targets[train_pos][expected_inner["train_positions"]],
    )


def test_deep_cv_smote_never_resamples_inner_or_outer_validation(tmp_path):
    config = make_config(
        tmp_path,
        imbalance_method="smote",
        smote_ratio_preset="moderate",
        preprocessing_options={"imbalance": {"smote_k_neighbors": 5}},
    )
    targets = np.array([0] * 20 + [1] * 8 + [0, 1] * 4)
    raw = pd.DataFrame(
        {
            "x1": np.arange(len(targets), dtype=float),
            "x2": np.arange(len(targets)) % 3,
            "cat": ["a", "b"] * 18,
            "target": targets,
        }
    )
    data = {
        "cv_raw_train": raw,
        "cv_y_train": targets,
        "target_encoder": None,
    }
    fold = _prepare_deep_cv_fold(
        data,
        config,
        np.arange(28),
        np.arange(28, 36),
        validation_fraction=0.25,
        random_state=19,
    )

    assert len(fold["y_train"]) > len(fold["inner_train_row_indices"])
    assert len(fold["y_inner_validation"]) == len(
        fold["inner_validation_row_indices"]
    )
    assert len(fold["y_outer_validation"]) == len(
        fold["outer_validation_row_indices"]
    )
    expected_inner_validation = transform_split_features(
        raw.loc[fold["inner_validation_row_indices"]],
        fold["artifacts"],
    ).to_numpy()
    expected_outer_validation = transform_split_features(
        raw.loc[fold["outer_validation_row_indices"]],
        fold["artifacts"],
    ).to_numpy()
    np.testing.assert_array_equal(
        fold["X_inner_validation"],
        expected_inner_validation,
    )
    np.testing.assert_array_equal(
        fold["X_outer_validation"],
        expected_outer_validation,
    )


def test_deep_cv_random_oversampling_cannot_duplicate_inner_validation(tmp_path):
    config = make_config(tmp_path, imbalance_method="random_oversample")
    targets = np.array([0] * 20 + [1] * 8 + [0, 1] * 4)
    raw = pd.DataFrame(
        {
            "x1": np.arange(len(targets), dtype=float),
            "x2": np.arange(len(targets)) % 3,
            "cat": ["a", "b"] * 18,
            "target": targets,
        }
    )
    fold = _prepare_deep_cv_fold(
        {
            "cv_raw_train": raw,
            "cv_y_train": targets,
            "target_encoder": None,
        },
        config,
        np.arange(28),
        np.arange(28, 36),
        validation_fraction=0.25,
        random_state=19,
    )

    resampled_train_ids = fold["X_train"][:, 0]
    inner_validation_ids = fold["X_inner_validation"][:, 0]
    assert len(resampled_train_ids) > len(np.unique(resampled_train_ids))
    assert set(resampled_train_ids).isdisjoint(set(inner_validation_ids))
    assert set(inner_validation_ids) == set(fold["inner_validation_row_indices"])


def test_deep_cv_preprocessing_fits_only_raw_inner_training(tmp_path):
    config = make_config(tmp_path, imbalance_method="none")
    targets = np.array([0, 1] * 15)
    train_pos = np.arange(24)
    outer_validation_pos = np.arange(24, 30)
    inner = _deep_cv_inner_split(targets[train_pos], 0.25, 23)
    raw = pd.DataFrame(
        {
            "x1": np.arange(len(targets), dtype=float),
            "x2": np.arange(len(targets)) % 3,
            "cat": ["inner-training"] * len(targets),
            "target": targets,
        }
    )
    raw.loc[train_pos[inner["validation_positions"]], "cat"] = "inner-validation"
    raw.loc[outer_validation_pos, "cat"] = "outer-validation"

    fold = _prepare_deep_cv_fold(
        {
            "cv_raw_train": raw,
            "cv_y_train": targets,
            "target_encoder": None,
        },
        config,
        train_pos,
        outer_validation_pos,
        validation_fraction=0.25,
        random_state=23,
    )

    learned = set(fold["artifacts"].encoder.categories_[0])
    assert "inner-training" in learned
    assert "inner-validation" not in learned
    assert "outer-validation" not in learned
    categorical_indices = [
        index
        for index, name in enumerate(fold["artifacts"].output_feature_names)
        if name.startswith("cat_")
    ]
    assert categorical_indices
    assert np.all(fold["X_inner_validation"][:, categorical_indices] == 0)
    assert np.all(fold["X_outer_validation"][:, categorical_indices] == 0)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_deep_cv_small_class_fallback_never_reuses_outer_validation(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["mamba_attention"],
        enable_cross_validation=True,
        cv_folds=2,
        model_params={
            "mamba_attention": {
                "hidden_dim": 8,
                "dropout": 0.0,
                "batch_size": 4,
                "epochs": 1,
                "warmup_epochs": 1,
                "early_stopping_patience": 1,
            }
        },
    )
    split_dir = save_training_bundle(tmp_path, config)
    source = pd.read_csv(config.input_file)
    source.loc[:11, "target"] = [0] * 10 + [1] * 2
    source.to_csv(config.input_file, index=False)
    original_target = np.array([0] * 10 + [1] * 2)
    np.save(split_dir / "y_train.npy", original_target)
    np.save(split_dir / "y_train_balanced.npy", original_target)
    progress = []

    result = train_saved_models(config, progress_callback=progress.append)

    assert result["results"][0]["status"] == "trained"
    output_dir = tmp_path / "outputs" / "training" / "MambaAttention"
    cv_results = pd.read_csv(output_dir / "cv_results.csv")
    assert not cv_results["cv_early_stopping_enabled"].any()
    assert set(cv_results["cv_inner_validation_role"]) == {"disabled"}
    assert (cv_results["cv_inner_validation_size"] == 0).all()
    assert (cv_results["cv_epochs_trained"] == 1).all()
    assert cv_results["cv_best_epoch"].isna().all()
    fold_epochs = [
        event
        for event in progress
        if event.get("step") == "epoch" and event.get("fold")
    ]
    assert fold_epochs
    assert all(event["validation_role"] == "none" for event in fold_epochs)
    assert all(event["val_macro_f1"] is None for event in fold_epochs)
    final_epochs = [
        event
        for event in progress
        if event.get("step") == "epoch" and not event.get("fold")
    ]
    assert final_epochs
    assert all(
        event["validation_role"] == "external_validation_checkpoint_selection"
        for event in final_epochs
    )


def test_train_saved_tree_saves_feature_importance(tmp_path):
    config = make_config(
        tmp_path,
        selected_models=["decision_tree"],
        enable_cross_validation=False,
    )
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    output_dir = tmp_path / "outputs" / "training" / "DecisionTree"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "feature_importance.csv").exists()
    assert (output_dir / "feature_importance.png").exists()
    assert (output_dir / "feature_importance.pdf").exists()


def test_train_saved_models_honors_cancellation(tmp_path):
    config = make_config(tmp_path, selected_models=["decision_tree"])
    save_training_bundle(tmp_path, config)

    try:
        train_saved_models(config, should_cancel=lambda: True)
    except TrainingCancelled:
        pass
    else:
        raise AssertionError("Expected cancellation to stop saved-artifact training.")


def test_train_saved_models_infers_string_target_as_classification(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["decision_tree"],
    )
    split_dir = save_training_bundle(tmp_path, config)
    string_targets = {
        "y_train_balanced.npy": np.array(["Minor", "Severe"] * 6),
        "y_val.npy": np.array(["Minor", "Severe"] * 2),
        "y_test.npy": np.array(["Minor", "Severe"] * 2),
    }
    for filename, values in string_targets.items():
        np.save(split_dir / filename, values)

    result = train_saved_models(config, save_outputs=False)

    assert result["results"][0]["status"] == "trained"
    log_text = (tmp_path / "logs" / "training_log.txt").read_text(encoding="utf-8")
    assert "current task_type=auto" in log_text
    assert "y_train dtype=<U6" in log_text
    assert "unique target classes=2" in log_text
    assert "detected target type=classification" in log_text


def test_train_saved_models_logs_continuous_target_before_blocking(tmp_path):
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["decision_tree"],
    )
    split_dir = save_training_bundle(tmp_path, config)
    np.save(split_dir / "y_train_balanced.npy", np.linspace(0.1, 11.7, 12))
    np.save(split_dir / "y_val.npy", np.linspace(12.1, 15.7, 4))
    np.save(split_dir / "y_test.npy", np.linspace(16.1, 19.7, 4))

    try:
        train_saved_models(config, save_outputs=False)
    except ValueError as exc:
        message = str(exc)
        assert "Training blocked." in message
        assert "Current task_type=auto" in message
        assert "Target=target" in message
        assert "Saved target=target" in message
        assert "Detected target type=regression" in message
    else:
        raise AssertionError("Expected continuous numeric target to block classification fitting.")

    log_text = (tmp_path / "logs" / "training_log.txt").read_text(encoding="utf-8")
    assert "current target column=target" in log_text
    assert "saved target column=target" in log_text
    assert "saved task_type=auto" in log_text
    assert "y_train dtype=float64" in log_text
    assert "unique target classes=12" in log_text
    assert "detected target type=regression" in log_text
