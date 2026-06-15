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
from sklearn.preprocessing import LabelEncoder

from app.core.evaluator import evaluate_predictions
from app.core.preprocessing import build_preprocessing_pipeline, save_artifacts
from app.core.project_config import ProjectConfig
from app.core.trainer import TrainingCancelled, train_saved_models, train_selected_models


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
    from app.__version__ import APP_NAME, __version__

    assert training_metadata["application"] == APP_NAME
    assert training_metadata["application_version"] == __version__
    assert training_metadata["report_footer"]["generated_by"] == APP_NAME
    assert training_metadata["report_footer"]["version"] == __version__
    assert training_metadata["report_footer"]["generated_on"]
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
def test_mamba_attention_saves_cv_outputs_without_fold_images(tmp_path):
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
    save_training_bundle(tmp_path, config)

    result = train_saved_models(config)

    output_dir = tmp_path / "outputs" / "training" / "MambaAttention"
    assert result["results"][0]["status"] == "trained"
    assert (output_dir / "cv_results.csv").exists()
    assert (output_dir / "cv_summary.json").exists()
    assert not list(output_dir.glob("fold*/confusion_matrix.*"))


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


def test_tabpfn_uses_one_estimator_value_and_internal_batching(tmp_path, monkeypatch):
    n_estimators = 12
    config = make_config(
        tmp_path,
        task_type="auto",
        selected_models=["tabpfn"],
        enable_cross_validation=True,
        cv_folds=2,
        random_state=17,
        model_params={"tabpfn": {"n_estimators": n_estimators}},
    )
    split_dir = save_encoded_string_training_bundle(tmp_path, config)
    test_features = np.load(split_dir / "X_test.npy")
    test_targets = np.load(split_dir / "y_test_encoded.npy")
    test_labels = np.load(
        split_dir / "y_test_original.npy",
        allow_pickle=True,
    )
    np.save(split_dir / "X_test.npy", np.tile(test_features, (126, 1)))
    np.save(split_dir / "y_test_encoded.npy", np.tile(test_targets, 126))
    np.save(split_dir / "y_test_original.npy", np.tile(test_labels, 126))

    class FakeTabPFNClassifier:
        instances = []
        prediction_batch_sizes = []

        def __init__(self, n_estimators, model_path):
            self.n_estimators = n_estimators
            self.model_path = model_path
            self.fit_size = 0
            self.__class__.instances.append(self)

        def fit(self, features, targets):
            self.fit_size = len(features)
            self.classes_ = np.unique(targets)
            return self

        def predict_proba(self, features):
            self.__class__.prediction_batch_sizes.append(len(features))
            probabilities = np.full(
                (len(features), len(self.classes_)),
                1.0 / len(self.classes_),
            )
            return probabilities

    fake_module = types.ModuleType("tabpfn")
    fake_module.TabPFNClassifier = FakeTabPFNClassifier
    monkeypatch.setitem(sys.modules, "tabpfn", fake_module)

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
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "assets"
        / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    assert {
        instance.model_path for instance in FakeTabPFNClassifier.instances
    } == {str(checkpoint.resolve())}
    assert all(size <= 500 for size in FakeTabPFNClassifier.prediction_batch_sizes)
    assert 500 in FakeTabPFNClassifier.prediction_batch_sizes
    saved_config = json.loads(
        (output_dir / "model_config.json").read_text(encoding="utf-8")
    )
    assert saved_config == {
        "n_estimators": n_estimators,
        "model_path": str(checkpoint.resolve()),
    }
    training_metadata = json.loads(
        (output_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    assert training_metadata["tabpfn_checkpoint_source"] == "bundled_app_asset"
    assert training_metadata["tabpfn_checkpoint_path"] == str(checkpoint.resolve())
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


def test_tabpfn_missing_bundled_checkpoint_saves_failure_reason(
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
        "app.core.trainer.get_app_resource_path",
        lambda *_args, **_kwargs: missing_checkpoint,
    )

    result = train_saved_models(config)

    model_result = result["results"][0]
    output_dir = tmp_path / "outputs" / "training" / "TabPFN_2_5"
    expected = "Bundled TabPFN checkpoint not found in app/assets."
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
