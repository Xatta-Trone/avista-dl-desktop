import json

import numpy as np
import pandas as pd

from app.core.edge_case_checker import run_edge_case_checks, run_saved_edge_case_checks
from app.core.error_handler import EdgeCaseReport, Issue
from app.core.project_config import ProjectConfig


def make_config(tmp_path, **overrides):
    values = {
        "project_name": "edge-demo",
        "project_dir": str(tmp_path),
        "input_file": str(tmp_path / "data.csv"),
        "output_dir": str(tmp_path / "outputs"),
        "target_column": "target",
        "feature_columns": ["feature"],
        "task_type": "classification",
        "split_method": "random",
        "imbalance_method": "none",
        "selected_models": [],
        "environment_mode": "base",
    }
    values.update(overrides)
    return ProjectConfig(**values)


def save_confirmed_artifacts(tmp_path, *, target_column="target"):
    data_dir = tmp_path / "data"
    split_dir = tmp_path / "outputs" / "data_split"
    data_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": range(10), "target": [0, 1] * 5}).to_csv(
        data_dir / "modeling_subset.csv",
        index=False,
    )
    (split_dir / "split_indices.json").write_text(
        json.dumps({"target_column": target_column}),
        encoding="utf-8",
    )
    (split_dir / "imbalance_config.json").write_text(
        json.dumps({"target_column": target_column}),
        encoding="utf-8",
    )
    X_train = np.arange(6, dtype=float).reshape(-1, 1)
    y_train = np.array([0, 1, 0, 1, 0, 1])
    X_val = np.array([[6.0], [7.0]])
    y_val = np.array([0, 1])
    X_test = np.array([[8.0], [9.0]])
    y_test = np.array([0, 1])
    for name, values in {
        "X_train_balanced.npy": X_train,
        "y_train_balanced.npy": y_train,
        "X_val.npy": X_val,
        "y_val.npy": y_val,
        "X_test.npy": X_test,
        "y_test.npy": y_test,
    }.items():
        np.save(split_dir / name, values)
    pd.DataFrame(
        [
            {"split": "Train Set (Balanced)", "class": 0, "count": 3, "percent": 50.0},
            {"split": "Train Set (Balanced)", "class": 1, "count": 3, "percent": 50.0},
            {"split": "Validation Set", "class": 0, "count": 1, "percent": 50.0},
            {"split": "Validation Set", "class": 1, "count": 1, "percent": 50.0},
            {"split": "Test Set", "class": 0, "count": 1, "percent": 50.0},
            {"split": "Test Set", "class": 1, "count": 1, "percent": 50.0},
        ]
    ).to_csv(split_dir / "class_distribution_after.csv", index=False)
    pd.DataFrame(
        [
            {
                "Class": 0,
                "Full count": 5,
                "Train count": 3,
                "Validation count": 1,
                "Test count": 1,
                "Status": "OK",
            },
            {
                "Class": 1,
                "Full count": 5,
                "Train count": 3,
                "Validation count": 1,
                "Test count": 1,
                "Status": "OK",
            },
        ]
    ).to_csv(split_dir / "class_coverage_report.csv", index=False)
    return split_dir


def test_issue_rejects_invalid_level():
    try:
        Issue(level="info", category="test", message="bad", suggestion="fix")
    except ValueError as exc:
        assert "Invalid issue level" in str(exc)
    else:
        raise AssertionError("Expected invalid issue level to raise ValueError.")


def test_report_to_dict_save_json_and_can_continue_for_warnings(tmp_path):
    report = EdgeCaseReport()
    report.add(
        "warning",
        "dataset",
        "Small dataset.",
        "Add more rows.",
        affected_column="sample_id",
    )

    saved_path = report.save_json(tmp_path / "edge_case_report.json")
    restored = EdgeCaseReport.from_dict(
        json.loads(saved_path.read_text(encoding="utf-8"))
    )

    assert report.can_continue is True
    assert report.to_dict()["warnings"] == 1
    assert json.loads(saved_path.read_text(encoding="utf-8"))["can_continue"] is True
    assert restored.warnings[0].affected_column == "sample_id"


def test_fatal_empty_dataframe_blocks_training(tmp_path):
    df = pd.DataFrame()
    config = make_config(tmp_path)

    report = run_edge_case_checks(df, config)

    assert report.fatals
    assert report.can_continue is False
    assert any(issue.category == "dataset" for issue in report.fatals)


def test_errors_block_training_for_missing_target_and_features(tmp_path):
    df = pd.DataFrame({"feature": [1, 2, 3], "target": [0, 1, 1]})
    config = make_config(
        tmp_path,
        target_column="missing_target",
        feature_columns=["missing_feature"],
    )

    report = run_edge_case_checks(df, config)

    assert len(report.errors) >= 2
    assert report.can_continue is False
    assert any("Target column" in issue.message for issue in report.errors)
    assert any("Selected feature columns" in issue.message for issue in report.errors)


def test_warning_checks_do_not_block_training(tmp_path):
    df = pd.DataFrame(
        {
            "feature": list(range(30)),
            "constant": ["same"] * 30,
            "target": [0] * 28 + [1] * 2,
        }
    )
    config = make_config(tmp_path, feature_columns=["feature", "constant"])

    report = run_edge_case_checks(df, config)

    assert report.warnings
    assert report.errors == []
    assert report.fatals == []
    assert report.can_continue is True


def test_split_and_imbalance_errors(tmp_path):
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3, 4, 5],
            "target": [0, 0, 0, 0, 1],
        }
    )
    config = make_config(
        tmp_path,
        split_method="stratified_group",
        group_column="missing_group",
        imbalance_method="smote",
    )

    report = run_edge_case_checks(df, config)

    assert report.can_continue is False
    assert any(issue.category == "split" for issue in report.errors)
    assert any(issue.category == "imbalance" for issue in report.errors)


def test_environment_checks_for_gpu_and_tabpfn(tmp_path):
    df = pd.DataFrame({"feature": range(3001), "target": [0, 1] * 1500 + [1]})
    config = make_config(
        tmp_path,
        selected_models=["tabpfn"],
        environment_mode="deep_gpu",
    )

    report = run_edge_case_checks(df, config, environment_info={"cuda_available": False})

    assert any("CUDA is unavailable" in issue.message for issue in report.errors)
    assert any("TabPFN" in issue.message for issue in report.warnings)
    assert report.can_continue is False


def test_missing_numeric_feature_blocks_training(tmp_path):
    df = pd.DataFrame(
        {
            "feature": list(range(99)) + [float("nan")],
            "target": [0, 1] * 50,
        }
    )

    report = run_edge_case_checks(df, make_config(tmp_path, split_method="stratified"))

    assert report.can_continue is False
    assert any(
        "Column 'feature' contains 1 empty values (1.0%)" in issue.message
        for issue in report.errors
    )


def test_empty_and_whitespace_feature_values_block_training(tmp_path):
    values = ["category"] * 100
    values[3] = ""
    values[7] = "   "
    df = pd.DataFrame({"feature": values, "target": [0, 1] * 50})

    report = run_edge_case_checks(df, make_config(tmp_path, split_method="stratified"))

    assert report.can_continue is False
    assert any(
        "Column 'feature' contains 2 empty values (2.0%)" in issue.message
        for issue in report.errors
    )


def test_target_missing_text_markers_block_training(tmp_path):
    target = [0, 1] * 46 + [0, None, "", "   ", "NA", "N/A", "null", "None"]
    df = pd.DataFrame({"feature": range(100), "target": target})

    report = run_edge_case_checks(df, make_config(tmp_path, split_method="random"))

    assert report.can_continue is False
    assert any(
        "Column 'target' contains 7 empty values (7.0%)" in issue.message
        for issue in report.errors
    )


def test_unused_column_missing_values_do_not_block_training(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(100),
            "target": [0, 1] * 50,
            "unused": [None] * 100,
        }
    )

    report = run_edge_case_checks(df, make_config(tmp_path, split_method="stratified"))

    assert report.errors == []
    assert report.fatals == []
    assert report.can_continue is True


def test_missing_training_class_is_fatal_but_eval_gaps_are_warnings(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0, 0, 0, 0, 1, 1, 1, 0, 0, 2],
            "event_date": pd.date_range("2026-01-01", periods=10, freq="D"),
        }
    )
    config = make_config(
        tmp_path,
        split_method="time",
        date_column="event_date",
        train_percent=70,
        validation_percent=10,
        test_percent=20,
    )

    report = run_edge_case_checks(df, config)

    assert any(
        issue.level == "fatal"
        and "Class '2' appears in validation/test but not training" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.level == "warning"
        and "Class '1' is absent from the validation set" in issue.message
        for issue in report.issues
    )
    assert any(
        issue.level == "warning"
        and "Class '1' is absent from the test set" in issue.message
        for issue in report.issues
    )
    assert report.can_continue is False


def test_stratified_rare_class_below_three_blocks_before_split(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(12),
            "target": [0] * 10 + [1] * 2,
        }
    )

    report = run_edge_case_checks(df, make_config(tmp_path, split_method="stratified"))

    assert any(
        issue.category == "split"
        and "Class '1' has only 2 samples" in issue.message
        and "Increase dataset size for this class" in issue.suggestion
        for issue in report.errors
    )


def test_saved_edge_checks_block_before_column_configuration(tmp_path):
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    config = make_config(tmp_path)

    report = run_saved_edge_case_checks(df, config)

    assert report.can_continue is False
    assert report.fatals[0].message == (
        "Please confirm Column Configuration before running edge-case checks."
    )


def test_saved_edge_checks_block_before_split_confirmation(tmp_path):
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})
    config = make_config(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.to_csv(data_dir / "modeling_subset.csv", index=False)

    report = run_saved_edge_case_checks(df, config)

    assert report.can_continue is False
    assert any(
        issue.message
        == "Please confirm Data Split & Imbalance before running edge-case checks."
        for issue in report.fatals
    )


def test_saved_edge_checks_ignore_unused_missing_columns(tmp_path):
    save_confirmed_artifacts(tmp_path)
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0, 1] * 5,
            "unused": [None] * 10,
        }
    )

    report = run_saved_edge_case_checks(df, make_config(tmp_path))

    assert report.can_continue is True


def test_saved_edge_checks_block_selected_missing_columns(tmp_path):
    save_confirmed_artifacts(tmp_path)
    df = pd.DataFrame({"feature": list(range(9)) + [None], "target": [0, 1] * 5})

    report = run_saved_edge_case_checks(df, make_config(tmp_path))

    assert report.can_continue is False
    assert any("Column 'feature' contains 1 empty values" in issue.message for issue in report.errors)


def test_saved_edge_checks_block_missing_split_artifacts(tmp_path):
    split_dir = save_confirmed_artifacts(tmp_path)
    (split_dir / "X_test.npy").unlink()
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    report = run_saved_edge_case_checks(df, make_config(tmp_path))

    assert report.can_continue is False
    assert any("Saved test artifact(s) are missing" in issue.message for issue in report.fatals)


def test_saved_edge_checks_block_mismatched_split_target(tmp_path):
    save_confirmed_artifacts(tmp_path, target_column="old_target")
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    report = run_saved_edge_case_checks(df, make_config(tmp_path))

    assert report.can_continue is False
    assert any(
        "does not match current target 'target'" in issue.message for issue in report.fatals
    )


def test_saved_edge_checks_allow_valid_confirmed_artifacts(tmp_path):
    save_confirmed_artifacts(tmp_path)
    df = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    report = run_saved_edge_case_checks(df, make_config(tmp_path))

    assert report.can_continue is True
    assert report.errors == []
    assert report.fatals == []
    assert report.context["column_configuration_confirmed"] is True
    assert report.context["split_confirmed"] is True
