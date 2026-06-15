import importlib.util

import pandas as pd

from app.core.imbalance import apply_imbalance_strategy
from app.core.preprocessing import build_preprocessing_pipeline
from app.core.project_config import ProjectConfig
from app.core.splitter import (
    build_class_coverage_report,
    class_coverage_issues,
    split_data,
    split_data_three_way,
)


def make_config(tmp_path, **overrides):
    values = {
        "project_name": "split-demo",
        "project_dir": str(tmp_path),
        "input_file": str(tmp_path / "data.csv"),
        "output_dir": str(tmp_path / "outputs"),
        "target_column": "target",
        "feature_columns": ["feature", "category"],
        "task_type": "classification",
        "split_method": "random",
        "imbalance_method": "none",
        "preprocessing_options": {},
    }
    values.update(overrides)
    return ProjectConfig(**values)


def make_df(rows=50):
    return pd.DataFrame(
        {
            "feature": list(range(rows)),
            "category": ["a" if i % 2 == 0 else "b" for i in range(rows)],
            "target": [0 if i < rows // 2 else 1 for i in range(rows)],
            "group": [f"group_{i // 5}" for i in range(rows)],
            "event_date": pd.date_range("2026-01-01", periods=rows, freq="D"),
        }
    )


def test_random_split(tmp_path):
    df = make_df()
    config = make_config(tmp_path, split_method="random")
    X, y, _ = build_preprocessing_pipeline(df, config)

    result = split_data(X, y, df, config)

    assert len(result["X_train"]) == 40
    assert len(result["X_test"]) == 10
    assert result["split_info"]["split_method"] == "random"


def test_stratified_split(tmp_path):
    df = make_df()
    config = make_config(tmp_path, split_method="stratified")
    X, y, _ = build_preprocessing_pipeline(df, config)

    result = split_data(X, y, df, config)

    assert result["y_train"].value_counts().to_dict() == {0: 20, 1: 20}
    assert result["y_test"].value_counts().to_dict() == {0: 5, 1: 5}


def test_group_split_has_no_group_leakage(tmp_path):
    df = make_df()
    config = make_config(tmp_path, split_method="group", group_column="group")
    X, y, _ = build_preprocessing_pipeline(df, config)

    result = split_data(X, y, df, config)

    train_groups = set(df.loc[result["train_index"], "group"])
    test_groups = set(df.loc[result["test_index"], "group"])
    assert train_groups.isdisjoint(test_groups)


def test_time_split_uses_latest_rows_for_test(tmp_path):
    df = make_df(rows=20)
    config = make_config(tmp_path, split_method="time", date_column="event_date")
    X, y, _ = build_preprocessing_pipeline(df, config)

    result = split_data(X, y, df, config)

    assert result["test_index"] == [16, 17, 18, 19]
    assert result["train_index"][0] == 0


def test_three_way_stratified_split(tmp_path):
    df = make_df(rows=100)
    config = make_config(
        tmp_path,
        split_method="stratified",
        train_percent=70,
        validation_percent=10,
        test_percent=20,
    )
    X, y, _ = build_preprocessing_pipeline(df, config)

    result = split_data_three_way(X, y, df, config)

    assert len(result["X_train"]) == 70
    assert len(result["X_val"]) == 10
    assert len(result["X_test"]) == 20
    all_indices = set(result["train_index"] + result["validation_index"] + result["test_index"])
    assert len(all_indices) == 100


def test_class_coverage_reports_missing_train_validation_and_test():
    coverage = build_class_coverage_report(
        pd.Series(["a", "a", "b", "b", "c"]),
        pd.Series(["a", "b"]),
        pd.Series(["a", "c"]),
        pd.Series(["a", "c"]),
    )
    issues = class_coverage_issues(coverage)

    statuses = dict(zip(coverage["Class"], coverage["Status"]))
    assert statuses["c"] == "Missing in train - blocking"
    assert "Missing in validation - warning" in statuses["b"]
    assert "Missing in test - warning" in statuses["b"]
    assert any(
        issue["level"] == "fatal"
        and "Class 'c' appears in validation/test but not training" in issue["message"]
        for issue in issues
    )
    assert any(
        issue["level"] == "warning"
        and "Class 'b' is absent from the validation set" in issue["message"]
        for issue in issues
    )
    assert any(
        issue["level"] == "warning"
        and "Class 'b' is absent from the test set" in issue["message"]
        for issue in issues
    )


def test_class_weights_do_not_modify_training_data(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(10),
            "category": ["a"] * 10,
            "target": [0] * 8 + [1] * 2,
        }
    )
    config = make_config(tmp_path, imbalance_method="class_weight")
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["X_resampled"].equals(X)
    assert result["y_resampled"].equals(y)
    assert result["class_weights"][0] < result["class_weights"][1]


def test_smote_fallback_or_validation(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(10),
            "category": ["a"] * 10,
            "target": [0] * 9 + [1],
        }
    )
    config = make_config(tmp_path, imbalance_method="smote")
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["X_resampled"].equals(X)
    assert result["y_resampled"].equals(y)
    assert result["imbalance_info"]["success"] is False
    if importlib.util.find_spec("imblearn") is None:
        assert "imbalanced-learn" in result["imbalance_info"]["message"]
    else:
        assert "SMOTE requires at least 2 samples" in result["imbalance_info"]["error"]


def test_random_oversample_changes_distribution(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(20),
            "category": ["a"] * 20,
            "target": [0] * 16 + [1] * 4,
        }
    )
    config = make_config(tmp_path, imbalance_method="random_oversample")
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["imbalance_info"]["success"] is True
    assert result["y_resampled"].value_counts().to_dict() == {0: 16, 1: 16}


def test_random_undersample_changes_distribution(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(20),
            "category": ["a"] * 20,
            "target": [0] * 16 + [1] * 4,
        }
    )
    config = make_config(tmp_path, imbalance_method="random_undersample")
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["imbalance_info"]["success"] is True
    assert result["y_resampled"].value_counts().to_dict() == {0: 4, 1: 4}


def test_random_oversample_ignores_non_training_strategy_classes(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(10),
            "category": ["a"] * 10,
            "target": [0] * 7 + [1] * 3,
        }
    )
    config = make_config(
        tmp_path,
        imbalance_method="random_oversample",
        preprocessing_options={
            "imbalance": {"sampling_strategy": {1: 7, 2: 7}}
        },
    )
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["imbalance_info"]["success"] is True
    assert result["y_resampled"].value_counts().to_dict() == {0: 7, 1: 7}
    assert 2 not in result["y_resampled"].unique()


def test_smote_changes_distribution_when_valid(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(30),
            "category": ["a"] * 30,
            "target": [0] * 24 + [1] * 6,
        }
    )
    config = make_config(
        tmp_path,
        feature_columns=["feature"],
        imbalance_method="smote",
        preprocessing_options={"imbalance": {"sampling_strategy": {1: 12}}},
    )
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["imbalance_info"]["success"] is True
    assert result["y_resampled"].value_counts().to_dict() == {0: 24, 1: 12}


def test_smote_nc_changes_distribution_when_valid(tmp_path):
    df = pd.DataFrame(
        {
            "feature": range(30),
            "category": ["a", "b"] * 15,
            "target": [0] * 24 + [1] * 6,
        }
    )
    config = make_config(
        tmp_path,
        imbalance_method="smote_nc",
        preprocessing_options={"imbalance": {"sampling_strategy": {1: 12}}},
    )
    X, y, artifacts = build_preprocessing_pipeline(df, config)

    result = apply_imbalance_strategy(X, y, artifacts, config)

    assert result["imbalance_info"]["success"] is True
    assert result["y_resampled"].value_counts().to_dict() == {0: 24, 1: 12}
