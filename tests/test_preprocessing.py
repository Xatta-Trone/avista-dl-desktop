import json

import pandas as pd
import pytest

from app.core.preprocessing import (
    PreprocessingArtifacts,
    build_preprocessing_pipeline,
    fit_split_preprocessing,
    invalidate_preprocessing_artifacts,
    load_artifacts,
    save_artifacts,
    save_numerical_scaling_artifacts,
    transform_split_features,
    transform_new_data,
)
from app.core.project_config import ProjectConfig


def make_config(tmp_path, **overrides):
    values = {
        "project_name": "preprocess-demo",
        "project_dir": str(tmp_path),
        "input_file": str(tmp_path / "data.csv"),
        "output_dir": str(tmp_path / "outputs"),
        "target_column": "target",
        "feature_columns": ["age", "city", "score"],
        "task_type": "classification",
        "preprocessing_options": {},
    }
    values.update(overrides)
    return ProjectConfig(**values)


def test_mixed_numeric_categorical_dataset(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(tmp_path)

    X, y, artifacts = build_preprocessing_pipeline(df, config)

    assert artifacts.numeric_columns == ["age", "score"]
    assert artifacts.categorical_columns == ["city"]
    assert "city_Austin" in artifacts.output_feature_names
    assert X.shape == (3, 4)
    assert y.tolist() == [0, 1, 0]


def test_missing_values_are_imputed(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, None, 40],
            "city": ["Austin", None, "Dallas"],
            "score": [1.0, None, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(tmp_path)

    X, _, artifacts = build_preprocessing_pipeline(df, config)

    assert X.isna().sum().sum() == 0
    assert "city_Unknown" in artifacts.output_feature_names


def test_classification_target_encoding(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Houston"],
            "score": [1.0, 2.0, 3.0],
            "target": ["class_b", "class_a", "class_b"],
        }
    )
    config = make_config(tmp_path, task_type="classification")

    _, y, artifacts = build_preprocessing_pipeline(df, config)

    assert y.tolist() == [1, 0, 1]
    assert artifacts.target_encoder.classes_.tolist() == ["class_a", "class_b"]


def test_regression_target_stays_numeric(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Houston"],
            "score": [1.0, 2.0, 3.0],
            "target": [10.5, 11.5, 12.5],
        }
    )
    config = make_config(tmp_path, task_type="regression")

    _, y, artifacts = build_preprocessing_pipeline(df, config)

    assert y.tolist() == [10.5, 11.5, 12.5]
    assert artifacts.target_encoder is None


def test_unknown_categories_are_safe_during_transform(tmp_path):
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    new_df = pd.DataFrame({"age": [50], "city": ["San Marcos"], "score": [4.0]})
    config = make_config(tmp_path)

    _, _, artifacts = build_preprocessing_pipeline(train_df, config)
    transformed = transform_new_data(new_df, artifacts)

    assert transformed.shape == (1, len(artifacts.output_feature_names))
    city_columns = [column for column in transformed.columns if column.startswith("city_")]
    assert transformed.loc[0, city_columns].sum() == 0


def test_artifact_save_load(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(tmp_path)

    _, _, artifacts = build_preprocessing_pipeline(df, config)
    path = save_artifacts(artifacts, tmp_path / "preprocessing.joblib")
    loaded = load_artifacts(path)

    assert isinstance(loaded, PreprocessingArtifacts)
    assert loaded.output_feature_names == artifacts.output_feature_names
    assert loaded.target_encoder.classes_.tolist() == artifacts.target_encoder.classes_.tolist()


def test_auto_feature_selection_excludes_configured_columns(tmp_path):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [20, 30, 40],
            "visit_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "site": ["A", "A", "B"],
            "subgroup": ["x", "y", "x"],
            "ignore_me": [9, 9, 9],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(
        tmp_path,
        feature_columns=[],
        id_columns=["id"],
        date_column="visit_date",
        group_column="site",
        subgroup_columns=["subgroup"],
        excluded_columns=["ignore_me"],
    )

    _, _, artifacts = build_preprocessing_pipeline(df, config)

    assert artifacts.feature_columns == ["age"]


def test_regression_target_rejects_text(tmp_path):
    df = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "city": ["Austin", "Dallas", "Houston"],
            "score": [1.0, 2.0, 3.0],
            "target": ["low", "medium", "high"],
        }
    )
    config = make_config(tmp_path, task_type="regression")

    with pytest.raises(ValueError, match="Regression target must be numeric"):
        build_preprocessing_pipeline(df, config)


def test_minmax_scaling_is_applied_only_to_numeric_columns(tmp_path):
    df = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [100.0, 200.0, 400.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(
        tmp_path,
        preprocessing_options={
            "numerical_scaling_method": "minmax",
            "scaled_numerical_columns": ["age"],
        },
    )

    X, _, artifacts = build_preprocessing_pipeline(df, config)

    assert artifacts.numeric_scaling_method == "minmax"
    assert artifacts.scaled_numeric_columns == ["age"]
    assert X["age"].tolist() == [0.0, 0.5, 1.0]
    assert X["score"].tolist() == [100.0, 200.0, 400.0]
    assert "city_Austin" in X.columns


def test_scaler_none_preserves_selected_columns_without_scaling(tmp_path):
    df = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [100.0, 200.0, 400.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(
        tmp_path,
        preprocessing_options={
            "numerical_scaling_method": "none",
            "scaled_numerical_columns": ["age"],
        },
    )

    X, _, artifacts = build_preprocessing_pipeline(df, config)

    assert artifacts.scaler is None
    assert artifacts.scaled_numeric_columns == ["age"]
    assert X["age"].tolist() == [10.0, 20.0, 30.0]
    assert X["score"].tolist() == [100.0, 200.0, 400.0]


def test_split_preprocessing_fits_scaler_on_train_only(tmp_path):
    train_df = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        },
        index=[0, 1, 2],
    )
    validation_df = pd.DataFrame(
        {
            "age": [40.0],
            "city": ["Austin"],
            "score": [4.0],
            "target": ["yes"],
        },
        index=[3],
    )
    config = make_config(
        tmp_path,
        preprocessing_options={
            "numerical_scaling_method": "standard",
            "scaled_numerical_columns": ["age", "score"],
        },
    )

    artifacts = fit_split_preprocessing(train_df, config)
    transformed_train = transform_split_features(train_df, artifacts)
    transformed_validation = transform_split_features(validation_df, artifacts)

    assert transformed_train["age"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed_train["score"].mean() == pytest.approx(0.0, abs=1e-9)
    assert transformed_validation.loc[3, "age"] > 1.0
    assert transformed_validation.loc[3, "score"] > 1.0
    assert artifacts.target_column == "target"


def test_numerical_scaling_artifacts_are_saved(tmp_path):
    df = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(
        tmp_path,
        preprocessing_options={
            "numerical_scaling_method": "standard",
            "scaled_numerical_columns": ["age", "score"],
        },
    )

    _, _, artifacts = build_preprocessing_pipeline(df, config)
    scaler_path, metadata_path = save_numerical_scaling_artifacts(tmp_path, artifacts)

    assert scaler_path is not None and scaler_path.exists()
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["numerical_scaling_method"] == "standard"
    assert metadata["scaled_numerical_columns"] == ["age", "score"]


def test_invalidate_preprocessing_artifacts_removes_saved_scaler_files(tmp_path):
    df = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0],
            "city": ["Austin", "Dallas", "Austin"],
            "score": [1.0, 2.0, 3.0],
            "target": ["no", "yes", "no"],
        }
    )
    config = make_config(
        tmp_path,
        preprocessing_options={
            "numerical_scaling_method": "standard",
            "scaled_numerical_columns": ["age", "score"],
        },
    )

    _, _, artifacts = build_preprocessing_pipeline(df, config)
    scaler_path, metadata_path = save_numerical_scaling_artifacts(tmp_path, artifacts)
    assert scaler_path is not None and scaler_path.exists()
    assert metadata_path.exists()

    invalidate_preprocessing_artifacts(tmp_path)

    assert not metadata_path.exists()
