import json
import shutil
from pathlib import Path

from app.__version__ import APP_NAME, __version__
from app.core.project_config import ProjectConfig


def test_project_config_save_load_avista_round_trip(tmp_path):
    project_dir = tmp_path / "demo"
    config = ProjectConfig(
        project_name="demo",
        project_dir=str(project_dir),
        input_file=str(project_dir / "data" / "input.csv"),
        output_dir=str(project_dir / "outputs"),
        target_column="target",
        feature_columns=["age", "score"],
        label_encoding_columns=["region"],
        id_columns=["record_id"],
        group_column="site",
        date_column="created_at",
        excluded_columns=["notes"],
        subgroup_columns=["gender", "region"],
        task_type="classification",
        split_method="stratified",
        imbalance_method="class_weight",
        selected_models=["logistic_regression", "random_forest"],
        model_params={
            "logistic_regression": {"max_iter": 500},
            "random_forest": {"n_estimators": 250},
        },
        enable_cross_validation=True,
        cv_folds=7,
        random_state=123,
        preprocessing_options={"scale_numeric": True, "encode_categorical": True},
        xai_options={"shap": True, "lime": False},
        environment_mode="deep_cpu",
    )

    saved_path = config.save()
    stored = json.loads(saved_path.read_text(encoding="utf-8"))
    loaded = ProjectConfig.load(saved_path)

    assert saved_path == project_dir / "demo.avista"
    assert stored["application"] == APP_NAME
    assert stored["application_version"] == __version__
    assert stored["project_file_version"] == "1.0"
    assert stored["project_dir"] == "."
    assert stored["project_file_path"] == "demo.avista"
    assert stored["input_file"] == "data/input.csv"
    assert stored["output_dir"] == "outputs"
    assert loaded == config


def test_project_config_save_accepts_custom_avista_path(tmp_path):
    config = ProjectConfig(
        project_name="custom-path-demo",
        project_dir=str(tmp_path / "project"),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "project" / "outputs"),
    )
    custom_path = tmp_path / "projects" / "Renamed.avista"

    saved_path = config.save(custom_path)

    assert saved_path == custom_path.resolve()
    assert config.project_dir == str(custom_path.parent.resolve())
    assert config.project_file == custom_path.resolve()
    assert ProjectConfig.load(custom_path) == config


def test_project_config_imports_legacy_json_and_converts_to_avista(tmp_path):
    legacy_path = tmp_path / "project_config.json"
    legacy_path.write_text(
        json.dumps(
            {
                "project_name": "legacy-demo",
                "project_dir": "C:/old/location",
                "input_file": "data/input.csv",
                "output_dir": "outputs",
            }
        ),
        encoding="utf-8",
    )

    config = ProjectConfig.load(legacy_path)

    converted = tmp_path / "legacy-demo.avista"
    assert converted.exists()
    assert config.project_file == converted.resolve()
    assert config.project_dir == str(tmp_path.resolve())
    assert config.application == APP_NAME
    assert config.application_version == __version__
    assert config.project_file_version == "1.0"


def test_moving_project_folder_keeps_dataset_resolvable(tmp_path):
    original = tmp_path / "original"
    dataset_path = original / "data" / "portable.csv"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("x,y\n1,0\n", encoding="utf-8")
    config = ProjectConfig(
        project_name="portable",
        project_dir=str(original),
        input_file=str(dataset_path),
        output_dir=str(original / "outputs"),
        dataset={
            "project_relative_path": "data/portable.csv",
            "original_source_path": "D:/external/portable.csv",
            "copied_project_path": "data/portable.csv",
            "copied_to_project": True,
            "file_size": dataset_path.stat().st_size,
            "copy_timestamp": "2026-07-12T15:30:00",
        },
    )
    config.save()
    moved = tmp_path / "moved"
    shutil.move(str(original), moved)

    loaded = ProjectConfig.load(moved / "portable.avista")

    assert loaded.project_dir == str(moved.resolve())
    assert loaded.input_file == str((moved / "data" / "portable.csv").resolve())
    assert Path(loaded.input_file).exists()


def test_project_config_migrates_legacy_xtab_to_avista(tmp_path):
    legacy_path = tmp_path / "legacy.xtab"
    legacy_path.write_text(
        json.dumps(
            {
                "project_name": "legacy",
                "project_dir": ".",
                "project_file_path": "legacy.xtab",
                "input_file": "",
                "output_dir": "outputs",
                "application": "LegacyApplication",
                "project_file_version": "1.0",
            }
        ),
        encoding="utf-8",
    )

    config = ProjectConfig.load(legacy_path)

    migrated_path = tmp_path / "legacy.avista"
    stored = json.loads(migrated_path.read_text(encoding="utf-8"))
    assert legacy_path.exists()
    assert config.project_file == migrated_path.resolve()
    assert stored["application"] == "AVISTA"
    assert stored["project_file_path"] == "legacy.avista"
