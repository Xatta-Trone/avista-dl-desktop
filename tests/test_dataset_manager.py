from datetime import datetime

import pytest

from app.core.dataset_manager import (
    DUPLICATE_APPEND_TIMESTAMP,
    DUPLICATE_CANCEL,
    DUPLICATE_OVERWRITE,
    copy_dataset_into_project,
)


def test_external_dataset_is_copied_into_project(tmp_path):
    source = tmp_path / "external" / "crash_data.csv"
    source.parent.mkdir()
    source.write_text("x,y\n1,0\n", encoding="utf-8")
    project_dir = tmp_path / "project"

    metadata = copy_dataset_into_project(source, project_dir)

    copied = project_dir / "data" / "crash_data.csv"
    assert copied.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
    assert metadata["project_relative_path"] == "data/crash_data.csv"
    assert metadata["copied_project_path"] == "data/crash_data.csv"
    assert metadata["original_source_path"] == str(source.resolve())
    assert metadata["copied_to_project"] is True
    assert metadata["file_size"] == source.stat().st_size
    assert metadata["copy_timestamp"]


def test_duplicate_dataset_defaults_to_timestamped_copy(tmp_path):
    first = tmp_path / "first" / "crash_data.csv"
    second = tmp_path / "second" / "crash_data.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("old", encoding="utf-8")
    second.write_text("new", encoding="utf-8")
    project_dir = tmp_path / "project"
    copy_dataset_into_project(first, project_dir)

    metadata = copy_dataset_into_project(
        second,
        project_dir,
        duplicate_policy=DUPLICATE_APPEND_TIMESTAMP,
        now=lambda: datetime(2026, 7, 12, 15, 30, 0),
    )

    assert metadata["project_relative_path"] == (
        "data/crash_data_20260712_153000.csv"
    )
    assert (project_dir / metadata["project_relative_path"]).read_text(
        encoding="utf-8"
    ) == "new"
    assert (project_dir / "data" / "crash_data.csv").read_text(
        encoding="utf-8"
    ) == "old"


def test_duplicate_dataset_can_overwrite_or_cancel(tmp_path):
    first = tmp_path / "first" / "data.csv"
    second = tmp_path / "second" / "data.csv"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("old", encoding="utf-8")
    second.write_text("new", encoding="utf-8")
    project_dir = tmp_path / "project"
    copy_dataset_into_project(first, project_dir)

    with pytest.raises(FileExistsError):
        copy_dataset_into_project(
            second,
            project_dir,
            duplicate_policy=DUPLICATE_CANCEL,
        )
    copy_dataset_into_project(
        second,
        project_dir,
        duplicate_policy=DUPLICATE_OVERWRITE,
    )

    assert (project_dir / "data" / "data.csv").read_text(encoding="utf-8") == "new"
