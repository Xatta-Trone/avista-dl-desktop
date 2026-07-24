import json
from pathlib import Path

import pytest

from scripts.prepare_release import (
    ReleaseMetadata,
    check_release_metadata,
    prepare_release,
    read_central_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _release_project(tmp_path: Path) -> Path:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    (app_dir / "__version__.py").write_text(
        '__version__ = "1.0.5"\n'
        'APP_NAME = "AVISTA"\n'
        'RELEASE_DATE = "July 24, 2026"\n',
        encoding="utf-8",
    )
    (tmp_path / "updates.json").write_text(
        json.dumps(
            {
                "latest_version": "1.0.5",
                "release_date": "July 24, 2026",
                "release_notes": ["Previous release"],
                "installer_url": (
                    "https://github.com/Xatta-Trone/avista-dl-desktop/"
                    "releases/download/v1.0.5/AVISTA_Setup.exe"
                ),
                "sha256": "b" * 64,
                "mandatory": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "# AVISTA\n\n"
        "The launch screen and About dialog identify the current release as "
        "**Version\n"
        "1.0.5**, released **July 24, 2026**. Product name, description, "
        "version, and\n"
        "release date come from `app/__version__.py`.\n",
        encoding="utf-8",
    )
    (tmp_path / "PROJECT_STATUS.md").write_text(
        "# Status\n\n"
        "The current AVISTA application and update-feed version is `1.0.5`, "
        "with the\n"
        "release date centralized as July 24, 2026. Focused "
        "centralized-version,\n"
        "update-feed verification passed.\n",
        encoding="utf-8",
    )
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n"
        "## 1.0.5 - 2026-07-24\n\n"
        "- Previous release\n",
        encoding="utf-8",
    )
    return tmp_path


def test_repository_release_metadata_is_synchronized():
    assert check_release_metadata(PROJECT_ROOT) == []
    metadata = read_central_metadata(PROJECT_ROOT)
    assert metadata.version == "1.0.5"
    assert metadata.release_date == "July 24, 2026"


def test_prepare_release_updates_central_and_derived_metadata(tmp_path):
    root = _release_project(tmp_path)
    metadata = ReleaseMetadata("1.0.6", "July 25, 2026")

    changed = prepare_release(
        root,
        metadata,
        [
            "Added synchronized release preparation",
            "Added a CI release metadata guard",
        ],
    )

    assert {path.name for path in changed} == {
        "__version__.py",
        "updates.json",
        "README.md",
        "PROJECT_STATUS.md",
        "CHANGELOG.md",
    }
    assert read_central_metadata(root) == metadata
    updates = json.loads((root / "updates.json").read_text(encoding="utf-8"))
    assert updates["latest_version"] == "1.0.6"
    assert updates["release_date"] == "July 25, 2026"
    assert updates["release_notes"] == [
        "Added synchronized release preparation",
        "Added a CI release metadata guard",
    ]
    assert "/releases/download/v1.0.6/AVISTA_Setup.exe" in updates[
        "installer_url"
    ]
    assert updates["sha256"] == ""
    assert "## 1.0.6 - 2026-07-25" in (
        root / "CHANGELOG.md"
    ).read_text(encoding="utf-8")
    assert check_release_metadata(root) == []


def test_prepare_release_requires_new_notes_when_version_advances(tmp_path):
    root = _release_project(tmp_path)
    original_version_source = (
        root / "app" / "__version__.py"
    ).read_text(encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="requires at least one --note",
    ):
        prepare_release(
            root,
            ReleaseMetadata("1.0.6", "July 25, 2026"),
        )

    assert (
        root / "app" / "__version__.py"
    ).read_text(encoding="utf-8") == original_version_source


def test_prepare_release_dry_run_does_not_write(tmp_path):
    root = _release_project(tmp_path)
    changed = prepare_release(
        root,
        ReleaseMetadata("1.0.6", "July 25, 2026"),
        ["Dry-run release"],
        dry_run=True,
    )

    assert len(changed) == 5
    assert read_central_metadata(root).version == "1.0.5"


def test_release_check_rejects_a_mismatched_tag(tmp_path):
    root = _release_project(tmp_path)

    problems = check_release_metadata(root, expected_tag="v1.0.6")

    assert problems == [
        "release tag 'v1.0.6' does not match central version 'v1.0.5'"
    ]


def test_prepare_release_accepts_installer_sha256(tmp_path):
    root = _release_project(tmp_path)
    digest = "a1" * 32

    prepare_release(
        root,
        ReleaseMetadata("1.0.5", "July 24, 2026"),
        sha256=digest,
    )

    updates = json.loads((root / "updates.json").read_text(encoding="utf-8"))
    assert updates["sha256"] == digest
