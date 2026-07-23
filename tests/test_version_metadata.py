from pathlib import Path
from types import SimpleNamespace

from app.__version__ import (
    APP_DESCRIPTION,
    APP_NAME,
    RELEASE_DATE,
    __version__,
)
from app.branding import report_footer
from app.core.error_handler import EdgeCaseReport
from app.core.project_config import ProjectConfig
from app.core.report_generator import collect_report_summary
from app.core.trainer import _project_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_report_and_edge_case_metadata_use_central_version(tmp_path):
    config = ProjectConfig(
        project_name="version-report",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )
    footer = report_footer()
    edge_metadata = EdgeCaseReport().to_dict()["report_footer"]
    summary = collect_report_summary(config)

    assert footer["generated_by"] == APP_NAME
    assert footer["description"] == APP_DESCRIPTION
    assert footer["version"] == __version__
    assert footer["release_date"] == RELEASE_DATE
    assert edge_metadata["generated_by"] == APP_NAME
    assert edge_metadata["description"] == APP_DESCRIPTION
    assert edge_metadata["version"] == __version__
    assert edge_metadata["release_date"] == RELEASE_DATE
    assert summary["description"] == APP_DESCRIPTION
    assert summary["version"] == __version__
    assert summary["release_date"] == RELEASE_DATE


def test_project_and_training_metadata_use_central_version(tmp_path):
    config = ProjectConfig(
        project_name="version-project",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )
    config.save()
    fallback = _project_metadata(
        SimpleNamespace(
            project_name="fallback",
            project_dir=str(tmp_path),
            project_file_path=str(tmp_path / "fallback.avista"),
            project_file_version="1.0",
        )
    )

    assert config.project_metadata()["application"] == APP_NAME
    assert (
        config.project_metadata()["application_description"]
        == APP_DESCRIPTION
    )
    assert config.project_metadata()["application_version"] == __version__
    assert (
        config.project_metadata()["application_release_date"]
        == RELEASE_DATE
    )
    assert fallback["application"] == APP_NAME
    assert fallback["application_description"] == APP_DESCRIPTION
    assert fallback["application_version"] == __version__
    assert fallback["application_release_date"] == RELEASE_DATE


def test_version_consumers_do_not_hard_code_release_number():
    version_literal = __version__
    consumers = (
        "app/gui/about_dialog.py",
        "app/core/report_generator.py",
        "app/gui/edge_case_report_page.py",
        "app/core/trainer.py",
        "app/core/project_config.py",
        "app/core/runtime_verification.py",
        "packaging/build_pyinstaller.ps1",
        "packaging/avista_pyinstaller.spec",
        "packaging/create_logo_icon.py",
        "packaging/avista_installer.iss",
    )

    for relative_path in consumers:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert version_literal not in text, relative_path


def test_branding_consumers_use_central_description():
    consumers = (
        "app/branding.py",
        "main.py",
        "app/gui/about_dialog.py",
        "app/gui/project_setup_page.py",
        "app/core/project_config.py",
        "app/core/report_generator.py",
        "app/core/runtime_verification.py",
        "app/core/trainer.py",
        "app/gui/edge_case_report_page.py",
        "packaging/build_pyinstaller.ps1",
        "packaging/avista_installer.iss",
    )

    for relative_path in consumers:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert APP_DESCRIPTION not in text, relative_path

    assert "APP_DESCRIPTION" in (
        PROJECT_ROOT / "app/__version__.py"
    ).read_text(encoding="utf-8")


def test_release_date_consumers_use_central_metadata():
    consumers = (
        "app/branding.py",
        "main.py",
        "app/gui/about_dialog.py",
        "app/core/project_config.py",
        "app/core/report_generator.py",
        "app/core/runtime_verification.py",
        "app/core/trainer.py",
        "packaging/build_pyinstaller.ps1",
        "packaging/avista_installer.iss",
    )

    for relative_path in consumers:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert RELEASE_DATE not in text, relative_path
        assert "RELEASE_DATE" in text or "MyAppReleaseDate" in text, relative_path
