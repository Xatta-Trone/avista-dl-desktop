from pathlib import Path
from types import SimpleNamespace

from app.__version__ import APP_NAME, __version__
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
    assert footer["version"] == __version__
    assert edge_metadata["generated_by"] == APP_NAME
    assert edge_metadata["version"] == __version__
    assert summary["version"] == __version__


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
    assert config.project_metadata()["application_version"] == __version__
    assert fallback["application"] == APP_NAME
    assert fallback["application_version"] == __version__


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
        "packaging/avista_installer.iss",
    )

    for relative_path in consumers:
        text = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert version_literal not in text, relative_path
