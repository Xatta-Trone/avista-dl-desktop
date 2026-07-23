import pytest

from app.__version__ import (
    APP_DESCRIPTION,
    APP_NAME,
    RELEASE_DATE,
    __version__,
)
from app.core.project_config import ProjectConfig
from main import create_splash_screen, load_startup_project


def test_command_line_avista_path_loads_project(tmp_path):
    config = ProjectConfig(
        project_name="startup",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )
    project_file = config.save()

    loaded = load_startup_project([str(project_file)])

    assert loaded is not None
    assert loaded.project_name == "startup"
    assert loaded.project_file == project_file
    assert loaded.project_dir == str(tmp_path.resolve())


def test_command_line_legacy_xtab_path_migrates_project(tmp_path):
    legacy_path = tmp_path / "startup.xtab"
    legacy_path.write_text(
        """
        {
          "project_name": "startup",
          "project_dir": ".",
          "input_file": "",
          "output_dir": "outputs"
        }
        """,
        encoding="utf-8",
    )

    loaded = load_startup_project([str(legacy_path)])

    assert loaded is not None
    assert loaded.project_file == (tmp_path / "startup.avista").resolve()


def test_command_line_project_rejects_non_project_path(tmp_path):
    legacy_path = tmp_path / "project_config.json"
    legacy_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=".avista"):
        load_startup_project([str(legacy_path)])


def test_splash_screen_uses_central_release_branding():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    splash = create_splash_screen()

    assert splash.pixmap().width() == 720
    assert splash.pixmap().height() == 360
    assert tuple(splash.property("avistaBrandingText")) == (
        APP_NAME,
        f"Version {__version__}",
        f"Release date: {RELEASE_DATE}",
        APP_DESCRIPTION,
    )
    splash.close()
    assert app is not None
