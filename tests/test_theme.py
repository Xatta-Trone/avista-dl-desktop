import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 is not installed.",
)


def test_theme_setting_defaults_to_light_and_persists(tmp_path):
    from app.core.user_settings import load_user_settings, save_user_settings
    from app.gui.theme import normalize_theme_name

    settings_file = tmp_path / "settings.json"
    settings = load_user_settings(settings_file)

    assert settings.theme_name == "light"

    settings.theme_name = "dark"
    save_user_settings(settings, settings_file)
    assert load_user_settings(settings_file).theme_name == "dark"

    assert normalize_theme_name("not-a-theme") == "light"


def test_theme_qss_and_transform_preserve_primary_button_text():
    from app.gui.theme import get_theme, theme_qss, transform_stylesheet

    dark = get_theme("dark")
    qss = theme_qss(dark)
    transformed = transform_stylesheet(
        """
        QWidget#card { background: #FFFFFF; color: #1F2937; }
        QPushButton#primary { background: #0F6CBD; color: #FFFFFF; }
        QLabel#muted { color: #5B6573; }
        """,
        dark,
    )

    assert dark.background in qss
    assert "background: #1D293D" in transformed
    assert "color: #FFFFFF" in transformed
    assert "color: #B7C0CC" in transformed


def test_main_window_theme_menu_applies_without_restart(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.user_settings import UserSettings
    from app.gui.main_window import MainWindow

    settings = UserSettings(theme_name="light")
    monkeypatch.setattr("app.gui.main_window.load_user_settings", lambda: settings)
    monkeypatch.setattr(
        "app.gui.theme.load_user_settings",
        lambda: settings,
    )

    def save_settings(updated):
        settings.theme_name = updated.theme_name

    monkeypatch.setattr("app.gui.theme.save_user_settings", save_settings)

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.light_theme_action.isChecked()
    window.set_theme(window.dark_theme_action)

    assert settings.theme_name == "dark"
    assert window.dark_theme_action.isChecked()
    assert app.property("avistaTheme") == "dark"
    assert "#111827" in app.styleSheet()

    window.set_theme(window.light_theme_action)
    assert settings.theme_name == "light"
    assert app.property("avistaTheme") == "light"
    window.close()
    assert app is not None


def test_all_sidebar_pages_accept_light_and_dark_themes(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.core.user_settings import UserSettings
    from app.gui.main_window import MainWindow

    settings = UserSettings(theme_name="light")
    monkeypatch.setattr("app.gui.main_window.load_user_settings", lambda: settings)
    monkeypatch.setattr("app.gui.theme.load_user_settings", lambda: settings)
    monkeypatch.setattr(
        "app.gui.theme.save_user_settings",
        lambda updated: setattr(settings, "theme_name", updated.theme_name),
    )

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    for action, expected in (
        (window.light_theme_action, "light"),
        (window.dark_theme_action, "dark"),
    ):
        window.set_theme(action)
        for index, (label, _, _) in enumerate(window.pages):
            window.navigate_to(index)
            app.processEvents()
            assert window.stack.currentIndex() == index
            assert window.nav_buttons[index].isChecked()
            assert label
        assert app.property("avistaTheme") == expected

    window.close()
    assert app is not None
