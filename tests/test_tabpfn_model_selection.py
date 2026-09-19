from pathlib import Path
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from app.core.project_config import ProjectConfig
from app.gui.main_window import MainWindow


def _status(path: Path | None = None, source: str = "unavailable"):
    return SimpleNamespace(
        active_checkpoint_path=path,
        active_checkpoint_source=source,
    )


def _window(tmp_path, monkeypatch, status):
    monkeypatch.setattr(
        "app.gui.model_selection_page.check_optional_packages",
        lambda *_args, **_kwargs: {
            "success": True,
            "packages": {"xgboost": True, "tabpfn": True, "torch": True},
            "missing": [],
            "error": None,
        },
    )
    current = {"status": status}
    monkeypatch.setattr(
        "app.gui.model_selection_page.get_tabpfn_model_status",
        lambda: current["status"],
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="tabpfn-selection",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.model_selection_page.refresh()
    return app, window, current


def test_missing_tabpfn_stays_visible_but_requires_setup(tmp_path, monkeypatch):
    app, window, _current = _window(tmp_path, monkeypatch, _status())
    page = window.model_selection_page

    assert page.model_checkboxes["tabpfn"].text() == "TabPFN 2.5"
    assert not page.model_checkboxes["tabpfn"].isEnabled()
    assert not page.model_checkboxes["tabpfn"].isChecked()
    assert page.tabpfn_checkpoint_status_label.text() == "Setup required"
    assert page.tabpfn_setup_button.text() == "Set Up TabPFN 2.5"
    assert not page.tabpfn_setup_button.isHidden()
    assert page.tabpfn_setup_button.objectName() == "primaryModelSelectionButton"

    page.model_checkboxes["random_forest"].setChecked(True)
    assert page.model_checkboxes["random_forest"].isEnabled()
    assert page.model_checkboxes["random_forest"].isChecked()
    window.close()
    assert app is not None


def test_tabpfn_setup_uses_existing_dialog_and_refreshes_after_success(
    tmp_path,
    monkeypatch,
):
    app, window, current = _window(tmp_path, monkeypatch, _status())
    page = window.model_selection_page
    opened = []
    monkeypatch.setattr(
        window,
        "show_tabpfn_model_status",
        lambda: opened.append("existing-dialog"),
    )

    page.tabpfn_setup_button.click()
    assert opened == ["existing-dialog"]

    checkpoint = tmp_path / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    current["status"] = _status(checkpoint, "user_cache")
    window.tabpfn_model_status_changed.emit(current["status"])
    app.processEvents()

    assert page.tabpfn_checkpoint_status_label.text() == "Ready"
    assert page.model_checkboxes["tabpfn"].isEnabled()
    assert page.tabpfn_setup_button.isHidden()
    window.close()


def test_legacy_tabpfn_checkpoint_remains_selectable(tmp_path, monkeypatch):
    checkpoint = tmp_path / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    app, window, _current = _window(
        tmp_path,
        monkeypatch,
        _status(checkpoint, "bundled_legacy"),
    )
    page = window.model_selection_page

    assert page.tabpfn_checkpoint_status_label.text() == "Legacy model available"
    assert page.model_checkboxes["tabpfn"].isEnabled()
    assert page.tabpfn_setup_button.isHidden()
    window.close()
    assert app is not None


def test_training_preflight_blocks_stale_tabpfn_selection(tmp_path, monkeypatch):
    app, window, _current = _window(
        tmp_path,
        monkeypatch,
        _status(tmp_path / "ready.ckpt", "user_cache"),
    )
    window.config.selected_models = ["tabpfn"]
    monkeypatch.setattr(
        "app.gui.training_page.get_tabpfn_model_status",
        lambda: _status(),
    )

    preflight = window.training_page._preflight_status(window.config)

    assert not preflight["ready"]
    assert any(
        "TabPFN 2.5 is not currently available" in message
        for message in preflight["messages"]
    )
    window.close()
    assert app is not None
