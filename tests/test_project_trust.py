from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from app.gui.project_trust import confirm_project_file_open


def test_project_trust_warning_describes_referenced_artifacts_and_defaults_to_no(
    monkeypatch,
):
    captured = {}

    def question(parent, title, message, buttons, default_button):
        captured.update(
            title=title,
            message=message,
            buttons=buttons,
            default_button=default_button,
        )
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", question)

    accepted = confirm_project_file_open(None, Path("received-project.avista"))

    assert accepted is False
    assert captured["title"] == "Open AVISTA Project"
    assert "datasets and saved model or preprocessing artifacts" in captured["message"]
    assert "only be loaded from sources you trust" in captured["message"]
    assert captured["default_button"] == QMessageBox.StandardButton.No
