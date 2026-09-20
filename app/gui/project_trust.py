"""Trust confirmation for opening existing AVISTA project files."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget


def confirm_project_file_open(
    parent: QWidget | None,
    project_path: str | Path,
) -> bool:
    """Confirm before an existing project may load referenced local resources."""

    path = Path(project_path)
    response = QMessageBox.question(
        parent,
        "Open AVISTA Project",
        (
            "This project may reference datasets and saved model or preprocessing "
            "artifacts on this computer. Saved artifacts can use formats that should "
            "only be loaded from sources you trust.\n\n"
            f"Project file:\n{path}\n\n"
            "Continue opening this project?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return response == QMessageBox.StandardButton.Yes
