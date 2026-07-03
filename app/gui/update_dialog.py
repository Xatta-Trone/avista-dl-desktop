"""Update dialogs for AVISTA."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from app.core.update_checker import UpdateMetadata
from app.core.user_settings import (
    UserSettings,
    load_user_settings,
    save_user_settings,
)
from app.gui.theme import apply_theme_to_widget, current_theme
from app.utils.resources import get_app_resource_path


def _application_icon() -> QIcon:
    return QIcon(str(get_app_resource_path("app/assets/logo.png")))


class UpdateAvailableDialog(QDialog):
    """Show release information and collect the user's update choice."""

    DOWNLOAD = 1
    LATER = 0
    SKIP = 2

    def __init__(
        self,
        *,
        current_version: str,
        metadata: UpdateMetadata,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.metadata = metadata
        self.choice = self.LATER
        self.setObjectName("updateAvailableDialog")
        self.setWindowTitle("AVISTA Update Available")
        self.setWindowIcon(_application_icon())
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(12)

        title = QLabel("AVISTA Update Available")
        title.setObjectName("updateTitle")
        layout.addWidget(title)

        notes = "\n".join(f"- {note}" for note in metadata.release_notes) or "- No release notes provided."
        details = QLabel(
            f"Current version: {current_version}\n"
            f"Latest version: {metadata.latest_version}\n"
            f"Release date: {metadata.release_date or 'Not specified'}\n\n"
            f"Release notes:\n{notes}"
        )
        details.setWordWrap(True)
        details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(details)

        self.auto_check_box = QCheckBox("Check for updates automatically")
        self.auto_check_box.setChecked(load_user_settings().auto_check_updates)
        self.auto_check_box.toggled.connect(self._save_auto_check)
        layout.addWidget(self.auto_check_box)

        buttons = QDialogButtonBox()
        self.download_button = QPushButton("Download and Install")
        self.later_button = QPushButton("Later")
        self.skip_button = QPushButton("Skip This Version")
        buttons.addButton(self.download_button, QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(self.later_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self.skip_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        self.download_button.clicked.connect(self._download)
        self.later_button.clicked.connect(self._later)
        self.skip_button.clicked.connect(self._skip)
        layout.addWidget(buttons)

        self.setStyleSheet(
            """
            QDialog { background: #F7F9FC; color: #1F2937; }
            QLabel#updateTitle { font-size: 22px; font-weight: 700; color: #1F2937; }
            QPushButton {
                min-width: 120px;
                min-height: 32px;
                border-radius: 6px;
                padding: 4px 12px;
            }
            """
        )
        apply_theme_to_widget(self, current_theme())

    def _download(self) -> None:
        self.choice = self.DOWNLOAD
        self.accept()

    def _later(self) -> None:
        self.choice = self.LATER
        self.accept()

    def _skip(self) -> None:
        settings = load_user_settings()
        settings.skipped_update_version = self.metadata.latest_version
        save_user_settings(settings)
        self.choice = self.SKIP
        self.accept()

    def _save_auto_check(self, checked: bool) -> None:
        settings = load_user_settings()
        settings.auto_check_updates = checked
        save_user_settings(settings)


class UpdateDownloadDialog(QDialog):
    """Show update download progress."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("updateDownloadDialog")
        self.setWindowTitle("Downloading AVISTA Update")
        self.setWindowIcon(_application_icon())
        self.setModal(True)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(12)
        self.status_label = QLabel("Downloading installer...")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)
        apply_theme_to_widget(self, current_theme())

    def update_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(downloaded)
            percent = min(100, int(downloaded * 100 / total))
            self.status_label.setText(f"Downloading installer... {percent}%")
        else:
            self.progress_bar.setRange(0, 0)
            self.status_label.setText("Downloading installer...")

    def show_finished(self, installer_path: str | Path) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.status_label.setText(f"Downloaded: {Path(installer_path).name}")


def set_auto_update_check_enabled(enabled: bool) -> UserSettings:
    """Persist the update auto-check preference."""

    settings = load_user_settings()
    settings.auto_check_updates = enabled
    save_user_settings(settings)
    return settings
