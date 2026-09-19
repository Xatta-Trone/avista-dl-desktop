"""TabPFN 2.5 checkpoint status and setup dialog."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from app.core.tabpfn_model_manager import (
    TABPFN_LICENSE_URL,
    TabPFNModelState,
    TabPFNModelStatus,
)
from app.gui.about_dialog import application_icon
from app.gui.theme import apply_theme_to_widget, current_theme


class TabPFNModelDialog(QDialog):
    """Display availability and collect user-initiated model actions."""

    download_requested = Signal(bool)
    verify_requested = Signal()

    def __init__(
        self,
        status: TabPFNModelStatus,
        *,
        startup: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.status = status
        self.startup = startup
        self.setObjectName("tabpfnModelDialog")
        self.setWindowTitle("TabPFN 2.5 Model")
        self.setWindowIcon(application_icon())
        self.setMinimumWidth(620)
        self.setModal(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(12)

        title = QLabel("TabPFN 2.5 Model")
        title.setObjectName("tabpfnModelTitle")
        layout.addWidget(title)

        explanation = QLabel(
            "AVISTA supports TabPFN 2.5 as an optional tabular foundation model.\n\n"
            "The model weights are distributed separately by Prior Labs under the "
            "TabPFN-2.5 License v1.1 and are not part of AVISTA's Apache-2.0 license. "
            "Setup may open a browser so you can review and accept Prior Labs' license."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.status_label = QLabel()
        self.status_label.setObjectName("tabpfnModelStatus")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(
            self.status_label.textInteractionFlags()
        )
        layout.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)

        self.download_button = QPushButton()
        self.download_button.clicked.connect(self._request_download)
        self.verify_button = QPushButton("Verify Model")
        self.verify_button.clicked.connect(self.verify_requested.emit)
        self.cache_button = QPushButton("Open Cache Folder")
        self.cache_button.clicked.connect(self.open_cache_folder)
        self.license_button = QPushButton("View License")
        self.license_button.clicked.connect(self.open_license)
        self.close_button = QPushButton("Remind Me Later" if startup else "Close")
        self.close_button.clicked.connect(self.close)
        for button in (
            self.download_button,
            self.verify_button,
            self.cache_button,
            self.license_button,
            self.close_button,
        ):
            button.setProperty("avistaButtonRole", "secondary")
            button.setMinimumHeight(34)

        utility_actions = QHBoxLayout()
        utility_actions.setSpacing(8)
        utility_actions.addWidget(self.verify_button)
        utility_actions.addWidget(self.cache_button)
        utility_actions.addWidget(self.license_button)
        utility_actions.addStretch(1)
        layout.addLayout(utility_actions)

        footer_actions = QHBoxLayout()
        footer_actions.setSpacing(8)
        footer_actions.addWidget(self.download_button)
        footer_actions.addStretch(1)
        footer_actions.addWidget(self.close_button)
        layout.addLayout(footer_actions)
        self.refresh_status(status)
        apply_theme_to_widget(self, current_theme())

    def refresh_status(self, status: TabPFNModelStatus) -> None:
        self.status = status
        if status.state == TabPFNModelState.AVAILABLE_IN_USER_CACHE:
            headline = "Installed / Ready"
        elif status.state == TabPFNModelState.AVAILABLE_AS_LEGACY_BUNDLED_COPY:
            headline = "Legacy bundled checkpoint available"
        elif status.state == TabPFNModelState.DOWNLOAD_IN_PROGRESS:
            headline = "Download in progress"
        elif status.state == TabPFNModelState.DOWNLOAD_FAILED:
            headline = "Download failed"
        else:
            headline = "Not installed"

        legacy = "Available" if status.legacy_bundled_checkpoint_exists else "Not available"
        cached = "Installed" if status.cache_checkpoint_exists else "Not installed"
        failure = (
            f"\nLast download error: {status.last_download_error}"
            if status.last_download_error
            else ""
        )
        migration_note = (
            "\n\nA legacy bundled checkpoint is currently available. Future AVISTA "
            "releases will use the separately downloaded user-cache copy."
            if status.legacy_bundled_checkpoint_exists and not status.cache_checkpoint_exists
            else ""
        )
        self.status_label.setText(
            f"Status: {headline}\n"
            f"Model: TabPFN {status.model_version}\n"
            f"Package: tabpfn {status.package_version}\n"
            f"Checkpoint: {status.checkpoint_filename}\n"
            f"License: {status.license_name}\n"
            f"Preferred cache: {status.cache_path}\n"
            f"User-cache checkpoint: {cached}\n"
            f"Legacy bundled checkpoint: {legacy}\n"
            f"Active source: {status.active_checkpoint_source}"
            f"{failure}{migration_note}"
        )
        installed = status.cache_checkpoint_exists
        self.download_button.setText("Re-download" if installed else "Download / Set Up")
        self.download_button.setProperty("redownload", installed)
        self._set_button_role(
            self.download_button,
            "secondary" if installed else "primary",
        )
        self._set_button_role(
            self.verify_button,
            "primary" if installed else "secondary",
        )
        self.verify_button.setVisible(installed)
        self.cache_button.setVisible(installed)
        self.progress.setVisible(status.state == TabPFNModelState.DOWNLOAD_IN_PROGRESS)
        self._set_action_buttons_enabled(status.state != TabPFNModelState.DOWNLOAD_IN_PROGRESS)

    def show_progress(self, message: str) -> None:
        self.progress.show()
        self.status_label.setText(message)
        self._set_action_buttons_enabled(False)

    def show_failure(self, message: str, status: TabPFNModelStatus) -> None:
        self.progress.hide()
        self.refresh_status(status)
        self.status_label.setText(
            "TabPFN 2.5 could not be downloaded or verified.\n\n"
            "The rest of AVISTA remains available. You can retry from Help → "
            f"TabPFN Model Status.\n\nDetails: {message}"
        )

    def _request_download(self) -> None:
        self.download_requested.emit(bool(self.download_button.property("redownload")))

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for button in (
            self.download_button,
            self.verify_button,
            self.cache_button,
            self.license_button,
        ):
            button.setEnabled(enabled)

    @staticmethod
    def _set_button_role(button: QPushButton, role: str) -> None:
        button.setProperty("avistaButtonRole", role)
        style = button.style()
        style.unpolish(button)
        style.polish(button)
        button.update()

    def open_license(self) -> None:
        QDesktopServices.openUrl(QUrl(TABPFN_LICENSE_URL))

    def open_cache_folder(self) -> None:
        path = Path(self.status.cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
