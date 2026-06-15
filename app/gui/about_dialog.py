"""Branded About dialog for AVISTA."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QVBoxLayout

from app.__version__ import APP_NAME, __version__
from app.branding import (
    APPLICATION_TAGLINE,
    DEVELOPERS,
    GITHUB_PROFILES,
)
from app.utils.resources import get_app_resource_path


LOGO_RESOURCE = "app/assets/logo.png"


def logo_path() -> Path:
    """Return the resolved bundled AVISTA logo path."""

    return get_app_resource_path(LOGO_RESOURCE)


def application_icon() -> QIcon:
    """Return the bundled AVISTA application icon."""

    return QIcon(str(logo_path()))


class AboutDialog(QDialog):
    """Display AVISTA identity, contributors, and clickable profile links."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("aboutAvistaDialog")
        self.setWindowTitle(f"About {APP_NAME}")
        self.setWindowIcon(application_icon())
        self.setModal(True)
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 24)
        layout.setSpacing(12)

        self.logo_label = QLabel()
        self.logo_label.setObjectName("aboutLogo")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(logo_path()))
        self.logo_label.setPixmap(
            pixmap.scaled(
                96,
                96,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(self.logo_label)

        self.title_label = QLabel(APP_NAME)
        self.title_label.setObjectName("aboutTitle")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.version_label = QLabel(f"Version {__version__}")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)

        self.description_label = QLabel(APPLICATION_TAGLINE)
        self.description_label.setObjectName("aboutDescription")
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        self.developers_label = QLabel(
            "<b>Developed by:</b><br>" + "<br>".join(DEVELOPERS)
        )
        self.developers_label.setTextFormat(Qt.TextFormat.RichText)
        self.developers_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.developers_label)

        self.github_label = QLabel(self._github_html())
        self.github_label.setObjectName("githubLinks")
        self.github_label.setTextFormat(Qt.TextFormat.RichText)
        self.github_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self.github_label.setOpenExternalLinks(False)
        self.github_label.linkActivated.connect(self.open_github_profile)
        self.github_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.github_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setStyleSheet(
            """
            QDialog { background: #F7F9FC; color: #1F2937; }
            QLabel#aboutTitle { font-size: 24px; font-weight: 700; color: #1F2937; }
            QLabel#aboutDescription { color: #4B5563; margin: 4px 12px 8px 12px; }
            QLabel#githubLinks { margin-top: 4px; }
            QPushButton {
                min-width: 90px;
                min-height: 32px;
                background: #0F6CBD;
                color: white;
                border: none;
                border-radius: 6px;
            }
            """
        )

    @staticmethod
    def _github_html() -> str:
        lines = ["<b>GitHub:</b>"]
        for name, url in GITHUB_PROFILES:
            lines.append(f'{name}:<br><a href="{url}">{url}</a>')
        return "<br><br>".join(lines)

    def open_github_profile(self, url: str) -> None:
        """Open a contributor profile in the default web browser."""

        QDesktopServices.openUrl(QUrl(url))
