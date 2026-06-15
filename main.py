"""Entry point for AVISTA."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen

from app.__version__ import APP_NAME, __version__
from app.branding import (
    APPLICATION_TAGLINE,
)
from app.core.project_config import ProjectConfig
from app.gui.about_dialog import application_icon
from app.gui.main_window import MainWindow


def load_startup_project(arguments: list[str]) -> ProjectConfig | None:
    """Load an optional AVISTA or legacy project command-line argument."""

    if not arguments:
        return None
    project_path = Path(arguments[0])
    if project_path.suffix.casefold() not in {".avista", ".xtab"}:
        raise ValueError(
            "Command-line project files must use .avista or legacy .xtab."
        )
    if not project_path.exists() or not project_path.is_file():
        raise FileNotFoundError(f"Project file does not exist: {project_path}")
    return ProjectConfig.load(project_path)


def create_splash_screen() -> QSplashScreen:
    """Create the branded startup splash screen."""

    pixmap = QPixmap(720, 360)
    pixmap.fill(QColor("#17324d"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#ffffff"))
    painter.setFont(QFont("Segoe UI", 34, QFont.Weight.Bold))
    painter.drawText(pixmap.rect().adjusted(48, 70, -48, -130), Qt.AlignLeft, APP_NAME)
    painter.setPen(QColor("#d9e7f2"))
    painter.setFont(QFont("Segoe UI", 15))
    painter.drawText(
        pixmap.rect().adjusted(48, 155, -48, -70),
        Qt.AlignLeft | Qt.TextWordWrap,
        APPLICATION_TAGLINE,
    )
    painter.setFont(QFont("Segoe UI", 10))
    painter.drawText(
        pixmap.rect().adjusted(48, 0, -48, -28),
        Qt.AlignLeft | Qt.AlignBottom,
        f"Version {__version__}",
    )
    painter.end()
    return QSplashScreen(pixmap)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(application_icon())
    splash = create_splash_screen()
    splash.show()
    app.processEvents()
    try:
        initial_config = load_startup_project(sys.argv[1:])
    except (OSError, ValueError) as exc:
        QMessageBox.critical(None, "AVISTA Project Error", str(exc))
        initial_config = None
    window = MainWindow(initial_config=initial_config)
    window.show()
    splash.finish(window)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
