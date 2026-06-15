"""Main PySide6 window for AVISTA."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.__version__ import APP_NAME
from app.core.project_config import ProjectConfig
from app.gui.about_dialog import AboutDialog, application_icon, logo_path
from app.gui.column_config_page import ColumnConfigPage
from app.gui.data_import_page import DataImportPage
from app.gui.data_split_imbalance_page import DataSplitImbalancePage
from app.gui.edge_case_report_page import EdgeCaseReportPage
from app.gui.environment_page import EnvironmentPage
from app.gui.icon_system import BACKGROUND, PAGE_ICONS, PRIMARY, TEXT, icon
from app.gui.model_selection_page import ModelSelectionPage
from app.gui.project_setup_page import ProjectSetupPage
from app.gui.report_page import ReportPage
from app.gui.training_page import TrainingPage


class MainWindow(QMainWindow):
    """Application shell with left navigation and shared page state."""

    def __init__(self, initial_config: ProjectConfig | None = None) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(application_icon())
        self.resize(1200, 760)

        self.config: ProjectConfig | None = None
        self.dataframe: pd.DataFrame | None = None
        self.environment_info: dict | None = None
        self._startup_environment_check_scheduled = False

        self.stack = QStackedWidget()
        self.nav_buttons: list[QPushButton] = []

        self.project_setup_page = ProjectSetupPage(self)
        self.environment_page = EnvironmentPage(self)
        self.data_import_page = DataImportPage(self)
        self.column_config_page = ColumnConfigPage(self)
        self.data_split_imbalance_page = DataSplitImbalancePage(self)
        self.model_selection_page = ModelSelectionPage(self)
        self.edge_case_report_page = EdgeCaseReportPage(self)
        self.training_page = TrainingPage(self)
        self.report_page = ReportPage(self)

        self.pages: list[tuple[str, QWidget, Callable[[], None] | None]] = [
            ("Project Setup", self.project_setup_page, None),
            ("Environment", self.environment_page, self.environment_page.refresh),
            ("Data Import", self.data_import_page, None),
            ("Column Configuration", self.column_config_page, self.column_config_page.refresh),
            ("Data Split & Imbalance", self.data_split_imbalance_page, self.data_split_imbalance_page.refresh),
            ("Model Selection", self.model_selection_page, self.model_selection_page.refresh),
            ("Edge-Case Report", self.edge_case_report_page, self.edge_case_report_page.refresh),
            ("Training", self.training_page, self.training_page.refresh),
            ("Report", self.report_page, self.report_page.refresh),
        ]

        for _, page, _ in self.pages:
            self.stack.addWidget(page)

        root = QWidget()
        root.setObjectName("applicationRoot")
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(root)
        help_menu = self.menuBar().addMenu("&Help")
        about_action = help_menu.addAction(f"About {APP_NAME}")
        about_action.triggered.connect(self.show_about_dialog)
        if initial_config is not None:
            self.set_config(initial_config)
            self.project_setup_page._populate_create_fields(initial_config)
            self.project_setup_page.existing_project_file_input.setText(
                str(initial_config.project_file)
            )
            self.project_setup_page._show_loaded(initial_config)

    def showEvent(self, event: QShowEvent) -> None:
        """Start environment diagnostics after the main window is visible."""

        super().showEvent(event)
        if not self._startup_environment_check_scheduled:
            self._startup_environment_check_scheduled = True
            QTimer.singleShot(
                0,
                self.environment_page.start_startup_environment_check,
            )

    def show_about_dialog(self) -> None:
        """Show AVISTA product and developer information."""

        self.create_about_dialog().exec()

    def create_about_dialog(self) -> AboutDialog:
        """Create the About dialog for display or focused GUI testing."""

        return AboutDialog(self)

    def set_config(self, config: ProjectConfig) -> None:
        previous_project_file = self.config.project_file if self.config else None
        self.config = config
        if previous_project_file != config.project_file:
            self.dataframe = None
        self.environment_page.refresh()
        self.data_import_page.refresh()
        self.data_import_page.restore_project_dataset()
        self.column_config_page.refresh()
        self.data_split_imbalance_page.refresh()
        self.model_selection_page.refresh()
        self.report_page.refresh()

    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        self.column_config_page.refresh()
        self.data_split_imbalance_page.refresh()

    def set_environment_info(self, info: dict) -> None:
        self.environment_info = info
        self.environment_page.refresh()

    def navigate_to(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        refresh = self.pages[index][2]
        if refresh is not None:
            refresh()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(6, 0, 0, 8)
        logo = QLabel()
        logo.setObjectName("sidebarLogo")
        logo.setPixmap(
            QPixmap(str(logo_path())).scaled(
                36,
                36,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        title = QLabel(APP_NAME)
        title.setObjectName("sidebarTitle")
        title.setAlignment(Qt.AlignVCenter)
        brand_row.addWidget(logo)
        brand_row.addWidget(title)
        brand_row.addStretch(1)
        layout.addLayout(brand_row)

        for index, (label, _, _) in enumerate(self.pages):
            button = QPushButton(label)
            button.setObjectName("sidebarButton")
            button.setIcon(icon(PAGE_ICONS[label], "#DCEBFA"))
            button.setIconSize(QSize(18, 18))
            button.setCheckable(True)
            button.setMinimumHeight(42)
            button.clicked.connect(lambda checked=False, page_index=index: self.navigate_to(page_index))
            self.nav_buttons.append(button)
            layout.addWidget(button)

        layout.addStretch(1)
        self.nav_buttons[0].setChecked(True)
        sidebar.setStyleSheet(
            f"""
            QWidget#sidebar {{
                background: #17324D;
                border: none;
            }}
            QLabel#sidebarTitle {{
                color: #FFFFFF;
                font-size: 22px;
                font-weight: 700;
                padding: 4px 8px 14px 8px;
            }}
            QPushButton#sidebarButton {{
                color: #DCEBFA;
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 0 12px;
                text-align: left;
                font-weight: 500;
            }}
            QPushButton#sidebarButton:hover {{
                background: #244A6B;
                color: #FFFFFF;
            }}
            QPushButton#sidebarButton:checked {{
                background: {PRIMARY};
                color: #FFFFFF;
                font-weight: 600;
            }}
            """
        )
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {BACKGROUND}; color: {TEXT}; }}")
        return sidebar


def default_output_dir(project_dir: str) -> str:
    return str(Path(project_dir) / "outputs") if project_dir else ""
