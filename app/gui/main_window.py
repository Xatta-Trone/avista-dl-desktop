"""Main PySide6 window for AVISTA."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

import pandas as pd
from PySide6.QtCore import QProcess, QThread, QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.__version__ import APP_NAME
from app.core.project_config import ProjectConfig
from app.core.update_checker import UpdateCheckResult, log_update_message
from app.core.user_settings import load_user_settings
from app.gui.about_dialog import AboutDialog, application_icon, logo_path
from app.gui.column_config_page import ColumnConfigPage
from app.gui.data_import_page import DataImportPage
from app.gui.data_split_imbalance_page import DataSplitImbalancePage
from app.gui.edge_case_report_page import EdgeCaseReportPage
from app.gui.environment_page import EnvironmentPage
from app.gui.icon_system import PAGE_ICONS, icon
from app.gui.model_selection_page import ModelSelectionPage
from app.gui.project_setup_page import ProjectSetupPage
from app.gui.report_page import ReportPage
from app.gui.theme import apply_theme, get_theme, load_theme_setting, save_theme_setting
from app.gui.training_page import TrainingPage
from app.gui.update_dialog import UpdateAvailableDialog, UpdateDownloadDialog
from app.gui.workers import UpdateCheckWorker, UpdateDownloadWorker


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
        self._startup_update_check_scheduled = False
        self._update_threads: list[QThread] = []
        self._update_workers: list[object] = []
        self._download_dialog: UpdateDownloadDialog | None = None
        self.theme_name = load_theme_setting()

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
        theme_menu = help_menu.addMenu("Theme")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)
        self.light_theme_action = QAction("Light", self, checkable=True)
        self.dark_theme_action = QAction("Dark", self, checkable=True)
        self.light_theme_action.setData("light")
        self.dark_theme_action.setData("dark")
        for action in (self.light_theme_action, self.dark_theme_action):
            self.theme_action_group.addAction(action)
            theme_menu.addAction(action)
        self.light_theme_action.setChecked(self.theme_name == "light")
        self.dark_theme_action.setChecked(self.theme_name == "dark")
        self.theme_action_group.triggered.connect(self.set_theme)
        update_action = help_menu.addAction("Check for Updates")
        update_action.triggered.connect(self.check_for_updates_manually)
        about_action = help_menu.addAction(f"About {APP_NAME}")
        about_action.triggered.connect(self.show_about_dialog)
        if initial_config is not None:
            self.set_config(initial_config)
            self.project_setup_page._populate_create_fields(initial_config)
            self.project_setup_page.existing_project_file_input.setText(
                str(initial_config.project_file)
            )
            self.project_setup_page._show_loaded(initial_config)
        self.apply_current_theme()

    def showEvent(self, event: QShowEvent) -> None:
        """Start environment diagnostics after the main window is visible."""

        super().showEvent(event)
        if not self._startup_environment_check_scheduled:
            self._startup_environment_check_scheduled = True
            QTimer.singleShot(
                0,
                self.environment_page.start_startup_environment_check,
            )
        if not self._startup_update_check_scheduled:
            self._startup_update_check_scheduled = True
            QTimer.singleShot(1000, self.start_startup_update_check)

    def show_about_dialog(self) -> None:
        """Show AVISTA product and developer information."""

        dialog = self.create_about_dialog()
        dialog.exec()

    def set_theme(self, action: QAction) -> None:
        """Persist and immediately apply a selected application theme."""

        theme_name = str(action.data() or "light")
        self.theme_name = save_theme_setting(theme_name)
        self.light_theme_action.setChecked(self.theme_name == "light")
        self.dark_theme_action.setChecked(self.theme_name == "dark")
        self.apply_current_theme()

    def apply_current_theme(self) -> None:
        """Apply the active theme to the app shell and page widgets."""

        app = QApplication.instance()
        if app is not None:
            apply_theme(app, self.theme_name)
        self.stack.setStyleSheet(
            f"QStackedWidget {{ background: {self.palette().window().color().name()}; }}"
        )
        self._apply_sidebar_theme()
        for _, page, _ in self.pages:
            if hasattr(page, "apply_theme"):
                page.apply_theme(self.theme_name)

    def start_startup_update_check(self) -> None:
        """Run one silent automatic update check after startup."""

        if not load_user_settings().auto_check_updates:
            return
        self._start_update_check(manual=False)

    def check_for_updates_manually(self) -> None:
        """Run a user-requested update check."""

        self._start_update_check(manual=True)

    def _start_update_check(self, *, manual: bool) -> None:
        thread = QThread(self)
        worker = UpdateCheckWorker(manual=manual)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(
            partial(
                self._handle_update_check_finished,
                manual=manual,
                thread=thread,
                worker=worker,
            )
        )
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        thread.finished.connect(partial(self._discard_update_worker, thread, worker))
        thread.finished.connect(thread.deleteLater)
        self._update_threads.append(thread)
        self._update_workers.append(worker)
        thread.start()

    def _handle_update_check_finished(
        self,
        result: UpdateCheckResult,
        *,
        manual: bool,
        thread: QThread | object,
        worker: object,
    ) -> None:
        """Compatibility helper for focused tests and direct GUI calls."""

        self._process_update_check_result(result, manual=manual)

    def _process_update_check_result(
        self,
        result: UpdateCheckResult,
        *,
        manual: bool,
    ) -> None:
        if result.error:
            if manual:
                QMessageBox.warning(
                    self,
                    "AVISTA Update Check",
                    "Could not check for updates. Please check your internet connection.",
                )
            return
        if not result.update_available or result.metadata is None:
            if manual:
                QMessageBox.information(
                    self,
                    "AVISTA Update Check",
                    "AVISTA is up to date.",
                )
            return
        if result.skipped and not manual:
            return
        dialog = UpdateAvailableDialog(
            current_version=result.current_version,
            metadata=result.metadata,
            parent=self,
        )
        dialog.exec()
        if dialog.choice == UpdateAvailableDialog.DOWNLOAD:
            self._start_update_download(result.metadata)

    def _start_update_download(self, metadata) -> None:
        self._download_dialog = UpdateDownloadDialog(self)
        self._download_dialog.show()
        thread = QThread(self)
        worker = UpdateDownloadWorker(metadata)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._download_dialog.update_progress)
        worker.finished.connect(
            partial(
                self._handle_update_download_finished,
                thread=thread,
                worker=worker,
            )
        )
        worker.failed.connect(
            partial(
                self._handle_update_download_failed,
                thread=thread,
                worker=worker,
            )
        )
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(partial(self._discard_update_worker, thread, worker))
        thread.finished.connect(thread.deleteLater)
        self._update_threads.append(thread)
        self._update_workers.append(worker)
        thread.start()

    def _handle_update_download_finished(
        self,
        installer_path: str,
        *,
        thread: QThread,
        worker: object,
    ) -> None:
        if self._download_dialog is not None:
            self._download_dialog.show_finished(installer_path)
        response = QMessageBox.question(
            self,
            "Install AVISTA Update",
            "AVISTA will close and launch the installer. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if response != QMessageBox.StandardButton.Yes:
            log_update_message("Install launch cancelled by user.")
            return
        launched = QProcess.startDetached(installer_path, [])
        log_update_message(
            f"Install launch status: {'started' if launched else 'failed'}; path={installer_path}"
        )
        if launched:
            QApplication.instance().quit()
        else:
            QMessageBox.critical(
                self,
                "AVISTA Update",
                "Could not launch the downloaded installer.",
            )

    def _handle_update_download_failed(
        self,
        message: str,
        *,
        thread: QThread,
        worker: object,
    ) -> None:
        if self._download_dialog is not None:
            self._download_dialog.close()
            self._download_dialog = None
        QMessageBox.critical(self, "AVISTA Update Download Failed", message)

    def _discard_update_worker(self, thread: QThread, worker: object) -> None:
        if thread in self._update_threads:
            self._update_threads.remove(thread)
        if worker in self._update_workers:
            self._update_workers.remove(worker)

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
        self.sidebar = sidebar
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
        self.sidebar_title = title
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
        return sidebar

    def _apply_sidebar_theme(self) -> None:
        theme = get_theme(self.theme_name)
        for index, (label, _, _) in enumerate(self.pages):
            if index < len(self.nav_buttons):
                self.nav_buttons[index].setIcon(icon(PAGE_ICONS[label], theme.sidebar_text))
        self.sidebar.setStyleSheet(
            f"""
            QWidget#sidebar {{
                background: {theme.sidebar};
                border: none;
            }}
            QLabel#sidebarTitle {{
                color: #FFFFFF;
                font-size: 22px;
                font-weight: 700;
                padding: 4px 8px 14px 8px;
            }}
            QPushButton#sidebarButton {{
                color: {theme.sidebar_text};
                background: transparent;
                border: none;
                border-radius: 7px;
                padding: 0 12px;
                text-align: left;
                font-weight: 500;
            }}
            QPushButton#sidebarButton:hover {{
                background: {theme.sidebar_hover};
                color: #FFFFFF;
            }}
            QPushButton#sidebarButton:checked {{
                background: {theme.primary};
                color: #FFFFFF;
                font-weight: 600;
            }}
            """
        )


def default_output_dir(project_dir: str) -> str:
    return str(Path(project_dir) / "outputs") if project_dir else ""
