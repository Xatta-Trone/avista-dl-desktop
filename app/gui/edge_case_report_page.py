"""AVISTA edge-case validation report page."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, QTimer, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.__version__ import APP_NAME, __version__
from app.core.error_handler import ERROR, FATAL, WARNING, EdgeCaseReport, Issue
from app.core.project_config import ProjectConfig
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon
from app.gui.workers import EdgeCaseCheckWorker

SUCCESS = "#1A7F37"
WARNING_COLOR = "#BF6A02"
ERROR_COLOR = "#CF222E"
MUTED = "#5B6573"


class EdgeCaseReportPage(QWidget):
    """Display saved validation results and run checks in a worker thread."""

    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.thread: QThread | None = None
        self.worker: EdgeCaseCheckWorker | None = None
        self.current_report: EdgeCaseReport | None = None
        self.current_report_path: Path | None = None
        self.report_metadata: dict[str, Any] = {}
        self.setObjectName("edgeCaseReportPage")

        self.notification_timer = QTimer(self)
        self.notification_timer.setSingleShot(True)
        self.notification_timer.timeout.connect(self.notification_card.hide)

        self.content_stack = QStackedWidget()
        self.empty_state = self._build_empty_state()
        self.report_content = self._build_report_content()
        self.content_stack.addWidget(self.empty_state)
        self.content_stack.addWidget(self.report_content)

        page = QWidget()
        page.setObjectName("edgeCaseScrollContent")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(24, 24, 24, 24)
        page_layout.setSpacing(16)
        title = QLabel("Edge-Case Report")
        title.setObjectName("edgeCaseTitle")
        subtitle = QLabel(
            "Review data quality, target integrity, split consistency, and training "
            "readiness before model execution."
        )
        subtitle.setObjectName("edgeCaseSubtitle")
        subtitle.setWordWrap(True)
        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)
        page_layout.addWidget(self.notification_card)
        page_layout.addWidget(self.content_stack)
        page_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setObjectName("edgeCaseScrollArea")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(scroll)
        self._apply_style()
        self._show_empty_state()

    def _build_empty_state(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 36, 0, 24)
        layout.addStretch(1)
        card = self._card("edgeCaseEmptyCard")
        card.setMaximumWidth(620)
        card_layout = card.layout()
        empty_icon = QLabel()
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setPixmap(icon("fa6s.shield-halved").pixmap(52, 52))
        heading = QLabel("No validation report available.")
        heading.setObjectName("edgeCaseEmptyTitle")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message = QLabel("Run Edge-Case Checks to generate a report.")
        message.setObjectName("edgeCaseMuted")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_run_button = self._button(
            "Run Edge-Case Checks",
            "fa6s.magnifying-glass-chart",
            primary=True,
        )
        self.empty_run_button.clicked.connect(self.run_checks)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.empty_run_button)
        button_row.addStretch(1)
        card_layout.addWidget(empty_icon)
        card_layout.addWidget(heading)
        card_layout.addWidget(message)
        card_layout.addLayout(button_row)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return wrapper

    def _build_report_content(self) -> QWidget:
        content = QWidget()
        content.setObjectName("edgeCaseGeneratedContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        layout.addWidget(self._build_configuration_card())
        layout.addWidget(self._build_validation_card())
        layout.addWidget(self._build_status_card())

        self.issue_tables: dict[str, QTableWidget] = {}
        self.issue_empty_states: dict[str, QFrame] = {}
        for level, title, icon_name, empty_message in (
            (FATAL, "Fatal Issues", "fa6s.circle-xmark", "No fatal issues detected."),
            (ERROR, "Errors", "fa6s.triangle-exclamation", "No errors detected."),
            (WARNING, "Warnings", "fa6s.circle-info", "No warnings detected."),
        ):
            card, table, empty = self._build_issue_card(
                level, title, icon_name, empty_message
            )
            self.issue_tables[level] = table
            self.issue_empty_states[level] = empty
            layout.addWidget(card)

        layout.addWidget(self._build_readiness_card())
        layout.addWidget(self._build_output_card())
        layout.addWidget(self._build_actions_card())
        return content

    def _build_configuration_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseConfigurationCard",
            "fa6s.shield",
            "Configuration Summary",
        )
        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        self.summary_values: dict[str, QLabel] = {}
        for key, label in (
            ("target", "Target Column"),
            ("features", "Selected Features"),
            ("split", "Split Confirmed"),
            ("imbalance", "Imbalance Method"),
            ("timestamp", "Timestamp"),
            ("project", "Project"),
        ):
            value = QLabel("Not available")
            value.setObjectName("edgeCaseValue")
            value.setWordWrap(True)
            self.summary_values[key] = value
            form.addRow(label, value)
        body.addLayout(form)

        self.target_label = QLabel()
        self.feature_count_label = QLabel()
        self.split_status_label = QLabel()
        self.imbalance_label = QLabel()
        for legacy_label in (
            self.target_label,
            self.feature_count_label,
            self.split_status_label,
            self.imbalance_label,
        ):
            legacy_label.hide()
            body.addWidget(legacy_label)
        return card

    def _build_validation_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseValidationCard",
            "fa6s.magnifying-glass-chart",
            "Edge-Case Validation",
        )
        controls = QHBoxLayout()
        self.run_button = self._button(
            "Run Edge-Case Checks",
            "fa6s.magnifying-glass-chart",
            primary=True,
        )
        self.run_button.clicked.connect(self.run_checks)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        self.progress_message = QLabel("Validation report is ready.")
        self.progress_message.setObjectName("edgeCaseMuted")
        self.progress_message.setWordWrap(True)
        self.status_label = self.progress_message
        controls.addWidget(self.run_button)
        controls.addStretch(1)
        body.addLayout(controls)
        body.addWidget(self.progress)
        body.addWidget(self.progress_message)
        return card

    def _build_status_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseStatusCard",
            "fa6s.clipboard-check",
            "Validation Status",
        )
        row = QHBoxLayout()
        row.setSpacing(12)
        self.status_tiles: dict[str, tuple[QFrame, QLabel]] = {}
        for key, title in (
            ("ready", "Training Ready"),
            ("fatals", "Fatal Issues"),
            ("errors", "Errors"),
            ("warnings", "Warnings"),
        ):
            tile = QFrame()
            tile.setObjectName("edgeCaseStatusTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(14, 12, 14, 12)
            tile_layout.setSpacing(5)
            tile_title = QLabel(title)
            tile_title.setObjectName("edgeCaseTileTitle")
            value = QLabel("0")
            value.setObjectName("edgeCaseTileValue")
            tile_layout.addWidget(tile_title)
            tile_layout.addWidget(value)
            self.status_tiles[key] = (tile, value)
            row.addWidget(tile, stretch=1)
        body.addLayout(row)
        return card

    def _build_issue_card(
        self,
        level: str,
        title: str,
        icon_name: str,
        empty_message: str,
    ) -> tuple[QWidget, QTableWidget, QFrame]:
        card, body = self._section_card(
            f"edgeCase{level.title()}Card",
            icon_name,
            title,
        )
        table = QTableWidget(0, 4)
        final_header = "Recommendation" if level == WARNING else "Recommended Fix"
        table.setHorizontalHeaderLabels(
            ["Issue", "Description", "Affected Column", final_header]
        )
        table.setObjectName("edgeCaseIssueTable")
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSortingEnabled(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionsClickable(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        table.setMinimumHeight(170)

        empty = QFrame()
        empty.setObjectName("edgeCaseSuccessState")
        empty_layout = QHBoxLayout(empty)
        empty_layout.setContentsMargins(18, 16, 18, 16)
        success_icon = QLabel()
        success_icon.setPixmap(icon("fa6s.circle-check", SUCCESS).pixmap(24, 24))
        success_label = QLabel(empty_message)
        success_label.setObjectName("edgeCaseSuccessText")
        empty_layout.addStretch(1)
        empty_layout.addWidget(success_icon)
        empty_layout.addWidget(success_label)
        empty_layout.addStretch(1)
        body.addWidget(table)
        body.addWidget(empty)
        return card, table, empty

    def _build_readiness_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseReadinessCard",
            "fa6s.rocket",
            "Training Readiness Assessment",
        )
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        self.readiness_values: dict[str, QLabel] = {}
        for key, label in (
            ("training", "Training Allowed"),
            ("target", "Target Ready"),
            ("split", "Split Ready"),
            ("encoding", "Encoding Ready"),
            ("coverage", "Class Coverage Ready"),
            ("imbalance", "Imbalance Ready"),
        ):
            value = QLabel()
            value.setObjectName("edgeCaseReadinessValue")
            self.readiness_values[key] = value
            form.addRow(label, value)
        body.addLayout(form)
        score_row = QHBoxLayout()
        self.readiness_score = QLabel("0 / 6 checks passed")
        self.readiness_score.setObjectName("edgeCaseReadinessScore")
        self.overall_status = QLabel("NOT READY")
        self.overall_status.setObjectName("edgeCaseOverallStatus")
        score_row.addWidget(self.readiness_score)
        score_row.addStretch(1)
        score_row.addWidget(self.overall_status)
        body.addLayout(score_row)
        return card

    def _build_output_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseOutputCard",
            "fa6s.file-shield",
            "Generated Report",
        )
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        self.output_values: dict[str, QLabel] = {}
        for key, label in (
            ("location", "Report Location"),
            ("generated", "Generated"),
            ("size", "File Size"),
            ("version", "Version"),
        ):
            value = QLabel("Not available")
            value.setObjectName("edgeCaseValue")
            value.setWordWrap(True)
            value.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self.output_values[key] = value
            form.addRow(label, value)
        body.addLayout(form)
        return card

    def _build_actions_card(self) -> QWidget:
        card, body = self._section_card(
            "edgeCaseActionsCard",
            "fa6s.bolt",
            "Actions",
        )
        row = QHBoxLayout()
        row.setSpacing(10)
        self.open_folder_button = self._button(
            "Open Report Folder", "fa6s.folder-open"
        )
        self.open_file_button = self._button("Open Report File", "fa6s.file")
        self.copy_button = self._button("Copy Summary", "fa6s.copy")
        self.export_pdf_button = self._button("Export PDF", "fa6s.file-pdf")
        self.rerun_button = self._button(
            "Re-run Checks", "fa6s.rotate-right", primary=True
        )
        self.open_folder_button.clicked.connect(self.open_report_folder)
        self.open_file_button.clicked.connect(self.open_report_file)
        self.copy_button.clicked.connect(self.copy_summary)
        self.export_pdf_button.clicked.connect(self.export_pdf)
        self.rerun_button.clicked.connect(self.run_checks)
        for button in (
            self.open_folder_button,
            self.open_file_button,
            self.copy_button,
            self.export_pdf_button,
            self.rerun_button,
        ):
            row.addWidget(button)
        row.addStretch(1)
        body.addLayout(row)
        return card

    @property
    def notification_card(self) -> QFrame:
        if not hasattr(self, "_notification_card"):
            card = QFrame()
            card.setObjectName("edgeCaseNotification")
            card.hide()
            layout = QHBoxLayout(card)
            layout.setContentsMargins(12, 9, 10, 9)
            layout.setSpacing(8)
            self.notification_icon = QLabel()
            self.notification_icon.setFixedSize(18, 18)
            self.notification_label = QLabel()
            self.notification_label.setWordWrap(True)
            self.notification_close = QPushButton()
            self.notification_close.setObjectName("edgeCaseNotificationClose")
            self.notification_close.setIcon(icon("fa6s.xmark", MUTED))
            self.notification_close.setFixedSize(28, 28)
            self.notification_close.clicked.connect(card.hide)
            layout.addWidget(self.notification_icon)
            layout.addWidget(self.notification_label, stretch=1)
            layout.addWidget(self.notification_close)
            self._notification_card = card
        return self._notification_card

    def refresh(self) -> None:
        config = self._latest_config()
        if config is None:
            self.current_report = None
            self.current_report_path = None
            self._show_empty_state()
            return
        path = self._report_path(config)
        if not path.exists():
            self.current_report = None
            self.current_report_path = path
            self._show_empty_state()
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            report = EdgeCaseReport.from_dict(data)
        except (OSError, TypeError, ValueError, json.JSONDecodeError, KeyError) as exc:
            self.current_report = None
            self.current_report_path = path
            self._show_empty_state()
            self._show_notification("error", f"Could not load validation report: {exc}")
            return
        if report.context.get("target_column") != config.target_column:
            self.current_report = None
            self.current_report_path = path
            self._show_empty_state()
            self._show_notification(
                "warning",
                "The saved report belongs to a different target. Run checks again.",
            )
            return
        self.report_metadata = data
        self._render_report(report, path, config)

    def run_checks(self) -> None:
        if self.thread is not None:
            return
        config = self._latest_config()
        dataframe = self.main_window.dataframe
        if config is None or dataframe is None:
            self._show_notification(
                "error",
                "Project configuration and dataset are required.",
            )
            return

        report_path = self._report_path(config)
        self.run_button.setEnabled(False)
        self.empty_run_button.setEnabled(False)
        self.rerun_button.setEnabled(False)
        self.progress.show()
        self.progress_message.setText("Running edge-case validation...")

        self.thread = QThread(self)
        self.worker = EdgeCaseCheckWorker(
            dataframe,
            config,
            self.main_window.environment_info,
            str(report_path),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_message.setText)
        self.worker.finished.connect(self._checks_finished)
        self.worker.failed.connect(self._checks_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(self._clear_thread_references)
        self.thread.start()

    def _checks_finished(self, report: EdgeCaseReport, path_text: str) -> None:
        config = self._latest_config()
        path = Path(path_text)
        self.report_metadata = self._read_report_metadata(path)
        self._set_running(False)
        if config is not None:
            self._render_report(report, path, config)
        if report.can_continue and report.warnings:
            self._show_notification(
                "warning",
                "Validation completed with warnings. Training remains available.",
            )
        elif report.can_continue:
            self._show_notification(
                "success",
                "Edge-case validation completed. Training is ready.",
            )
        elif report.fatals or report.errors:
            self._show_notification(
                "error",
                "Validation completed with blocking issues. Review the report below.",
            )

    def _checks_failed(self, message: str) -> None:
        self._set_running(False)
        self.progress_message.setText("Validation failed.")
        self._show_notification("error", f"Edge-case validation failed: {message}")

    def _clear_thread_references(self) -> None:
        self.thread = None
        self.worker = None

    def _set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.empty_run_button.setEnabled(not running)
        self.rerun_button.setEnabled(not running)
        self.progress.setVisible(running)

    def _render_report(
        self,
        report: EdgeCaseReport,
        path: Path,
        config: ProjectConfig,
    ) -> None:
        self.current_report = report
        self.current_report_path = path
        self.content_stack.setCurrentWidget(self.report_content)
        context = report.context
        split_confirmed = bool(context.get("split_confirmed"))
        generated = self._generated_datetime(path)
        generated_text = generated.strftime("%Y-%m-%d %I:%M %p")
        imbalance = str(context.get("imbalance_method") or "none")

        self.summary_values["target"].setText(
            str(context.get("target_column") or "Not confirmed")
        )
        self.summary_values["features"].setText(
            str(context.get("feature_count", len(config.feature_columns or [])))
        )
        self.summary_values["split"].setText("Yes" if split_confirmed else "No")
        self.summary_values["imbalance"].setText(imbalance)
        self.summary_values["timestamp"].setText(generated_text)
        self.summary_values["project"].setText(config.project_name)

        self.target_label.setText(
            f"Current target column: {context.get('target_column') or 'not confirmed'}"
        )
        self.feature_count_label.setText(
            f"Selected feature columns: {context.get('feature_count', 0)}"
        )
        self.split_status_label.setText(
            f"Data Split & Imbalance confirmed: {'Yes' if split_confirmed else 'No'}"
        )
        self.imbalance_label.setText(f"Imbalance method: {imbalance}")
        self.status_label.setText(
            f"Can continue: {report.can_continue}. "
            f"Fatals: {len(report.fatals)}, Errors: {len(report.errors)}, "
            f"Warnings: {len(report.warnings)}. Saved {path}"
        )

        self._set_status_tile(
            "ready",
            "YES" if report.can_continue else "NO",
            SUCCESS if report.can_continue else ERROR_COLOR,
        )
        self._set_status_tile(
            "fatals",
            str(len(report.fatals)),
            SUCCESS if not report.fatals else ERROR_COLOR,
        )
        self._set_status_tile(
            "errors",
            str(len(report.errors)),
            SUCCESS if not report.errors else ERROR_COLOR,
        )
        self._set_status_tile(
            "warnings",
            str(len(report.warnings)),
            SUCCESS if not report.warnings else WARNING_COLOR,
        )
        for level, issues in (
            (FATAL, report.fatals),
            (ERROR, report.errors),
            (WARNING, report.warnings),
        ):
            self._populate_issue_table(level, issues)
        self._render_readiness(report)
        self._render_output(path, generated)

    def _populate_issue_table(self, level: str, issues: list[Issue]) -> None:
        table = self.issue_tables[level]
        table.setSortingEnabled(False)
        table.setRowCount(len(issues))
        for row, issue in enumerate(issues):
            values = (
                issue.category.replace("_", " ").title(),
                issue.message,
                self._affected_column(issue),
                issue.suggestion,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row, column, item)
        table.setSortingEnabled(True)
        table.setVisible(bool(issues))
        self.issue_empty_states[level].setVisible(not issues)

    def _render_readiness(self, report: EdgeCaseReport) -> None:
        context = report.context
        blocking_categories = {
            issue.category for issue in report.issues if issue.level in {FATAL, ERROR}
        }
        states = {
            "training": report.can_continue,
            "target": bool(context.get("target_column"))
            and not blocking_categories.intersection({"target", "configuration"}),
            "split": bool(context.get("split_confirmed"))
            and not blocking_categories.intersection({"split", "artifacts"}),
            "encoding": bool(context.get("column_configuration_confirmed"))
            and not blocking_categories.intersection(
                {"features", "target", "configuration", "artifacts"}
            ),
            "coverage": not blocking_categories.intersection({"split"}),
            "imbalance": bool(context.get("split_confirmed"))
            and not blocking_categories.intersection({"imbalance", "artifacts"}),
        }
        for key, passed in states.items():
            label = self.readiness_values[key]
            label.setText("PASS" if passed else "FAIL")
            label.setPixmap(
                icon(
                    "fa6s.circle-check" if passed else "fa6s.circle-xmark",
                    SUCCESS if passed else ERROR_COLOR,
                ).pixmap(18, 18)
            )
            label.setToolTip("Ready" if passed else "Review blocking validation issues")
        passed_count = sum(states.values())
        self.readiness_score.setText(f"{passed_count} / {len(states)} checks passed")
        self.overall_status.setText(
            "READY FOR TRAINING" if report.can_continue else "NOT READY FOR TRAINING"
        )
        color = SUCCESS if report.can_continue else ERROR_COLOR
        background = "#DAFBE1" if report.can_continue else "#FFEBE9"
        self.overall_status.setStyleSheet(
            f"color: {color}; background: {background}; border-radius: 9px;"
            "padding: 4px 10px; font-weight: 700;"
        )

    def _render_output(self, path: Path, generated: datetime) -> None:
        footer = dict(self.report_metadata.get("report_footer") or {})
        self.output_values["location"].setText(str(path))
        self.output_values["generated"].setText(
            generated.strftime("%Y-%m-%d %I:%M %p")
        )
        self.output_values["size"].setText(self._format_size(path.stat().st_size))
        self.output_values["version"].setText(
            f"{footer.get('generated_by', APP_NAME)} "
            f"{footer.get('version', __version__)}"
        )

    def open_report_folder(self) -> None:
        if self.current_report_path is not None:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.current_report_path.parent.resolve()))
            )

    def open_report_file(self) -> None:
        if self.current_report_path is not None and self.current_report_path.exists():
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.current_report_path.resolve()))
            )

    def copy_summary(self) -> None:
        if self.current_report is None:
            return
        QApplication.clipboard().setText(self._summary_text())
        self._show_notification("success", "Validation summary copied.")

    def export_pdf(self) -> None:
        if self.current_report is None or self.current_report_path is None:
            return
        default_path = self.current_report_path.with_suffix(".pdf")
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Export Edge-Case Report",
            str(default_path),
            "PDF Files (*.pdf)",
        )
        if not selected:
            return
        output_path = Path(selected)
        if output_path.suffix.casefold() != ".pdf":
            output_path = output_path.with_suffix(".pdf")
        document = QTextDocument()
        document.setHtml(self._report_html())
        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(str(output_path))
        document.print_(printer)
        self._show_notification("success", f"PDF exported to {output_path.name}.")

    def _show_notification(self, level: str, message: str) -> None:
        colors = {
            "success": (SUCCESS, "#F0FFF4", "fa6s.circle-check", 5000),
            "warning": (WARNING_COLOR, "#FFF8E6", "fa6s.triangle-exclamation", 8000),
            "error": (ERROR_COLOR, "#FFF1F0", "fa6s.circle-xmark", 0),
        }
        foreground, background, icon_name, timeout = colors[level]
        self.notification_icon.setPixmap(icon(icon_name, foreground).pixmap(17, 17))
        self.notification_label.setText(message)
        self.notification_card.setStyleSheet(
            f"QFrame#edgeCaseNotification {{ background: {background}; "
            f"border: 1px solid {BORDER}; border-left: 3px solid {foreground}; "
            "border-radius: 7px; }"
            "QLabel { border: none; background: transparent; }"
        )
        self.notification_card.show()
        self.notification_timer.stop()
        if timeout:
            self.notification_timer.start(timeout)

    def _show_empty_state(self) -> None:
        self.content_stack.setCurrentWidget(self.empty_state)

    def _latest_config(self) -> ProjectConfig | None:
        config = self.main_window.config
        if config is None:
            return None
        path = config.project_file
        if path.exists():
            try:
                config = ProjectConfig.load(path)
                self.main_window.config = config
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        return config

    def _report_path(self, config: ProjectConfig) -> Path:
        return (
            Path(config.project_dir)
            / "outputs"
            / "edge_cases"
            / "edge_case_report.json"
        )

    def _read_report_metadata(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, json.JSONDecodeError):
            return {}

    def _generated_datetime(self, path: Path) -> datetime:
        generated = (
            (self.report_metadata.get("report_footer") or {}).get("generated_on")
        )
        if generated:
            try:
                return datetime.fromisoformat(str(generated))
            except ValueError:
                pass
        return datetime.fromtimestamp(path.stat().st_mtime)

    def _affected_column(self, issue: Issue) -> str:
        if issue.affected_column:
            return issue.affected_column
        quoted = re.findall(r"'([^']+)'", issue.message)
        if quoted:
            return ", ".join(quoted)
        if issue.category == "target" and self.current_report:
            return str(self.current_report.context.get("target_column") or "Target")
        return "General"

    def _summary_text(self) -> str:
        assert self.current_report is not None
        report = self.current_report
        return "\n".join(
            (
                f"{APP_NAME} Edge-Case Report",
                f"Project: {report.context.get('project_name', 'Unknown')}",
                f"Target: {report.context.get('target_column', 'Not confirmed')}",
                f"Training ready: {'Yes' if report.can_continue else 'No'}",
                f"Fatal issues: {len(report.fatals)}",
                f"Errors: {len(report.errors)}",
                f"Warnings: {len(report.warnings)}",
                f"Report: {self.current_report_path}",
            )
        )

    def _report_html(self) -> str:
        assert self.current_report is not None
        rows = []
        for issue in self.current_report.issues:
            rows.append(
                "<tr>"
                f"<td>{self._html(issue.level.title())}</td>"
                f"<td>{self._html(issue.category.title())}</td>"
                f"<td>{self._html(issue.message)}</td>"
                f"<td>{self._html(self._affected_column(issue))}</td>"
                f"<td>{self._html(issue.suggestion)}</td>"
                "</tr>"
            )
        issue_rows = "".join(rows) or (
            "<tr><td colspan='5'>No validation issues detected.</td></tr>"
        )
        return (
            f"<h1>{APP_NAME} Edge-Case Report</h1>"
            f"<p>{self._html(self._summary_text()).replace(chr(10), '<br>')}</p>"
            "<table border='1' cellspacing='0' cellpadding='6' width='100%'>"
            "<tr><th>Level</th><th>Issue</th><th>Description</th>"
            "<th>Affected Column</th><th>Recommendation</th></tr>"
            f"{issue_rows}</table>"
            f"<p>Version {__version__}</p>"
        )

    @staticmethod
    def _html(value: Any) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    @staticmethod
    def _format_size(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:,.1f} {unit}"
            value /= 1024
        return f"{size} B"

    def _set_status_tile(self, key: str, text: str, color: str) -> None:
        tile, value = self.status_tiles[key]
        value.setText(text)
        value.setStyleSheet(f"color: {color};")
        tile.setStyleSheet(
            f"QFrame#edgeCaseStatusTile {{ border: 1px solid {BORDER}; "
            f"border-top: 3px solid {color}; border-radius: 7px; "
            "background: #FFFFFF; }}"
        )

    def _section_card(
        self,
        object_name: str,
        icon_name: str,
        title: str,
    ) -> tuple[QWidget, QVBoxLayout]:
        card = self._card(object_name)
        body = card.layout()
        header = QHBoxLayout()
        header.setSpacing(9)
        icon_label = QLabel()
        icon_label.setFixedSize(22, 22)
        icon_label.setPixmap(icon(icon_name).pixmap(20, 20))
        title_label = QLabel(title)
        title_label.setObjectName("edgeCaseCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        body.addLayout(header)
        return card, body

    def _card(self, object_name: str) -> QWidget:
        card = QWidget()
        card.setObjectName(object_name)
        card.setProperty("edgeCaseCard", True)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(31, 41, 55, 24))
        shadow.setOffset(0, 3)
        card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        return card

    def _button(self, text: str, icon_name: str, *, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(
            "edgeCasePrimaryButton" if primary else "edgeCaseSecondaryButton"
        )
        button.setIcon(icon(icon_name, "#FFFFFF" if primary else PRIMARY))
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#edgeCaseReportPage,
            QWidget#edgeCaseScrollContent,
            QWidget#edgeCaseGeneratedContent {{
                background: {BACKGROUND};
                color: {TEXT};
            }}
            QScrollArea#edgeCaseScrollArea {{ background: {BACKGROUND}; }}
            QLabel#edgeCaseTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#edgeCaseSubtitle,
            QLabel#edgeCaseMuted {{ color: {MUTED}; font-size: 12px; }}
            QWidget[edgeCaseCard="true"] {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QLabel#edgeCaseCardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#edgeCaseEmptyTitle {{
                color: {TEXT};
                font-size: 17px;
                font-weight: 700;
            }}
            QLabel#edgeCaseValue {{ color: {TEXT}; font-weight: 600; }}
            QLabel#edgeCaseTileTitle {{ color: {MUTED}; font-size: 11px; }}
            QLabel#edgeCaseTileValue {{ font-size: 22px; font-weight: 700; }}
            QLabel#edgeCaseSuccessText {{ color: {SUCCESS}; font-weight: 700; }}
            QFrame#edgeCaseSuccessState {{
                background: #F0FFF4;
                border: 1px solid #B7E4C7;
                border-radius: 7px;
            }}
            QLabel#edgeCaseReadinessValue {{ font-weight: 700; }}
            QLabel#edgeCaseReadinessScore {{ color: {MUTED}; font-weight: 600; }}
            QPushButton {{
                min-height: 40px;
                border-radius: 7px;
                padding: 0 16px;
                font-weight: 600;
            }}
            QPushButton#edgeCasePrimaryButton {{
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
            }}
            QPushButton#edgeCasePrimaryButton:hover {{ background: #00A6A6; }}
            QPushButton#edgeCaseSecondaryButton {{
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton#edgeCaseSecondaryButton:hover {{
                background: #EFF6FF;
                border-color: {PRIMARY};
            }}
            QPushButton:disabled {{ background: #D0D7DE; color: #6B7280; }}
            QPushButton#edgeCaseNotificationClose {{
                min-height: 0;
                padding: 0;
                border: none;
                background: transparent;
            }}
            QProgressBar {{
                border: none;
                background: #DCE6F2;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 4px; }}
            QTableWidget#edgeCaseIssueTable {{
                background: #FFFFFF;
                alternate-background-color: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5EAF0;
                selection-background-color: #DCEBFA;
                selection-color: {TEXT};
            }}
            QTableWidget#edgeCaseIssueTable::item {{
                padding: 8px;
                border: none;
            }}
            QTableWidget#edgeCaseIssueTable::item:hover {{
                background: #EFF6FF;
            }}
            QHeaderView::section {{
                background: #EEF3F8;
                color: {TEXT};
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 8px;
                font-weight: 700;
            }}
            """
        )
