"""Comprehensive AVISTA model report page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.project_config import ProjectConfig
from app.core.report_generator import (
    PERFORMANCE_COLUMNS,
    ReportArtifacts,
    collect_model_performance,
    collect_report_summary,
    generate_project_report,
    available_diagnostic_models,
    load_model_diagnostic,
)
from app.gui.icon_system import (
    BACKGROUND,
    BORDER,
    FEEDBACK_COLORS,
    FEEDBACK_ICONS,
    PRIMARY,
    TEXT,
    icon,
)


class ReportGenerationWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: ProjectConfig) -> None:
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        try:
            artifacts = generate_project_report(
                self.config,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(artifacts)
        except Exception as exc:
            self.failed.emit(str(exc))


class AspectRatioPixmapLabel(QLabel):
    """Display the complete source image while adapting to the card width."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.setWordWrap(True)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._rescale()

    def clear_source_pixmap(self, message: str) -> None:
        self._source_pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(message)
        self.setMinimumHeight(210)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap.isNull():
            return
        width = max(1, self.contentsRect().width() - 8)
        scaled = self._source_pixmap.scaledToWidth(
            width,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setMinimumHeight(scaled.height() + 8)
        self.setPixmap(scaled)


class FigurePreviewContainer(QWidget):
    """Center a complete figure at a constrained share of the card width."""

    def __init__(
        self,
        preview: AspectRatioPixmapLabel,
        *,
        width_ratio: float = 0.5,
        maximum_width: int = 900,
    ) -> None:
        super().__init__()
        self.preview = preview
        self.width_ratio = width_ratio
        self.maximum_preview_width = maximum_width
        self.setObjectName("reportFigurePreviewContainer")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch(1)
        layout.addWidget(preview, alignment=Qt.AlignmentFlag.AlignTop)
        layout.addStretch(1)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_preview_width()

    def update_preview_width(self) -> None:
        available = max(1, self.contentsRect().width())
        preferred = int(available * self.width_ratio)
        target = min(self.maximum_preview_width, preferred, available)
        self.preview.setFixedWidth(max(1, target))
        self.preview._rescale()


class ReportPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.thread: QThread | None = None
        self.worker: ReportGenerationWorker | None = None
        self.artifacts: ReportArtifacts | None = None
        self.preview_containers: dict[AspectRatioPixmapLabel, FigurePreviewContainer] = {}
        self._diagnostic_stacked = False
        self.setObjectName("reportPage")

        self.summary_values = {
            key: QLabel("Not available")
            for key in (
                "project_name",
                "target_column",
                "dataset_rows",
                "feature_count",
                "train_rows",
                "validation_rows",
                "test_rows",
                "imbalance_method",
                "cv_enabled",
                "models_trained",
                "generated_on",
                "version",
            )
        }
        self.performance_table = QTableWidget(0, len(PERFORMANCE_COLUMNS))
        self.performance_table.setObjectName("reportPerformanceTable")
        self.performance_table.setHorizontalHeaderLabels(PERFORMANCE_COLUMNS)
        self.performance_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.performance_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.performance_table.setAlternatingRowColors(True)
        self.performance_table.setMinimumHeight(240)
        self.performance_table.setMaximumHeight(360)
        self.performance_table.verticalHeader().hide()
        self.performance_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.performance_table.horizontalHeader().setStretchLastSection(True)

        self.roc_preview = self._preview_label("ROC comparison is not available.")
        self.pr_preview = self._preview_label(
            "Precision-recall comparison is not available."
        )
        self.deep_preview = self._preview_label(
            "Deep learning training curves are not available."
        )
        self.confusion_summary = self._section_status(
            "No saved test confusion matrices found."
        )
        self.feature_summary = self._section_status(
            "No saved feature importance outputs found."
        )
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("reportDiagnosticModelSelector")
        self.model_selector.currentTextChanged.connect(self._update_diagnostic)
        self.split_selector = QComboBox()
        self.split_selector.setObjectName("reportDiagnosticSplitSelector")
        self.split_selector.addItems(["Train", "Validation", "Test"])
        self.split_selector.setCurrentText("Test")
        self.split_selector.currentTextChanged.connect(self._update_diagnostic)
        self.diagnostic_confusion = self._preview_label(
            "Select a model with saved diagnostic outputs."
        )
        self.diagnostic_table = QTableWidget(0, 0)
        self.diagnostic_table.setObjectName("reportDiagnosticTable")
        self.diagnostic_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.diagnostic_table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        self.diagnostic_table.setAlternatingRowColors(True)
        self.diagnostic_table.setMinimumHeight(190)
        self.diagnostic_table.setMaximumHeight(320)
        self.diagnostic_table.verticalHeader().hide()
        self.diagnostic_warning = QLabel("")
        self.diagnostic_warning.setObjectName("reportDiagnosticWarning")
        self.diagnostic_warning.setWordWrap(True)
        self.diagnostic_warning.hide()

        self.generate_button = self._button(
            "Generate Report", "fa6s.file-circle-plus", "primaryReportButton", "#FFFFFF"
        )
        self.generate_button.clicked.connect(self.generate_report)
        self.open_folder_button = self._button(
            "Open Report Folder", "fa6s.folder-open", "secondaryReportButton", PRIMARY
        )
        self.open_folder_button.clicked.connect(self.open_report_folder)
        self.open_markdown_button = self._button(
            "Open Markdown", "fa6s.file-lines", "secondaryReportButton", PRIMARY
        )
        self.open_markdown_button.clicked.connect(self.open_markdown)
        self.open_pdf_button = self._button(
            "Open PDF", "fa6s.file-pdf", "secondaryReportButton", PRIMARY
        )
        self.open_pdf_button.clicked.connect(self.open_pdf)
        self.generation_progress = QProgressBar()
        self.generation_progress.setObjectName("reportProgress")
        self.generation_progress.setRange(0, 0)
        self.generation_progress.hide()
        self.generation_status = QLabel("Generate a report from saved training outputs.")
        self.generation_status.setObjectName("reportGenerationStatus")
        self.generation_status.setWordWrap(True)

        self.notification_timer = QTimer(self)
        self.notification_timer.setSingleShot(True)
        self.notification_timer.setInterval(5000)
        self.notification_timer.timeout.connect(self._dismiss_notification)
        self.notification_card = self._notification_card()

        content = QWidget()
        content.setObjectName("reportContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Report")
        title.setObjectName("reportTitle")
        layout.addWidget(title)
        layout.addWidget(self.notification_card)
        self.summary_card = self._summary_card()
        self.performance_card = self._performance_card()
        self.roc_card = self._figure_card(
            "reportRocCard",
            "ROC Curve Comparison",
            "Comparison based on saved test-set predictions/probabilities.",
            "fa6s.chart-line",
            self.roc_preview,
        )
        self.pr_card = self._figure_card(
            "reportPrCard",
            "Precision-Recall Curve Comparison",
            "Comparison based on saved test-set predictions/probabilities.",
            "fa6s.chart-area",
            self.pr_preview,
        )
        self.deep_card = self._figure_card(
            "reportDeepCard",
            "Training/Loss Curves",
            "Combined saved histories for AVISTA deep tabular models.",
            "fa6s.wave-square",
            self.deep_preview,
        )
        self.diagnostic_card = self._diagnostic_card()
        self.confusion_card = self._figure_card(
            "reportConfusionCard",
            "Confusion Matrix Summary",
            "Compact test confusion matrices using original class labels.",
            "fa6s.table-cells",
            self.confusion_summary,
        )
        self.feature_card = self._figure_card(
            "reportFeatureCard",
            "Feature Importance Summary",
            "Available tree-model feature importance outputs.",
            "fa6s.bars-progress",
            self.feature_summary,
        )
        self.export_card = self._export_card()
        for card in (
            self.summary_card,
            self.performance_card,
            self.roc_card,
            self.pr_card,
            self.deep_card,
            self.diagnostic_card,
            self.confusion_card,
            self.feature_card,
            self.export_card,
        ):
            layout.addWidget(card)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        self._apply_style()
        self.refresh()

    def refresh(self) -> None:
        config = self._latest_config()
        self.generate_button.setEnabled(config is not None and self.thread is None)
        if config is None:
            self._render_summary({})
            self._populate_performance_table(collect_model_performance(Path()))
            self._set_output_buttons(None)
            self._populate_diagnostic_models(Path())
            return
        summary = collect_report_summary(config)
        self._render_summary(summary)
        performance = collect_model_performance(
            Path(config.project_dir) / "outputs" / "training"
        )
        self._populate_performance_table(performance)
        self._populate_diagnostic_models(
            Path(config.project_dir) / "outputs" / "training"
        )
        self._load_existing_report(config)

    def generate_report(self) -> None:
        config = self._latest_config()
        if config is None:
            self._show_notification("Create or open a project first.", "error")
            return
        self.generate_button.setEnabled(False)
        self.generation_progress.show()
        self.generation_status.setText("Starting report generation...")
        self.thread = QThread(self)
        self.worker = ReportGenerationWorker(config)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.generation_status.setText)
        self.worker.finished.connect(self._on_report_generated)
        self.worker.failed.connect(self._on_report_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(self._clear_thread_references)
        self.thread.start()

    def open_report_folder(self) -> None:
        self._open_path(self._report_dir())

    def open_markdown(self) -> None:
        self._open_report_file("AVISTA_Report.md", "Markdown report")

    def open_pdf(self) -> None:
        self._open_report_file("AVISTA_Report.pdf", "PDF report")

    def _open_report_file(self, filename: str, label: str) -> None:
        path = self._report_dir() / filename
        if not path.exists():
            self._show_notification(f"{label} is not available yet.", "warning")
            return
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        if path.is_dir() or path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
        else:
            self._show_notification(f"Not available: {path}", "warning")

    def _on_report_generated(self, artifacts: ReportArtifacts) -> None:
        self.artifacts = artifacts
        self.generation_status.setText("Report generated successfully.")
        self._render_summary(artifacts.summary)
        self._populate_performance_table(artifacts.performance)
        self._render_artifacts(artifacts)
        self._show_notification(
            "Report generated successfully. "
            f"Markdown saved to: {artifacts.markdown_path}. "
            f"PDF saved to: {artifacts.pdf_path}",
            "success",
        )

    def _on_report_failed(self, message: str) -> None:
        self.generation_status.setText(f"Report generation failed: {message}")
        self._show_notification(f"Report generation failed: {message}", "error")

    def _thread_finished(self) -> None:
        self.generation_progress.hide()
        self.generate_button.setEnabled(self._latest_config() is not None)

    def _clear_thread_references(self) -> None:
        self.thread = None
        self.worker = None

    def _summary_card(self) -> QWidget:
        card, layout = self._card(
            "reportSummaryCard",
            "Report Summary",
            "Project, dataset, split, model, and reproducibility metadata.",
            "fa6s.file-lines",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(22)
        grid.setVerticalSpacing(10)
        fields = [
            ("Project name", "project_name"),
            ("Target column", "target_column"),
            ("Dataset rows", "dataset_rows"),
            ("Feature count", "feature_count"),
            ("Train rows", "train_rows"),
            ("Validation rows", "validation_rows"),
            ("Test rows", "test_rows"),
            ("Imbalance method", "imbalance_method"),
            ("Cross validation", "cv_enabled"),
            ("Models trained", "models_trained"),
            ("Generated timestamp", "generated_on"),
            ("AVISTA version", "version"),
        ]
        for index, (label, key) in enumerate(fields):
            row, column = divmod(index, 2)
            panel = QFrame()
            panel.setObjectName("reportSummaryTile")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(10, 8, 10, 8)
            name = QLabel(label)
            name.setObjectName("reportSummaryLabel")
            value = self.summary_values[key]
            value.setObjectName("reportSummaryValue")
            value.setWordWrap(True)
            panel_layout.addWidget(name)
            panel_layout.addWidget(value)
            grid.addWidget(panel, row, column)
        layout.addLayout(grid)
        return card

    def _performance_card(self) -> QWidget:
        card, layout = self._card(
            "reportPerformanceCard",
            "Model Performance Table",
            "Combined saved metrics for all completed, failed, and skipped models.",
            "fa6s.table",
        )
        layout.addWidget(self.performance_table)
        return card

    def _figure_card(
        self,
        object_name: str,
        title: str,
        subtitle: str,
        icon_name: str,
        content: QWidget,
    ) -> QWidget:
        card, layout = self._card(object_name, title, subtitle, icon_name)
        if isinstance(content, AspectRatioPixmapLabel):
            preview_container = FigurePreviewContainer(content)
            self.preview_containers[content] = preview_container
            layout.addWidget(preview_container)
        else:
            layout.addWidget(content)
        return card

    def _diagnostic_card(self) -> QWidget:
        card, layout = self._card(
            "reportDiagnosticCard",
            "Model Diagnostic Report",
            "Saved confusion matrix and classification metrics for one model and split.",
            "fa6s.stethoscope",
        )
        controls = QHBoxLayout()
        controls.setSpacing(10)
        model_label = QLabel("Select Model")
        model_label.setObjectName("reportDiagnosticFieldLabel")
        split_label = QLabel("Select Split")
        split_label.setObjectName("reportDiagnosticFieldLabel")
        controls.addWidget(model_label)
        controls.addWidget(self.model_selector, stretch=1)
        controls.addWidget(split_label)
        controls.addWidget(self.split_selector)
        layout.addLayout(controls)
        layout.addWidget(self.diagnostic_warning)
        self.diagnostic_body = QWidget()
        self.diagnostic_body.setObjectName("reportDiagnosticBody")
        self.diagnostic_grid = QGridLayout(self.diagnostic_body)
        self.diagnostic_grid.setContentsMargins(0, 0, 0, 0)
        self.diagnostic_grid.setHorizontalSpacing(16)
        self.diagnostic_grid.setVerticalSpacing(12)
        self.diagnostic_preview_container = FigurePreviewContainer(
            self.diagnostic_confusion,
            width_ratio=1.0,
            maximum_width=720,
        )
        self.diagnostic_report_panel = QWidget()
        diagnostic_report_layout = QVBoxLayout(self.diagnostic_report_panel)
        diagnostic_report_layout.setContentsMargins(0, 0, 0, 0)
        report_heading = QLabel("Classification Report")
        report_heading.setObjectName("reportDiagnosticHeading")
        diagnostic_report_layout.addWidget(report_heading)
        diagnostic_report_layout.addWidget(self.diagnostic_table)
        self.diagnostic_grid.addWidget(self.diagnostic_preview_container, 0, 0)
        self.diagnostic_grid.addWidget(self.diagnostic_report_panel, 0, 1)
        self.diagnostic_grid.setColumnStretch(0, 1)
        self.diagnostic_grid.setColumnStretch(1, 1)
        layout.addWidget(self.diagnostic_body)
        return card

    def _export_card(self) -> QWidget:
        card, layout = self._card(
            "reportExportCard",
            "Export Report",
            "Generate and open the comprehensive Markdown and PDF report.",
            "fa6s.file-export",
        )
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(self.generate_button)
        buttons.addWidget(self.open_folder_button)
        buttons.addWidget(self.open_markdown_button)
        buttons.addWidget(self.open_pdf_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        layout.addWidget(self.generation_progress)
        layout.addWidget(self.generation_status)
        return card

    def _card(
        self,
        object_name: str,
        title: str,
        subtitle: str,
        icon_name: str,
    ) -> tuple[QWidget, QVBoxLayout]:
        card = QWidget()
        card.setObjectName(object_name)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setObjectName(f"{object_name}Icon")
        icon_label.setPixmap(icon(icon_name, PRIMARY).pixmap(22, 22))
        title_label = QLabel(title)
        title_label.setObjectName("reportCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("reportCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return card, layout

    def _preview_label(self, message: str) -> AspectRatioPixmapLabel:
        label = AspectRatioPixmapLabel(message)
        label.setObjectName("reportFigurePreview")
        return label

    def _section_status(self, message: str) -> QLabel:
        label = QLabel(message)
        label.setObjectName("reportSectionStatus")
        label.setWordWrap(True)
        return label

    def _button(
        self,
        text: str,
        icon_name: str,
        object_name: str,
        color: str,
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setIcon(icon(icon_name, color))
        button.setIconSize(QSize(16, 16))
        return button

    def _notification_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("reportNotification")
        card.hide()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 9)
        self.notification_icon = QLabel()
        self.notification_icon.setFixedSize(18, 18)
        self.notification_label = QLabel("")
        self.notification_label.setWordWrap(True)
        layout.addWidget(self.notification_icon)
        layout.addWidget(self.notification_label, stretch=1)
        return card

    def _show_notification(self, message: str, level: str) -> None:
        color, background = FEEDBACK_COLORS[level]
        self.notification_icon.setPixmap(
            icon(FEEDBACK_ICONS[level], color).pixmap(16, 16)
        )
        self.notification_label.setText(message)
        self.notification_card.setStyleSheet(
            f"QFrame {{ background: {background}; border: 1px solid {BORDER}; "
            f"border-left: 3px solid {color}; border-radius: 7px; }}"
            "QLabel { border: none; background: transparent; }"
        )
        self.notification_card.show()
        self.notification_timer.stop()
        if level == "success":
            self.notification_timer.start()

    def _dismiss_notification(self) -> None:
        self.notification_card.hide()

    def _render_summary(self, summary: dict[str, Any]) -> None:
        for key, label in self.summary_values.items():
            value = summary.get(key, "Not available")
            if key == "cv_enabled" and key in summary:
                value = (
                    f"Enabled ({summary.get('cv_folds', 0)} folds)"
                    if value
                    else "Disabled"
                )
            label.setText(str(value))

    def _populate_performance_table(self, frame) -> None:
        self.performance_table.setRowCount(len(frame))
        for row_index, row in frame.iterrows():
            for column_index, column in enumerate(PERFORMANCE_COLUMNS):
                value = row.get(column)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    text = "Not available"
                elif isinstance(value, float):
                    text = f"{value:.4f}"
                else:
                    text = str(value)
                self.performance_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(text),
                )

    def _render_artifacts(self, artifacts: ReportArtifacts) -> None:
        self._set_preview(self.roc_preview, artifacts.generated_files.get("roc"))
        self._set_preview(self.pr_preview, artifacts.generated_files.get("pr"))
        self._set_preview(self.deep_preview, artifacts.generated_files.get("deep"))
        training_dir = artifacts.output_dir.parent / "training"
        confusion_count = len(list(training_dir.glob("*/test/confusion_matrix.csv")))
        feature_count = len(list(training_dir.glob("*/feature_importance.csv")))
        self.confusion_summary.setText(
            f"{confusion_count} saved test confusion matrix figure(s) included."
            if confusion_count
            else "Not available"
        )
        self.feature_summary.setText(
            f"{feature_count} feature importance summary figure(s) included."
            if feature_count
            else "Not available"
        )
        self._set_output_buttons(artifacts.output_dir)

    def _set_preview(self, label: QLabel, path: Path | None) -> None:
        if path and path.exists():
            label.setText("")
            label.set_source_pixmap(QPixmap(str(path)))
        else:
            label.clear_source_pixmap("Not available")
        container = self.preview_containers.get(label)
        if container is not None:
            container.update_preview_width()

    def _load_existing_report(self, config: ProjectConfig) -> None:
        output_dir = Path(config.project_dir) / "outputs" / "report"
        self._set_output_buttons(output_dir if output_dir.exists() else None)
        for label, filename in (
            (self.roc_preview, "roc_curve_comparison.png"),
            (self.pr_preview, "pr_curve_comparison.png"),
            (self.deep_preview, "deep_training_curves.png"),
        ):
            self._set_preview(label, output_dir / filename)

    def _populate_diagnostic_models(self, training_dir: Path) -> None:
        current = self.model_selector.currentText()
        models = list(available_diagnostic_models(training_dir))
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(models)
        if current in models:
            self.model_selector.setCurrentText(current)
        self.model_selector.blockSignals(False)
        self._update_diagnostic()

    def _update_diagnostic(self, *_args) -> None:
        config = self._latest_config()
        model_name = self.model_selector.currentText()
        split_name = self.split_selector.currentText() or "Test"
        if config is None or not model_name:
            self._set_preview(self.diagnostic_confusion, None)
            self._populate_diagnostic_table(pd.DataFrame())
            self._show_diagnostic_warning("No saved model diagnostics are available.")
            return
        diagnostic = load_model_diagnostic(
            Path(config.project_dir) / "outputs" / "training",
            model_name,
            split_name,
            Path(config.project_dir) / "outputs" / "report",
        )
        self._set_preview(self.diagnostic_confusion, diagnostic.confusion_path)
        self._populate_diagnostic_table(diagnostic.classification_report)
        self._show_diagnostic_warning(diagnostic.warning)

    def _populate_diagnostic_table(self, frame) -> None:
        self.diagnostic_table.clear()
        self.diagnostic_table.setRowCount(len(frame))
        self.diagnostic_table.setColumnCount(len(frame.columns))
        self.diagnostic_table.setHorizontalHeaderLabels(
            [str(column) for column in frame.columns]
        )
        for row_index, row in frame.iterrows():
            for column_index, value in enumerate(row):
                text = (
                    f"{value:.4f}"
                    if isinstance(value, (float, np.floating))
                    else str(value)
                )
                self.diagnostic_table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(text),
                )
        self.diagnostic_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

    def _show_diagnostic_warning(self, message: str) -> None:
        self.diagnostic_warning.setText(message)
        self.diagnostic_warning.setVisible(bool(message))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_diagnostic_layout()

    def _update_diagnostic_layout(self) -> None:
        if not hasattr(self, "diagnostic_grid"):
            return
        stacked = self.diagnostic_card.width() < 900
        if stacked == self._diagnostic_stacked:
            self.diagnostic_preview_container.update_preview_width()
            return
        self._diagnostic_stacked = stacked
        self.diagnostic_grid.removeWidget(self.diagnostic_preview_container)
        self.diagnostic_grid.removeWidget(self.diagnostic_report_panel)
        if stacked:
            self.diagnostic_preview_container.setMaximumWidth(16777215)
            self.diagnostic_grid.addWidget(self.diagnostic_preview_container, 0, 0)
            self.diagnostic_grid.addWidget(self.diagnostic_report_panel, 1, 0)
            self.diagnostic_grid.setColumnStretch(0, 1)
            self.diagnostic_grid.setColumnStretch(1, 0)
        else:
            half_width = max(1, int(self.diagnostic_body.width() * 0.5))
            self.diagnostic_preview_container.setMaximumWidth(half_width)
            self.diagnostic_grid.addWidget(self.diagnostic_preview_container, 0, 0)
            self.diagnostic_grid.addWidget(self.diagnostic_report_panel, 0, 1)
            self.diagnostic_grid.setColumnStretch(0, 1)
            self.diagnostic_grid.setColumnStretch(1, 1)
        self.diagnostic_preview_container.update_preview_width()

    def _set_output_buttons(self, output_dir: Path | None) -> None:
        exists = bool(output_dir and output_dir.exists())
        self.open_folder_button.setEnabled(exists)
        self.open_markdown_button.setEnabled(
            bool(output_dir and (output_dir / "AVISTA_Report.md").exists())
        )
        self.open_pdf_button.setEnabled(
            bool(output_dir and (output_dir / "AVISTA_Report.pdf").exists())
        )

    def _latest_config(self) -> ProjectConfig | None:
        config = self.main_window.config
        if config is None:
            return None
        if config.project_file.exists():
            try:
                config = ProjectConfig.load(config.project_file)
                self.main_window.config = config
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                return None
        return config

    def _report_dir(self) -> Path:
        config = self._latest_config()
        return (
            Path(config.project_dir) / "outputs" / "report"
            if config
            else Path("outputs") / "report"
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#reportContent {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#reportTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#reportCardTitle {{
                color: {TEXT}; font-size: 16px; font-weight: 700;
            }}
            QLabel#reportCardSubtitle,
            QLabel#reportGenerationStatus {{
                color: #5B6573; font-size: 12px;
            }}
            QWidget#reportSummaryCard,
            QWidget#reportPerformanceCard,
            QWidget#reportRocCard,
            QWidget#reportPrCard,
            QWidget#reportDeepCard,
            QWidget#reportDiagnosticCard,
            QWidget#reportConfusionCard,
            QWidget#reportFeatureCard,
            QWidget#reportExportCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QFrame#reportSummaryTile {{
                background: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QLabel#reportSummaryLabel {{
                color: #5B6573; font-size: 11px; font-weight: 600;
            }}
            QLabel#reportSummaryValue {{ color: {TEXT}; font-weight: 700; }}
            QLabel#reportFigurePreview {{
                color: #6B7280;
                background: #F8FAFC;
                border: 1px dashed {BORDER};
                border-radius: 8px;
            }}
            QWidget#reportFigurePreviewContainer,
            QWidget#reportDiagnosticBody {{
                background: transparent;
                border: none;
            }}
            QLabel#reportSectionStatus {{
                color: #5B6573;
                background: #F8FAFC;
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 12px;
            }}
            QLabel#reportDiagnosticWarning {{
                color: #9A6700;
                background: #FFF8E6;
                border: 1px solid #F2CC60;
                border-left: 3px solid #D97706;
                border-radius: 7px;
                padding: 9px 11px;
            }}
            QLabel#reportDiagnosticFieldLabel,
            QLabel#reportDiagnosticHeading {{
                color: {TEXT}; font-weight: 700;
            }}
            QComboBox {{
                min-height: 36px;
                color: {TEXT};
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 0 10px;
            }}
            QTableWidget#reportPerformanceTable {{
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5E7EB;
                color: {TEXT};
            }}
            QTableWidget#reportDiagnosticTable {{
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5E7EB;
                color: {TEXT};
            }}
            QTableWidget#reportDiagnosticTable QHeaderView::section {{
                background: #EEF3F8;
                color: {TEXT};
                font-weight: 700;
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 6px;
            }}
            QTableWidget#reportPerformanceTable QHeaderView::section {{
                background: #EEF3F8;
                color: {TEXT};
                font-weight: 700;
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 6px;
            }}
            QProgressBar#reportProgress {{
                min-height: 8px; max-height: 8px;
                background: #E8EEF5; border: none; border-radius: 4px;
            }}
            QProgressBar#reportProgress::chunk {{
                background: {PRIMARY}; border-radius: 4px;
            }}
            QPushButton {{
                min-height: 42px;
                border-radius: 7px;
                padding: 0 14px;
                font-weight: 600;
            }}
            QPushButton#primaryReportButton {{
                color: #FFFFFF; background: {PRIMARY}; border: none;
            }}
            QPushButton#primaryReportButton:hover {{ background: #00A6A6; }}
            QPushButton#secondaryReportButton {{
                color: {PRIMARY}; background: #FFFFFF; border: 1px solid {BORDER};
            }}
            QPushButton#secondaryReportButton:hover {{
                background: #EFF6FF; border-color: {PRIMARY};
            }}
            QPushButton:disabled {{
                color: #8C959F; background: #EAEEF2; border-color: #D0D7DE;
            }}
            """
        )
