"""AVISTA saved-artifact model training page."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QThread, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.project_config import ProjectConfig
from app.gui.icon_system import (
    BACKGROUND,
    BORDER,
    FEEDBACK_COLORS,
    FEEDBACK_ICONS,
    PRIMARY,
    TEXT,
    icon,
)
from app.gui.workers import TrainingWorker


DEEP_MODELS = {"MambaAttention", "FT-Transformer", "AutoInt", "TabResNet"}
SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
ERROR_COLOR = "#DC2626"
INFO_COLOR = PRIMARY
DISABLED_COLOR = "#8C959F"

RESULT_COLUMNS = [
    "Model",
    "Status",
    "Train Accuracy",
    "Train Macro-F1",
    "Validation Accuracy",
    "Validation Macro-F1",
    "Test Accuracy",
    "Test Macro-F1",
    "CV Accuracy Mean",
    "CV Accuracy Std",
    "CV Macro-F1 Mean",
    "CV Macro-F1 Std",
    "ROC-AUC",
    "Saved",
]


class TrainingPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.thread: QThread | None = None
        self.worker: TrainingWorker | None = None
        self._training_running = False
        self._spinner_angle = 0
        self.setObjectName("trainingPage")

        self.status_values = {
            key: QLabel("No")
            for key in (
                "column_confirmed",
                "split_confirmed",
                "edge_passed",
                "target",
                "feature_count",
                "train_rows",
                "validation_rows",
                "test_rows",
                "imbalance_method",
                "cv_enabled",
                "cv_folds",
                "selected_models",
            )
        }
        self.readiness_value = QLabel("Not Ready")
        self.readiness_value.setObjectName("trainingReadinessBadge")
        self.preflight_message = QLabel("")
        self.preflight_message.setObjectName("trainingReadinessMessage")
        self.preflight_message.setWordWrap(True)

        self.save_outputs = QCheckBox("Save Outputs")
        self.save_outputs.setChecked(True)
        self.use_saved_data = QCheckBox("Use Saved Split Data")
        self.use_saved_data.setChecked(True)
        self.use_saved_data.setEnabled(False)

        self.start_button = self._button(
            "Start Training", "fa6s.play", "primaryTrainingButton", "#FFFFFF"
        )
        self._start_play_icon = self.start_button.icon()
        self._spinner_source = icon("fa6s.spinner", "#FFFFFF").pixmap(16, 16)
        self.spinner_timer = QTimer(self)
        self.spinner_timer.setInterval(80)
        self.spinner_timer.timeout.connect(self._rotate_start_spinner)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = self._button(
            "Stop Training", "fa6s.stop", "dangerTrainingButton", "#FFFFFF"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        self.open_output_button = self._button(
            "Open Output Folder",
            "fa6s.folder-open",
            "secondaryTrainingButton",
            PRIMARY,
        )
        self.open_output_button.clicked.connect(self.open_output_folder)

        self.current_model_label = self._progress_badge("Model: None")
        self.current_fold_label = self._progress_badge("Fold: None")
        self.current_epoch_label = self._progress_badge("Epoch: None")
        self.current_step_label = self._progress_badge("Step: Idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("trainingProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.log_text = QTextEdit()
        self.log_text.setObjectName("trainingLog")
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(170)
        self.log_text.setMaximumHeight(250)

        self.curve_model_name = ""
        self.curve_epochs: list[int] = []
        self.curve_train_loss: list[float] = []
        self.curve_train_accuracy: list[float | None] = []
        self.curve_validation_loss: list[float | None] = []
        self.curve_validation_macro_f1: list[float] = []
        self.curve_validation_accuracy: list[float | None] = []
        self.curve_message = QLabel(
            "No deep learning model currently training."
        )
        self.curve_message.setObjectName("trainingCurveEmptyState")
        self.curve_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.curve_message.setMinimumHeight(160)
        self.curve_figure = Figure(figsize=(9, 4.2), facecolor="#FFFFFF")
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        self.curve_canvas.setMinimumHeight(300)
        self.curve_canvas.setVisible(False)

        self.result_table = QTableWidget(0, len(RESULT_COLUMNS))
        self.result_table.setObjectName("trainingResultsTable")
        self.result_table.setHorizontalHeaderLabels(RESULT_COLUMNS)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.result_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setMinimumHeight(230)
        self.result_table.setMaximumHeight(340)
        self.result_table.verticalHeader().hide()
        self.result_table.verticalHeader().setDefaultSectionSize(28)
        self.result_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_rows: dict[str, int] = {}

        self.saved_outputs_label = QLabel("Not available")
        self.output_folder_value = self.saved_outputs_label
        self.models_saved_value = QLabel("0")
        self.reports_generated_value = QLabel("0")
        self.output_timestamp_value = QLabel("Not available")
        for label in (
            self.output_folder_value,
            self.models_saved_value,
            self.reports_generated_value,
            self.output_timestamp_value,
        ):
            label.setObjectName("trainingOutputValue")
            label.setWordWrap(True)

        self.open_folder_button = self._button(
            "Open Folder", "fa6s.folder-open", "secondaryTrainingButton", PRIMARY
        )
        self.open_folder_button.clicked.connect(self.open_output_folder)
        self.open_results_button = self._button(
            "Open Results CSV", "fa6s.file-csv", "secondaryTrainingButton", PRIMARY
        )
        self.open_results_button.clicked.connect(self.open_results_csv)
        self.open_report_button = self._button(
            "Open Report", "fa6s.file-lines", "secondaryTrainingButton", PRIMARY
        )
        self.open_report_button.clicked.connect(self.open_training_report)

        self.notification_timer = QTimer(self)
        self.notification_timer.setSingleShot(True)
        self.notification_timer.timeout.connect(self._dismiss_notification)
        self.notification_card = self._notification_card()

        content = QWidget()
        content.setObjectName("trainingContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Training")
        title.setObjectName("trainingTitle")
        layout.addWidget(title)
        layout.addWidget(self.notification_card)
        self.readiness_card = self._readiness_card()
        self.controls_card = self._controls_card()
        self.progress_card = self._progress_card()
        self.curves_card = self._training_curves_card()
        self.results_card = self._results_card()
        self.outputs_card = self._outputs_card()
        for card in (
            self.readiness_card,
            self.controls_card,
            self.progress_card,
            self.curves_card,
            self.results_card,
            self.outputs_card,
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
        self._update_output_summary(None)
        self._update_output_button_state(None)

    def refresh(self) -> None:
        config = self._latest_config()
        status = self._preflight_status(config)
        self._render_preflight(status)
        if not self._training_running:
            self.start_button.setEnabled(status["ready"] and self.thread is None)
        self._update_output_summary(config)
        self._update_output_button_state(config)

    def start_training(self) -> None:
        config = self._latest_config()
        status = self._preflight_status(config)
        self._render_preflight(status)
        if not status["ready"]:
            message = "Training blocked: " + " ".join(status["messages"])
            self._append_log(message, "error")
            self._show_notification(message, "error")
            return
        if not self.use_saved_data.isChecked():
            message = "Training requires the confirmed saved split data."
            self._append_log(message, "error")
            self._show_notification(message, "error")
            return

        self.result_table.setRowCount(0)
        self.result_rows.clear()
        self.progress_bar.setValue(0)
        self._set_running(True)
        self._show_notification("Training started", "success")
        self.thread = QThread(self)
        self.worker = TrainingWorker(
            config,
            save_outputs=self.save_outputs.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.started.connect(
            lambda: self._append_log("Background training started.", "info")
        )
        self.worker.progress_message.connect(self._append_log)
        self.worker.progress_update.connect(self._on_progress)
        self.worker.model_started.connect(self._on_model_started)
        self.worker.model_result_ready.connect(self._on_model_result_ready)
        self.worker.finished.connect(self._on_training_finished)
        self.worker.cancelled.connect(self._on_training_cancelled)
        self.worker.failed.connect(self._on_training_failed)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.cancelled.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker.finished.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(self._clear_thread_references)
        self.thread.start()

    def stop_training(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            self._append_log("Stop requested.", "warning")
            self._show_notification(
                "Stop requested. Training will stop at a safe checkpoint.",
                "warning",
            )

    def open_output_folder(self) -> None:
        config = self._latest_config()
        output_dir = self._training_output_dir(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._open_path(output_dir)

    def open_results_csv(self) -> None:
        self._open_output_file("training_results.csv", "Results CSV")

    def open_training_report(self) -> None:
        self._open_output_file("training_results.json", "training report")

    def _open_output_file(self, filename: str, label: str) -> None:
        path = self._training_output_dir(self._latest_config()) / filename
        if not path.exists():
            self._show_notification(f"{label} is not available yet.", "warning")
            return
        self._open_path(path)

    def _open_path(self, path: Path) -> None:
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self._show_notification(f"Could not open {path}.", "error")

    def _readiness_card(self) -> QWidget:
        card, layout = self._card(
            "trainingReadinessCard",
            "Training Readiness",
            "Confirmed project settings and saved artifacts required for training.",
            "fa6s.circle-check",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        tile_specs = [
            ("Target Selected", self.status_values["target"], "fa6s.bullseye"),
            ("Features", self.status_values["feature_count"], "fa6s.table-columns"),
            ("Train Rows", self.status_values["train_rows"], "fa6s.database"),
            ("Validation Rows", self.status_values["validation_rows"], "fa6s.database"),
            ("Test Rows", self.status_values["test_rows"], "fa6s.database"),
            ("Selected Models", self.status_values["selected_models"], "fa6s.brain"),
            ("Cross Validation", self.status_values["cv_enabled"], "fa6s.rotate"),
            ("Readiness Status", self.readiness_value, "fa6s.shield"),
        ]
        for index, (title, value, icon_name) in enumerate(tile_specs):
            grid.addWidget(
                self._status_tile(title, value, icon_name),
                index // 4,
                index % 4,
            )
        layout.addLayout(grid)
        layout.addWidget(self.preflight_message)
        return card

    def _controls_card(self) -> QWidget:
        card, layout = self._card(
            "trainingControlsCard",
            "Training Controls",
            "Run the selected models from the confirmed split artifacts.",
            "fa6s.play",
        )
        options = QHBoxLayout()
        options.addWidget(self.save_outputs)
        options.addWidget(self.use_saved_data)
        options.addStretch(1)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_output_button)
        buttons.addStretch(1)
        layout.addLayout(options)
        layout.addLayout(buttons)
        return card

    def _progress_card(self) -> QWidget:
        card, layout = self._card(
            "trainingProgressCard",
            "Live Training Progress",
            "Model, fold, epoch, and worker events update without blocking the UI.",
            "fa6s.chart-line",
        )
        badges = QHBoxLayout()
        badges.setSpacing(8)
        badges.addWidget(self.current_model_label)
        badges.addWidget(self.current_fold_label)
        badges.addWidget(self.current_epoch_label)
        badges.addWidget(self.current_step_label)
        badges.addStretch(1)
        layout.addLayout(badges)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_text)
        return card

    def _training_curves_card(self) -> QWidget:
        card, layout = self._card(
            "trainingCurvesCard",
            "Deep Learning Training Curves",
            "Realtime epoch metrics for supported AVISTA deep tabular models.",
            "fa6s.wave-square",
        )
        layout.addWidget(self.curve_message)
        layout.addWidget(self.curve_canvas)
        return card

    def _results_card(self) -> QWidget:
        card, layout = self._card(
            "trainingResultsCard",
            "Model Results",
            "Rows appear as each model starts and update immediately on completion.",
            "fa6s.table",
        )
        layout.addWidget(self.result_table)
        return card

    def _outputs_card(self) -> QWidget:
        card, layout = self._card(
            "trainingOutputsCard",
            "Training Outputs",
            "Saved models, result summaries, reports, and generated artifacts.",
            "fa6s.folder-open",
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(10)
        fields = [
            ("Output Folder", self.output_folder_value),
            ("Models Saved", self.models_saved_value),
            ("Reports Generated", self.reports_generated_value),
            ("Timestamp", self.output_timestamp_value),
        ]
        for index, (name, value) in enumerate(fields):
            label = QLabel(name)
            label.setObjectName("trainingOutputLabel")
            grid.addWidget(label, index, 0)
            grid.addWidget(value, index, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        buttons.addWidget(self.open_folder_button)
        buttons.addWidget(self.open_results_button)
        buttons.addWidget(self.open_report_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
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
        card.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
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
        icon_label.setProperty("iconColor", PRIMARY)
        icon_label.setPixmap(icon(icon_name, PRIMARY).pixmap(22, 22))
        title_label = QLabel(title)
        title_label.setObjectName("trainingCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("trainingCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return card, layout

    def _status_tile(self, title: str, value: QLabel, icon_name: str) -> QFrame:
        tile = QFrame()
        tile.setObjectName("trainingStatusTile")
        layout = QVBoxLayout(tile)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(5)
        heading = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setObjectName("trainingStatusTileIcon")
        icon_label.setProperty("iconColor", PRIMARY)
        icon_label.setPixmap(icon(icon_name, PRIMARY).pixmap(14, 14))
        title_label = QLabel(title)
        title_label.setObjectName("trainingTileTitle")
        heading.addWidget(icon_label)
        heading.addWidget(title_label)
        heading.addStretch(1)
        value.setObjectName(
            "trainingReadinessBadge"
            if value is self.readiness_value
            else "trainingTileValue"
        )
        value.setWordWrap(True)
        layout.addLayout(heading)
        layout.addWidget(value)
        return tile

    def _button(
        self,
        text: str,
        icon_name: str,
        object_name: str,
        icon_color: str,
    ) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName(object_name)
        button.setProperty("iconColor", icon_color)
        button.setProperty("disabledIconColor", DISABLED_COLOR)
        button.setIcon(self._button_icon(icon_name, icon_color))
        button.setIconSize(QSize(16, 16))
        return button

    def _button_icon(self, icon_name: str, normal_color: str) -> QIcon:
        result = QIcon()
        result.addPixmap(
            icon(icon_name, normal_color).pixmap(16, 16),
            QIcon.Mode.Normal,
            QIcon.State.Off,
        )
        result.addPixmap(
            icon(icon_name, DISABLED_COLOR).pixmap(16, 16),
            QIcon.Mode.Disabled,
            QIcon.State.Off,
        )
        return result

    def _progress_badge(self, text: str) -> QLabel:
        badge = QLabel(text)
        badge.setObjectName("trainingProgressBadge")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return badge

    def _notification_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("trainingNotification")
        card.hide()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(8)
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
            self.notification_timer.start(5000)
        elif level == "warning":
            self.notification_timer.start(8000)

    def _dismiss_notification(self) -> None:
        self.notification_card.hide()

    def _preflight_status(self, config: ProjectConfig | None) -> dict:
        messages = []
        if config is None:
            return self._empty_status(["Project configuration is required."])

        project_dir = Path(config.project_dir)
        split_dir = project_dir / "outputs" / "data_split"
        edge_path = project_dir / "outputs" / "edge_cases" / "edge_case_report.json"
        modeling_subset = project_dir / "data" / "modeling_subset.csv"
        column_confirmed = modeling_subset.exists() and bool(
            config.feature_columns and config.target_column
        )
        required_split_files = [
            "split_indices.json",
            "imbalance_config.json",
            "class_coverage_report.csv",
            "X_train_balanced.npy",
            "y_train_balanced.npy",
            "X_val.npy",
            "y_val.npy",
            "X_test.npy",
            "y_test.npy",
            "preprocessing_artifact.joblib",
        ]
        split_confirmed = all((split_dir / name).exists() for name in required_split_files)
        edge_passed = False
        if edge_path.exists():
            try:
                edge_report = json.loads(edge_path.read_text(encoding="utf-8"))
                edge_passed = bool(edge_report["can_continue"]) and (
                    (edge_report.get("context") or {}).get("target_column")
                    == config.target_column
                )
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                edge_passed = False

        rows = {"train": 0, "validation": 0, "test": 0}
        if split_confirmed:
            rows = {
                "train": len(np.load(split_dir / "y_train_balanced.npy", allow_pickle=True)),
                "validation": len(np.load(split_dir / "y_val.npy", allow_pickle=True)),
                "test": len(np.load(split_dir / "y_test.npy", allow_pickle=True)),
            }

        if not column_confirmed:
            messages.append("Please confirm Column Configuration.")
        if not split_confirmed:
            messages.append("Please confirm Data Split & Imbalance.")
        if not edge_path.exists():
            messages.append("Run the Edge-Case Report.")
        elif not edge_passed:
            messages.append("Resolve blocking Edge-Case Report issues.")
        if not config.selected_models:
            messages.append("Select at least one model.")
        if split_confirmed and config.enable_cross_validation:
            counts = pd_series_counts(split_dir / "y_train_balanced.npy")
            insufficient = counts[counts < int(config.cv_folds)]
            if not insufficient.empty:
                class_name = insufficient.index[0]
                messages.append(
                    f"Class '{class_name}' has only {int(insufficient.iloc[0])} samples "
                    f"but CV folds = {config.cv_folds}."
                )

        return {
            "ready": not messages,
            "messages": messages,
            "column_confirmed": column_confirmed,
            "split_confirmed": split_confirmed,
            "edge_passed": edge_passed,
            "target": config.target_column or "Not selected",
            "feature_count": len(config.feature_columns or []),
            "train_rows": rows["train"],
            "validation_rows": rows["validation"],
            "test_rows": rows["test"],
            "imbalance_method": config.imbalance_method or "none",
            "cv_enabled": bool(config.enable_cross_validation),
            "cv_folds": int(config.cv_folds),
            "selected_models": ", ".join(config.selected_models) or "none",
            "selected_model_count": len(config.selected_models or []),
        }

    def _empty_status(self, messages: list[str]) -> dict:
        return {
            "ready": False,
            "messages": messages,
            "column_confirmed": False,
            "split_confirmed": False,
            "edge_passed": False,
            "target": "Not selected",
            "feature_count": 0,
            "train_rows": 0,
            "validation_rows": 0,
            "test_rows": 0,
            "imbalance_method": "none",
            "cv_enabled": False,
            "cv_folds": 0,
            "selected_models": "none",
            "selected_model_count": 0,
        }

    def _render_preflight(self, status: dict) -> None:
        for key in ("column_confirmed", "split_confirmed", "edge_passed"):
            self.status_values[key].setText("Yes" if status[key] else "No")
        for key in (
            "target",
            "feature_count",
            "train_rows",
            "validation_rows",
            "test_rows",
            "imbalance_method",
            "cv_folds",
        ):
            self.status_values[key].setText(str(status[key]))
        self.status_values["selected_models"].setText(
            str(status["selected_model_count"])
        )
        self.status_values["cv_enabled"].setText(
            f"{status['cv_folds']} folds" if status["cv_enabled"] else "Disabled"
        )
        if status["ready"]:
            self.readiness_value.setText("Ready To Train")
            self.readiness_value.setProperty("ready", True)
            self.preflight_message.setText("All required training artifacts are confirmed.")
        else:
            self.readiness_value.setText("Not Ready")
            self.readiness_value.setProperty("ready", False)
            self.preflight_message.setText(" ".join(status["messages"]))
        self.readiness_value.style().unpolish(self.readiness_value)
        self.readiness_value.style().polish(self.readiness_value)

    def _on_progress(self, progress: dict) -> None:
        model = progress.get("model") or "None"
        fold = int(progress.get("fold", 0))
        total_folds = int(progress.get("total_folds", 0))
        epoch = progress.get("epoch")
        total_epochs = progress.get("total_epochs")
        step = str(progress.get("step", "idle")).replace("_", " ").title()
        self.current_model_label.setText(f"Model: {model}")
        self.current_fold_label.setText(
            f"Fold: {fold}/{total_folds}" if total_folds else "Fold: None"
        )
        self.current_epoch_label.setText(
            f"Epoch: {epoch}/{total_epochs}" if epoch and total_epochs else "Epoch: None"
        )
        self.current_step_label.setText(f"Step: {step}")
        self.progress_bar.setValue(int(progress.get("percent", 0)))
        if progress.get("step") == "started":
            if model in DEEP_MODELS:
                self._start_deep_curve(model)
            else:
                self._show_non_deep_curve_message()
        if (
            progress.get("event") == "epoch_progress"
            and model in DEEP_MODELS
            and int(progress.get("fold", 0)) == 0
        ):
            self._append_epoch_progress(progress)

    def _on_model_started(self, model_name: str) -> None:
        row = self._result_row(model_name)
        self._set_result_row(
            row,
            {"model_name": model_name, "status": "Running", "saved": False},
        )
        self._append_log(f"{model_name} started.", "info")

    def _on_model_result_ready(self, result: dict) -> None:
        model_name = str(result.get("model_name", "unknown"))
        status = str(result.get("status", "")).lower()
        row = self._result_row(model_name)
        self._set_result_row(row, result)
        if status in {"trained", "completed", "success"}:
            self._append_log(f"{model_name} completed.", "success")
            self._show_notification(f"{model_name} completed", "success")
        elif status == "skipped":
            self._append_log(f"{model_name} skipped.", "warning")
            self._show_notification(f"{model_name} skipped", "warning")
        elif status == "failed":
            error = str(result.get("error", "Unknown training error"))
            self._append_log(f"{model_name} failed: {error}", "error")
            self._show_notification(f"{model_name} failed: {error}", "error")

    def _on_model_finished(self, _model_name: str, result: dict) -> None:
        """Backward-compatible result handler used by older integrations."""
        self._on_model_result_ready(result)

    def _result_row(self, model_name: str) -> int:
        if model_name in self.result_rows:
            return self.result_rows[model_name]
        row = self.result_table.rowCount()
        self.result_table.insertRow(row)
        self.result_rows[model_name] = row
        return row

    def _set_result_row(self, row: int, result: dict) -> None:
        train = result.get("train_metrics", {})
        validation = result.get("validation_metrics", {})
        test = result.get("test_metrics", {})
        cv = result.get("cv_summary", {})
        values = [
            result.get("model_name", ""),
            result.get("status", ""),
            metric_text(train, "accuracy"),
            metric_text(train, "macro_f1"),
            metric_text(validation, "accuracy"),
            metric_text(validation, "macro_f1"),
            metric_text(test, "accuracy"),
            metric_text(test, "macro_f1"),
            summary_text(cv, "accuracy", "mean"),
            summary_text(cv, "accuracy", "std"),
            summary_text(cv, "macro_f1", "mean"),
            summary_text(cv, "macro_f1", "std"),
            metric_text(test, "roc_auc"),
            "Yes" if result.get("saved") else "No",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 1:
                detail = str(result.get("error") or result.get("reason") or "")
                if detail:
                    item.setToolTip(detail)
            self.result_table.setItem(row, column, item)
        self.result_table.scrollToItem(self.result_table.item(row, 0))

    def _start_deep_curve(self, model_name: str) -> None:
        self.curve_model_name = model_name
        self.curve_epochs.clear()
        self.curve_train_loss.clear()
        self.curve_train_accuracy.clear()
        self.curve_validation_loss.clear()
        self.curve_validation_macro_f1.clear()
        self.curve_validation_accuracy.clear()
        self.curve_message.setVisible(False)
        self.curve_canvas.setVisible(True)
        self._draw_training_curve()
        self._append_log(f"Live training curve started for {model_name}", "info")

    def _show_non_deep_curve_message(self) -> None:
        self.curve_model_name = ""
        self.curve_message.setText("No deep learning model currently training.")
        self.curve_message.setVisible(True)
        self.curve_canvas.setVisible(False)

    def _append_epoch_progress(self, progress: dict) -> None:
        if self.curve_model_name != progress.get("model"):
            self._start_deep_curve(str(progress.get("model", "Deep Model")))
        epoch = int(progress["epoch"])
        self.curve_epochs.append(epoch)
        self.curve_train_loss.append(float(progress["train_loss"]))
        self.curve_train_accuracy.append(
            float(progress["train_accuracy"])
            if progress.get("train_accuracy") is not None
            else None
        )
        self.curve_validation_loss.append(
            float(progress["validation_loss"])
            if progress.get("validation_loss") is not None
            else None
        )
        self.curve_validation_macro_f1.append(
            float(progress["validation_macro_f1"])
        )
        self.curve_validation_accuracy.append(
            float(progress["validation_accuracy"])
            if progress.get("validation_accuracy") is not None
            else None
        )
        self._draw_training_curve()
        self._append_log(f"Training curves updated: epoch {epoch}", "info")

    def _draw_training_curve(self) -> None:
        self.curve_figure.clear()
        accuracy_axis, loss_axis = self.curve_figure.subplots(1, 2)
        self.curve_figure.suptitle(
            f"Training Curves - {self.curve_model_name or 'Deep Model'}",
            fontsize=12,
            fontweight="semibold",
        )
        self.curve_figure.patch.set_facecolor("#FFFFFF")
        if self.curve_epochs:
            self._plot_optional_series(
                accuracy_axis,
                self.curve_train_accuracy,
                "Train Accuracy",
                "#0F6CBD",
            )
            self._plot_optional_series(
                accuracy_axis,
                self.curve_validation_accuracy,
                "Validation Accuracy",
                "#00A6A6",
            )
            loss_axis.plot(
                self.curve_epochs,
                self.curve_train_loss,
                label="Train Loss",
                color="#0F6CBD",
                linewidth=2,
            )
            self._plot_optional_series(
                loss_axis,
                self.curve_validation_loss,
                "Validation Loss",
                "#D97706",
            )
        for axis, title, ylabel in (
            (accuracy_axis, "Accuracy", "Accuracy"),
            (loss_axis, "Loss", "Loss"),
        ):
            axis.set_facecolor("#FFFFFF")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, color="#D0D7DE", alpha=0.55, linewidth=0.7)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend(loc="best", frameon=False)
        accuracy_values = [
            value
            for values in (
                self.curve_train_accuracy,
                self.curve_validation_accuracy,
            )
            for value in values
            if value is not None
        ]
        if accuracy_values and all(0.0 <= value <= 1.0 for value in accuracy_values):
            accuracy_axis.set_ylim(0.0, 1.0)
        self.curve_figure.tight_layout(rect=(0, 0, 1, 0.93))
        self.curve_canvas.draw_idle()

    def _plot_optional_series(
        self,
        axis,
        values: list[float | None],
        label: str,
        color: str,
    ) -> None:
        points = [
            (epoch, value)
            for epoch, value in zip(self.curve_epochs, values)
            if value is not None
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=label,
                color=color,
                linewidth=2,
            )

    def _on_training_finished(self, results: dict) -> None:
        self._finish_running_state()
        for result in results.get("results", []):
            model_name = str(result.get("model_name", "unknown"))
            if model_name not in self.result_rows:
                self._on_model_result_ready(result)
        self.progress_bar.setValue(100)
        self.current_step_label.setText("Step: Complete")
        count = len(results.get("results", []))
        self._append_log(f"Training finished for {count} model result(s).", "success")
        self._show_notification("Results exported", "success")
        self._update_output_summary(self._latest_config(), results)

    def _on_training_cancelled(self) -> None:
        self._finish_running_state()
        self.current_step_label.setText("Step: Cancelled")
        self._append_log("Training cancelled.", "error")
        self._show_notification("Training cancelled", "error")

    def _on_training_failed(self, message: str) -> None:
        self._finish_running_state()
        self.current_step_label.setText("Step: Failed")
        self._append_log(f"Training failed: {message}", "error")
        self._show_notification(f"Training failed: {message}", "error")

    def _thread_finished(self) -> None:
        self._finish_running_state()
        self.refresh()

    def _clear_thread_references(self) -> None:
        self.thread = None
        self.worker = None
        self.refresh()

    def _set_running(self, running: bool) -> None:
        self._training_running = running
        if running:
            self.start_button.setText("Training...")
            self.start_button.setEnabled(False)
            self._spinner_angle = 0
            self._rotate_start_spinner()
            self.spinner_timer.start()
        else:
            self.spinner_timer.stop()
            self.start_button.setText("Start Training")
            self.start_button.setIcon(self._start_play_icon)
            status = self._preflight_status(self._latest_config())
            self.start_button.setEnabled(status["ready"] and self.thread is None)
        self.stop_button.setEnabled(running)
        self.save_outputs.setEnabled(not running)
        self._update_output_button_state(self._latest_config())

    def _finish_running_state(self) -> None:
        self._set_running(False)

    def _rotate_start_spinner(self) -> None:
        canvas = QPixmap(20, 20)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(10, 10)
        painter.rotate(self._spinner_angle)
        painter.translate(-8, -8)
        painter.drawPixmap(0, 0, self._spinner_source)
        painter.end()
        spinner_icon = QIcon()
        spinner_icon.addPixmap(canvas, QIcon.Mode.Normal, QIcon.State.Off)
        spinner_icon.addPixmap(canvas, QIcon.Mode.Disabled, QIcon.State.Off)
        self.start_button.setIcon(spinner_icon)
        self._spinner_angle = (self._spinner_angle + 30) % 360

    def _update_output_button_state(self, config: ProjectConfig | None) -> None:
        exists = self._training_output_dir(config).exists()
        self.open_output_button.setEnabled(exists)
        self.open_folder_button.setEnabled(exists)

    def _append_log(self, message: str, level: str | None = None) -> None:
        level = level or self._log_level(message)
        timestamp = datetime.now().strftime("%H:%M")
        color = {
            "info": INFO_COLOR,
            "success": INFO_COLOR,
            "warning": WARNING_COLOR,
            "error": ERROR_COLOR,
        }[level]
        scrollbar = self.log_text.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        escaped = (
            str(message)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        self.log_text.append(
            f'<span style="color:{color};">[{timestamp}] {escaped}</span>'
        )
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _log_level(self, message: str) -> str:
        lowered = message.casefold()
        if any(word in lowered for word in ("failed", "error", "cancelled")):
            return "error"
        if any(word in lowered for word in ("warning", "skipped", "stop requested")):
            return "warning"
        if any(word in lowered for word in ("completed", "finished", "saved", "exported")):
            return "success"
        return "info"

    def _update_output_summary(
        self,
        config: ProjectConfig | None,
        results: dict | None = None,
    ) -> None:
        output_dir = self._training_output_dir(config)
        self.output_folder_value.setText(str(output_dir))
        result_items = list((results or {}).get("results", []))
        summary_path = output_dir / "training_results.json"
        if not result_items and summary_path.exists():
            try:
                result_items = list(
                    json.loads(summary_path.read_text(encoding="utf-8")).get(
                        "results", []
                    )
                )
            except (OSError, TypeError, json.JSONDecodeError):
                result_items = []
        self.models_saved_value.setText(
            str(sum(bool(item.get("saved")) for item in result_items))
        )
        reports = 0
        if output_dir.exists():
            reports = sum(
                1
                for path in output_dir.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in {".csv", ".json", ".png", ".pdf"}
            )
        self.reports_generated_value.setText(str(reports))
        timestamp_path = (
            summary_path
            if summary_path.exists()
            else max(
                (path for path in output_dir.rglob("*") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                default=None,
            )
        )
        if timestamp_path is None:
            self.output_timestamp_value.setText("Not available")
        else:
            self.output_timestamp_value.setText(
                datetime.fromtimestamp(timestamp_path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )
        self.open_results_button.setEnabled(
            (output_dir / "training_results.csv").exists()
        )
        self.open_report_button.setEnabled(summary_path.exists())

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
                return None
        return config

    def _training_output_dir(self, config: ProjectConfig | None) -> Path:
        if config is None:
            return Path("outputs") / "training"
        return Path(config.project_dir) / "outputs" / "training"

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#trainingContent {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#trainingTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#trainingCardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#trainingCardSubtitle,
            QLabel#trainingReadinessMessage {{
                color: #5B6573;
                font-size: 12px;
            }}
            QWidget#trainingReadinessCard,
            QWidget#trainingControlsCard,
            QWidget#trainingProgressCard,
            QWidget#trainingCurvesCard,
            QWidget#trainingResultsCard,
            QWidget#trainingOutputsCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QFrame#trainingStatusTile {{
                background: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QLabel#trainingTileTitle {{
                color: #5B6573;
                font-size: 11px;
                font-weight: 600;
            }}
            QLabel#trainingTileValue {{
                color: {TEXT};
                font-size: 14px;
                font-weight: 700;
            }}
            QLabel#trainingReadinessBadge {{
                color: {ERROR_COLOR};
                background: #FFF1F0;
                border-radius: 9px;
                padding: 3px 8px;
                font-weight: 700;
            }}
            QLabel#trainingReadinessBadge[ready="true"] {{
                color: #1A7F37;
                background: #DAFBE1;
            }}
            QLabel#trainingProgressBadge {{
                color: {PRIMARY};
                background: #EFF6FF;
                border: 1px solid #B6D4F0;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: 12px;
                font-weight: 600;
            }}
            QLabel#trainingCurveEmptyState {{
                color: #6B7280;
                background: #F8FAFC;
                border: 1px dashed {BORDER};
                border-radius: 8px;
            }}
            QLabel#trainingOutputLabel {{
                color: #5B6573;
                font-weight: 600;
            }}
            QLabel#trainingOutputValue {{ color: {TEXT}; }}
            QCheckBox {{ color: {TEXT}; spacing: 8px; }}
            QProgressBar#trainingProgressBar {{
                min-height: 20px;
                max-height: 20px;
                color: {TEXT};
                background: #E8EEF5;
                border: none;
                border-radius: 10px;
                text-align: center;
                font-weight: 600;
            }}
            QProgressBar#trainingProgressBar::chunk {{
                background: {PRIMARY};
                border-radius: 10px;
            }}
            QTextEdit#trainingLog {{
                color: {TEXT};
                background: #FBFCFE;
                border: 1px solid {BORDER};
                border-radius: 7px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 11px;
            }}
            QTableWidget#trainingResultsTable {{
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5E7EB;
                color: {TEXT};
                selection-background-color: #EAF3FC;
                selection-color: {TEXT};
            }}
            QTableWidget#trainingResultsTable QHeaderView::section {{
                background: #EEF3F8;
                color: {TEXT};
                font-weight: 700;
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 6px;
            }}
            QPushButton {{
                min-height: 38px;
                border-radius: 7px;
                padding: 0 14px;
                font-weight: 600;
            }}
            QPushButton#primaryTrainingButton {{
                min-height: 44px;
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
            }}
            QPushButton#primaryTrainingButton:hover {{ background: #00A6A6; }}
            QPushButton#dangerTrainingButton {{
                min-height: 44px;
                color: #FFFFFF;
                background: {ERROR_COLOR};
                border: none;
            }}
            QPushButton#dangerTrainingButton:hover {{ background: #B91C1C; }}
            QPushButton#secondaryTrainingButton {{
                min-height: 44px;
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton#secondaryTrainingButton:hover {{
                background: #EFF6FF;
                border-color: {PRIMARY};
            }}
            QPushButton:disabled {{
                color: #8C959F;
                background: #EAEEF2;
                border-color: #D0D7DE;
            }}
            """
        )


def pd_series_counts(path: Path):
    import pandas as pd

    return pd.Series(np.load(path, allow_pickle=True)).value_counts()


def metric_text(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    return "" if value is None else f"{float(value):.4f}"


def summary_text(summary: dict, metric: str, statistic: str) -> str:
    value = (summary.get(metric) or {}).get(statistic)
    return "" if value is None else f"{float(value):.4f}"
