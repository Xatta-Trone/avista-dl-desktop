"""Column and modeling configuration page."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.target_encoding import invalidate_target_artifacts
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon


MAX_UNIQUE_PREVIEW = 100


class TargetDistributionPlot(QWidget):
    """Matplotlib target-distribution chart suitable for export."""

    def __init__(self) -> None:
        super().__init__()
        self.figure = Figure(figsize=(9, 4.2), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(320)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.target_column: str | None = None
        self.counts = pd.Series(dtype="int64")
        self.set_series(None)

    def set_series(self, series: pd.Series | None) -> None:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        self.target_column = str(series.name) if series is not None else None
        if series is None:
            self.counts = pd.Series(dtype="int64")
            axis.text(
                0.5,
                0.5,
                "Select a target column to view its distribution.",
                ha="center",
                va="center",
                color="#5B6573",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        values = series.dropna()
        self.counts = values.astype(str).value_counts(dropna=False)
        if self.counts.empty:
            axis.text(
                0.5,
                0.5,
                "No non-null target values to display.",
                ha="center",
                va="center",
                color="#5B6573",
                transform=axis.transAxes,
            )
            axis.set_title(f"Target Distribution: {series.name}")
            axis.set_axis_off()
            self.figure.tight_layout()
            self.canvas.draw_idle()
            return

        labels = self.counts.index.tolist()
        counts = self.counts.astype(int).tolist()
        total = sum(counts)
        bars = axis.bar(range(len(labels)), counts, color=PRIMARY, edgecolor="#0B4F8A")
        axis.set_title(f"Target Distribution: {series.name}", fontweight="bold")
        axis.set_xlabel("Class")
        axis.set_ylabel("Count")
        axis.set_xticks(range(len(labels)), labels, rotation=40, ha="right")
        axis.grid(axis="y", alpha=0.25, linestyle="--")
        axis.set_axisbelow(True)
        offset = max(counts) * 0.02 if counts else 0
        for bar, count in zip(bars, counts):
            percent = count / total * 100 if total else 0.0
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"{count:,}\n({percent:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.margins(y=0.18)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save(self, path: str | Path) -> None:
        """Save the chart at publication-oriented resolution."""

        self.figure.savefig(path, dpi=300, bbox_inches="tight")


class ColumnConfigPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self._refreshing = False
        self.setObjectName("columnConfigPage")
        self.success_notification_timer = QTimer(self)
        self.success_notification_timer.setSingleShot(True)
        self.success_notification_timer.setInterval(5000)
        self.success_notification_timer.timeout.connect(
            self._dismiss_success_notifications
        )

        self.available_columns_list = QListWidget()
        self.selected_columns_list = QListWidget()
        for widget in (self.available_columns_list, self.selected_columns_list):
            widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            widget.setMinimumHeight(220)

        self.target_input = QComboBox()
        self.target_input.setMinimumHeight(38)
        self.target_input.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.target_input.currentTextChanged.connect(self._target_changed)
        self.target_plot = TargetDistributionPlot()

        self.label_encoding_columns_list = QListWidget()
        self.label_encoding_columns_list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.label_encoding_columns_list.setMinimumHeight(190)
        self.label_encoding_columns_list.itemClicked.connect(
            self._label_encoding_candidate_clicked
        )
        self.unique_column_name = QLabel("Select a candidate column")
        self.unique_column_name.setObjectName("uniqueColumnName")
        self.unique_summary_label = QLabel("")
        self.unique_summary_label.setWordWrap(True)
        self.unique_values_list = QListWidget()
        self.unique_values_list.setMinimumHeight(150)
        self.unique_values_message = QLabel("")
        self.unique_values_message.setWordWrap(True)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.feedback_card = self._feedback_card()
        self.status_card = self._status_card()

        self.empty_state_card = self._empty_state_card()
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)
        controls_layout.addWidget(self._selection_card())
        controls_layout.addWidget(self._target_card())
        controls_layout.addWidget(self._encoding_card())

        self.confirm_button = QPushButton("Confirm Modeling Columns")
        self.confirm_button.setObjectName("primaryColumnConfigButton")
        self.confirm_button.setIcon(icon("fa6s.circle-check", "#FFFFFF"))
        self.confirm_button.setMinimumHeight(44)
        self.confirm_button.clicked.connect(self.confirm_modeling_columns)
        controls_layout.addWidget(self.confirm_button)
        controls_layout.addWidget(self.feedback_card)
        controls_layout.addWidget(self.status_card)

        content = QWidget()
        content.setObjectName("columnConfigContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)
        title = QLabel("Column Configuration")
        title.setObjectName("columnConfigTitle")
        subtitle = QLabel(
            "Select modeling columns, choose the target, and configure categorical encoding."
        )
        subtitle.setObjectName("columnConfigSubtitle")
        content_layout.addWidget(title)
        content_layout.addWidget(subtitle)
        content_layout.addWidget(
            self.empty_state_card,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        content_layout.addWidget(self.controls_widget)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self._apply_style()
        self._show_empty_state()

    def refresh(self) -> None:
        df = self.main_window.dataframe
        config = self.main_window.config
        if config is None or df is None:
            self._show_empty_state()
            self.target_plot.set_series(None)
            return

        self.empty_state_card.hide()
        self.controls_widget.show()
        self._refreshing = True
        try:
            columns = sorted(
                (str(column) for column in df.columns),
                key=str.casefold,
            )
            configured_target = config.target_column
            selected_features = [
                column
                for column in config.feature_columns
                if column in columns and column != configured_target
            ]
            selected = sorted(
                selected_features
                + (
                [configured_target]
                if configured_target and configured_target in columns
                else []
                ),
                key=str.casefold,
            )
            self._set_transfer_columns(columns, selected)
            self._refresh_target_options(configured_target)
            self._refresh_label_encoding_options(config.label_encoding_columns)
            self._select_combo(self.target_input, config.target_column)
        finally:
            self._refreshing = False
        self._update_target_plot()

    def add_selected(self) -> None:
        self._move_items(
            self.available_columns_list,
            self.selected_columns_list,
            sort_destination=True,
        )
        self._refresh_after_selection_change()

    def remove_selected(self) -> None:
        self._move_items(
            self.selected_columns_list,
            self.available_columns_list,
            sort_destination=True,
        )
        self._refresh_after_selection_change()

    def add_all(self) -> None:
        items = self._all_items(self.available_columns_list)
        for text in items:
            self.selected_columns_list.addItem(text)
        self.available_columns_list.clear()
        self._sort_list(self.selected_columns_list)
        self._refresh_after_selection_change()

    def clear_selected(self) -> None:
        for text in self._all_items(self.selected_columns_list):
            self.available_columns_list.addItem(text)
        self.selected_columns_list.clear()
        self._sort_list(self.available_columns_list)
        self._refresh_after_selection_change()

    def confirm_modeling_columns(self) -> None:
        config = self.main_window.config
        df = self.main_window.dataframe
        if config is None or df is None:
            self._show_confirmation_error(
                "Project configuration and loaded data are required."
            )
            return

        selected = self._all_items(self.selected_columns_list)
        if not selected:
            self._show_confirmation_error(
                "Select at least one modeling column before confirming."
            )
            return
        target = self.target_input.currentText() or None
        if not target:
            self._show_confirmation_error("Select a target column before confirming.")
            return
        features = [column for column in selected if column != target]
        if not features:
            self._show_confirmation_error(
                "Select at least one feature column in addition to the target column."
            )
            return

        label_encoding_columns = [
            self.label_encoding_columns_list.item(index).text()
            for index in range(self.label_encoding_columns_list.count())
            if self.label_encoding_columns_list.item(index).checkState()
            == Qt.CheckState.Checked
            and self.label_encoding_columns_list.item(index).text() != target
        ]
        metadata = {
            column: self._column_encoding_metadata(df[column])
            for column in label_encoding_columns
        }

        previous_target = config.target_column
        config.feature_columns = features
        config.target_column = target
        config.label_encoding_columns = label_encoding_columns
        config.preprocessing_options = dict(config.preprocessing_options or {})
        config.preprocessing_options["label_encoding_metadata"] = metadata
        subset_columns = list(features) + [target]
        data_dir = Path(config.project_dir) / "data"
        subset_path = data_dir / "modeling_subset.csv"
        try:
            invalidate_target_artifacts(
                Path(config.project_dir) / "outputs" / "data_split",
                previous_target,
                target,
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            df.loc[:, subset_columns].to_csv(subset_path, index=False)
            config.save()
        except Exception as exc:
            self._show_confirmation_error(f"Could not save modeling subset: {exc}")
            return

        self._show_success_feedback(
            [
                "Modeling configuration saved successfully.",
                f"Selected features: {len(features)}",
                f"Target column: {target}",
                f"Label-encoded columns: {len(label_encoding_columns)}",
            ]
        )
        self._show_status_message(
            f"Modeling subset saved to: {subset_path}",
            level="success",
        )

    def _target_changed(self) -> None:
        if self._refreshing:
            return
        self._refresh_label_encoding_options()
        self._update_target_plot()

    def _update_target_plot(self) -> None:
        df = self.main_window.dataframe
        target = self.target_input.currentText()
        self.target_plot.set_series(
            df[target]
            if df is not None and target and target in df.columns
            else None
        )

    def _selection_card(self) -> QWidget:
        card, layout = self._card(
            "modelingColumnsCard",
            "Modeling Columns",
            "Move columns into the modeling set; both lists remain alphabetically sorted.",
            "fa6s.table-columns",
        )
        row = QHBoxLayout()
        row.setSpacing(16)
        left = QVBoxLayout()
        left.addWidget(QLabel("Available Columns"))
        left.addWidget(self.available_columns_list)
        row.addLayout(left, stretch=1)

        buttons = QVBoxLayout()
        buttons.addStretch(1)
        for label, callback in [
            ("Add Selected", self.add_selected),
            ("Remove Selected", self.remove_selected),
            ("Add All", self.add_all),
            ("Clear Selected", self.clear_selected),
        ]:
            button = QPushButton(label)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch(1)
        row.addLayout(buttons)

        right = QVBoxLayout()
        right.addWidget(QLabel("Selected Modeling Columns"))
        right.addWidget(self.selected_columns_list)
        row.addLayout(right, stretch=1)
        layout.addLayout(row)
        return card

    def _target_card(self) -> QWidget:
        card, layout = self._card(
            "targetColumnCard",
            "Target Column",
            "Choose the outcome column and review its complete class distribution.",
            "fa6s.bullseye",
        )
        layout.addWidget(self.target_input)
        layout.addWidget(self.target_plot)
        return card

    def _encoding_card(self) -> QWidget:
        card, layout = self._card(
            "encodingOptionsCard",
            "Categorical Encoding Options",
            "Check columns to encode; select a row to inspect its unique values.",
            "fa6s.tags",
        )
        row = QHBoxLayout()
        row.setSpacing(16)
        candidates = QVBoxLayout()
        candidates.addWidget(QLabel("Label Encoding Candidates"))
        candidates.addWidget(self.label_encoding_columns_list)
        row.addLayout(candidates, stretch=1)

        preview = QFrame()
        preview.setObjectName("uniqueValuesPanel")
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(16, 14, 16, 14)
        preview_layout.setSpacing(8)
        preview_layout.addWidget(self.unique_column_name)
        preview_layout.addWidget(self.unique_summary_label)
        preview_layout.addWidget(self.unique_values_list)
        preview_layout.addWidget(self.unique_values_message)
        row.addWidget(preview, stretch=1)
        layout.addLayout(row)
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
        icon_label.setPixmap(icon(icon_name).pixmap(24, 24))
        title_label = QLabel(title)
        title_label.setObjectName("columnCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("columnCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return card, layout

    def _empty_state_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("columnConfigEmptyCard")
        card.setMaximumWidth(620)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)
        empty_icon = QLabel()
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setPixmap(icon("fa6s.table-columns").pixmap(48, 48))
        message = QLabel(
            "Please select or create a project and load a dataset first."
        )
        message.setObjectName("columnConfigEmptyMessage")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(empty_icon)
        layout.addWidget(message)
        return card

    def _feedback_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("columnConfigFeedbackCard")
        card.setMaximumHeight(136)
        card.hide()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        self.feedback_icons: list[QLabel] = []
        self.feedback_labels: list[QLabel] = []
        for _ in range(4):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            icon_label = QLabel()
            icon_label.setObjectName("successLineIcon")
            icon_label.setFixedSize(18, 18)
            text_label = QLabel("")
            text_label.setWordWrap(True)
            row_layout.addWidget(icon_label)
            row_layout.addWidget(text_label, stretch=1)
            layout.addWidget(row)
            self.feedback_icons.append(icon_label)
            self.feedback_labels.append(text_label)
        self.feedback_icon = self.feedback_icons[0]
        self.feedback_label = self.feedback_labels[0]
        return card

    def _status_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("columnConfigStatusCard")
        card.setMaximumHeight(52)
        card.hide()
        layout = QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        self.status_icon = QLabel()
        self.status_icon.setObjectName("successLineIcon")
        self.status_icon.setFixedSize(18, 18)
        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_label, stretch=1)
        return card

    def _refresh_after_selection_change(self) -> None:
        self._refresh_target_options()
        self._refresh_label_encoding_options()

    def _set_transfer_columns(self, columns: list[str], selected: list[str]) -> None:
        self.available_columns_list.clear()
        self.selected_columns_list.clear()
        selected_set = set(selected)
        self.available_columns_list.addItems(
            sorted(
                (column for column in columns if column not in selected_set),
                key=str.casefold,
            )
        )
        self.selected_columns_list.addItems(sorted(selected, key=str.casefold))

    def _refresh_target_options(self, preferred: str | None = None) -> None:
        current = preferred if preferred is not None else self.target_input.currentText()
        selected = self._all_items(self.selected_columns_list)
        self.target_input.blockSignals(True)
        self.target_input.clear()
        self.target_input.addItem("")
        self.target_input.addItems(selected)
        self._select_combo(self.target_input, current if current in selected else None)
        self.target_input.blockSignals(False)
        self._update_target_plot()

    def _refresh_label_encoding_options(
        self, preferred: list[str] | None = None
    ) -> None:
        df = self.main_window.dataframe
        if df is None:
            self.label_encoding_columns_list.clear()
            self._clear_unique_preview()
            return
        target = self.target_input.currentText()
        previous_checked = (
            set(preferred)
            if preferred is not None
            else {
                self.label_encoding_columns_list.item(index).text()
                for index in range(self.label_encoding_columns_list.count())
                if self.label_encoding_columns_list.item(index).checkState()
                == Qt.CheckState.Checked
            }
        )
        candidates = [
            column
            for column in self._all_items(self.selected_columns_list)
            if column != target
            and column in df.columns
            and self._is_categorical_or_text(df[column])
        ]
        self.label_encoding_columns_list.clear()
        for column in candidates:
            item = QListWidgetItem(column)
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEnabled
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if column in previous_checked
                else Qt.CheckState.Unchecked
            )
            self.label_encoding_columns_list.addItem(item)
        self._clear_unique_preview()

    def _label_encoding_candidate_clicked(self, item: QListWidgetItem) -> None:
        self._show_unique_values(item.text())

    def _show_unique_values(self, column: str) -> None:
        df = self.main_window.dataframe
        if df is None or column not in df.columns:
            self._clear_unique_preview()
            return
        series = df[column]
        counts = series.value_counts(dropna=False)
        total_rows = len(series)
        self.unique_column_name.setText(column)
        self.unique_summary_label.setText(
            f"Unique values: {series.nunique(dropna=True):,}\n"
            f"Missing values: {int(series.isna().sum()):,}"
        )
        self.unique_values_list.clear()
        shown = counts.iloc[:MAX_UNIQUE_PREVIEW]
        for value, count in shown.items():
            label = "Missing/Null" if pd.isna(value) else str(value)
            percentage = count / total_rows * 100 if total_rows else 0.0
            self.unique_values_list.addItem(
                f"{label} \u2014 {int(count):,} rows ({percentage:.1f}%)"
            )
        if len(counts) > MAX_UNIQUE_PREVIEW:
            self.unique_values_message.setText(
                f"Showing first {MAX_UNIQUE_PREVIEW} of "
                f"{len(counts)} unique values."
            )
        else:
            self.unique_values_message.setText(
                f"Showing all {len(counts)} unique values."
            )

    def _clear_unique_preview(self) -> None:
        self.unique_column_name.setText("Select a candidate column")
        self.unique_summary_label.setText("")
        self.unique_values_list.clear()
        self.unique_values_message.setText("")

    def _column_encoding_metadata(self, series: pd.Series) -> dict[str, Any]:
        unique_values = series.drop_duplicates().tolist()
        return {
            "unique_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "preview_values": [
                None if pd.isna(value) else str(value)
                for value in unique_values[:MAX_UNIQUE_PREVIEW]
            ],
            "preview_limit": MAX_UNIQUE_PREVIEW,
        }

    def _is_categorical_or_text(self, series: pd.Series) -> bool:
        return bool(
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series)
        )

    def _show_confirmation_error(self, message: str) -> None:
        self.success_notification_timer.stop()
        for index, label in enumerate(self.feedback_labels):
            label.parentWidget().setVisible(index == 0)
        self.feedback_icon.setPixmap(
            icon("fa6s.circle-xmark", "#CF222E").pixmap(16, 16)
        )
        self.feedback_label.setText(f"Error: {message}")
        self.feedback_card.setStyleSheet(self._feedback_style("error"))
        self.feedback_card.show()
        self._show_status_message(message, level="error")

    def _show_success_feedback(self, messages: list[str]) -> None:
        for index, (icon_label, text_label) in enumerate(
            zip(self.feedback_icons, self.feedback_labels)
        ):
            visible = index < len(messages)
            text_label.parentWidget().setVisible(visible)
            if visible:
                icon_label.setPixmap(
                    icon("fa6s.circle-check", "#2DA44E").pixmap(16, 16)
                )
                text_label.setText(messages[index])
        self.feedback_card.setStyleSheet(self._feedback_style("success"))
        self.feedback_card.show()
        self._restart_success_notification_timer()

    def _show_status_message(self, message: str, *, level: str) -> None:
        success = level == "success"
        color = "#2DA44E" if success else "#CF222E"
        icon_name = "fa6s.circle-check" if success else "fa6s.circle-xmark"
        self.status_icon.setPixmap(icon(icon_name, color).pixmap(16, 16))
        self.status_label.setText(message)
        self.status_card.setStyleSheet(self._feedback_style(level))
        self.status_card.show()
        if success:
            self._restart_success_notification_timer()

    def _restart_success_notification_timer(self) -> None:
        self.success_notification_timer.stop()
        self.success_notification_timer.start()

    def _dismiss_success_notifications(self) -> None:
        self.feedback_card.hide()
        self.status_card.hide()

    def _feedback_style(self, level: str) -> str:
        border, background = (
            ("#2DA44E", "#F8FFF9")
            if level == "success"
            else ("#CF222E", "#FFF8F7")
        )
        return (
            f"QFrame {{ background: {background}; border: 1px solid {BORDER};"
            f" border-left: 3px solid {border}; border-radius: 7px; }}"
            "QWidget { border: none; background: transparent; }"
            f"QLabel {{ color: {TEXT}; border: none; background: transparent; }}"
        )

    def _show_empty_state(self) -> None:
        self.controls_widget.hide()
        self.empty_state_card.show()

    def _move_items(
        self,
        source: QListWidget,
        destination: QListWidget,
        *,
        sort_destination: bool,
    ) -> None:
        rows = sorted((source.row(item) for item in source.selectedItems()), reverse=True)
        moved = [source.takeItem(row).text() for row in rows]
        for text in reversed(moved):
            if not destination.findItems(text, Qt.MatchFlag.MatchExactly):
                destination.addItem(text)
        if sort_destination:
            self._sort_list(destination)

    def _sort_list(self, list_widget: QListWidget) -> None:
        items = sorted(self._all_items(list_widget), key=str.casefold)
        list_widget.clear()
        list_widget.addItems(items)

    def _select_combo(self, combo: QComboBox, value: str | None) -> None:
        index = combo.findText(value or "")
        if index >= 0:
            combo.setCurrentIndex(index)

    def _all_items(self, list_widget: QListWidget) -> list[str]:
        return [
            list_widget.item(index).text()
            for index in range(list_widget.count())
        ]

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#columnConfigContent {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#columnConfigTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#columnConfigSubtitle,
            QLabel#columnCardSubtitle {{ color: #5B6573; font-size: 12px; }}
            QWidget#modelingColumnsCard,
            QWidget#targetColumnCard,
            QWidget#encodingOptionsCard,
            QWidget#columnConfigEmptyCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#columnCardTitle,
            QLabel#uniqueColumnName {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#columnConfigEmptyMessage {{ color: #5B6573; font-size: 13px; }}
            QFrame#uniqueValuesPanel {{
                background: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QListWidget, QComboBox {{
                background: #FFFFFF;
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 5px;
            }}
            QListWidget::item {{ min-height: 26px; padding: 3px 6px; }}
            QListWidget::item:selected {{ background: #DCEBFA; color: {TEXT}; }}
            QPushButton {{
                min-height: 36px;
                border-radius: 6px;
                padding: 0 12px;
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton:hover {{ background: #EFF6FF; border-color: {PRIMARY}; }}
            QPushButton#primaryColumnConfigButton {{
                min-height: 44px;
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
                border-radius: 7px;
                font-weight: 600;
            }}
            QPushButton#primaryColumnConfigButton:hover {{ background: #00A6A6; }}
            """
        )
