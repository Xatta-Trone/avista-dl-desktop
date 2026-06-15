"""Data import page."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.core.data_loader import load_dataset, summarize_dataframe
from app.core.dataset_manager import (
    DUPLICATE_APPEND_TIMESTAMP,
    DUPLICATE_CANCEL,
    DUPLICATE_OVERWRITE,
    copy_dataset_into_project,
    project_dataset_path,
)
from app.gui.dataframe_model import PandasPreviewModel, display_value
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon


DEFAULT_ROWS_PER_PAGE = 50
MAX_RENDER_ROWS = 200


class DataImportPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.current_page = 0
        self.summary: dict | None = None
        self.setObjectName("dataImportPage")

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.info_card = self._message_card()
        self.empty_state_card = self._empty_state_card()
        self.empty_state_area = QWidget()
        self.empty_state_area.setObjectName("dataImportEmptyStateArea")
        empty_area_layout = QVBoxLayout(self.empty_state_area)
        empty_area_layout.setContentsMargins(0, 24, 0, 0)
        empty_area_layout.addStretch(1)
        empty_row = QHBoxLayout()
        empty_row.addStretch(1)
        empty_row.addWidget(self.empty_state_card)
        empty_row.addStretch(1)
        empty_area_layout.addLayout(empty_row)
        empty_area_layout.addStretch(2)
        self.cards_widget = QWidget()
        self.cards_widget.setObjectName("datasetSummaryArea")
        self.cards_layout = QGridLayout(self.cards_widget)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(16)

        self.table = QTableView()
        self.preview_model = PandasPreviewModel(parent=self.table)
        self.table.setModel(self.preview_model)
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.horizontalHeader().setMinimumSectionSize(120)
        self.table.horizontalHeader().setDefaultSectionSize(180)
        self.table.horizontalHeader().setFixedHeight(62)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.previous_button = QPushButton("Previous Page")
        self.previous_button.clicked.connect(self.previous_page)
        self.next_button = QPushButton("Next Page")
        self.next_button.clicked.connect(self.next_page)
        self.page_label = QLabel("Page 0 of 0")
        self.rows_per_page_combo = QComboBox()
        self.rows_per_page_combo.addItems(["25", "50", "100", "200"])
        self.rows_per_page_combo.setCurrentText(str(DEFAULT_ROWS_PER_PAGE))
        self.rows_per_page_combo.currentTextChanged.connect(self._rows_per_page_changed)

        self.pagination_widget = QWidget()
        self.pagination_widget.setObjectName("paginationControls")
        self.pagination_widget.setLayout(self._pagination_layout())
        self.preview_card = self._preview_card()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Data Import")
        title.setObjectName("dataImportTitle")
        subtitle = QLabel(
            "Review the managed project dataset, summary statistics, and preview."
        )
        subtitle.setObjectName("dataImportSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.info_card)
        layout.addWidget(self.empty_state_area, stretch=1)

        self.load_button = QPushButton("Load Project Dataset")
        self.load_button.clicked.connect(self.load_selected_file)
        self.replace_dataset_button = QPushButton("Replace Dataset")
        self.replace_dataset_button.clicked.connect(self.replace_dataset)
        self.load_button.hide()
        self.replace_dataset_button.hide()

        layout.addWidget(self.cards_widget)
        layout.addWidget(self.preview_card, stretch=1)
        self._apply_style()
        self._show_no_project_state()
        self.update_pagination_controls()

    def refresh(self) -> None:
        config = self.main_window.config
        if config is None:
            self._show_no_project_state()
            return
        dataset_path = project_dataset_path(config)
        if dataset_path and dataset_path.exists():
            if self.main_window.dataframe is not None and self.summary is not None:
                self._show_loaded_state()
            else:
                self._show_info(
                    f"Loading managed project dataset: "
                    f"{config.dataset.get('project_relative_path', dataset_path.name)}"
                )
            return
        self._show_missing_dataset_state()

    def load_selected_file(self) -> None:
        config = self.main_window.config
        if not config:
            self._show_no_project_state()
            return

        try:
            dataset_path = project_dataset_path(config)
            if dataset_path is None or not dataset_path.exists():
                self._show_missing_dataset_state()
                return
            config.input_file = str(dataset_path)
            df = load_dataset(dataset_path)
            self.main_window.set_dataframe(df)
            self.summary = summarize_dataframe(df)
            self.current_page = 0
            self._populate_cards(self.summary, df)
            self.render_preview_page()
            self._show_loaded_state()
        except Exception as exc:
            self._show_message(f"Failed to load dataset: {exc}", "error")

    def restore_project_dataset(self) -> None:
        """Load the configured project dataset without another file dialog."""

        self.load_selected_file()

    def replace_dataset(self) -> None:
        config = self.main_window.config
        if config is None:
            self._show_no_project_state()
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Replace Project Dataset",
            "",
            "Tabular files (*.csv *.xlsx *.parquet *.feather *.fst);;All files (*)",
        )
        if not selected:
            return
        destination = Path(config.project_dir) / "data" / Path(selected).name
        policy = self._duplicate_policy(destination)
        if policy == DUPLICATE_CANCEL:
            return
        try:
            dataset = copy_dataset_into_project(
                selected,
                config.project_dir,
                duplicate_policy=policy,
            )
            config.dataset = dataset
            config.input_file = str(
                Path(config.project_dir) / dataset["project_relative_path"]
            )
            config.save()
            self.load_selected_file()
            self._show_message(
                "Dataset copied into project:\n"
                f"{dataset['project_relative_path']}\n"
                f"Loaded {self.summary['rows']:,} rows and "
                f"{self.summary['columns']:,} columns.",
                "success",
            )
        except Exception as exc:
            self._show_message(f"Failed to replace dataset: {exc}", "error")

    def _duplicate_policy(self, destination: Path) -> str:
        if not destination.exists():
            return DUPLICATE_APPEND_TIMESTAMP
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Dataset Already Exists")
        dialog.setText(f"A dataset named '{destination.name}' already exists.")
        overwrite = dialog.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        keep_both = dialog.addButton("Keep Both", QMessageBox.ButtonRole.AcceptRole)
        cancel = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(keep_both)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is overwrite:
            return DUPLICATE_OVERWRITE
        if clicked is cancel:
            return DUPLICATE_CANCEL
        return DUPLICATE_APPEND_TIMESTAMP

    def render_preview_page(self) -> None:
        """Render only the current bounded page slice."""

        df = self.main_window.dataframe
        if df is None or self.summary is None:
            self.preview_model.set_preview(pd.DataFrame(), [])
            self.update_pagination_controls()
            return

        preview_df = self.get_current_page_slice()
        headers = [self._header_text(column, self.summary) for column in preview_df.columns]
        self.table.setSortingEnabled(False)
        self.preview_model.set_preview(preview_df, headers)
        self.update_pagination_controls()
        self._show_loaded_state()

    def next_page(self) -> None:
        total_pages = self._total_pages()
        if self.current_page + 1 < total_pages:
            self.current_page += 1
            self.render_preview_page()

    def previous_page(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            self.render_preview_page()

    def update_pagination_controls(self) -> None:
        total_pages = self._total_pages()
        if total_pages == 0:
            self.current_page = 0
            self.page_label.setText("Page 0 of 0")
            self.previous_button.setEnabled(False)
            self.next_button.setEnabled(False)
            return

        self.current_page = min(self.current_page, total_pages - 1)
        self.page_label.setText(f"Page {self.current_page + 1} of {total_pages}")
        self.previous_button.setEnabled(self.current_page > 0)
        self.next_button.setEnabled(self.current_page + 1 < total_pages)

    def get_current_page_slice(self) -> pd.DataFrame:
        """Return at most 200 rows for the current page."""

        df = self.main_window.dataframe
        if df is None:
            return pd.DataFrame()
        page_size = self._rows_per_page()
        start = self.current_page * page_size
        end = min(start + page_size, len(df))
        return df.iloc[start:end]

    def _pagination_layout(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self.previous_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.page_label)
        layout.addStretch(1)
        layout.addWidget(QLabel("Rows per page"))
        layout.addWidget(self.rows_per_page_combo)
        return layout

    def _rows_per_page_changed(self, _value: str | None = None) -> None:
        self.current_page = 0
        self.render_preview_page()

    def _rows_per_page(self) -> int:
        try:
            requested = int(self.rows_per_page_combo.currentText())
        except ValueError:
            requested = DEFAULT_ROWS_PER_PAGE
        return max(1, min(requested, MAX_RENDER_ROWS))

    def _total_pages(self) -> int:
        df = self.main_window.dataframe
        if df is None or len(df) == 0:
            return 0
        return math.ceil(len(df) / self._rows_per_page())

    def _populate_cards(self, summary: dict, df: pd.DataFrame) -> None:
        self._clear_cards()
        duplicate_id_count, duplicate_id_percent = self._duplicate_id_summary(df)
        cards = [
            ("Total rows", f"{summary['rows']:,}"),
            ("Total columns", f"{summary['columns']:,}"),
            ("Numeric columns", f"{len(summary['numeric_columns']):,} ({summary['numeric_percent']:.1f}%)"),
            (
                "Categorical columns",
                f"{len(summary['categorical_columns']):,} ({summary['categorical_percent']:.1f}%)",
            ),
            ("Missing values", f"{summary['total_missing_values']:,} ({summary['total_missing_percent']:.1f}%)"),
            ("Duplicate rows", f"{summary['duplicate_rows']:,} ({summary['duplicate_row_percent']:.1f}%)"),
            ("Memory usage", f"{summary['memory_usage_mb']:.2f} MB"),
        ]
        if duplicate_id_count is not None:
            cards.append(("Duplicate IDs", f"{duplicate_id_count:,} ({duplicate_id_percent:.1f}%)"))

        for index, (label, value) in enumerate(cards):
            self.cards_layout.addWidget(self._summary_card(label, value), index // 4, index % 4)

    def _summary_card(self, label: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("datasetSummaryCard")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(frame)
        shadow.setBlurRadius(18)
        shadow.setColor(QColor(31, 41, 55, 24))
        shadow.setOffset(0, 3)
        frame.setGraphicsEffect(shadow)
        frame.setStyleSheet(
            f"""
            QFrame#datasetSummaryCard {{
                border: 1px solid {BORDER};
                border-radius: 10px;
                background: #FFFFFF;
            }}
            QLabel#cardLabel {{
                color: #5B6573;
                font-size: 11px;
                border: none;
            }}
            QLabel#cardValue {{
                color: {TEXT};
                font-size: 18px;
                font-weight: 700;
                border: none;
            }}
            """
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)
        label_widget = QLabel(label)
        label_widget.setObjectName("cardLabel")
        value_widget = QLabel(value)
        value_widget.setObjectName("cardValue")
        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        return frame

    def _header_text(self, column, summary: dict) -> str:
        missing_count = summary["missing_counts"].get(column, 0)
        missing_percent = summary["column_missing_percent"].get(column, 0.0)
        display_type = summary["simplified_column_types"].get(column, "Unknown")
        return f"{column}\nType: {display_type}\nMissing: {missing_count} ({missing_percent:.1f}%)"

    def _duplicate_id_summary(self, df: pd.DataFrame) -> tuple[int | None, float]:
        config = self.main_window.config
        id_columns = [column for column in (getattr(config, "id_columns", []) or []) if column in df.columns]
        if not id_columns:
            return None, 0.0
        duplicate_ids = int(df.duplicated(subset=id_columns).sum())
        duplicate_percent = (duplicate_ids / len(df) * 100) if len(df) else 0.0
        return duplicate_ids, float(duplicate_percent)

    def _clear_cards(self) -> None:
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _message_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("dataImportMessageCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        self.status_icon = QLabel()
        self.status_icon.setFixedWidth(24)
        layout.addWidget(self.status_icon)
        layout.addWidget(self.status_label, stretch=1)
        return card

    def _empty_state_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("dataImportEmptyStateCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card.setMinimumWidth(420)
        card.setMaximumWidth(600)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(31, 41, 55, 30))
        shadow.setOffset(0, 5)
        card.setGraphicsEffect(shadow)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(36, 32, 36, 32)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.empty_state_icon = QLabel()
        self.empty_state_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_icon.setPixmap(
            icon("fa6s.folder-open", PRIMARY).pixmap(56, 56)
        )
        layout.addWidget(self.empty_state_icon)

        self.empty_state_title = QLabel("No project selected")
        self.empty_state_title.setObjectName("dataImportEmptyStateTitle")
        self.empty_state_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_state_title)

        self.empty_state_message = QLabel(
            "Please create a new AVISTA project or open an existing project "
            "before importing data."
        )
        self.empty_state_message.setObjectName("dataImportEmptyStateMessage")
        self.empty_state_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state_message.setWordWrap(True)
        layout.addWidget(self.empty_state_message)

        self.go_to_project_setup_button = QPushButton("Go to Project Setup")
        self.go_to_project_setup_button.setObjectName("dataImportPrimaryButton")
        self.go_to_project_setup_button.setIcon(icon("fa6s.diagram-project", "#FFFFFF"))
        self.go_to_project_setup_button.setMinimumHeight(42)
        self.go_to_project_setup_button.clicked.connect(
            lambda: self.main_window.navigate_to(0)
        )
        layout.addWidget(
            self.go_to_project_setup_button,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        return card

    def _preview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("dataPreviewCard")
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        header_icon = QLabel()
        header_icon.setPixmap(icon("fa6s.table").pixmap(22, 22))
        header_title = QLabel("Data Preview")
        header_title.setObjectName("dataPreviewTitle")
        header.addWidget(header_icon)
        header.addWidget(header_title)
        header.addStretch(1)
        layout.addLayout(header)
        layout.addWidget(self.pagination_widget)
        layout.addWidget(self.table, stretch=1)
        return card

    def _show_no_project_state(self) -> None:
        self._clear_loaded_view()
        self.info_card.hide()
        self.empty_state_area.show()

    def _show_missing_dataset_state(self) -> None:
        self._clear_loaded_view()
        self._show_message(
            "Project dataset could not be found. "
            "Please replace the dataset from Project Setup.",
            "warning",
        )

    def _show_loaded_state(self) -> None:
        self.info_card.hide()
        self.empty_state_area.hide()
        self.cards_widget.show()
        self.preview_card.show()
        self.pagination_widget.show()
        self.table.show()

    def _show_info(self, message: str) -> None:
        self._clear_loaded_view()
        self.empty_state_area.hide()
        self._show_message(message, "info")

    def _show_message(self, message: str, level: str) -> None:
        colors = {
            "info": (PRIMARY, "#EFF6FF", "fa6s.circle-info"),
            "warning": ("#BF6A02", "#FFF8E6", "fa6s.triangle-exclamation"),
            "error": ("#CF222E", "#FFF1F0", "fa6s.circle-xmark"),
            "success": ("#2DA44E", "#F0FFF4", "fa6s.circle-check"),
        }
        border, background, icon_name = colors[level]
        self.status_label.setText(message)
        self.empty_state_area.hide()
        self.status_icon.setPixmap(icon(icon_name, border).pixmap(20, 20))
        self.info_card.setStyleSheet(
            f"""
            QFrame#dataImportMessageCard {{
                background: {background};
                border: 1px solid {border};
                border-radius: 10px;
            }}
            QLabel {{
                color: {TEXT};
                border: none;
                background: transparent;
            }}
            """
        )
        self.info_card.show()

    def _clear_loaded_view(self) -> None:
        self.cards_widget.hide()
        self.preview_card.hide()
        self.pagination_widget.hide()
        self.table.hide()
        self.preview_model.set_preview(pd.DataFrame(), [])
        self.current_page = 0
        self.summary = None
        self.update_pagination_controls()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#dataImportPage {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#dataImportTitle {{
                color: {TEXT};
                font-size: 24px;
                font-weight: 700;
            }}
            QLabel#dataImportSubtitle {{ color: #5B6573; font-size: 12px; }}
            QFrame#dataPreviewCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#dataPreviewTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QFrame#dataImportEmptyStateCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 12px;
            }}
            QLabel#dataImportEmptyStateTitle {{
                color: {TEXT};
                font-size: 20px;
                font-weight: 700;
            }}
            QLabel#dataImportEmptyStateMessage {{
                color: #5B6573;
                font-size: 12px;
            }}
            QPushButton#dataImportPrimaryButton {{
                min-width: 180px;
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
                border-radius: 7px;
                padding: 0 18px;
                font-weight: 600;
            }}
            QPushButton#dataImportPrimaryButton:hover {{ background: #00A6A6; }}
            QTableView {{
                background: #FFFFFF;
                alternate-background-color: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5E7EB;
            }}
            QHeaderView::section {{
                background: #F0F4F8;
                color: {TEXT};
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 6px;
                font-weight: 600;
            }}
            QPushButton {{
                min-height: 34px;
                border-radius: 6px;
                padding: 0 12px;
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton:hover {{ background: #EFF6FF; border-color: {PRIMARY}; }}
            QComboBox {{
                min-height: 32px;
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 0 8px;
            }}
            """
        )


def _table_value_text(value) -> str:
    """Backward-compatible wrapper used by GUI tests."""

    return display_value(value)
