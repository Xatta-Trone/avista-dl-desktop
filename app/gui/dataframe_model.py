"""Qt table model for a bounded pandas DataFrame preview."""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class PandasPreviewModel(QAbstractTableModel):
    """Display a preview DataFrame slice without owning the full dataset."""

    def __init__(
        self,
        preview_df: pd.DataFrame | None = None,
        header_labels: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._preview_df = preview_df if preview_df is not None else pd.DataFrame()
        self._header_labels = header_labels or [str(column) for column in self._preview_df.columns]

    def set_preview(self, preview_df: pd.DataFrame, header_labels: list[str]) -> None:
        self.beginResetModel()
        self._preview_df = preview_df
        self._header_labels = header_labels
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._preview_df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._preview_df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._preview_df.iat[index.row(), index.column()]
        return display_value(value)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._header_labels):
                return self._header_labels[section]
            return None
        if 0 <= section < len(self._preview_df.index):
            return str(self._preview_df.index[section])
        return None


def display_value(value: Any) -> str:
    """Format missing values consistently for the preview."""

    try:
        if value is None or pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return str(value)
