"""Shared pytest isolation for optional GUI state."""

from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def cleanup_qt_top_level_widgets():
    """Delete closed Qt windows so timers and signals cannot leak across tests."""

    yield

    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
