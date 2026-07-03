import importlib.util
import inspect

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 is not installed.",
)


def _result(*, available, skipped=False, error="", current="1.0.0", latest="1.0.1"):
    from app.core.update_checker import UpdateCheckResult, UpdateMetadata

    metadata = (
        UpdateMetadata(
            latest_version=latest,
            release_date="2026-07-03",
            release_notes=["Update checker"],
            installer_url="https://example.com/AVISTA_Setup.exe",
        )
        if latest
        else None
    )
    return UpdateCheckResult(
        current_version=current,
        metadata=metadata,
        update_available=available,
        skipped=skipped,
        error=error,
    )


def test_startup_up_to_date_does_not_show_popup(monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    window = MainWindow()

    window._handle_update_check_finished(
        _result(available=False, current="1.0.1", latest="1.0.1"),
        manual=False,
        thread=object(),
        worker=object(),
    )

    assert messages == []
    window.close()
    assert app is not None


def test_manual_up_to_date_shows_message(monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    window = MainWindow()

    window._handle_update_check_finished(
        _result(available=False, current="1.0.1", latest="1.0.1"),
        manual=True,
        thread=object(),
        worker=object(),
    )

    assert messages
    assert messages[0][2] == "AVISTA is up to date."
    window.close()
    assert app is not None


def test_newer_startup_result_shows_update_dialog(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    shown = []

    class FakeDialog:
        DOWNLOAD = 1
        LATER = 0

        def __init__(self, *, current_version, metadata, parent=None):
            shown.append((current_version, metadata.latest_version, parent))
            self.choice = self.LATER

        def exec(self):
            return 0

    monkeypatch.setattr("app.gui.main_window.UpdateAvailableDialog", FakeDialog)
    window = MainWindow()

    window._handle_update_check_finished(
        _result(available=True),
        manual=False,
        thread=object(),
        worker=object(),
    )

    assert shown
    assert shown[0][0] == "1.0.0"
    assert shown[0][1] == "1.0.1"
    window.close()
    assert app is not None


def test_skipped_startup_result_does_not_show_update_dialog(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    shown = []
    monkeypatch.setattr(
        "app.gui.main_window.UpdateAvailableDialog",
        lambda *args, **kwargs: shown.append((args, kwargs)),
    )
    window = MainWindow()

    window._handle_update_check_finished(
        _result(available=True, skipped=True),
        manual=False,
        thread=object(),
        worker=object(),
    )

    assert shown == []
    window.close()
    assert app is not None


def test_update_check_uses_qthread_worker():
    from app.gui.main_window import MainWindow
    from app.gui.workers import UpdateCheckWorker

    start_source = inspect.getsource(MainWindow._start_update_check)
    worker_source = inspect.getsource(UpdateCheckWorker.run)

    assert "QThread" in start_source
    assert "worker.moveToThread(thread)" in start_source
    assert "thread.started.connect(worker.run)" in start_source
    assert "check_for_updates(" in worker_source
