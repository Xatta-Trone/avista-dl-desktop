from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from app.core import tabpfn_model_manager as manager
from app.core.tabpfn_model_manager import (
    TABPFN_CHECKPOINT_FILENAME,
    TabPFNModelState,
    TabPFNModelStatus,
    TabPFNVerificationResult,
)
from app.gui.main_window import MainWindow
from app.gui.tabpfn_model_dialog import TabPFNModelDialog
from app.gui.workers import TabPFNModelWorker


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        with archive.open("checkpoint/data.pkl", "w") as handle:
            block = b"0" * 1_000_000
            for _ in range(11):
                handle.write(block)
    return path


def _status(
    tmp_path: Path,
    state: TabPFNModelState,
    *,
    cached: bool,
) -> TabPFNModelStatus:
    cache_path = tmp_path / "cache" / TABPFN_CHECKPOINT_FILENAME
    return TabPFNModelStatus(
        state=state,
        model_version="2.5",
        package_version="8.0.8",
        checkpoint_filename=TABPFN_CHECKPOINT_FILENAME,
        cache_dir=cache_path.parent,
        cache_path=cache_path,
        cache_checkpoint_exists=cached,
        active_checkpoint_path=cache_path if cached else None,
        active_checkpoint_source="user_cache" if cached else "unavailable",
        license_name="TabPFN-2.5 License v1.1",
        license_url=manager.TABPFN_LICENSE_URL,
        download_authentication_required=True,
        last_download_error="",
    )


@pytest.fixture(autouse=True)
def _reset_download_state():
    manager.reset_tabpfn_download_state()
    yield
    manager.reset_tabpfn_download_state()


def test_tabpfn_status_resolver_reports_cache_and_missing_without_bundle_fallback(
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    _checkpoint(tmp_path / "bundle" / TABPFN_CHECKPOINT_FILENAME)

    missing_status = manager.get_tabpfn_model_status(cache_dir=cache_dir)
    assert missing_status.state == TabPFNModelState.MISSING
    assert missing_status.active_checkpoint_source == "unavailable"
    assert missing_status.active_checkpoint_path is None

    _checkpoint(cache_dir / TABPFN_CHECKPOINT_FILENAME)
    cache_status = manager.get_tabpfn_model_status(cache_dir=cache_dir)
    assert cache_status.state == TabPFNModelState.AVAILABLE_IN_USER_CACHE
    assert cache_status.active_checkpoint_source == "user_cache"
    assert cache_status.active_checkpoint_path == cache_dir / TABPFN_CHECKPOINT_FILENAME

    (cache_dir / TABPFN_CHECKPOINT_FILENAME).unlink()
    missing_status = manager.get_tabpfn_model_status(cache_dir=cache_dir)
    assert missing_status.state == TabPFNModelState.MISSING
    assert missing_status.active_checkpoint_path is None


def test_tabpfn_official_download_uses_v25_api_and_refreshes_status(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(manager, "_ensure_tabpfn_license_for_gui", lambda **kwargs: None)

    def fake_download(to, *, version, which, model_name):
        calls.append((to, version.value, which, model_name))
        _checkpoint(to)
        return "ok"

    monkeypatch.setattr("tabpfn.model_loading.download_model", fake_download)
    result = manager.download_tabpfn_model(
        cache_dir=tmp_path / "empty-cache",
        run_smoke_test=False,
    )

    assert calls and calls[0][1:] == (
        "v2.5",
        "classifier",
        TABPFN_CHECKPOINT_FILENAME,
    )
    assert result.valid
    assert result.path == tmp_path / "empty-cache" / TABPFN_CHECKPOINT_FILENAME
    assert manager.get_tabpfn_model_status(cache_dir=tmp_path / "empty-cache").state == TabPFNModelState.AVAILABLE_IN_USER_CACHE


def test_tabpfn_download_failure_is_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "_ensure_tabpfn_license_for_gui", lambda **kwargs: None)
    monkeypatch.setattr(
        "tabpfn.model_loading.download_model",
        lambda *args, **kwargs: [PermissionError("license access denied")],
    )

    with pytest.raises(RuntimeError, match="license access denied"):
        manager.download_tabpfn_model(
            cache_dir=tmp_path / "empty-cache",
            run_smoke_test=False,
        )

    status = manager.get_tabpfn_model_status(cache_dir=tmp_path / "empty-cache")
    assert status.state == TabPFNModelState.DOWNLOAD_FAILED
    assert "license access denied" in status.last_download_error


def test_tabpfn_gui_auth_uses_official_callback_and_token_cache(monkeypatch):
    from tabpfn import browser_auth

    ensure_calls = []
    saved_tokens = []

    def fake_ensure(*, hf_repo_id):
        ensure_calls.append(hf_repo_id)
        if len(ensure_calls) == 1:
            raise RuntimeError("interactive authentication required")
        return True

    class FakeServer:
        server_address = ("localhost", 43123)

        def server_close(self):
            return None

    def fake_create(gui_url, event, received_token):
        received_token[0] = "test-token-never-logged"
        event.set()
        return FakeServer(), 43123

    monkeypatch.setattr(browser_auth, "ensure_license_accepted", fake_ensure)
    monkeypatch.setattr(browser_auth, "get_cached_token", lambda: None)
    monkeypatch.setattr(
        browser_auth,
        "_create_callback_server",
        fake_create,
    )
    monkeypatch.setattr(browser_auth, "_serve_until_event", lambda server, event: None)
    monkeypatch.setattr(browser_auth, "save_token", saved_tokens.append)
    opened = []
    monkeypatch.setattr(manager.webbrowser, "open", lambda url: opened.append(url) or True)

    manager._ensure_tabpfn_license_for_gui(timeout_seconds=1)

    assert ensure_calls == ["tabpfn_2_5", "tabpfn_2_5"]
    assert saved_tokens == ["test-token-never-logged"]
    assert opened and "hf_repo_id=tabpfn_2_5" in opened[0]


def test_startup_status_check_never_downloads_and_only_offers_when_cache_missing(
    tmp_path,
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    missing = _status(
        tmp_path,
        TabPFNModelState.MISSING,
        cached=False,
    )
    monkeypatch.setattr("app.gui.main_window.get_tabpfn_model_status", lambda: missing)
    window = MainWindow()
    shown = []
    downloads = []
    monkeypatch.setattr(window, "_show_tabpfn_model_dialog", lambda *, startup: shown.append(startup))
    monkeypatch.setattr(window, "_start_tabpfn_model_action", lambda action: downloads.append(action))

    window.check_startup_tabpfn_model_status()
    assert shown == [True]
    assert downloads == []

    cached = _status(
        tmp_path,
        TabPFNModelState.AVAILABLE_IN_USER_CACHE,
        cached=True,
    )
    monkeypatch.setattr("app.gui.main_window.get_tabpfn_model_status", lambda: cached)
    shown.clear()
    window.check_startup_tabpfn_model_status()
    assert shown == []
    window.close()
    assert app is not None


def test_help_menu_contains_tabpfn_status_and_opens_dialog(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    status = _status(tmp_path, TabPFNModelState.MISSING, cached=False)
    monkeypatch.setattr("app.gui.main_window.get_tabpfn_model_status", lambda: status)
    window = MainWindow()
    shown = []
    monkeypatch.setattr(window, "_show_tabpfn_model_dialog", lambda *, startup: shown.append(startup))

    assert window.tabpfn_model_action.text() == "TabPFN Model Status"
    window.show_tabpfn_model_status()
    assert shown == [False]
    window.close()
    assert app is not None


def test_tabpfn_dialog_actions_follow_cache_status(tmp_path):
    app = QApplication.instance() or QApplication([])
    missing = TabPFNModelDialog(
        _status(tmp_path, TabPFNModelState.MISSING, cached=False)
    )
    assert missing.download_button.text() == "Download / Set Up"
    assert missing.download_button.property("avistaButtonRole") == "primary"
    assert missing.verify_button.property("avistaButtonRole") == "secondary"
    assert missing.verify_button.isHidden()
    assert missing.cache_button.isHidden()

    installed = TabPFNModelDialog(
        _status(
            tmp_path,
            TabPFNModelState.AVAILABLE_IN_USER_CACHE,
            cached=True,
        )
    )
    assert installed.download_button.text() == "Re-download"
    assert installed.download_button.property("avistaButtonRole") == "secondary"
    assert installed.verify_button.property("avistaButtonRole") == "primary"
    assert installed.cache_button.property("avistaButtonRole") == "secondary"
    assert installed.license_button.property("avistaButtonRole") == "secondary"
    assert installed.close_button.property("avistaButtonRole") == "secondary"
    assert not installed.verify_button.isHidden()
    assert not installed.cache_button.isHidden()
    missing.close()
    installed.close()
    assert app is not None


def test_tabpfn_dialog_button_roles_render_in_light_and_dark_themes(tmp_path):
    from app.gui.theme import apply_theme

    app = QApplication.instance() or QApplication([])
    dialog = TabPFNModelDialog(
        _status(
            tmp_path,
            TabPFNModelState.AVAILABLE_IN_USER_CACHE,
            cached=True,
        )
    )

    for theme_name in ("light", "dark"):
        tokens = apply_theme(app, theme_name)
        dialog.show()
        app.processEvents()
        assert dialog.verify_button.property("avistaButtonRole") == "primary"
        assert dialog.download_button.property("avistaButtonRole") == "secondary"
        assert tokens.primary in app.styleSheet()
        assert tokens.accent in app.styleSheet()
        assert not dialog.grab().isNull()

    dialog.close()
    assert app is not None


def test_tabpfn_worker_success_and_failure_signals(tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path / "cache" / TABPFN_CHECKPOINT_FILENAME)
    verification = TabPFNVerificationResult(
        checkpoint,
        True,
        checkpoint.stat().st_size,
        "abc123",
        True,
        True,
        0,
        "",
    )
    status = _status(
        tmp_path,
        TabPFNModelState.AVAILABLE_IN_USER_CACHE,
        cached=True,
    )
    monkeypatch.setattr("app.gui.workers.get_tabpfn_model_status", lambda: status)
    monkeypatch.setattr("app.gui.workers.download_tabpfn_model", lambda **kwargs: verification)
    completed = []
    worker = TabPFNModelWorker(action="download")
    worker.finished.connect(lambda result, refreshed: completed.append((result, refreshed)))
    worker.run()
    assert completed == [(verification, status)]

    monkeypatch.setattr(
        "app.gui.workers.download_tabpfn_model",
        lambda **kwargs: (_ for _ in ()).throw(ConnectionError("offline")),
    )
    failures = []
    failed_worker = TabPFNModelWorker(action="download")
    failed_worker.failed.connect(lambda message, refreshed: failures.append((message, refreshed)))
    failed_worker.run()
    assert failures and failures[0][0] == "offline"
    assert failures[0][1] == status
