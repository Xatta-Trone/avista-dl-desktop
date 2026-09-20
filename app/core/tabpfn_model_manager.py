"""Status, acquisition, and verification for the TabPFN 2.5 checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.parse
import webbrowser
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from app.core.user_settings import app_settings_dir
from app.utils.resources import TABPFN_CHECKPOINT_FILENAME


TABPFN_MODEL_VERSION = "2.5"
TABPFN_LICENSE_NAME = "TabPFN-2.5 License v1.1"
TABPFN_LICENSE_URL = "https://huggingface.co/Prior-Labs/tabpfn_2_5"
MINIMUM_PLAUSIBLE_CHECKPOINT_BYTES = 10_000_000


class TabPFNModelState(str, Enum):
    """Availability state for the supported TabPFN checkpoint."""

    AVAILABLE_IN_USER_CACHE = "AVAILABLE_IN_USER_CACHE"
    MISSING = "MISSING"
    DOWNLOAD_IN_PROGRESS = "DOWNLOAD_IN_PROGRESS"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"


@dataclass(frozen=True)
class TabPFNModelStatus:
    """Resolved checkpoint state without credentials or other secrets."""

    state: TabPFNModelState
    model_version: str
    package_version: str
    checkpoint_filename: str
    cache_dir: Path
    cache_path: Path
    cache_checkpoint_exists: bool
    active_checkpoint_path: Path | None
    active_checkpoint_source: str
    license_name: str
    license_url: str
    download_authentication_required: bool
    last_download_error: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        for key in (
            "cache_dir",
            "cache_path",
            "active_checkpoint_path",
        ):
            value = data[key]
            data[key] = str(value) if value is not None else None
        return data


@dataclass(frozen=True)
class TabPFNVerificationResult:
    """Result of a structural or explicit model smoke verification."""

    path: Path
    valid: bool
    size_bytes: int
    sha256: str
    initialized: bool
    smoke_tested: bool
    prediction: int | None
    error: str


_download_in_progress = False
_last_download_error = ""


def get_tabpfn_cache_dir() -> Path:
    """Return the cache directory used by tabpfn 8.0.8 without importing torch."""

    configured = os.environ.get("TABPFN_MODEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        if appdata:
            return (Path(appdata) / "tabpfn").resolve()
    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Caches" / "tabpfn").resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return (Path(xdg_cache) / "tabpfn").resolve()
    return (Path.home() / ".cache" / "tabpfn").resolve()


def get_tabpfn_cache_path(cache_dir: str | Path | None = None) -> Path:
    directory = Path(cache_dir).resolve() if cache_dir is not None else get_tabpfn_cache_dir()
    return directory / TABPFN_CHECKPOINT_FILENAME


def _package_version() -> str:
    try:
        return version("tabpfn")
    except PackageNotFoundError:
        return "not installed"


def _plausible_checkpoint(path: Path) -> bool:
    try:
        return (
            path.is_file()
            and path.name == TABPFN_CHECKPOINT_FILENAME
            and path.stat().st_size >= MINIMUM_PLAUSIBLE_CHECKPOINT_BYTES
            and zipfile.is_zipfile(path)
        )
    except OSError:
        return False


def get_tabpfn_model_status(
    *,
    cache_dir: str | Path | None = None,
) -> TabPFNModelStatus:
    """Resolve availability exclusively from the official TabPFN user cache."""

    cache_path = get_tabpfn_cache_path(cache_dir)
    cache_exists = _plausible_checkpoint(cache_path)
    if _download_in_progress:
        state = TabPFNModelState.DOWNLOAD_IN_PROGRESS
    elif cache_exists:
        state = TabPFNModelState.AVAILABLE_IN_USER_CACHE
    elif _last_download_error:
        state = TabPFNModelState.DOWNLOAD_FAILED
    else:
        state = TabPFNModelState.MISSING

    if cache_exists:
        active_path = cache_path.resolve()
        active_source = "user_cache"
    else:
        active_path = None
        active_source = "unavailable"

    status = TabPFNModelStatus(
        state=state,
        model_version=TABPFN_MODEL_VERSION,
        package_version=_package_version(),
        checkpoint_filename=TABPFN_CHECKPOINT_FILENAME,
        cache_dir=cache_path.parent,
        cache_path=cache_path,
        cache_checkpoint_exists=cache_exists,
        active_checkpoint_path=active_path,
        active_checkpoint_source=active_source,
        license_name=TABPFN_LICENSE_NAME,
        license_url=TABPFN_LICENSE_URL,
        download_authentication_required=True,
        last_download_error=_last_download_error,
    )
    log_tabpfn_model_event(
        "Cache status checked",
        state=state.value,
        cache_path=cache_path,
        active_source=active_source,
    )
    return status


def resolve_tabpfn_model_checkpoint() -> tuple[Path, str]:
    """Return the verified official user-cache checkpoint."""

    status = get_tabpfn_model_status()
    if status.active_checkpoint_path is None:
        raise FileNotFoundError(
            "TabPFN 2.5 checkpoint is unavailable. Expected official user-cache "
            f"path: {status.cache_path}. Use Help > TabPFN Model Status to set it up."
        )
    log_tabpfn_model_event(
        "Checkpoint source selected",
        source=status.active_checkpoint_source,
        path=status.active_checkpoint_path,
    )
    return status.active_checkpoint_path, status.active_checkpoint_source


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_tabpfn_checkpoint(
    path: str | Path,
    *,
    run_smoke_test: bool = False,
) -> TabPFNVerificationResult:
    """Verify structure and optionally perform a tiny real fit/predict."""

    checkpoint = Path(path).resolve()
    size = checkpoint.stat().st_size if checkpoint.is_file() else 0
    sha256 = checkpoint_sha256(checkpoint) if _plausible_checkpoint(checkpoint) else ""
    if not _plausible_checkpoint(checkpoint):
        return TabPFNVerificationResult(
            checkpoint, False, size, sha256, False, False, None,
            "Checkpoint is missing, unexpectedly named, or implausibly small.",
        )

    initialized = False
    prediction: int | None = None
    try:
        from tabpfn import TabPFNClassifier

        model = TabPFNClassifier(
            n_estimators=1,
            model_path=str(checkpoint),
            device="cpu",
            random_state=42,
        )
        initialized = True
        if run_smoke_test:
            import numpy as np
            import pandas as pd

            X = pd.DataFrame(
                {"number": [0.0, 0.2, 0.8, 1.0, 0.1, 0.9], "group": ["a", "a", "b", "b", "a", "b"]}
            )
            y = np.array([0, 0, 1, 1, 0, 1])
            model.categorical_features_indices = [1]
            model.fit(X, y)
            prediction = int(model.predict(X.iloc[[0]])[0])
    except Exception as exc:
        log_tabpfn_model_event("Verification failed", path=checkpoint, error=repr(exc))
        return TabPFNVerificationResult(
            checkpoint, False, size, sha256, initialized, run_smoke_test, prediction, str(exc)
        )

    log_tabpfn_model_event(
        "Verification succeeded",
        path=checkpoint,
        smoke_tested=run_smoke_test,
    )
    return TabPFNVerificationResult(
        checkpoint, True, size, sha256, initialized, run_smoke_test, prediction, ""
    )


def download_tabpfn_model(
    *,
    cache_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    run_smoke_test: bool = True,
) -> TabPFNVerificationResult:
    """Download through tabpfn's official gated v2.5 acquisition mechanism."""

    global _download_in_progress, _last_download_error
    destination = get_tabpfn_cache_path(cache_dir)
    _download_in_progress = True
    _last_download_error = ""
    log_tabpfn_model_event("Download requested", destination=destination)

    def progress(message: str) -> None:
        log_tabpfn_model_event(message, destination=destination)
        if progress_callback is not None:
            progress_callback(message)

    try:
        progress("Checking license and access...")
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import download_model

        _ensure_tabpfn_license_for_gui(progress_callback=progress)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="tabpfn-v2.5-download-",
            dir=destination.parent,
        ) as temporary_directory:
            temporary_path = Path(temporary_directory) / TABPFN_CHECKPOINT_FILENAME
            progress("Authenticating and downloading TabPFN 2.5...")
            result = download_model(
                temporary_path,
                version=ModelVersion.V2_5,
                which="classifier",
                model_name=TABPFN_CHECKPOINT_FILENAME,
            )
            if result != "ok":
                details = "; ".join(f"{type(error).__name__}: {error}" for error in result)
                raise RuntimeError(details or "Official TabPFN downloader returned no checkpoint.")
            progress("Verifying downloaded checkpoint...")
            verification = verify_tabpfn_checkpoint(
                temporary_path,
                run_smoke_test=run_smoke_test,
            )
            if not verification.valid:
                raise RuntimeError(verification.error)
            shutil.move(str(temporary_path), str(destination))

        final_result = verify_tabpfn_checkpoint(
            destination,
            run_smoke_test=False,
        )
        progress("TabPFN 2.5 is ready.")
        _last_download_error = ""
        return TabPFNVerificationResult(
            final_result.path,
            final_result.valid,
            final_result.size_bytes,
            final_result.sha256,
            verification.initialized,
            verification.smoke_tested,
            verification.prediction,
            final_result.error,
        )
    except Exception as exc:
        _last_download_error = str(exc)
        log_tabpfn_model_event("Download failed", destination=destination, error=repr(exc))
        raise RuntimeError(str(exc)) from exc
    finally:
        _download_in_progress = False


def _ensure_tabpfn_license_for_gui(
    *,
    progress_callback: Callable[[str], None] | None = None,
    timeout_seconds: int = 600,
) -> None:
    """Use tabpfn 8.0.8's official browser callback from a GUI process."""

    from tabpfn import browser_auth

    try:
        browser_auth.ensure_license_accepted(hf_repo_id="tabpfn_2_5")
        return
    except Exception as initial_error:
        if browser_auth.get_cached_token() is not None:
            raise initial_error

    required_helpers = (
        "_create_callback_server",
        "_serve_until_event",
        "save_token",
        "ensure_license_accepted",
    )
    if any(not hasattr(browser_auth, name) for name in required_helpers):
        raise RuntimeError(
            "Installed TabPFN does not expose its expected v8.0.8 browser-auth "
            "helpers. Update AVISTA's pinned integration before downloading."
        )

    if progress_callback is not None:
        progress_callback("Opening Prior Labs license and authentication in your browser...")
    auth_event = threading.Event()
    received_token: list[str | None] = [None]
    gui_url = "https://ux.priorlabs.ai"
    httpd, port = browser_auth._create_callback_server(  # noqa: SLF001
        gui_url,
        auth_event,
        received_token,
    )
    server_thread = threading.Thread(
        target=browser_auth._serve_until_event,  # noqa: SLF001
        args=(httpd, auth_event),
        daemon=True,
    )
    server_thread.start()
    callback_url = f"http://localhost:{port}"
    login_url = (
        f"{gui_url}/login?callback={urllib.parse.quote(callback_url, safe=':/')}"
        "&hf_repo_id=tabpfn_2_5"
    )
    try:
        if not webbrowser.open(login_url):
            raise RuntimeError(
                "Could not open the Prior Labs authentication page in the default browser."
            )
        if not auth_event.wait(timeout_seconds) or not received_token[0]:
            raise TimeoutError(
                "Prior Labs authentication did not complete. Review and accept the "
                "TabPFN 2.5 license in the browser, then retry."
            )
    finally:
        auth_event.set()
        httpd.server_close()
    browser_auth.save_token(received_token[0])
    browser_auth.ensure_license_accepted(hf_repo_id="tabpfn_2_5")


def reset_tabpfn_download_state() -> None:
    """Clear transient process state; intended for retry and focused tests."""

    global _download_in_progress, _last_download_error
    _download_in_progress = False
    _last_download_error = ""


def log_tabpfn_model_event(event: str, **details: Any) -> None:
    """Append a credential-free TabPFN model-management event."""

    log_path = app_settings_dir() / "logs" / "tabpfn_model.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        safe_details = {
            key: str(value)
            for key, value in details.items()
            if "token" not in key.casefold() and "credential" not in key.casefold()
        }
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **safe_details,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass
