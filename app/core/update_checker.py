"""Update metadata checks, downloads, and logging for AVISTA."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.__version__ import __version__
from app.core.user_settings import UserSettings, mark_update_check_now, save_user_settings


UPDATE_METADATA_URL = (
    "https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json"
)


@dataclass(frozen=True)
class UpdateMetadata:
    latest_version: str
    release_date: str
    release_notes: list[str]
    installer_url: str
    sha256: str = ""
    mandatory: bool = False


@dataclass(frozen=True)
class UpdateCheckResult:
    current_version: str
    metadata: UpdateMetadata | None
    update_available: bool
    skipped: bool = False
    error: str = ""


def compare_semantic_versions(current_version: str, latest_version: str) -> int:
    """Compare semantic versions, returning -1, 0, or 1."""

    current = _semantic_parts(current_version)
    latest = _semantic_parts(latest_version)
    width = max(len(current), len(latest), 3)
    current += (0,) * (width - len(current))
    latest += (0,) * (width - len(latest))
    if current < latest:
        return -1
    if current > latest:
        return 1
    return 0


def parse_update_metadata(data: dict[str, Any]) -> UpdateMetadata:
    """Validate and normalize update metadata."""

    latest_version = str(data.get("latest_version") or "").strip()
    installer_url = str(data.get("installer_url") or "").strip()
    if not latest_version:
        raise ValueError("Update metadata is missing latest_version.")
    if not _is_https_url(installer_url):
        raise ValueError("Update installer_url must use HTTPS.")
    notes = data.get("release_notes") or []
    if isinstance(notes, str):
        notes = [notes]
    if not isinstance(notes, list):
        raise ValueError("Update release_notes must be a list or string.")
    return UpdateMetadata(
        latest_version=latest_version,
        release_date=str(data.get("release_date") or "").strip(),
        release_notes=[str(note) for note in notes],
        installer_url=installer_url,
        sha256=str(data.get("sha256") or "").strip().lower(),
        mandatory=bool(data.get("mandatory", False)),
    )


def fetch_update_metadata(
    metadata_url: str = UPDATE_METADATA_URL,
    *,
    timeout: float = 10.0,
) -> UpdateMetadata:
    """Fetch GitHub-hosted update metadata."""

    if not _is_https_url(metadata_url):
        raise ValueError("Update metadata URL must use HTTPS.")
    request = Request(metadata_url, headers={"User-Agent": f"AVISTA/{__version__}"})
    with urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Update metadata root must be a JSON object.")
    return parse_update_metadata(data)


def check_for_updates(
    *,
    current_version: str = __version__,
    metadata_url: str = UPDATE_METADATA_URL,
    settings: UserSettings | None = None,
    manual: bool = False,
    fetcher: Callable[[str], UpdateMetadata] | None = None,
) -> UpdateCheckResult:
    """Check whether a newer AVISTA release is available."""

    settings = settings or UserSettings()
    try:
        metadata = fetcher(metadata_url) if fetcher else fetch_update_metadata(metadata_url)
        available = compare_semantic_versions(current_version, metadata.latest_version) < 0
        skipped = (
            available
            and not manual
            and not metadata.mandatory
            and bool(settings.skipped_update_version)
            and settings.skipped_update_version == metadata.latest_version
        )
        result = UpdateCheckResult(
            current_version=current_version,
            metadata=metadata,
            update_available=available,
            skipped=skipped,
        )
        log_update_check(result)
        if not available:
            log_update_message("AVISTA is up to date.")
    except (OSError, URLError, ValueError, json.JSONDecodeError) as exc:
        result = UpdateCheckResult(
            current_version=current_version,
            metadata=None,
            update_available=False,
            error=str(exc),
        )
        log_update_check(result)
    mark_update_check_now(settings)
    save_user_settings(settings)
    return result


def download_installer(
    metadata: UpdateMetadata,
    *,
    destination_dir: str | Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    timeout: float = 30.0,
) -> Path:
    """Download the update installer and verify SHA256 when supplied."""

    if not _is_https_url(metadata.installer_url):
        raise ValueError("Installer URL must use HTTPS.")
    destination = Path(destination_dir) if destination_dir else Path(tempfile.gettempdir())
    destination.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(metadata.installer_url).path).name or "AVISTA_Setup.exe"
    output_path = destination / filename
    log_update_message(f"Download started: {metadata.installer_url}")
    request = Request(metadata.installer_url, headers={"User-Agent": f"AVISTA/{__version__}"})
    with urlopen(request, timeout=timeout) as response, output_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total)
    if metadata.sha256:
        verify_sha256(output_path, metadata.sha256)
    log_update_message(f"Download completed: {output_path}")
    return output_path


def verify_sha256(path: str | Path, expected_sha256: str) -> None:
    """Raise ValueError when the file hash does not match."""

    expected = expected_sha256.strip().lower()
    actual = calculate_sha256(path)
    if actual != expected:
        message = f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        log_update_message(message)
        raise ValueError(message)


def calculate_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def app_log_dir() -> Path:
    """Return the app-level log directory."""

    if getattr(sys, "frozen", False):
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            return Path(local_appdata) / "AVISTA" / "logs"
    return Path.cwd() / "logs"


def update_log_path() -> Path:
    return app_log_dir() / "update.log"


def log_update_check(result: UpdateCheckResult) -> None:
    metadata = result.metadata
    latest = metadata.latest_version if metadata else "unknown"
    lines = [
        f"check_timestamp={datetime.now(timezone.utc).isoformat()}",
        f"current_version={result.current_version}",
        f"latest_version={latest}",
        f"update_available={'yes' if result.update_available else 'no'}",
        f"skipped={'yes' if result.skipped else 'no'}",
    ]
    if result.error:
        lines.append(f"error={result.error}")
    log_update_message("; ".join(lines))


def log_update_message(message: str) -> None:
    path = update_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def _semantic_parts(version: str) -> tuple[int, ...]:
    cleaned = version.strip().lower()
    if cleaned.startswith("v"):
        cleaned = cleaned[1:]
    parts: list[int] = []
    for piece in cleaned.split("."):
        digits = ""
        for character in piece:
            if character.isdigit():
                digits += character
            else:
                break
        parts.append(int(digits or "0"))
    return tuple(parts)


def _is_https_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)
