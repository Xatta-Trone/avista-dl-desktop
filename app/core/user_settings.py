"""Application-level user settings for AVISTA."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class UserSettings:
    """Small app-level settings file stored outside project files."""

    skipped_update_version: str = ""
    last_update_check: str = ""
    auto_check_updates: bool = True
    theme_name: str = "light"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserSettings":
        return cls(
            skipped_update_version=str(data.get("skipped_update_version") or ""),
            last_update_check=str(data.get("last_update_check") or ""),
            auto_check_updates=bool(data.get("auto_check_updates", True)),
            theme_name=str(data.get("theme_name") or "light").strip().lower()
            or "light",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def app_settings_dir() -> Path:
    """Return the per-user AVISTA settings directory."""

    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "AVISTA"
    return Path.home() / ".avista"


def settings_path() -> Path:
    """Return the JSON settings path."""

    return app_settings_dir() / "settings.json"


def load_user_settings(path: str | Path | None = None) -> UserSettings:
    """Load user settings, returning defaults when the file is absent or invalid."""

    target = Path(path) if path is not None else settings_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return UserSettings()
    if not isinstance(data, dict):
        return UserSettings()
    return UserSettings.from_dict(data)


def save_user_settings(
    settings: UserSettings,
    path: str | Path | None = None,
) -> Path:
    """Persist user settings as app-level JSON."""

    target = Path(path) if path is not None else settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
    return target


def mark_update_check_now(settings: UserSettings) -> UserSettings:
    """Update the last-check timestamp in-place and return the settings."""

    settings.last_update_check = datetime.now(timezone.utc).isoformat()
    return settings
