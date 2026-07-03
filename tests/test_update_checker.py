import hashlib

import pytest

from app.core.update_checker import (
    UPDATE_METADATA_URL,
    UpdateMetadata,
    calculate_sha256,
    check_for_updates,
    compare_semantic_versions,
    parse_update_metadata,
    verify_sha256,
)
from app.core.user_settings import UserSettings


def _metadata(version="1.0.1", *, mandatory=False):
    return UpdateMetadata(
        latest_version=version,
        release_date="2026-07-03",
        release_notes=["Added automatic update checker"],
        installer_url="https://github.com/Xatta-Trone/avista-dl-desktop/releases/download/v1.0.1/AVISTA_Setup.exe",
        sha256="",
        mandatory=mandatory,
    )


def test_semantic_version_comparison():
    assert compare_semantic_versions("1.0.0", "1.0.1") < 0
    assert compare_semantic_versions("v1.2.0", "1.2") == 0
    assert compare_semantic_versions("1.10.0", "1.2.9") > 0


def test_updates_json_parsing_requires_https_installer():
    metadata = parse_update_metadata(
        {
            "latest_version": "1.0.1",
            "release_date": "2026-07-03",
            "release_notes": ["Fixed Report page figure scaling"],
            "installer_url": "https://example.com/AVISTA_Setup.exe",
            "sha256": "",
            "mandatory": False,
        }
    )

    assert metadata.latest_version == "1.0.1"
    assert metadata.installer_url == "https://example.com/AVISTA_Setup.exe"
    assert metadata.release_notes == ["Fixed Report page figure scaling"]

    with pytest.raises(ValueError, match="HTTPS"):
        parse_update_metadata(
            {
                "latest_version": "1.0.1",
                "installer_url": "http://example.com/AVISTA_Setup.exe",
            }
        )


def test_update_available_and_installer_url_parsed(monkeypatch):
    saved = []
    monkeypatch.setattr(
        "app.core.update_checker.save_user_settings",
        lambda settings: saved.append(settings),
    )
    result = check_for_updates(
        current_version="1.0.0",
        settings=UserSettings(),
        fetcher=lambda url: _metadata("1.0.1"),
    )

    assert result.update_available is True
    assert result.metadata is not None
    assert result.metadata.installer_url.endswith("AVISTA_Setup.exe")
    assert saved


def test_skip_version_suppresses_automatic_popup_result(monkeypatch):
    monkeypatch.setattr("app.core.update_checker.save_user_settings", lambda settings: None)
    settings = UserSettings(skipped_update_version="1.0.1")

    result = check_for_updates(
        current_version="1.0.0",
        settings=settings,
        manual=False,
        fetcher=lambda url: _metadata("1.0.1"),
    )

    assert result.update_available is True
    assert result.skipped is True

    manual = check_for_updates(
        current_version="1.0.0",
        settings=settings,
        manual=True,
        fetcher=lambda url: _metadata("1.0.1"),
    )
    assert manual.update_available is True
    assert manual.skipped is False


def test_mandatory_update_ignores_skipped_version(monkeypatch):
    monkeypatch.setattr("app.core.update_checker.save_user_settings", lambda settings: None)
    result = check_for_updates(
        current_version="1.0.0",
        settings=UserSettings(skipped_update_version="1.0.1"),
        manual=False,
        fetcher=lambda url: _metadata("1.0.1", mandatory=True),
    )

    assert result.update_available is True
    assert result.skipped is False


def test_sha256_verification(tmp_path):
    installer = tmp_path / "AVISTA_Setup.exe"
    installer.write_bytes(b"avista installer")
    expected = hashlib.sha256(b"avista installer").hexdigest()

    assert calculate_sha256(installer) == expected
    verify_sha256(installer, expected)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_sha256(installer, "0" * 64)


def test_default_update_metadata_url_is_github_https():
    assert UPDATE_METADATA_URL == (
        "https://raw.githubusercontent.com/Xatta-Trone/avista-dl-desktop/main/updates.json"
    )
