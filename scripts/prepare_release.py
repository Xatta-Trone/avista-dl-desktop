"""Prepare and verify synchronized AVISTA release metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
GITHUB_RELEASE_BASE = (
    "https://github.com/Xatta-Trone/avista-dl-desktop/releases/download"
)


@dataclass(frozen=True)
class ReleaseMetadata:
    version: str
    release_date: str

    @property
    def iso_date(self) -> str:
        return datetime.strptime(
            self.release_date,
            "%B %d, %Y",
        ).strftime("%Y-%m-%d")

    @property
    def installer_url(self) -> str:
        return (
            f"{GITHUB_RELEASE_BASE}/v{self.version}/AVISTA_Setup.exe"
        )


def read_central_metadata(root: Path = PROJECT_ROOT) -> ReleaseMetadata:
    """Read canonical release values without importing application modules."""

    text = (root / "app" / "__version__.py").read_text(encoding="utf-8")
    version_match = re.search(
        r'(?m)^__version__\s*=\s*"([^"]+)"',
        text,
    )
    date_match = re.search(
        r'(?m)^RELEASE_DATE\s*=\s*"([^"]+)"',
        text,
    )
    if not version_match or not date_match:
        raise ValueError(
            "app/__version__.py must define __version__ and RELEASE_DATE."
        )
    return validate_metadata(
        ReleaseMetadata(
            version=version_match.group(1),
            release_date=date_match.group(1),
        )
    )


def validate_metadata(metadata: ReleaseMetadata) -> ReleaseMetadata:
    """Validate the supported semantic version and display-date formats."""

    if not VERSION_PATTERN.fullmatch(metadata.version):
        raise ValueError(
            "Version must use X.Y.Z numeric format, for example 1.0.6."
        )
    try:
        datetime.strptime(metadata.release_date, "%B %d, %Y")
    except ValueError as exc:
        raise ValueError(
            "Release date must use 'Month D, YYYY', for example "
            "'July 25, 2026'."
        ) from exc
    return metadata


def prepare_release(
    root: Path,
    metadata: ReleaseMetadata,
    release_notes: Sequence[str] | None = None,
    *,
    sha256: str | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Synchronize canonical and derived release files."""

    metadata = validate_metadata(metadata)
    updates_path = root / "updates.json"
    existing_updates = json.loads(updates_path.read_text(encoding="utf-8"))
    notes = _resolve_release_notes(
        metadata,
        existing_updates,
        release_notes,
    )

    rendered = {
        root / "app" / "__version__.py": _render_version_source(
            root,
            metadata,
        ),
        updates_path: _render_update_feed(
            existing_updates,
            metadata,
            notes,
            sha256,
        ),
        root / "README.md": _render_readme(root, metadata),
        root / "PROJECT_STATUS.md": _render_project_status(
            root,
            metadata,
        ),
        root / "CHANGELOG.md": _render_changelog(
            root,
            metadata,
            notes,
        ),
    }
    changed = [
        path
        for path, content in rendered.items()
        if path.read_text(encoding="utf-8") != content
    ]
    if not dry_run:
        for path in changed:
            path.write_text(rendered[path], encoding="utf-8")
    return changed


def check_release_metadata(
    root: Path = PROJECT_ROOT,
    expected_tag: str | None = None,
) -> list[str]:
    """Return actionable synchronization problems without changing files."""

    metadata = read_central_metadata(root)
    problems: list[str] = []
    updates = json.loads((root / "updates.json").read_text(encoding="utf-8"))
    if updates.get("latest_version") != metadata.version:
        problems.append(
            "updates.json latest_version does not match app/__version__.py"
        )
    if updates.get("release_date") != metadata.release_date:
        problems.append(
            "updates.json release_date does not match app/__version__.py"
        )
    if updates.get("installer_url") != metadata.installer_url:
        problems.append(
            "updates.json installer_url does not match the central version"
        )
    if expected_tag and expected_tag != f"v{metadata.version}":
        problems.append(
            f"release tag {expected_tag!r} does not match central version "
            f"'v{metadata.version}'"
        )
    sha256 = str(updates.get("sha256", "") or "")
    if sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        problems.append(
            "updates.json sha256 must be empty or a 64-character hex digest"
        )
    readme = (root / "README.md").read_text(encoding="utf-8")
    if (
        f"Version\n{metadata.version}**, released "
        f"**{metadata.release_date}**" not in readme
    ):
        problems.append("README release banner is not synchronized")
    status = (root / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    if (
        "current AVISTA application and update-feed version is "
        f"`{metadata.version}`" not in status
    ):
        problems.append("PROJECT_STATUS current version is not synchronized")
    if f"release date centralized as {metadata.release_date}" not in status:
        problems.append("PROJECT_STATUS release date is not synchronized")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {metadata.version} - {metadata.iso_date}" not in changelog:
        problems.append("CHANGELOG has no heading for the central release")
    if not isinstance(updates.get("release_notes"), list) or not updates.get(
        "release_notes"
    ):
        problems.append("updates.json release_notes must be a non-empty list")
    return problems


def _resolve_release_notes(
    metadata: ReleaseMetadata,
    updates: dict,
    release_notes: Sequence[str] | None,
) -> list[str]:
    cleaned = [
        str(note).strip()
        for note in (release_notes or [])
        if str(note).strip()
    ]
    if cleaned:
        return cleaned
    if updates.get("latest_version") == metadata.version:
        existing = updates.get("release_notes", [])
        if isinstance(existing, list) and existing:
            return [str(note).strip() for note in existing if str(note).strip()]
    raise ValueError(
        "Advancing the release requires at least one --note so old release "
        "notes cannot be published under a new version."
    )


def _render_version_source(
    root: Path,
    metadata: ReleaseMetadata,
) -> str:
    path = root / "app" / "__version__.py"
    text = path.read_text(encoding="utf-8")
    text, version_count = re.subn(
        r'(?m)^__version__\s*=\s*"[^"]+"',
        f'__version__ = "{metadata.version}"',
        text,
        count=1,
    )
    text, date_count = re.subn(
        r'(?m)^RELEASE_DATE\s*=\s*"[^"]+"',
        f'RELEASE_DATE = "{metadata.release_date}"',
        text,
        count=1,
    )
    if version_count != 1 or date_count != 1:
        raise ValueError(
            "Could not update __version__ and RELEASE_DATE in "
            "app/__version__.py."
        )
    return text


def _render_update_feed(
    updates: dict,
    metadata: ReleaseMetadata,
    release_notes: Sequence[str],
    sha256: str | None,
) -> str:
    synchronized = dict(updates)
    previous_version = synchronized.get("latest_version")
    synchronized["latest_version"] = metadata.version
    synchronized["release_date"] = metadata.release_date
    synchronized["release_notes"] = list(release_notes)
    synchronized["installer_url"] = metadata.installer_url
    if sha256 is not None:
        normalized_sha256 = sha256.strip()
        if normalized_sha256 and not re.fullmatch(
            r"[0-9a-fA-F]{64}",
            normalized_sha256,
        ):
            raise ValueError(
                "--sha256 must be empty or a 64-character hex digest."
            )
        synchronized["sha256"] = normalized_sha256.lower()
    elif previous_version != metadata.version:
        synchronized["sha256"] = ""
    else:
        synchronized.setdefault("sha256", "")
    synchronized.setdefault("mandatory", False)
    return json.dumps(synchronized, indent=2) + "\n"


def _render_readme(root: Path, metadata: ReleaseMetadata) -> str:
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    replacement = (
        "The launch screen and About dialog identify the current release as "
        "**Version\n"
        f"{metadata.version}**, released **{metadata.release_date}**. "
        "Product name, description, version, and\n"
        "release date come from `app/__version__.py`."
    )
    updated, count = re.subn(
        r"The launch screen and About dialog identify the current release as "
        r"\*\*Version\s*\n[^\n]+\nrelease date come from "
        r"`app/__version__\.py`\.",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not locate the README release banner.")
    return updated


def _render_project_status(
    root: Path,
    metadata: ReleaseMetadata,
) -> str:
    path = root / "PROJECT_STATUS.md"
    text = path.read_text(encoding="utf-8")
    replacement = (
        "The current AVISTA application and update-feed version is "
        f"`{metadata.version}`, with the\n"
        f"release date centralized as {metadata.release_date}. "
        "Focused centralized-version,"
    )
    updated, count = re.subn(
        r"The current AVISTA application and update-feed version is "
        r"`[^`]+`, with the\nrelease date centralized as [^.]+\."
        r" Focused centralized-version,",
        replacement,
        text,
        count=1,
    )
    if count != 1:
        raise ValueError(
            "Could not locate the current release block in PROJECT_STATUS.md."
        )
    return updated


def _render_changelog(
    root: Path,
    metadata: ReleaseMetadata,
    release_notes: Sequence[str],
) -> str:
    path = root / "CHANGELOG.md"
    text = path.read_text(encoding="utf-8")
    heading = f"## {metadata.version} - {metadata.iso_date}"
    if heading in text:
        return text
    bullets = "\n".join(
        textwrap.fill(
            f"- {note}",
            width=88,
            subsequent_indent="  ",
        )
        for note in release_notes
    )
    return text.replace(
        "# Changelog\n",
        f"# Changelog\n\n{heading}\n\n{bullets}\n",
        1,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize AVISTA release metadata from app/__version__.py or "
            "prepare a new central release."
        )
    )
    parser.add_argument("--version", help="New X.Y.Z application version.")
    parser.add_argument(
        "--release-date",
        help="Display date such as 'July 25, 2026'.",
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Release-note line; repeat for multiple notes.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify synchronization without modifying files.",
    )
    parser.add_argument(
        "--expected-tag",
        help="Optional vX.Y.Z tag that must match the central version.",
    )
    parser.add_argument(
        "--sha256",
        help="Optional installer SHA256 to publish after the build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would change without writing them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.check:
        if (
            args.version
            or args.release_date
            or args.note
            or args.sha256 is not None
            or args.dry_run
        ):
            raise ValueError("--check cannot be combined with update options.")
        problems = check_release_metadata(
            PROJECT_ROOT,
            expected_tag=args.expected_tag,
        )
        if problems:
            print("Release metadata is not synchronized:", file=sys.stderr)
            for problem in problems:
                print(f"- {problem}", file=sys.stderr)
            print(
                "Run scripts/prepare_release.py with the intended version, "
                "date, and release notes.",
                file=sys.stderr,
            )
            return 1
        metadata = read_central_metadata(PROJECT_ROOT)
        print(
            f"AVISTA v{metadata.version} release metadata is synchronized."
        )
        return 0

    if bool(args.version) != bool(args.release_date):
        raise ValueError(
            "--version and --release-date must be supplied together."
        )
    if args.expected_tag:
        raise ValueError("--expected-tag is only valid with --check.")
    metadata = (
        ReleaseMetadata(args.version, args.release_date)
        if args.version
        else read_central_metadata(PROJECT_ROOT)
    )
    changed = prepare_release(
        PROJECT_ROOT,
        metadata,
        args.note,
        sha256=args.sha256,
        dry_run=args.dry_run,
    )
    action = "Would update" if args.dry_run else "Updated"
    if changed:
        print(f"{action}:")
        for path in changed:
            print(f"- {path.relative_to(PROJECT_ROOT)}")
    else:
        print("Release metadata is already synchronized.")
    print(
        f"Release target: v{metadata.version} ({metadata.release_date})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Release preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
