"""Structured issue reporting for validation checks."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.branding import report_footer


WARNING = "warning"
ERROR = "error"
FATAL = "fatal"
MESSAGE_LEVELS = {WARNING, ERROR, FATAL}


@dataclass
class Issue:
    """A single validation issue."""

    level: str
    category: str
    message: str
    suggestion: str
    affected_column: str | None = None

    def __post_init__(self) -> None:
        if self.level not in MESSAGE_LEVELS:
            raise ValueError(f"Invalid issue level '{self.level}'. Expected one of {sorted(MESSAGE_LEVELS)}.")


@dataclass
class EdgeCaseReport:
    """Collection of validation issues and blocking status."""

    issues: list[Issue] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == WARNING]

    @property
    def errors(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == ERROR]

    @property
    def fatals(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.level == FATAL]

    @property
    def can_continue(self) -> bool:
        return not self.errors and not self.fatals

    def add(
        self,
        level: str,
        category: str,
        message: str,
        suggestion: str,
        affected_column: str | None = None,
    ) -> None:
        self.issues.append(
            Issue(
                level=level,
                category=category,
                message=message,
                suggestion=suggestion,
                affected_column=affected_column,
            )
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeCaseReport":
        """Restore a report saved by current or earlier AVISTA versions."""

        issues = [
            Issue(
                level=str(item["level"]),
                category=str(item["category"]),
                message=str(item["message"]),
                suggestion=str(item["suggestion"]),
                affected_column=item.get("affected_column"),
            )
            for item in data.get("issues", [])
        ]
        return cls(issues=issues, context=dict(data.get("context") or {}))

    def to_dict(self) -> dict:
        return {
            "issues": [asdict(issue) for issue in self.issues],
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "fatals": len(self.fatals),
            "can_continue": self.can_continue,
            "context": self.context,
            "report_footer": report_footer(),
        }

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path
