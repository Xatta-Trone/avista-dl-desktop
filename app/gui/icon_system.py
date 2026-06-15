"""Shared Font Awesome icons and AVISTA UI colors."""

from __future__ import annotations

import qtawesome as qta
from PySide6.QtGui import QIcon


PRIMARY = "#0F6CBD"
ACCENT = "#00A6A6"
BACKGROUND = "#F7F9FC"
TEXT = "#1F2937"
BORDER = "#D0D7DE"

PAGE_ICONS = {
    "Project Setup": "fa6s.diagram-project",
    "Environment": "fa6s.microchip",
    "Data Import": "fa6s.database",
    "Column Configuration": "fa6s.table-columns",
    "Data Split & Imbalance": "fa6s.code-branch",
    "Model Selection": "fa6s.brain",
    "Edge-Case Report": "fa6s.shield-halved",
    "Training": "fa6s.circle-play",
    "Report": "fa6s.file-lines",
}

FEEDBACK_ICONS = {
    "success": "fa6s.circle-check",
    "warning": "fa6s.triangle-exclamation",
    "error": "fa6s.circle-xmark",
    "info": "fa6s.circle-info",
}

FEEDBACK_COLORS = {
    "success": ("#2DA44E", "#F0FFF4"),
    "warning": ("#BF6A02", "#FFF8E6"),
    "error": ("#CF222E", "#FFF1F0"),
    "info": (PRIMARY, "#EFF6FF"),
}


def icon(name: str, color: str = PRIMARY) -> QIcon:
    """Return a Font Awesome icon with AVISTA's default primary color."""

    return qta.icon(name, color=color)
