"""Central light and dark theme support for AVISTA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

from app.core.user_settings import load_user_settings, save_user_settings


LIGHT = "light"
DARK = "dark"
THEME_NAMES = {LIGHT, DARK}


@dataclass(frozen=True)
class ThemeTokens:
    name: str
    primary: str
    accent: str
    background: str
    surface: str
    card: str
    elevated: str
    border: str
    text_primary: str
    text_secondary: str
    muted: str
    input: str
    table_alternate: str
    table_header: str
    selection: str
    success: str
    success_bg: str
    warning: str
    warning_bg: str
    error: str
    error_bg: str
    info_bg: str
    disabled_bg: str
    disabled_text: str
    sidebar: str
    sidebar_hover: str
    sidebar_text: str


THEMES: dict[str, ThemeTokens] = {
    LIGHT: ThemeTokens(
        name=LIGHT,
        primary="#0F6CBD",
        accent="#00A6A6",
        background="#F7F9FC",
        surface="#FFFFFF",
        card="#FFFFFF",
        elevated="#FBFCFE",
        border="#D0D7DE",
        text_primary="#0F172A",
        text_secondary="#5B6573",
        muted="#6B7280",
        input="#FFFFFF",
        table_alternate="#F8FAFC",
        table_header="#EEF3F8",
        selection="#DCEBFA",
        success="#16A34A",
        success_bg="#F0FFF4",
        warning="#D97706",
        warning_bg="#FFF8E6",
        error="#DC2626",
        error_bg="#FFF1F0",
        info_bg="#EFF6FF",
        disabled_bg="#EAEEF2",
        disabled_text="#8C959F",
        sidebar="#17324D",
        sidebar_hover="#244A6B",
        sidebar_text="#DCEBFA",
    ),
    DARK: ThemeTokens(
        name=DARK,
        primary="#0F6CBD",
        accent="#00A6A6",
        background="#111827",
        surface="#162033",
        card="#1D293D",
        elevated="#24324A",
        border="#3A4658",
        text_primary="#E5E7EB",
        text_secondary="#B7C0CC",
        muted="#93A4B7",
        input="#111827",
        table_alternate="#18243A",
        table_header="#223047",
        selection="#12385A",
        success="#4ADE80",
        success_bg="#123524",
        warning="#FBBF24",
        warning_bg="#3A2A0B",
        error="#F87171",
        error_bg="#3B171B",
        info_bg="#102D4A",
        disabled_bg="#2E3748",
        disabled_text="#8996A8",
        sidebar="#0B1728",
        sidebar_hover="#14314E",
        sidebar_text="#DCEBFA",
    ),
}


COLOR_MAP_KEYS = {
    "primary": [
        "#0F6CBD",
        "#0969DA",
        "#0B4F8A",
        "#2F6F9F",
    ],
    "accent": ["#00A6A6"],
    "background": ["#F7F9FC"],
    "card": [],
    "elevated": ["#FBFCFE", "#F8FAFC"],
    "border": ["#D0D7DE", "#E5E7EB", "#D8DEE4", "#B6D4F0"],
    "text_primary": ["#0F172A", "#1F2937"],
    "text_secondary": ["#5B6573", "#4B5563"],
    "muted": ["#6B7280"],
    "input": [],
    "table_alternate": ["#F8FAFC"],
    "table_header": ["#EEF3F8", "#F0F4F8"],
    "selection": ["#DCEBFA", "#EAF3FC", "#EFF6FF"],
    "success": ["#2DA44E", "#16A34A", "#1A7F37"],
    "success_bg": ["#F0FFF4", "#F8FFF9", "#DAFBE1"],
    "warning": ["#BF6A02", "#D97706", "#B45309", "#9A6700"],
    "warning_bg": ["#FFF8E6"],
    "error": ["#CF222E", "#DC2626", "#B91C1C"],
    "error_bg": ["#FFF1F0", "#FFF8F7", "#FFEBE9"],
    "disabled_bg": ["#EAEEF2", "#E8EEF5"],
    "disabled_text": ["#8C959F"],
}


def normalize_theme_name(theme_name: str | None) -> str:
    name = str(theme_name or LIGHT).strip().lower()
    return name if name in THEME_NAMES else LIGHT


def get_theme(theme_name: str | None = None) -> ThemeTokens:
    return THEMES[normalize_theme_name(theme_name)]


def load_theme_setting() -> str:
    return normalize_theme_name(load_user_settings().theme_name)


def save_theme_setting(theme_name: str) -> str:
    normalized = normalize_theme_name(theme_name)
    settings = load_user_settings()
    settings.theme_name = normalized
    save_user_settings(settings)
    return normalized


def current_theme() -> ThemeTokens:
    return get_theme(load_theme_setting())


def apply_theme(app: QApplication, theme_name: str | None = None) -> ThemeTokens:
    """Apply the selected theme to the QApplication and visible widgets."""

    tokens = get_theme(theme_name or load_theme_setting())
    app.setProperty("avistaTheme", tokens.name)
    app.setPalette(_palette(tokens))
    app.setStyleSheet(theme_qss(tokens))
    for widget in app.topLevelWidgets():
        apply_theme_to_widget(widget, tokens)
    return tokens


def apply_theme_to_widget(
    widget: QWidget,
    tokens: ThemeTokens | str | None = None,
) -> None:
    theme = get_theme(tokens if isinstance(tokens, str) else getattr(tokens, "name", None))
    for item in _walk_widgets(widget):
        original = item.property("_avista_base_stylesheet")
        if original is None:
            original = item.styleSheet()
            item.setProperty("_avista_base_stylesheet", original)
        if original:
            item.setStyleSheet(transform_stylesheet(str(original), theme))


def transform_stylesheet(stylesheet: str, tokens: ThemeTokens | str | None = None) -> str:
    theme = get_theme(tokens if isinstance(tokens, str) else getattr(tokens, "name", None))
    transformed = stylesheet
    for token_name, colors in COLOR_MAP_KEYS.items():
        value = getattr(theme, token_name)
        for color in colors:
            transformed = transformed.replace(color, value)
            transformed = transformed.replace(color.lower(), value)
    transformed = _replace_color_property(transformed, "background", "#FFFFFF", theme.card)
    transformed = _replace_color_property(
        transformed,
        "alternate-background-color",
        "#FFFFFF",
        theme.table_alternate,
    )
    transformed = transformed.replace("color: white", "color: #FFFFFF")
    return transformed


def theme_qss(tokens: ThemeTokens) -> str:
    return f"""
        QMainWindow, QDialog, QWidget {{
            background: {tokens.background};
            color: {tokens.text_primary};
        }}
        QMenuBar, QMenu {{
            background: {tokens.surface};
            color: {tokens.text_primary};
            border: 1px solid {tokens.border};
        }}
        QMenuBar::item:selected, QMenu::item:selected {{
            background: {tokens.selection};
        }}
        QToolTip {{
            background: {tokens.elevated};
            color: {tokens.text_primary};
            border: 1px solid {tokens.border};
        }}
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox,
        QComboBox, QListWidget, QTableWidget, QTableView {{
            background: {tokens.input};
            color: {tokens.text_primary};
            border: 1px solid {tokens.border};
            selection-background-color: {tokens.selection};
            selection-color: {tokens.text_primary};
        }}
        QHeaderView::section {{
            background: {tokens.table_header};
            color: {tokens.text_primary};
            border: 1px solid {tokens.border};
        }}
        QScrollArea, QAbstractScrollArea {{
            background: {tokens.background};
            border-color: {tokens.border};
        }}
        QTabWidget::pane {{
            border: 1px solid {tokens.border};
            background: {tokens.card};
        }}
        QTabBar::tab {{
            background: {tokens.elevated};
            color: {tokens.text_secondary};
            border: 1px solid {tokens.border};
            padding: 6px 12px;
        }}
        QTabBar::tab:selected {{
            background: {tokens.card};
            color: {tokens.primary};
        }}
        QPushButton {{
            color: {tokens.primary};
            background: {tokens.card};
            border: 1px solid {tokens.border};
            border-radius: 6px;
        }}
        QPushButton:hover {{
            background: {tokens.selection};
            border-color: {tokens.primary};
        }}
        QPushButton:disabled {{
            color: {tokens.disabled_text};
            background: {tokens.disabled_bg};
            border-color: {tokens.border};
        }}
        QProgressBar {{
            background: {tokens.disabled_bg};
            color: {tokens.text_primary};
            border: none;
            border-radius: 4px;
        }}
        QProgressBar::chunk {{
            background: {tokens.primary};
            border-radius: 4px;
        }}
    """


def apply_matplotlib_theme(figure, theme_name: str | None = None) -> None:
    """Style embedded preview figures for the active UI theme."""

    theme = get_theme(theme_name or load_theme_setting())
    figure.patch.set_facecolor(theme.card)
    for axis in figure.axes:
        axis.set_facecolor(theme.card)
        axis.tick_params(colors=theme.text_secondary)
        axis.xaxis.label.set_color(theme.text_primary)
        axis.yaxis.label.set_color(theme.text_primary)
        axis.title.set_color(theme.text_primary)
        for spine in axis.spines.values():
            spine.set_color(theme.border)
        axis.grid(True, color=theme.border, alpha=0.55, linewidth=0.7)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(theme.card)
            legend.get_frame().set_edgecolor(theme.border)
            for text in legend.get_texts():
                text.set_color(theme.text_primary)


def _palette(tokens: ThemeTokens) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(tokens.background))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(tokens.text_primary))
    palette.setColor(QPalette.ColorRole.Base, QColor(tokens.input))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(tokens.table_alternate))
    palette.setColor(QPalette.ColorRole.Text, QColor(tokens.text_primary))
    palette.setColor(QPalette.ColorRole.Button, QColor(tokens.card))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(tokens.primary))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(tokens.selection))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(tokens.text_primary))
    return palette


def _walk_widgets(widget: QWidget) -> Iterable[QWidget]:
    yield widget
    yield from widget.findChildren(QWidget)


def _replace_color_property(
    stylesheet: str,
    property_name: str,
    source: str,
    target: str,
) -> str:
    pattern = re.compile(
        rf"({re.escape(property_name)}\s*:\s*){re.escape(source)}",
        re.IGNORECASE,
    )
    return pattern.sub(rf"\1{target}", stylesheet)
