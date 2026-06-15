"""Professional system and GPU status page."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFormLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.environment_manager import collect_environment_info, save_environment_info
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon
from app.gui.workers import EnvironmentCheckWorker, EnvironmentRepairWorker
from app.utils.resources import is_packaged_application

logger = logging.getLogger(__name__)


class SummaryCard(QWidget):
    """Reusable environment summary card with a status badge."""

    def __init__(
        self,
        object_name: str,
        title: str,
        subtitle: str,
        icon_name: str,
        fields: list[tuple[str, str]],
    ) -> None:
        super().__init__()
        self.setObjectName(object_name)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.values: dict[str, QLabel] = {}
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        header = QHBoxLayout()
        self.icon_label = QLabel()
        self.icon_label.setPixmap(icon(icon_name).pixmap(22, 22))
        title_label = QLabel(title)
        title_label.setObjectName("environmentCardTitle")
        self.badge = QLabel("Warning")
        self.badge.setObjectName("environmentBadge")
        header.addWidget(self.icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        header.addWidget(self.badge)
        layout.addLayout(header)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("environmentCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)

        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, label in fields:
            value = QLabel("Unknown")
            value.setWordWrap(True)
            value.setTextInteractionFlags(value.textInteractionFlags())
            self.values[key] = value
            form.addRow(label, value)
        layout.addLayout(form)
        self.message_label = QLabel("")
        self.message_label.setObjectName("environmentCardMessage")
        self.message_label.setWordWrap(True)
        self.message_label.hide()
        layout.addWidget(self.message_label)
        layout.addStretch(1)

    def set_status(self, text: str, level: str) -> None:
        colors = {
            "ok": ("#1A7F37", "#DAFBE1"),
            "warning": ("#9A6700", "#FFF8C5"),
            "error": ("#CF222E", "#FFEBE9"),
            "unavailable": ("#57606A", "#EAEEF2"),
        }
        foreground, background = colors[level]
        self.badge.setText(text)
        self.badge.setStyleSheet(
            f"color: {foreground}; background: {background};"
            "border-radius: 9px; padding: 2px 8px; font-weight: 600;"
        )


class EnvironmentPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.thread: QThread | None = None
        self.worker: EnvironmentCheckWorker | EnvironmentRepairWorker | None = None
        self._environment_check_origin: str | None = None
        self.setObjectName("environmentPage")

        self.status_label = QLabel("System information is ready. GPU check has not been run.")
        self.status_label.setWordWrap(True)
        self.results_label = self.status_label

        self.cpu_card = SummaryCard(
            "cpuEnvironmentCard",
            "CPU",
            "Processor identity, core availability, and current utilization.",
            "fa6s.microchip",
            [
                ("cpu_name", "CPU name"),
                ("physical_cores", "Physical cores"),
                ("logical_cores", "Logical cores"),
                ("cpu_usage_percent", "Current usage"),
                ("architecture", "Architecture"),
            ],
        )
        self.gpu_card = SummaryCard(
            "gpuEnvironmentCard",
            "GPU",
            "Graphics hardware, CUDA runtime, memory, and validation status.",
            "fa6s.fan",
            [
                ("gpu_detected", "GPU detected"),
                ("gpu_name", "GPU name"),
                ("cuda_available", "CUDA available"),
                ("torch_installed", "PyTorch installed"),
                ("torch_cuda_version", "PyTorch CUDA"),
                ("gpu_count", "CUDA devices"),
                ("driver_version", "Driver version"),
                ("gpu_memory_total_mb", "Memory total"),
                ("gpu_memory_used_mb", "Memory used"),
                ("gpu_memory_free_mb", "Memory free"),
                ("tensor_test_passed", "Tensor test"),
            ],
        )
        self.memory_card = SummaryCard(
            "memoryEnvironmentCard",
            "Memory",
            "System memory pressure and available project-drive capacity.",
            "fa6s.memory",
            [
                ("ram_total_bytes", "Total RAM"),
                ("ram_available_bytes", "Available RAM"),
                ("ram_used_percent", "RAM used"),
                ("disk_free_bytes", "Project drive free"),
            ],
        )

        self.run_gpu_button = QPushButton("Run GPU Check")
        self.run_gpu_button.setObjectName("primaryEnvironmentButton")
        self.run_gpu_button.setIcon(icon("fa6s.gauge-high", "#FFFFFF"))
        self.run_gpu_button.clicked.connect(self.run_gpu_check)
        self.repair_gpu_button = QPushButton("Repair GPU Runtime")
        self.repair_gpu_button.setObjectName("primaryEnvironmentButton")
        self.repair_gpu_button.setIcon(icon("fa6s.screwdriver-wrench", "#FFFFFF"))
        self.repair_gpu_button.clicked.connect(self.repair_gpu_runtime)
        self.repair_gpu_button.hide()
        self.refresh_system_button = QPushButton("Refresh System Info")
        self.refresh_system_button.setObjectName("secondaryEnvironmentButton")
        self.refresh_system_button.setIcon(icon("fa6s.rotate-right", PRIMARY))
        self.refresh_system_button.clicked.connect(self.refresh_system_info)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        self.progress.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("Environment")
        title.setObjectName("environmentTitle")
        subtitle = QLabel(
            "Review runtime, system resources, and GPU readiness."
        )
        subtitle.setObjectName("environmentSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        cards = QHBoxLayout()
        cards.setContentsMargins(0, 0, 0, 0)
        cards.setSpacing(16)
        cards.addWidget(self.cpu_card, stretch=1)
        cards.addWidget(self.gpu_card, stretch=1)
        cards.addWidget(self.memory_card, stretch=1)
        layout.addLayout(cards)

        controls = QHBoxLayout()
        controls.setSpacing(10)
        controls.addWidget(self.run_gpu_button)
        controls.addWidget(self.repair_gpu_button)
        controls.addWidget(self.refresh_system_button)
        controls.addStretch(1)
        layout.addLayout(controls)
        layout.addWidget(self.progress)
        layout.addWidget(self.status_label)
        self._apply_style()
        self.refresh_system_info()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(30_000)
        self.refresh_timer.timeout.connect(self.refresh_system_info)
        self.refresh_timer.start()

    def refresh(self) -> None:
        if self.main_window.environment_info:
            self._show_info(self.main_window.environment_info)
        if self._environment_check_origin == "startup":
            self.status_label.setText("Environment check running...")

    def refresh_system_info(self) -> None:
        project_dir = self._project_dir()
        info = collect_environment_info(project_dir=project_dir)
        existing = dict(self.main_window.environment_info or {})
        existing.update(info)
        self.main_window.environment_info = existing
        self._show_system_info(existing)
        if info.get("system_info_error"):
            self.status_label.setText(info["system_info_error"])
        elif self.thread is None:
            self.status_label.setText("System information refreshed.")

    def run_gpu_check(self) -> None:
        self._start_environment_check(origin="manual")

    def start_startup_environment_check(self) -> None:
        """Run the read-only startup environment check in the background."""

        if self.thread is not None:
            return
        logger.info("Startup environment check started")
        self._start_environment_check(origin="startup")

    def _start_environment_check(self, *, origin: str) -> None:
        if self.thread is not None:
            return
        self._environment_check_origin = origin
        self.run_gpu_button.setEnabled(False)
        self.repair_gpu_button.setEnabled(False)
        self.refresh_system_button.setEnabled(False)
        self.progress.show()
        self.status_label.setText(
            "Environment check running..."
            if origin == "startup"
            else "Checking GPU..."
        )

        self.thread = QThread(self)
        self.worker = EnvironmentCheckWorker(self._project_dir())
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._gpu_check_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(self._clear_thread_references)
        self.thread.start()

    def repair_gpu_runtime(self) -> None:
        if self.thread is not None:
            return
        project_dir = self._project_dir()
        if not project_dir:
            self.status_label.setText(
                "Open or create a project before repairing the GPU runtime."
            )
            return
        self.run_gpu_button.setEnabled(False)
        self.repair_gpu_button.setEnabled(False)
        self.refresh_system_button.setEnabled(False)
        self.progress.show()
        self.status_label.setText("Repairing GPU runtime...")

        self.thread = QThread(self)
        self.worker = EnvironmentRepairWorker(project_dir)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._gpu_repair_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.destroyed.connect(self._clear_thread_references)
        self.thread.start()

    def _gpu_check_finished(self, info: dict[str, Any]) -> None:
        origin = self._environment_check_origin
        self.main_window.environment_info = info
        self._show_info(info)
        saved_path = None
        try:
            saved_path = save_environment_info(self._environment_storage_root(), info)
        except Exception as exc:
            info["environment_save_error"] = str(exc)

        error = info.get("gpu_check_error")
        if error:
            if origin == "startup":
                logger.error("Startup environment check failed: %s", error)
                prefix = "Startup environment check failed"
            else:
                prefix = "GPU check failed"
            self.status_label.setText(
                f"{prefix}: {error}"
                + (f"\nSaved details: {saved_path}" if saved_path else "")
            )
        else:
            if origin == "startup":
                logger.info("Startup environment check completed")
                prefix = "Startup environment check complete"
            else:
                prefix = "GPU check complete"
            self.status_label.setText(
                f"{prefix}: {_gpu_status_text(info)}"
                + (f"\nSaved details: {saved_path}" if saved_path else "")
            )
        self._environment_check_origin = None
        self.progress.hide()
        self.run_gpu_button.setEnabled(True)
        self.repair_gpu_button.setEnabled(True)
        self.refresh_system_button.setEnabled(True)

    def _gpu_repair_finished(self, info: dict[str, Any]) -> None:
        self.main_window.environment_info = info
        self._show_info(info)
        saved_path = None
        project_dir = self._project_dir()
        if project_dir:
            try:
                saved_path = save_environment_info(project_dir, info)
            except Exception as exc:
                info["environment_save_error"] = str(exc)

        repair_result = dict(info.get("repair_result") or {})
        message = repair_result.get("message", "GPU runtime repair completed.")
        error = info.get("gpu_check_error")
        if error:
            self.status_label.setText(
                f"GPU runtime repair failed: {error}"
                + (f"\nSaved details: {saved_path}" if saved_path else "")
            )
        else:
            self.status_label.setText(
                f"{message}\nGPU status: {_gpu_status_text(info)}"
                + (f"\nSaved details: {saved_path}" if saved_path else "")
            )
        self.progress.hide()
        self.run_gpu_button.setEnabled(True)
        self.repair_gpu_button.setEnabled(True)
        self.refresh_system_button.setEnabled(True)

    def _clear_thread_references(self) -> None:
        self.thread = None
        self.worker = None

    def _show_info(self, info: dict[str, Any]) -> None:
        self._show_system_info(info)
        self._show_gpu_info(info)

    def _show_system_info(self, info: dict[str, Any]) -> None:
        cpu = self.cpu_card.values
        cpu["cpu_name"].setText(_display_value(info.get("cpu_name")))
        cpu["physical_cores"].setText(_display_value(info.get("physical_cores")))
        cpu["logical_cores"].setText(_display_value(info.get("logical_cores")))
        cpu["cpu_usage_percent"].setText(_percent(info.get("cpu_usage_percent")))
        cpu["architecture"].setText(_display_value(info.get("architecture")))
        self.cpu_card.set_status(
            "OK" if info.get("psutil_available") else "Warning",
            "ok" if info.get("psutil_available") else "warning",
        )

        memory = self.memory_card.values
        memory["ram_total_bytes"].setText(_bytes_value(info.get("ram_total_bytes")))
        memory["ram_available_bytes"].setText(
            _bytes_value(info.get("ram_available_bytes"))
        )
        memory["ram_used_percent"].setText(_percent(info.get("ram_used_percent")))
        memory["disk_free_bytes"].setText(_bytes_value(info.get("disk_free_bytes")))
        self.memory_card.set_status(
            "OK" if info.get("psutil_available") else "Warning",
            "ok" if info.get("psutil_available") else "warning",
        )

    def _show_gpu_info(self, info: dict[str, Any]) -> None:
        values = self.gpu_card.values
        detected = bool(
            info.get("gpu_count")
            or info.get("nvidia_gpu_detected")
            or info.get("gpu_name")
        )
        values["gpu_detected"].setText(_yes_no(detected))
        values["gpu_name"].setText(_display_value(info.get("gpu_name")))
        values["cuda_available"].setText(_yes_no(info.get("cuda_available")))
        values["torch_installed"].setText(_yes_no(info.get("torch_installed")))
        values["torch_cuda_version"].setText(
            _display_value(info.get("torch_cuda_version"))
        )
        values["gpu_count"].setText(_display_value(info.get("gpu_count"), empty="0"))
        values["driver_version"].setText(_display_value(info.get("driver_version")))
        for key in (
            "gpu_memory_total_mb",
            "gpu_memory_used_mb",
            "gpu_memory_free_mb",
        ):
            values[key].setText(_memory_value(info.get(key)))
        values["tensor_test_passed"].setText(
            _yes_no(info.get("tensor_test_passed"))
        )
        if info.get("gpu_check_error"):
            self.gpu_card.set_status("Error", "error")
            self.gpu_card.message_label.setText(
                f"GPU check failed: {info['gpu_check_error']}"
            )
            self.gpu_card.message_label.show()
            self.repair_gpu_button.hide()
        elif info.get("cuda_available") and info.get("tensor_test_passed"):
            self.gpu_card.set_status("OK", "ok")
            self.gpu_card.message_label.setText(
                "CUDA is available and the GPU runtime is ready."
            )
            self.gpu_card.message_label.show()
            self.repair_gpu_button.hide()
        elif info.get("nvidia_gpu_detected"):
            self.gpu_card.set_status("Warning", "warning")
            self.gpu_card.message_label.setText(
                "NVIDIA GPU detected, but CUDA PyTorch is not active."
            )
            self.gpu_card.message_label.show()
            self.repair_gpu_button.show()
        else:
            self.gpu_card.set_status("Not available", "unavailable")
            self.gpu_card.message_label.setText("No NVIDIA GPU detected.")
            self.gpu_card.message_label.show()
            self.repair_gpu_button.hide()

    def _project_dir(self) -> str | None:
        config = self.main_window.config
        return str(Path(config.project_dir)) if config else None

    def _environment_storage_root(self) -> Path:
        project_dir = self._project_dir()
        if project_dir:
            return Path(project_dir)
        if is_packaged_application():
            local_app_data = os.environ.get("LOCALAPPDATA")
            return (
                Path(local_app_data) / "AVISTA"
                if local_app_data
                else Path.home() / ".avista"
            )
        return Path.cwd()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#environmentPage {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#environmentTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#environmentSubtitle,
            QLabel#environmentCardSubtitle {{
                color: #5B6573;
                font-size: 12px;
            }}
            QLabel#environmentCardMessage {{ color: #5B6573; margin-top: 6px; }}
            QWidget#cpuEnvironmentCard,
            QWidget#gpuEnvironmentCard,
            QWidget#memoryEnvironmentCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#environmentCardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton {{
                min-height: 42px;
                border-radius: 7px;
                padding: 0 18px;
                font-weight: 600;
            }}
            QPushButton#primaryEnvironmentButton {{
                color: white;
                background: {PRIMARY};
                border: none;
            }}
            QPushButton#primaryEnvironmentButton:hover {{ background: #00A6A6; }}
            QPushButton#secondaryEnvironmentButton {{
                color: {PRIMARY};
                background: white;
                border: 1px solid {BORDER};
            }}
            QPushButton#secondaryEnvironmentButton:hover {{
                background: #EFF6FF;
                border-color: {PRIMARY};
            }}
            QPushButton:disabled {{ background: #D0D7DE; color: #6B7280; }}
            QProgressBar {{ border: none; background: #DCE6F2; border-radius: 4px; }}
            QProgressBar::chunk {{ background: {PRIMARY}; border-radius: 4px; }}
            """
        )


def _gpu_status_text(info: dict) -> str:
    cuda_available = bool(info.get("cuda_available"))
    tensor_test_passed = bool(info.get("tensor_test_passed"))
    nvidia_gpu_detected = bool(info.get("nvidia_gpu_detected"))

    if cuda_available and tensor_test_passed:
        return "Ready"
    if cuda_available and not tensor_test_passed:
        return "CUDA Detected but Validation Failed"
    if not cuda_available and nvidia_gpu_detected:
        return "NVIDIA GPU Found, CUDA PyTorch Not Active"
    return "CPU Mode"


def _display_value(value: Any, empty: str = "Unknown") -> str:
    if value is None or value == "":
        return empty
    return str(value)


def _yes_no(value: Any) -> str:
    return "Yes" if bool(value) else "No"


def _memory_value(value: Any) -> str:
    if value is None or value == "":
        return "Unknown"
    return f"{float(value):,.0f} MB"


def _bytes_value(value: Any) -> str:
    if value is None or value == "":
        return "Unknown"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return "Unknown"


def _percent(value: Any) -> str:
    if value is None or value == "":
        return "Unknown"
    return f"{float(value):.1f}%"
