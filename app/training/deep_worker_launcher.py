"""Build source and packaged commands for the GUI-free deep-learning worker."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.model_registry import get_model_spec
from app.utils.resources import is_packaged_application


DEEP_WORKER_EXECUTABLE = "AVISTADeepWorker.exe"


@dataclass(frozen=True)
class DeepWorkerLaunch:
    """Resolved process details for one deep-learning model launch."""

    executable: Path
    arguments: tuple[str, ...]
    working_directory: Path
    packaged: bool
    config_path: Path
    output_directory: Path
    log_path: Path

    @property
    def command(self) -> list[str]:
        return [str(self.executable), *self.arguments]

    @property
    def executable_exists(self) -> bool:
        return self.executable.is_file()

    @property
    def mode(self) -> str:
        return "packaged" if self.packaged else "source"


def resolve_source_worker_script() -> Path:
    """Return the repository worker script used by source-mode launches."""

    return (Path(__file__).resolve().parent / "run_torch_model.py").resolve()


def resolve_packaged_deep_worker() -> Path:
    """Resolve the dedicated worker beside the installed AVISTA executable."""

    return (Path(sys.executable).resolve().parent / DEEP_WORKER_EXECUTABLE).resolve()


def build_deep_worker_launch(config: Any, model_name: str) -> DeepWorkerLaunch:
    """Resolve the executable, arguments, working directory, and log path."""

    project_dir = Path(config.project_dir).resolve()
    config_path = Path(config.project_file).resolve()
    spec = get_model_spec(model_name)
    output_dir = (
        project_dir
        / "outputs"
        / "training"
        / _torch_output_name(spec.display_name)
    ).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_name = (
        f"{_safe_filename(spec.display_name)}_worker_{timestamp}.log"
    )
    log_path = (project_dir / "logs" / "training" / log_name).resolve()
    packaged = is_packaged_application()

    common_arguments = (
        "--project-dir",
        str(project_dir),
        "--config",
        str(config_path),
        "--model",
        spec.display_name,
        "--output-dir",
        str(output_dir),
        "--log-path",
        str(log_path),
    )
    if packaged:
        executable = resolve_packaged_deep_worker()
        arguments = common_arguments
    else:
        executable = Path(sys.executable).resolve()
        arguments = (
            "-u",
            str(resolve_source_worker_script()),
            *common_arguments,
        )

    return DeepWorkerLaunch(
        executable=executable,
        arguments=tuple(arguments),
        working_directory=project_dir,
        packaged=packaged,
        config_path=config_path,
        output_directory=output_dir,
        log_path=log_path,
    )


def build_deep_worker_command(
    config: Any,
    model_name: str,
) -> tuple[str, list[str]]:
    """Return the centralized executable/arguments tuple for callers and tests."""

    launch = build_deep_worker_launch(config, model_name)
    return str(launch.executable), list(launch.arguments)


def sanitized_worker_arguments(arguments: tuple[str, ...] | list[str]) -> list[str]:
    """Redact project-local paths while retaining useful launch diagnostics."""

    redactions = {
        "--project-dir": "<project-dir>",
        "--config": "<project-config>",
        "--output-dir": "<model-output-dir>",
        "--log-path": "<worker-log>",
    }
    sanitized: list[str] = []
    redact_next: str | None = None
    for argument in arguments:
        if redact_next is not None:
            sanitized.append(redact_next)
            redact_next = None
            continue
        sanitized.append(str(argument))
        redact_next = redactions.get(str(argument))
    return sanitized


def _torch_output_name(display_name: str) -> str:
    if display_name == "FT-Transformer":
        return display_name
    if display_name == "TabPFN 2.5":
        return "TabPFN_2_5"
    return "".join(character for character in display_name if character.isalnum())


def _safe_filename(display_name: str) -> str:
    value = "".join(
        character if character.isalnum() else "_"
        for character in display_name
    )
    return "_".join(part for part in value.split("_") if part) or "DeepModel"
