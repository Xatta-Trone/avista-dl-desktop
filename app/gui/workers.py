"""Background workers for long-running GUI tasks."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from app.branding import report_footer
from app.core.dependency_manager import install_optional_package
from app.core.edge_case_checker import run_saved_edge_case_checks
from app.core.environment_manager import collect_environment_info
from app.core.gpu_checker import check_gpu, repair_gpu_torch
from app.core.model_registry import get_model_spec
from app.core.project_config import ProjectConfig
from app.core.runtime_verification import collect_runtime_verification
from app.core.trainer import TrainingCancelled, train_saved_models
from app.core.update_checker import (
    UPDATE_METADATA_URL,
    UpdateMetadata,
    check_for_updates,
    download_installer,
)
from app.core.user_settings import load_user_settings
from app.training.deep_worker_launcher import (
    DeepWorkerLaunch,
    build_deep_worker_launch,
    sanitized_worker_arguments,
)
from app.utils.resources import (
    resolve_tabpfn_checkpoint,
    tabpfn_checkpoint_candidates,
)


class UpdateCheckWorker(QObject):
    """Check GitHub update metadata without blocking the GUI."""

    finished = Signal(object)

    def __init__(
        self,
        *,
        manual: bool = False,
        metadata_url: str = UPDATE_METADATA_URL,
    ) -> None:
        super().__init__()
        self.manual = manual
        self.metadata_url = metadata_url

    @Slot()
    def run(self) -> None:
        settings = load_user_settings()
        result = check_for_updates(
            metadata_url=self.metadata_url,
            settings=settings,
            manual=self.manual,
        )
        self.finished.emit(result)


class UpdateDownloadWorker(QObject):
    """Download an update installer without blocking the GUI."""

    progress = Signal(int, int)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, metadata: UpdateMetadata) -> None:
        super().__init__()
        self.metadata = metadata

    @Slot()
    def run(self) -> None:
        try:
            path = download_installer(
                self.metadata,
                progress_callback=lambda downloaded, total: self.progress.emit(
                    downloaded,
                    total,
                ),
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(path))


class DependencyInstallWorker(QObject):
    """Install one optional package without blocking the GUI thread."""

    finished = Signal(dict)

    def __init__(
        self,
        package_name: str,
        *,
        project_dir: str,
        environment_mode: str,
        app_root: str,
    ) -> None:
        super().__init__()
        self.package_name = package_name
        self.project_dir = project_dir
        self.environment_mode = environment_mode
        self.app_root = app_root

    @Slot()
    def run(self) -> None:
        try:
            result = install_optional_package(
                self.package_name,
                project_dir=self.project_dir,
                environment_mode=self.environment_mode,
                app_root=self.app_root,
            )
        except Exception as exc:
            result = {
                "success": False,
                "package": self.package_name,
                "error": str(exc),
            }
        self.finished.emit(result)


class EnvironmentCheckWorker(QObject):
    """Collect system and GPU diagnostics without blocking the GUI thread."""

    finished = Signal(dict)

    def __init__(self, project_dir: str | None = None) -> None:
        super().__init__()
        self.project_dir = project_dir

    @Slot()
    def run(self) -> None:
        try:
            info = collect_environment_info(project_dir=self.project_dir)
            info.update(check_gpu())
            info.update(collect_runtime_verification(info))
            info["gpu_check_error"] = None
        except Exception as exc:
            info = collect_environment_info(project_dir=self.project_dir)
            info.update(collect_runtime_verification(info))
            info.update(
                {
                    "cuda_available": False,
                    "tensor_test_passed": False,
                    "gpu_check_error": str(exc),
                    "error": str(exc),
                }
            )
        self.finished.emit(info)


class EnvironmentRepairWorker(QObject):
    """Repair the active AVISTA GPU runtime, then rerun GPU detection."""

    finished = Signal(dict)

    def __init__(self, project_dir: str) -> None:
        super().__init__()
        self.project_dir = project_dir

    @Slot()
    def run(self) -> None:
        try:
            repair_result = repair_gpu_torch(self.project_dir)
            info = collect_environment_info(project_dir=self.project_dir)
            info.update(check_gpu())
            info["repair_result"] = repair_result
            info["gpu_check_error"] = None
        except Exception as exc:
            info = collect_environment_info(project_dir=self.project_dir)
            info.update(
                {
                    "cuda_available": False,
                    "tensor_test_passed": False,
                    "gpu_check_error": str(exc),
                    "error": str(exc),
                    "repair_result": {
                        "success": False,
                        "message": f"GPU runtime repair failed: {exc}",
                    },
                }
            )
        self.finished.emit(info)


class EdgeCaseCheckWorker(QObject):
    """Run saved-artifact validation without blocking the GUI thread."""

    progress = Signal(str)
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(
        self,
        dataframe: Any,
        config: Any,
        environment_info: dict[str, Any] | None,
        report_path: str,
    ) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.config = config
        self.environment_info = environment_info
        self.report_path = report_path

    @Slot()
    def run(self) -> None:
        try:
            for message in (
                "Running edge-case validation...",
                "Checking target integrity...",
                "Checking missing classes...",
                "Checking train/validation mismatch...",
                "Checking encoding consistency...",
            ):
                self.progress.emit(message)
            report = run_saved_edge_case_checks(
                self.dataframe,
                self.config,
                self.environment_info,
            )
            path = report.save_json(self.report_path)
            self.progress.emit("Completed successfully.")
            self.finished.emit(report, str(path))
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingWorker(QObject):
    """Run model training in a background QThread."""

    started = Signal()
    progress_message = Signal(str)
    progress_update = Signal(dict)
    model_started = Signal(str)
    model_finished = Signal(str, dict)
    model_result_ready = Signal(dict)
    finished = Signal(dict)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        config: Any,
        *,
        save_outputs: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.save_outputs = save_outputs
        self._cancel_requested = False
        self._torch_process: subprocess.Popen[str] | None = None

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True
        if self._torch_process is not None and self._torch_process.poll() is None:
            self._torch_process.terminate()
        self.progress_message.emit("Stop requested. Training will stop at the next safe checkpoint.")

    @Slot()
    def run(self) -> None:
        self.started.emit()
        self._log("Training worker started.")

        try:
            results = self._run_selected_models()

            self._log("Training worker finished.")
            self.finished.emit(results)
        except TrainingCancelled:
            self._log("Training cancelled.")
            self.cancelled.emit()
        except Exception as exc:
            message = str(exc)
            self._log(f"Training failed: {message}")
            self.failed.emit(message)

    def _run_selected_models(self) -> dict[str, Any]:
        selected = list(getattr(self.config, "selected_models", []) or [])
        torch_models = []
        in_process_models = []
        for model_name in selected:
            spec = get_model_spec(model_name)
            if spec.name in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
                "tabpfn",
            }:
                torch_models.append(spec.name)
            else:
                in_process_models.append(spec.name)

        combined_results = []
        for model_name in in_process_models:
            if self._cancel_requested:
                raise TrainingCancelled("Training cancelled by user.")
            display_name = get_model_spec(model_name).display_name
            self.model_started.emit(display_name)
            sklearn_config = ProjectConfig(**self.config.__dict__)
            sklearn_config.selected_models = [model_name]
            try:
                summary = train_saved_models(
                    sklearn_config,
                    save_outputs=self.save_outputs,
                    progress_callback=self._on_progress,
                    should_cancel=lambda: self._cancel_requested,
                )
                model_results = list(summary.get("results", []))
                if not model_results:
                    raise RuntimeError(
                        f"{display_name} completed without returning a model result."
                    )
            except TrainingCancelled:
                raise
            except Exception as exc:
                model_results = [
                    {
                        "model_name": display_name,
                        "status": "failed",
                        "error": str(exc),
                        "saved": False,
                    }
                ]
            for model_result in model_results:
                combined_results.append(model_result)
                self._emit_model_result(model_result)

        for model_name in torch_models:
            if self._cancel_requested:
                raise TrainingCancelled("Training cancelled by user.")
            self.model_started.emit(get_model_spec(model_name).display_name)
            model_result = self._run_torch_subprocess(model_name)
            combined_results.append(model_result)
            self._emit_model_result(model_result)

        combined = {
            "status": "completed",
            "output_root": str(Path(self.config.project_dir) / "outputs" / "training"),
            **self.config.project_metadata(),
            "report_footer": report_footer(),
            "results": combined_results,
        }
        if self.save_outputs:
            output_root = Path(combined["output_root"])
            output_root.mkdir(parents=True, exist_ok=True)
            (output_root / "training_results.json").write_text(
                json.dumps(combined, indent=2, default=str),
                encoding="utf-8",
            )
            _write_training_results_csv(
                output_root / "training_results.csv",
                combined_results,
            )
        return combined

    def _emit_model_result(self, result: dict[str, Any]) -> None:
        model_name = str(result.get("model_name", "unknown"))
        self.model_result_ready.emit(result)
        self.model_finished.emit(model_name, result)
        self._log(f"Model result updated: {model_name}")

    def _run_torch_subprocess(self, model_name: str) -> dict[str, Any]:
        spec = get_model_spec(model_name)
        display_name = spec.display_name
        launch = build_deep_worker_launch(self.config, model_name)
        command = launch.command
        output_dir = launch.output_directory
        runtime = _worker_runtime_context(self.config, model_name)
        launch_details = _launch_diagnostics(launch, runtime)
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONFAULTHANDLER": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        launch.working_directory.mkdir(parents=True, exist_ok=True)
        _append_worker_log(
            launch.log_path,
            "launch",
            {
                **launch_details,
                "arguments": list(launch.arguments),
                "environment_overrides": {
                    key: environment[key]
                    for key in (
                        "PYTHONFAULTHANDLER",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "PYTHONUNBUFFERED",
                    )
                },
            },
        )
        self._log(
            f"Starting {display_name} with {launch.executable.name} "
            f"({launch.mode} mode). Worker log: {launch.log_path}"
        )
        if not _worker_target_exists(launch):
            failure = _subprocess_failure_result(
                display_name,
                None,
                "",
                child_error=(
                    "The packaged deep-learning worker is missing. "
                    "Repair or reinstall AVISTA."
                ),
                launch=launch,
                process_started=False,
                runtime=runtime,
                exception_type="FileNotFoundError",
            )
            _save_subprocess_failure(output_dir, failure)
            _append_worker_log(launch.log_path, "launch_failed", failure)
            self._log(failure["error"])
            return failure

        process_started = False
        try:
            self._torch_process = subprocess.Popen(
                command,
                cwd=str(launch.working_directory),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            process_started = True
            _append_worker_log(
                launch.log_path,
                "process_started",
                {"pid": getattr(self._torch_process, "pid", None)},
            )
        except (OSError, ValueError) as exc:
            failure = _subprocess_failure_result(
                display_name,
                None,
                str(exc),
                child_error=(
                    f"{display_name} training could not start. "
                    f"{type(exc).__name__}: {exc}"
                ),
                launch=launch,
                process_started=False,
                runtime=runtime,
                exception_type=type(exc).__name__,
            )
            _save_subprocess_failure(output_dir, failure)
            _append_worker_log(launch.log_path, "launch_failed", failure)
            self._log(failure["error"])
            return failure
        result: dict[str, Any] | None = None
        child_error = ""
        child_exception_type = ""
        child_traceback = ""
        last_valid_event: dict[str, Any] | None = None
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        assert self._torch_process.stderr is not None
        stderr_thread = threading.Thread(
            target=_drain_stream,
            args=(self._torch_process.stderr, stderr_lines),
            daemon=True,
        )
        stderr_thread.start()
        try:
            assert self._torch_process.stdout is not None
            for line in self._torch_process.stdout:
                stdout_lines.append(line)
                _append_worker_log(
                    launch.log_path,
                    "stdout",
                    {"line": line.rstrip()},
                )
                if self._cancel_requested:
                    self._torch_process.terminate()
                    _append_worker_log(
                        launch.log_path,
                        "cancelled",
                        {"message": "Termination requested by the user."},
                    )
                    raise TrainingCancelled("Training cancelled by user.")
                payload = _parse_json_line(line)
                if payload is None:
                    self._log(line.rstrip())
                    continue
                last_valid_event = payload
                event = payload.get("event")
                if event in {"progress", "epoch_progress"}:
                    self._on_progress(_subprocess_progress(payload))
                elif event == "runtime":
                    for key in (
                        "torch_version",
                        "cuda_available",
                        "cuda_version",
                        "device",
                    ):
                        if key in payload:
                            runtime[key] = payload[key]
                    runtime["last_successful_stage"] = "runtime_detection"
                elif event == "stage":
                    runtime["last_successful_stage"] = str(
                        payload.get("stage") or "worker_stage"
                    )
                elif event == "started":
                    runtime["last_successful_stage"] = "worker_started"
                    self._on_progress(
                        {
                            "model": display_name,
                            "fold": 0,
                            "total_folds": 0,
                            "step": "started",
                            "percent": 0,
                            "message": f"{display_name} started in subprocess",
                        }
                    )
                elif event == "result":
                    result = dict(payload.get("result") or {})
                elif event == "curve_saved":
                    self._log(
                        f"Training curve saved to {payload.get('path', output_dir)}"
                    )
                elif event == "failed":
                    child_error = str(
                        payload.get("error", f"{display_name} subprocess failed.")
                    )
                    child_exception_type = str(payload.get("exception_type") or "")
                    child_traceback = str(payload.get("traceback") or "")
                    self._log(child_error)
            return_code = self._torch_process.wait()
            stderr_thread.join(timeout=5)
        finally:
            self._torch_process = None

        stderr_text = "".join(stderr_lines)
        for line in stderr_lines:
            _append_worker_log(
                launch.log_path,
                "stderr",
                {"line": line.rstrip()},
            )
        if return_code != 0 or result is None:
            failure = _subprocess_failure_result(
                display_name,
                return_code,
                stderr_text,
                child_error=child_error,
                launch=launch,
                process_started=process_started,
                stdout_text="".join(stdout_lines),
                last_valid_event=last_valid_event,
                runtime=runtime,
                exception_type=child_exception_type,
                traceback_text=child_traceback,
            )
            _save_subprocess_failure(output_dir, failure)
            _append_worker_log(launch.log_path, "process_failed", failure)
            self._log(
                f"{failure['error']} Worker log: {launch.log_path}"
            )
            return failure
        result.setdefault("worker_log_path", str(launch.log_path))
        result.setdefault("worker_mode", launch.mode)
        _append_worker_log(
            launch.log_path,
            "process_complete",
            {
                "return_code": return_code,
                "model_status": result.get("status"),
            },
        )
        return result

    def _on_progress(self, progress: dict[str, Any]) -> None:
        self.progress_update.emit(progress)
        message = str(progress.get("message", "")).strip()
        if message:
            self._log(message)
        if progress.get("step") == "started" and progress.get("model"):
            self.model_started.emit(str(progress["model"]))

    def _log(self, message: str) -> None:
        project_dir = Path(getattr(self.config, "project_dir"))
        log_dir = project_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="seconds")
        with (log_dir / "training_log.txt").open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
        self.progress_message.emit(message)


def build_torch_subprocess_command(config: Any, model_name: str) -> list[str]:
    """Compatibility wrapper around the centralized deep-worker launcher."""

    return build_deep_worker_launch(config, model_name).command


def _parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _subprocess_progress(payload: dict[str, Any]) -> dict[str, Any]:
    epoch = payload.get("epoch")
    display_name = str(payload.get("model", "Torch model"))
    progress = {
        "model": display_name,
        "fold": int(payload.get("fold", 0)),
        "total_folds": int(payload.get("total_folds", 0)),
        "step": f"epoch {epoch}" if epoch is not None else payload.get("step", "training"),
        "percent": int(payload.get("percent", 0)),
        "message": payload.get("message")
        or (
            f"{display_name} epoch {payload['epoch']}"
            if payload.get("epoch") is not None
            else f"{display_name} training"
        ),
    }
    if payload.get("event") == "epoch_progress":
        progress.update(
            {
                "event": "epoch_progress",
                "epoch": int(payload["epoch"]),
                "total_epochs": int(payload["total_epochs"]),
                "train_loss": float(payload["train_loss"]),
                "train_accuracy": (
                    float(payload["train_accuracy"])
                    if payload.get("train_accuracy") is not None
                    else None
                ),
                "validation_loss": (
                    float(payload["validation_loss"])
                    if payload.get("validation_loss") is not None
                    else None
                ),
                "validation_macro_f1": float(
                    payload["validation_macro_f1"]
                ),
                "validation_accuracy": (
                    float(payload["validation_accuracy"])
                    if payload.get("validation_accuracy") is not None
                    else None
                ),
            }
        )
    return progress


def _save_subprocess_failure(output_dir: Path, failure: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_reason.json").write_text(
        json.dumps(failure, indent=2),
        encoding="utf-8",
    )


def _subprocess_failure_result(
    display_name: str,
    return_code: int | None,
    stderr_text: str,
    *,
    child_error: str = "",
    launch: DeepWorkerLaunch | None = None,
    process_started: bool = False,
    stdout_text: str = "",
    last_valid_event: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    exception_type: str = "",
    traceback_text: str = "",
) -> dict[str, Any]:
    runtime = runtime or {}
    status_name, status_explanation = _windows_status(return_code)
    return_code_hex = _return_code_hex(return_code)
    if child_error:
        error = child_error
    elif return_code is not None and status_name:
        process_name = (
            launch.executable.name
            if launch is not None
            else "AVISTADeepWorker.exe"
        )
        error = (
            f"{display_name} training could not start.\n\n"
            "The packaged deep-learning worker terminated unexpectedly.\n\n"
            f"Process: {process_name}\n"
            f"Exit code: {return_code} ({return_code_hex})\n"
            f"Windows status: {status_name} / {status_explanation}\n"
            f"Worker log: {launch.log_path if launch is not None else 'Unavailable'}\n\n"
            "No Python traceback was returned. Review the worker log for the "
            "launch command, loaded libraries, CUDA status, and the last "
            "successful initialization step."
        )
    else:
        error = (
            f"{display_name} deep-learning worker failed. "
            f"Review the worker log for launch and runtime details."
        )
    return {
        "model_name": display_name,
        "status": "failed",
        "error": error,
        "return_code": return_code,
        "return_code_decimal": return_code,
        "return_code_hex": return_code_hex,
        "windows_status": status_name,
        "windows_status_explanation": status_explanation,
        "native_process_termination": bool(status_name),
        "executable_path": str(launch.executable) if launch is not None else "",
        "sanitized_arguments": (
            sanitized_worker_arguments(launch.arguments)
            if launch is not None
            else []
        ),
        "working_directory": (
            str(launch.working_directory) if launch is not None else ""
        ),
        "packaged_mode": launch.packaged if launch is not None else None,
        "worker_mode": launch.mode if launch is not None else "unknown",
        "process_start_success": process_started,
        "worker_executable_exists": (
            _worker_target_exists(launch) if launch is not None else False
        ),
        "worker_log_path": str(launch.log_path) if launch is not None else "",
        "stderr_log_path": str(launch.log_path) if launch is not None else "",
        "worker_config_path": (
            str(launch.config_path) if launch is not None else ""
        ),
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "last_valid_json_event": last_valid_event,
        "exception_type": exception_type,
        "traceback": traceback_text,
        "required_assets": runtime.get("required_assets", {}),
        "cuda_requested": runtime.get("cuda_requested"),
        "torch_version": runtime.get("torch_version"),
        "cuda_available": runtime.get("cuda_available"),
        "cuda_version": runtime.get("cuda_version"),
        "device": runtime.get("device"),
        "last_successful_stage": runtime.get("last_successful_stage"),
        "saved": False,
    }


def _windows_status(return_code: int | None) -> tuple[str, str]:
    if return_code is None:
        return "", ""
    statuses = {
        0xC0000409: (
            "STATUS_STACK_BUFFER_OVERRUN",
            "Native fast-fail termination",
        ),
        0xC0000374: (
            "STATUS_HEAP_CORRUPTION",
            "Native heap-corruption termination",
        ),
        0xC0000005: (
            "STATUS_ACCESS_VIOLATION",
            "Native access-violation termination",
        ),
    }
    return statuses.get(return_code & 0xFFFFFFFF, ("", ""))


def _return_code_hex(return_code: int | None) -> str:
    return "" if return_code is None else f"0x{return_code & 0xFFFFFFFF:08X}"


def _worker_target_exists(launch: DeepWorkerLaunch) -> bool:
    if launch.packaged:
        return launch.executable_exists
    if not launch.executable_exists or len(launch.arguments) < 2:
        return False
    return Path(launch.arguments[1]).is_file()


def _worker_runtime_context(config: Any, model_name: str) -> dict[str, Any]:
    parameters = (
        (getattr(config, "model_params", {}) or {}).get(model_name)
        or {}
    )
    device = str(parameters.get("device") or "").strip()
    cuda_requested = device.casefold().startswith("cuda") or bool(
        parameters.get("use_gpu", False)
    )
    try:
        checkpoint = resolve_tabpfn_checkpoint()
    except FileNotFoundError:
        checkpoint = tabpfn_checkpoint_candidates()[0]
    return {
        "cuda_requested": cuda_requested,
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "device": device or "automatic",
        "last_successful_stage": "launcher_initialized",
        "required_assets": {
            "tabpfn_checkpoint": {
                "required": model_name == "tabpfn",
                "exists": checkpoint.is_file(),
            }
        },
    }


def _launch_diagnostics(
    launch: DeepWorkerLaunch,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_worker_mode": launch.mode,
        "executable": str(launch.executable),
        "sanitized_arguments": sanitized_worker_arguments(launch.arguments),
        "working_directory": str(launch.working_directory),
        "worker_executable_exists": _worker_target_exists(launch),
        "worker_config_path": str(launch.config_path),
        "worker_log_path": str(launch.log_path),
        **runtime,
    }


def _append_worker_log(
    log_path: Path,
    event: str,
    details: dict[str, Any],
) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"[{timestamp}] {event}: "
                f"{json.dumps(details, default=str, sort_keys=True)}\n"
            )
    except OSError:
        return


def _drain_stream(stream: Any, collected: list[str]) -> None:
    for line in stream:
        collected.append(line)


def _write_training_results_csv(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    columns = [
        "model",
        "status",
        "train_accuracy",
        "train_macro_f1",
        "validation_accuracy",
        "validation_macro_f1",
        "test_accuracy",
        "test_macro_f1",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_macro_f1_mean",
        "cv_macro_f1_std",
        "roc_auc",
        "saved",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for result in results:
            train = result.get("train_metrics") or {}
            validation = result.get("validation_metrics") or {}
            test = result.get("test_metrics") or {}
            cv = result.get("cv_summary") or {}
            writer.writerow(
                {
                    "model": result.get("model_name", ""),
                    "status": result.get("status", ""),
                    "train_accuracy": train.get("accuracy", ""),
                    "train_macro_f1": train.get("macro_f1", ""),
                    "validation_accuracy": validation.get("accuracy", ""),
                    "validation_macro_f1": validation.get("macro_f1", ""),
                    "test_accuracy": test.get("accuracy", ""),
                    "test_macro_f1": test.get("macro_f1", ""),
                    "cv_accuracy_mean": (cv.get("accuracy") or {}).get("mean", ""),
                    "cv_accuracy_std": (cv.get("accuracy") or {}).get("std", ""),
                    "cv_macro_f1_mean": (cv.get("macro_f1") or {}).get("mean", ""),
                    "cv_macro_f1_std": (cv.get("macro_f1") or {}).get("std", ""),
                    "roc_auc": test.get("roc_auc", ""),
                    "saved": bool(result.get("saved")),
                }
            )
