"""Background workers for long-running GUI tasks."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
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
        command = build_torch_subprocess_command(self.config, model_name)
        output_dir = (
            Path(self.config.project_dir)
            / "outputs"
            / "training"
            / _torch_output_name(display_name)
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONFAULTHANDLER": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        self._log(f"Starting {display_name} subprocess: {command[0]}")
        try:
            self._torch_process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            failure = _subprocess_failure_result(display_name, None, str(exc))
            _save_subprocess_failure(output_dir, failure)
            self._log(f"{failure['error']} {exc}")
            return failure
        result: dict[str, Any] | None = None
        child_error = ""
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
                if self._cancel_requested:
                    self._torch_process.terminate()
                    raise TrainingCancelled("Training cancelled by user.")
                payload = _parse_json_line(line)
                if payload is None:
                    self._log(line.rstrip())
                    continue
                event = payload.get("event")
                if event in {"progress", "epoch_progress"}:
                    self._on_progress(_subprocess_progress(payload))
                elif event == "started":
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
                    self._log(child_error)
            return_code = self._torch_process.wait()
            stderr_thread.join(timeout=5)
        finally:
            self._torch_process = None

        stderr_text = "".join(stderr_lines)
        if return_code != 0 or result is None:
            failure = _subprocess_failure_result(
                display_name,
                return_code,
                stderr_text,
                child_error=child_error,
            )
            _save_subprocess_failure(output_dir, failure)
            self._log(
                f"{failure['error']} Return code: {return_code}. "
                f"stderr: {stderr_text[-1000:].strip()}"
            )
            return failure
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
    project_dir = Path(config.project_dir).resolve()
    config_path = Path(config.project_file).resolve()
    spec = get_model_spec(model_name)
    output_dir = (
        project_dir / "outputs" / "training" / _torch_output_name(spec.display_name)
    )
    return [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
        "--project-dir",
        str(project_dir),
        "--config",
        str(config_path),
        "--model",
        spec.display_name,
        "--output-dir",
        str(output_dir),
    ]


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
) -> dict[str, Any]:
    return {
        "model_name": display_name,
        "status": "failed",
        "error": child_error
        or f"{display_name} failed in subprocess. GUI remained stable.",
        "return_code": return_code,
        "stderr_tail": stderr_text[-4000:],
        "saved": False,
    }


def _torch_output_name(display_name: str) -> str:
    if display_name == "FT-Transformer":
        return display_name
    if display_name == "TabPFN 2.5":
        return "TabPFN_2_5"
    return "".join(character for character in display_name if character.isalnum())


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
