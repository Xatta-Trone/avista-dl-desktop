"""Run one torch model outside the AVISTA desktop GUI process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from app.__version__ import APP_NAME, __version__
from app.core.project_config import ProjectConfig
from app.core.trainer import train_saved_models
from app.utils.resources import is_packaged_application


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=str), flush=True)


def _failure_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "failure_reason.json"


def _append_worker_log(
    log_path: Path,
    event: str,
    message: str = "",
    **details: Any,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    suffix = f" {json.dumps(details, default=str, sort_keys=True)}" if details else ""
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {event}: {message}{suffix}\n")


def _runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device": "cpu",
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        snapshot.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_available": cuda_available,
                "cuda_version": str(torch.version.cuda or ""),
                "device": (
                    str(torch.cuda.get_device_name(0))
                    if cuda_available
                    else "cpu"
                ),
            }
        )
    except Exception as exc:
        snapshot["torch_detection_error"] = (
            f"{type(exc).__name__}: {exc}"
        )
    return snapshot


def _save_training_history(
    output_dir: Path,
    history: list[dict[str, Any]],
    *,
    model_name: str,
    save_plots: bool,
) -> None:
    if not history:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(history)
    columns = [
        column
        for column in (
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_macro_f1",
            "validation_accuracy",
        )
        if column in frame.columns
    ]
    frame[columns].to_csv(output_dir / "training_history.csv", index=False)
    if not save_plots:
        return

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, (accuracy_axis, loss_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        constrained_layout=True,
    )
    figure.suptitle(f"Training Curves - {model_name}")
    if "train_accuracy" in frame and frame["train_accuracy"].notna().any():
        accuracy_axis.plot(
            frame["epoch"],
            frame["train_accuracy"],
            label="Train Accuracy",
            color="#0F6CBD",
            linewidth=2,
        )
    if "validation_accuracy" in frame and frame["validation_accuracy"].notna().any():
        accuracy_axis.plot(
            frame["epoch"],
            frame["validation_accuracy"],
            label="Validation Accuracy",
            color="#00A6A6",
            linewidth=2,
        )
    loss_axis.plot(
        frame["epoch"],
        frame["train_loss"],
        label="Train Loss",
        color="#0F6CBD",
        linewidth=2,
    )
    if "validation_loss" in frame and frame["validation_loss"].notna().any():
        loss_axis.plot(
            frame["epoch"],
            frame["validation_loss"],
            label="Validation Loss",
            color="#D97706",
            linewidth=2,
        )
    accuracy_axis.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    loss_axis.set(xlabel="Epoch", ylabel="Loss", title="Loss Curve")
    accuracy_axis.grid(True, alpha=0.25)
    loss_axis.grid(True, alpha=0.3)
    accuracy_handles, _ = accuracy_axis.get_legend_handles_labels()
    if accuracy_handles:
        accuracy_axis.legend(loc="best")
    loss_axis.legend(loc="best")
    accuracy_columns = [
        column
        for column in ("train_accuracy", "validation_accuracy")
        if column in frame.columns
    ]
    accuracy_values = frame[accuracy_columns].to_numpy().ravel()
    accuracy_values = accuracy_values[pd.notna(accuracy_values)]
    if len(accuracy_values) and all(
        0.0 <= value <= 1.0 for value in accuracy_values
    ):
        accuracy_axis.set_ylim(0.0, 1.0)
    figure.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "training_curves.pdf", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--log-path")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    log_path = (
        Path(args.log_path).resolve()
        if args.log_path
        else (
            project_dir
            / "logs"
            / "training"
            / "deep_worker.log"
        ).resolve()
    )
    model_name = str(args.model).strip()
    history: list[dict[str, Any]] = []
    model_aliases = {
        "mambaattention": ("mamba_attention", "MambaAttention", "MambaAttention"),
        "mamba_attention": ("mamba_attention", "MambaAttention", "MambaAttention"),
        "ft-transformer": ("ft_transformer", "FT-Transformer", "FT-Transformer"),
        "ft_transformer": ("ft_transformer", "FT-Transformer", "FT-Transformer"),
        "autoint": ("autoint", "AutoInt", "AutoInt"),
        "tabresnet": ("tab_resnet", "TabResNet", "TabResNet"),
        "tab_resnet": ("tab_resnet", "TabResNet", "TabResNet"),
        "tabpfn": ("tabpfn", "TabPFN 2.5", "TabPFN_2_5"),
        "tabpfn 2.5": ("tabpfn", "TabPFN 2.5", "TabPFN_2_5"),
    }
    model_info = model_aliases.get(model_name.casefold())
    canonical_name = model_info[0] if model_info else ""
    display_name = model_info[1] if model_info else model_name
    output_name = model_info[2] if model_info else model_name

    def emit(payload: dict[str, Any]) -> None:
        _emit(payload)
        _append_worker_log(
            log_path,
            str(payload.get("event") or "event"),
            str(payload.get("message") or payload.get("stage") or ""),
            payload=payload,
        )

    _append_worker_log(
        log_path,
        "startup",
        f"{APP_NAME} deep-learning worker starting",
        avista_version=__version__,
        packaged=is_packaged_application(),
        executable=sys.executable,
        arguments=sys.argv[1:],
        working_directory=os.getcwd(),
        config_path=str(config_path),
        output_directory=str(output_dir),
    )
    runtime = _runtime_snapshot()
    emit({"event": "runtime", **runtime})

    try:
        if model_info is None:
            raise ValueError(f"Unsupported torch model '{model_name}'.")
        emit(
            {
                "event": "stage",
                "stage": "configuration_loading",
                "model": display_name,
            }
        )
        config = ProjectConfig.load(config_path)
        config.project_dir = str(project_dir)
        config.selected_models = [canonical_name]
        expected_output = project_dir / "outputs" / "training" / output_name
        if output_dir != expected_output.resolve():
            raise ValueError(
                f"Output directory must be '{expected_output}', got '{output_dir}'."
            )

        model_parameters = (
            (config.model_params or {}).get(canonical_name) or {}
        )
        cuda_requested = str(model_parameters.get("device", "")).casefold().startswith(
            "cuda"
        ) or bool(model_parameters.get("use_gpu", False))
        emit(
            {
                "event": "stage",
                "stage": "dataset_loading",
                "model": display_name,
                "cuda_requested": cuda_requested,
            }
        )
        emit(
            {
                "event": "stage",
                "stage": "model_initialization",
                "model": display_name,
            }
        )
        emit({"event": "started", "model": display_name})

        def progress(payload: dict[str, Any]) -> None:
            event_name = (
                "epoch_progress"
                if payload.get("step") == "epoch"
                else "progress"
            )
            event = {"event": event_name, **payload}
            message = str(payload.get("message", ""))
            if payload.get("step") == "epoch" and payload.get("epoch") is None:
                parts = message.rsplit("epoch ", 1)
                if len(parts) == 2:
                    epoch_text = parts[1].split(":", 1)[0].split("/", 1)[0]
                    if epoch_text.isdigit():
                        event["epoch"] = int(epoch_text)
            if event_name == "epoch_progress":
                event["total_epochs"] = int(
                    (config.model_params.get(canonical_name) or {}).get(
                        "epochs",
                        80,
                    )
                )
                event["validation_macro_f1"] = event.pop(
                    "val_macro_f1",
                    event.get("validation_macro_f1"),
                )
                emit(event)
                if int(event.get("fold", 0)) == 0:
                    history.append(
                        {
                            "epoch": int(event["epoch"]),
                            "train_loss": float(event["train_loss"]),
                            "train_accuracy": (
                                float(event["train_accuracy"])
                                if event.get("train_accuracy") is not None
                                else None
                            ),
                            "validation_loss": (
                                float(event["validation_loss"])
                                if event.get("validation_loss") is not None
                                else None
                            ),
                            "validation_macro_f1": float(
                                event["validation_macro_f1"]
                            ),
                            "validation_accuracy": (
                                float(event["validation_accuracy"])
                                if event.get("validation_accuracy") is not None
                                else None
                            ),
                        }
                    )
                    _save_training_history(
                        output_dir,
                        history,
                        model_name=display_name,
                        save_plots=False,
                    )
                return
            emit(event)

        emit(
            {
                "event": "stage",
                "stage": "training",
                "model": display_name,
            }
        )
        summary = train_saved_models(
            config,
            save_outputs=True,
            progress_callback=progress,
        )
        result = summary["results"][0]
        if result.get("status") not in {"trained", "skipped"}:
            raise RuntimeError(
                str(result.get("error") or result.get("reason") or "Training failed.")
            )
        _save_training_history(
            output_dir,
            history,
            model_name=display_name,
            save_plots=True,
        )
        emit(
            {
                "event": "curve_saved",
                "model": display_name,
                "path": str(output_dir / "training_curves.png"),
            }
        )
        emit({"event": "result", "result": result})
        emit({"event": "complete", "model": display_name})
        _append_worker_log(
            log_path,
            "exit",
            "Worker completed successfully",
            return_code=0,
        )
        return 0
    except Exception as exc:
        traceback_text = traceback.format_exc()
        failure = {
            "model_name": display_name,
            "error": str(exc),
            "source": "torch_subprocess",
            "exception_type": type(exc).__name__,
            "traceback": traceback_text,
        }
        _failure_path(output_dir).write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        emit({"event": "failed", **failure})
        print(traceback_text, file=sys.stderr, flush=True)
        _append_worker_log(
            log_path,
            "exit",
            "Worker failed",
            return_code=1,
            exception_type=type(exc).__name__,
        )
        return 1
    finally:
        try:
            _save_training_history(
                output_dir,
                history,
                model_name=display_name,
                save_plots=bool(history),
            )
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
