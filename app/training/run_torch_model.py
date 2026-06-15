"""Run one torch model outside the PySide6 process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.project_config import ProjectConfig
from app.core.trainer import train_saved_models


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, default=str), flush=True)


def _failure_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "failure_reason.json"


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
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
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

    try:
        if model_info is None:
            raise ValueError(f"Unsupported torch model '{model_name}'.")
        config = ProjectConfig.load(config_path)
        config.project_dir = str(project_dir)
        config.selected_models = [canonical_name]
        expected_output = project_dir / "outputs" / "training" / output_name
        if output_dir != expected_output.resolve():
            raise ValueError(
                f"Output directory must be '{expected_output}', got '{output_dir}'."
            )

        _emit({"event": "started", "model": display_name})

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
                _emit(event)
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
            _emit(event)

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
        _emit(
            {
                "event": "curve_saved",
                "model": display_name,
                "path": str(output_dir / "training_curves.png"),
            }
        )
        _emit({"event": "result", "result": result})
        _emit({"event": "complete", "model": display_name})
        return 0
    except Exception as exc:
        failure = {
            "model_name": display_name,
            "error": str(exc),
            "source": "torch_subprocess",
        }
        _failure_path(output_dir).write_text(
            json.dumps(failure, indent=2),
            encoding="utf-8",
        )
        _emit({"event": "failed", **failure})
        print(str(exc), file=sys.stderr, flush=True)
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
