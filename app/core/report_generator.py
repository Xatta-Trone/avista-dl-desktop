"""Generate comprehensive AVISTA reports from saved project artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from app.__version__ import APP_NAME, __version__
from app.branding import report_footer
from app.core.project_config import ProjectConfig


PERFORMANCE_COLUMNS = [
    "Model",
    "Status",
    "Train Accuracy",
    "Train Macro-F1",
    "Validation Accuracy",
    "Validation Macro-F1",
    "Test Accuracy",
    "Test Macro-F1",
    "CV Accuracy Mean",
    "CV Accuracy Std",
    "CV Macro-F1 Mean",
    "CV Macro-F1 Std",
    "ROC-AUC",
    "Saved",
]
DEEP_OUTPUT_NAMES = {
    "MambaAttention": "MambaAttention",
    "FT-Transformer": "FT-Transformer",
    "AutoInt": "AutoInt",
    "TabResNet": "TabResNet",
}


@dataclass
class ReportArtifacts:
    output_dir: Path
    markdown_path: Path
    pdf_path: Path
    performance_csv: Path
    summary: dict[str, Any]
    performance: pd.DataFrame
    generated_files: dict[str, Path]


@dataclass
class ModelDiagnostic:
    model_name: str
    split_name: str
    confusion_path: Path | None
    classification_report: pd.DataFrame
    warning: str = ""


def generate_project_report(
    config: ProjectConfig,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> ReportArtifacts:
    project_dir = Path(config.project_dir)
    training_dir = project_dir / "outputs" / "training"
    output_dir = project_dir / "outputs" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_on = datetime.now()

    _progress(progress_callback, "Collecting project and training metadata...")
    summary = collect_report_summary(config, generated_on)
    performance = collect_model_performance(training_dir)
    performance_path = output_dir / "model_performance_summary.csv"
    performance.to_csv(performance_path, index=False)

    _progress(progress_callback, "Creating ROC comparison...")
    roc_path = create_curve_comparison(
        training_dir,
        output_dir,
        curve_name="roc",
    )
    _progress(progress_callback, "Creating precision-recall comparison...")
    pr_path = create_curve_comparison(
        training_dir,
        output_dir,
        curve_name="pr",
    )
    _progress(progress_callback, "Creating deep-learning curve comparison...")
    deep_path = create_deep_training_comparison(training_dir, output_dir)
    confusion_paths = collect_confusion_matrices(training_dir)
    classification_reports = collect_classification_reports(training_dir)
    feature_importance_paths = collect_feature_importance(training_dir)

    generated_files = {
        "performance": performance_path,
        **({"roc": roc_path} if roc_path else {}),
        **({"pr": pr_path} if pr_path else {}),
        **({"deep": deep_path} if deep_path else {}),
    }
    _progress(progress_callback, "Writing Markdown report...")
    markdown_path = output_dir / "AVISTA_Report.md"
    markdown_path.write_text(
        build_markdown_report(
            config,
            summary,
            performance,
            generated_files,
            confusion_paths,
            classification_reports,
            feature_importance_paths,
        ),
        encoding="utf-8",
    )
    _progress(progress_callback, "Writing PDF report...")
    pdf_path = output_dir / "AVISTA_Report.pdf"
    write_pdf_report(
        pdf_path,
        summary,
        performance,
        generated_files,
        confusion_paths,
        classification_reports,
        feature_importance_paths,
    )
    _progress(progress_callback, "Report generated successfully.")
    return ReportArtifacts(
        output_dir=output_dir,
        markdown_path=markdown_path,
        pdf_path=pdf_path,
        performance_csv=performance_path,
        summary=summary,
        performance=performance,
        generated_files=generated_files,
    )


def collect_report_summary(
    config: ProjectConfig,
    generated_on: datetime | None = None,
) -> dict[str, Any]:
    project_dir = Path(config.project_dir)
    split_dir = project_dir / "outputs" / "data_split"
    rows = {"train": 0, "validation": 0, "test": 0}
    for key, filename in (
        ("train", "y_train_balanced.npy"),
        ("validation", "y_val.npy"),
        ("test", "y_test.npy"),
    ):
        path = split_dir / filename
        if path.exists():
            try:
                rows[key] = int(len(np.load(path, allow_pickle=True)))
            except (OSError, ValueError):
                pass
    dataset_rows = _dataset_rows(project_dir, config, rows)
    performance = collect_model_performance(project_dir / "outputs" / "training")
    footer = report_footer(generated_on)
    return {
        "project_name": config.project_name,
        "target_column": config.target_column or "Not available",
        "dataset_rows": dataset_rows,
        "feature_count": len(config.feature_columns or []),
        "train_rows": rows["train"],
        "validation_rows": rows["validation"],
        "test_rows": rows["test"],
        "imbalance_method": config.imbalance_method or "none",
        "cv_enabled": bool(config.enable_cross_validation),
        "cv_folds": int(config.cv_folds),
        "models_trained": int(
            performance["Status"].str.casefold().eq("trained").sum()
            if not performance.empty
            else 0
        ),
        "generated_on": footer["generated_on"],
        "version": __version__,
    }


def collect_model_performance(training_dir: Path) -> pd.DataFrame:
    summary_path = training_dir / "training_results.json"
    results: list[dict[str, Any]] = []
    if summary_path.exists():
        payload = _read_json(summary_path)
        results = list(payload.get("results") or [])
    if not results and training_dir.exists():
        results = _scan_model_results(training_dir)
    rows = []
    for result in results:
        train = result.get("train_metrics") or {}
        validation = result.get("validation_metrics") or {}
        test = result.get("test_metrics") or {}
        cv = result.get("cv_summary") or {}
        rows.append(
            {
                "Model": result.get("model_name", "Unknown"),
                "Status": result.get("status", "Not available"),
                "Train Accuracy": train.get("accuracy"),
                "Train Macro-F1": train.get("macro_f1"),
                "Validation Accuracy": validation.get("accuracy"),
                "Validation Macro-F1": validation.get("macro_f1"),
                "Test Accuracy": test.get("accuracy"),
                "Test Macro-F1": test.get("macro_f1"),
                "CV Accuracy Mean": (cv.get("accuracy") or {}).get("mean"),
                "CV Accuracy Std": (cv.get("accuracy") or {}).get("std"),
                "CV Macro-F1 Mean": (cv.get("macro_f1") or {}).get("mean"),
                "CV Macro-F1 Std": (cv.get("macro_f1") or {}).get("std"),
                "ROC-AUC": test.get("roc_auc"),
                "Saved": "Yes" if result.get("saved") else "No",
            }
        )
    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)


def create_curve_comparison(
    training_dir: Path,
    output_dir: Path,
    *,
    curve_name: str,
) -> Path | None:
    is_roc = curve_name == "roc"
    filename = "roc_curve.csv" if is_roc else "pr_curve.csv"
    x_column = "fpr" if is_roc else "recall"
    y_column = "tpr" if is_roc else "precision"
    score_column = "auc" if is_roc else "average_precision"
    curves: list[tuple[str, np.ndarray, np.ndarray, float | None]] = []
    for model_dir in _model_directories(training_dir):
        files = _curve_files(model_dir, filename)
        runs = []
        scores = []
        for path in files:
            frame = _read_csv(path)
            if frame.empty or not {x_column, y_column}.issubset(frame.columns):
                continue
            runs.append(_macro_curve(frame, x_column, y_column))
            if score_column in frame:
                score_values = pd.to_numeric(frame[score_column], errors="coerce").dropna()
                if not score_values.empty:
                    scores.append(float(score_values.mean()))
        if not runs:
            continue
        grid = np.linspace(0.0, 1.0, 201)
        interpolated = np.vstack(
            [np.interp(grid, x_values, y_values) for x_values, y_values in runs]
        )
        curves.append(
            (
                _model_display_name(model_dir),
                grid,
                interpolated.mean(axis=0),
                float(np.mean(scores)) if scores else None,
            )
        )
    if not curves:
        return None

    base = output_dir / (
        "roc_curve_comparison" if is_roc else "pr_curve_comparison"
    )
    figure, axis = plt.subplots(figsize=(11.5, 7.2), constrained_layout=True)
    for model, x_values, mean_values, score in curves:
        score_label = "AUC" if is_roc else "AP"
        suffix = f" ({score_label}={score:.3f})" if score is not None else ""
        axis.plot(x_values, mean_values, linewidth=2.4, label=f"{model}{suffix}")
    if is_roc:
        axis.plot([0, 1], [0, 1], "--", color="0.45", label="No-skill")
    axis.set(
        title="ROC Curve Comparison" if is_roc else "Precision-Recall Curve Comparison",
        xlabel="False Positive Rate" if is_roc else "Recall",
        ylabel="True Positive Rate" if is_roc else "Precision",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=max(1, min(3, len(curves) + int(is_roc))),
        frameon=True,
        fontsize=9,
    )
    _save_figure(figure, base)
    plt.close(figure)
    return base.with_suffix(".png")


def create_deep_training_comparison(
    training_dir: Path,
    output_dir: Path,
) -> Path | None:
    histories = []
    for display_name, output_name in DEEP_OUTPUT_NAMES.items():
        model_dir = training_dir / output_name
        files = sorted(model_dir.rglob("training_history.csv")) if model_dir.exists() else []
        runs = [_read_csv(path) for path in files]
        runs = [frame for frame in runs if not frame.empty and "epoch" in frame]
        if runs:
            histories.append((display_name, runs))
    if not histories:
        return None

    base = output_dir / "deep_training_curves"
    figure, (loss_axis, metric_axis) = plt.subplots(
        1, 2, figsize=(14.0, 6.2), constrained_layout=True
    )
    for model, runs in histories:
        _plot_history_group(loss_axis, model, runs, "train_loss")
        metric = (
            "validation_accuracy"
            if any("validation_accuracy" in frame for frame in runs)
            else "validation_macro_f1"
        )
        _plot_history_group(metric_axis, model, runs, metric)
    loss_axis.set(title="Training Loss", xlabel="Epoch", ylabel="Loss")
    metric_axis.set(title="Validation Performance", xlabel="Epoch", ylabel="Score")
    for axis in (loss_axis, metric_axis):
        axis.grid(alpha=0.25)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=2,
            fontsize=9,
        )
    _save_figure(figure, base)
    plt.close(figure)
    return base.with_suffix(".png")


def collect_confusion_matrices(training_dir: Path) -> dict[str, Path]:
    paths = {}
    for model_dir in _model_directories(training_dir):
        path = model_dir / "test" / "confusion_matrix.png"
        if path.exists():
            paths[_model_display_name(model_dir)] = path
            continue
        csv_path = model_dir / "test" / "confusion_matrix.csv"
        if csv_path.exists():
            frame = _read_csv(csv_path, index_col=0)
            if not frame.empty:
                generated = training_dir.parent / "report" / (
                    f"confusion_matrix_{model_dir.name}.png"
                )
                _plot_confusion(frame, generated, _model_display_name(model_dir))
                paths[_model_display_name(model_dir)] = generated
    return paths


def collect_classification_reports(
    training_dir: Path,
    *,
    split_name: str = "test",
) -> dict[str, pd.DataFrame]:
    reports = {}
    for model_dir in _model_directories(training_dir):
        frame = _read_classification_report(
            model_dir / split_name / "classification_report.csv"
        )
        if not frame.empty:
            reports[_model_display_name(model_dir)] = frame
    return reports


def available_diagnostic_models(training_dir: Path) -> dict[str, Path]:
    return {
        _model_display_name(model_dir): model_dir
        for model_dir in _model_directories(training_dir)
        if any((model_dir / split).exists() for split in ("train", "validation", "test"))
    }


def load_model_diagnostic(
    training_dir: Path,
    model_name: str,
    split_name: str,
    preview_dir: Path,
) -> ModelDiagnostic:
    model_dir = available_diagnostic_models(training_dir).get(model_name)
    normalized_split = split_name.casefold()
    if model_dir is None:
        return ModelDiagnostic(
            model_name,
            normalized_split,
            None,
            pd.DataFrame(),
            f"No saved outputs found for {model_name}.",
        )
    split_dir = model_dir / normalized_split
    report = _read_classification_report(split_dir / "classification_report.csv")
    confusion_png = split_dir / "confusion_matrix.png"
    if confusion_png.exists():
        confusion_path = confusion_png
    else:
        matrix = _read_csv(split_dir / "confusion_matrix.csv", index_col=0)
        if matrix.empty:
            matrix = _confusion_from_predictions(split_dir / "predictions.csv")
        if matrix.empty:
            confusion_path = None
        else:
            preview_dir.mkdir(parents=True, exist_ok=True)
            confusion_path = preview_dir / (
                f"diagnostic_{model_dir.name}_{normalized_split}_confusion.png"
            )
            _plot_confusion(
                matrix,
                confusion_path,
                model_name,
                split_name=normalized_split,
            )
    missing = []
    if confusion_path is None:
        missing.append("confusion matrix")
    if report.empty:
        missing.append("classification report")
    warning = (
        f"Not available for {model_name} ({normalized_split.title()}): "
        + ", ".join(missing)
        if missing
        else ""
    )
    return ModelDiagnostic(
        model_name,
        normalized_split,
        confusion_path,
        report,
        warning,
    )


def collect_feature_importance(training_dir: Path) -> dict[str, Path]:
    paths = {}
    for model_dir in _model_directories(training_dir):
        png = model_dir / "feature_importance.png"
        if png.exists():
            paths[_model_display_name(model_dir)] = png
            continue
        csv_path = model_dir / "feature_importance.csv"
        if csv_path.exists():
            frame = _read_csv(csv_path)
            if {"feature", "importance"}.issubset(frame.columns):
                generated = training_dir.parent / "report" / (
                    f"feature_importance_{model_dir.name}.png"
                )
                _plot_feature_importance(frame, generated, _model_display_name(model_dir))
                paths[_model_display_name(model_dir)] = generated
    return paths


def build_markdown_report(
    config: ProjectConfig,
    summary: dict[str, Any],
    performance: pd.DataFrame,
    generated_files: dict[str, Path],
    confusion_paths: dict[str, Path],
    classification_reports: dict[str, pd.DataFrame],
    feature_importance_paths: dict[str, Path],
) -> str:
    lines = [
        "# AVISTA Model Report",
        "",
        "## Project Summary",
        "",
        _markdown_pairs(
            [
                ("Project name", summary["project_name"]),
                ("Target column", summary["target_column"]),
                ("Report generated", summary["generated_on"]),
                ("AVISTA version", summary["version"]),
            ]
        ),
        "",
        "## Dataset Summary",
        "",
        _markdown_pairs(
            [
                ("Dataset rows", summary["dataset_rows"]),
                ("Feature count", summary["feature_count"]),
                ("Train rows", summary["train_rows"]),
                ("Validation rows", summary["validation_rows"]),
                ("Test rows", summary["test_rows"]),
            ]
        ),
        "",
        "## Modeling Configuration",
        "",
        f"- Selected models: {', '.join(config.selected_models or []) or 'Not available'}",
        f"- Cross-validation: {'Enabled' if summary['cv_enabled'] else 'Disabled'}",
        f"- CV folds: {summary['cv_folds'] if summary['cv_enabled'] else 'Not applicable'}",
        "",
        "## Split and Imbalance Summary",
        "",
        f"- Split method: {config.split_method or 'Not available'}",
        f"- Imbalance method: {summary['imbalance_method']}",
        "",
        "## Model Performance Summary",
        "",
        "Unless otherwise stated, model comparison figures and diagnostic tables are based on the test set.",
        "",
        _markdown_table(performance),
        "",
        "## ROC Curve Comparison",
        "",
        _markdown_image(generated_files.get("roc"), "ROC Curve Comparison"),
        "",
        "## Precision-Recall Curve Comparison",
        "",
        _markdown_image(generated_files.get("pr"), "Precision-Recall Curve Comparison"),
        "",
        "## Deep Learning Training Curves",
        "",
        _markdown_image(generated_files.get("deep"), "Deep Learning Training Curves"),
        "",
        "## Confusion Matrices",
        "",
    ]
    lines.extend(_markdown_image_group(confusion_paths))
    lines.extend(["", "## Classification Reports", ""])
    lines.extend(_markdown_report_group(classification_reports))
    lines.extend(["", "## Feature Importance", ""])
    lines.extend(_markdown_image_group(feature_importance_paths))
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            *[
                f"- {path.name}"
                for path in generated_files.values()
            ],
            "- model_performance_summary.csv",
            "- AVISTA_Report.md",
            "- AVISTA_Report.pdf",
            "",
            "## Reproducibility Metadata",
            "",
            f"- Project file: {config.project_file}",
            f"- Random state: {config.random_state}",
            f"- Target column: {config.target_column or 'Not available'}",
            "",
            "---",
            "",
            "Generated by AVISTA",
            f"Version {summary['version']}",
            f"Generated on {summary['generated_on']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_pdf_report(
    path: Path,
    summary: dict[str, Any],
    performance: pd.DataFrame,
    generated_files: dict[str, Path],
    confusion_paths: dict[str, Path],
    classification_reports: dict[str, pd.DataFrame],
    feature_importance_paths: dict[str, Path],
) -> None:
    with PdfPages(path) as pdf:
        _pdf_text_page(pdf, "AVISTA Model Report", [
            f"Project: {summary['project_name']}",
            f"Target: {summary['target_column']}",
            f"Dataset rows: {summary['dataset_rows']}",
            f"Features: {summary['feature_count']}",
            (
                "Split rows: "
                f"{summary['train_rows']} train / "
                f"{summary['validation_rows']} validation / "
                f"{summary['test_rows']} test"
            ),
            f"Imbalance method: {summary['imbalance_method']}",
            (
                f"Cross-validation: {summary['cv_folds']} folds"
                if summary["cv_enabled"]
                else "Cross-validation: Disabled"
            ),
            f"Models trained: {summary['models_trained']}",
            f"Generated on: {summary['generated_on']}",
            f"AVISTA version: {summary['version']}",
            (
                "Unless otherwise stated, model comparison figures and "
                "diagnostic tables are based on the test set."
            ),
        ], page_number=1)
        _pdf_table_pages(pdf, performance, start_page=2)
        page_number = 2 + max(1, int(np.ceil(max(1, len(performance)) / 18)))
        for title, key in (
            ("ROC Curve Comparison", "roc"),
            ("Precision-Recall Curve Comparison", "pr"),
            ("Deep Learning Training Curves", "deep"),
        ):
            image = generated_files.get(key)
            if image and image.exists():
                _pdf_image_page(pdf, title, image, page_number)
                page_number += 1
        for title, paths in (
            ("Confusion Matrices", confusion_paths),
            ("Feature Importance", feature_importance_paths),
        ):
            if paths:
                for model, image in paths.items():
                    _pdf_image_page(pdf, f"{title} - {model}", image, page_number)
                    page_number += 1
            else:
                _pdf_text_page(
                    pdf,
                    title,
                    ["Not available"],
                    page_number=page_number,
                )
                page_number += 1
        for model, report in classification_reports.items():
            page_number = _pdf_classification_report_pages(
                pdf,
                model,
                report,
                start_page=page_number,
            )


def _scan_model_results(training_dir: Path) -> list[dict[str, Any]]:
    results = []
    for model_dir in _model_directories(training_dir):
        train = _read_json(model_dir / "train" / "metrics.json")
        validation = _read_json(model_dir / "validation" / "metrics.json")
        test = _read_json(model_dir / "test" / "metrics.json")
        cv = _read_json(model_dir / "cv_summary.json")
        failure = _read_json(model_dir / "failure_reason.json")
        if not any((train, validation, test, cv, failure)):
            continue
        results.append(
            {
                "model_name": _model_display_name(model_dir),
                "status": "failed" if failure else "trained",
                "train_metrics": train,
                "validation_metrics": validation,
                "test_metrics": test,
                "cv_summary": cv,
                "saved": not bool(failure),
            }
        )
    return results


def _model_directories(training_dir: Path) -> list[Path]:
    if not training_dir.exists():
        return []
    return sorted(path for path in training_dir.iterdir() if path.is_dir())


def _model_display_name(path: Path) -> str:
    config = _read_json(path / "model_config.json")
    return str(config.get("display_name") or path.name.replace("_", " "))


def _curve_files(model_dir: Path, filename: str) -> list[Path]:
    main = model_dir / "test" / filename
    return [main] if main.exists() else []


def _macro_curve(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, 201)
    groups = (
        frame.groupby("class", dropna=False)
        if "class" in frame
        else [("all", frame)]
    )
    values = []
    for _, group in groups:
        x_values = pd.to_numeric(group[x_column], errors="coerce").to_numpy()
        y_values = pd.to_numeric(group[y_column], errors="coerce").to_numpy()
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        x_values, y_values = x_values[valid], y_values[valid]
        if len(x_values) < 2:
            continue
        order = np.argsort(x_values)
        values.append(np.interp(grid, x_values[order], y_values[order]))
    return grid, np.mean(values, axis=0) if values else np.zeros_like(grid)


def _plot_history_group(
    axis,
    model: str,
    runs: list[pd.DataFrame],
    column: str,
) -> None:
    available = [frame for frame in runs if column in frame and frame[column].notna().any()]
    if not available:
        return
    max_epoch = max(int(frame["epoch"].max()) for frame in available)
    grid = np.arange(1, max_epoch + 1)
    values = np.vstack(
        [
            np.interp(
                grid,
                pd.to_numeric(frame["epoch"], errors="coerce"),
                pd.to_numeric(frame[column], errors="coerce"),
            )
            for frame in available
        ]
    )
    mean = values.mean(axis=0)
    axis.plot(grid, mean, linewidth=2.4, label=model)


def _plot_confusion(
    frame: pd.DataFrame,
    path: Path,
    model: str,
    *,
    split_name: str = "test",
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(frame.to_numpy(), cmap="Blues")
    axis.set(
        title=f"{split_name.title()} Confusion Matrix - {model}",
        xlabel="Predicted class",
        ylabel="Actual class",
        xticks=range(len(frame.columns)),
        yticks=range(len(frame.index)),
    )
    axis.set_xticklabels(frame.columns, rotation=45, ha="right")
    axis.set_yticklabels(frame.index)
    figure.colorbar(image, ax=axis)
    _save_png(figure, path)
    plt.close(figure)


def _plot_feature_importance(frame: pd.DataFrame, path: Path, model: str) -> None:
    shown = frame.sort_values("importance", ascending=False).head(20)
    shown = shown.sort_values("importance")
    figure, axis = plt.subplots(figsize=(7.0, max(4.5, len(shown) * 0.3 + 1.5)))
    axis.barh(shown["feature"], shown["importance"], color="#0F6CBD")
    axis.set(title=f"Feature Importance - {model}", xlabel="Importance")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    _save_png(figure, path)
    plt.close(figure)


def _save_figure(figure, base: Path) -> None:
    figure.savefig(
        base.with_suffix(".png"),
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.18,
    )
    figure.savefig(
        base.with_suffix(".pdf"),
        dpi=320,
        bbox_inches="tight",
        pad_inches=0.18,
    )


def _save_png(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight")


def _pdf_text_page(
    pdf: PdfPages,
    title: str,
    lines: list[str],
    *,
    page_number: int,
) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.94, title, fontsize=20, weight="bold", color="#17324D")
    y = 0.88
    for line in lines:
        figure.text(0.09, y, line, fontsize=11, va="top", wrap=True)
        y -= 0.045
    _pdf_footer(figure, page_number)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _pdf_table_pages(pdf: PdfPages, frame: pd.DataFrame, *, start_page: int) -> None:
    shown_columns = [
        "Model",
        "Status",
        "Validation Accuracy",
        "Validation Macro-F1",
        "Test Accuracy",
        "Test Macro-F1",
        "ROC-AUC",
        "Saved",
    ]
    display = frame[shown_columns].copy() if not frame.empty else pd.DataFrame(
        [["Not available"] + [""] * (len(shown_columns) - 1)],
        columns=shown_columns,
    )
    for page_offset, start in enumerate(range(0, max(1, len(display)), 18)):
        chunk = display.iloc[start : start + 18].fillna("Not available")
        figure, axis = plt.subplots(figsize=(11.69, 8.27))
        axis.axis("off")
        axis.set_title("Model Performance Summary", fontsize=18, weight="bold", pad=18)
        cell_text = [
            [
                f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            ]
            for row in chunk.to_numpy()
        ]
        table = axis.table(
            cellText=cell_text,
            colLabels=shown_columns,
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        for column in range(len(shown_columns)):
            table[(0, column)].set_facecolor("#DCEBFA")
            table[(0, column)].set_text_props(weight="bold")
        _pdf_footer(figure, start_page + page_offset)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)


def _pdf_image_page(pdf: PdfPages, title: str, path: Path, page_number: int) -> None:
    figure, axis = plt.subplots(figsize=(8.27, 11.69))
    axis.axis("off")
    axis.set_title(title, fontsize=18, weight="bold", pad=16)
    axis.imshow(plt.imread(path))
    _pdf_footer(figure, page_number)
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def _pdf_classification_report_pages(
    pdf: PdfPages,
    model: str,
    frame: pd.DataFrame,
    *,
    start_page: int,
) -> int:
    display = frame.fillna("Not available")
    for page_offset, start in enumerate(range(0, max(1, len(display)), 20)):
        chunk = display.iloc[start : start + 20]
        figure, axis = plt.subplots(figsize=(11.69, 8.27))
        axis.axis("off")
        axis.set_title(
            f"Test Classification Report - {model}",
            fontsize=18,
            weight="bold",
            pad=18,
        )
        columns = list(chunk.columns)
        rows = [
            [
                f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)
                for value in row
            ]
            for row in chunk.to_numpy()
        ]
        table = axis.table(
            cellText=rows,
            colLabels=columns,
            loc="upper center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)
        for column in range(len(columns)):
            table[(0, column)].set_facecolor("#DCEBFA")
            table[(0, column)].set_text_props(weight="bold")
        _pdf_footer(figure, start_page + page_offset)
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)
    return start_page + max(1, int(np.ceil(max(1, len(display)) / 20)))


def _pdf_footer(figure, page_number: int) -> None:
    figure.text(
        0.5,
        0.025,
        f"Generated by {APP_NAME} | Version {__version__} | Page {page_number}",
        ha="center",
        fontsize=8,
        color="#5B6573",
    )


def _markdown_pairs(items: list[tuple[str, Any]]) -> str:
    return "\n".join(f"- {name}: {value}" for name, value in items)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Not available"
    display = frame.fillna("Not available").copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else value
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


def _markdown_image(path: Path | None, alt: str) -> str:
    return f"![{alt}]({path.name})" if path and path.exists() else "Not available"


def _markdown_image_group(paths: dict[str, Path]) -> list[str]:
    if not paths:
        return ["Not available"]
    lines = []
    for model, path in paths.items():
        if path.parent.name == "report":
            relative = path.name
        else:
            parts = list(path.parts)
            relative = path.name
            if "training" in parts:
                training_index = parts.index("training")
                relative = "../training/" + Path(
                    *parts[training_index + 1 :]
                ).as_posix()
        lines.extend([f"### {model}", "", f"![{model}]({relative})", ""])
    return lines


def _markdown_report_group(reports: dict[str, pd.DataFrame]) -> list[str]:
    if not reports:
        return ["Not available"]
    lines = []
    for model, frame in reports.items():
        lines.extend([f"### {model} - Test", "", _markdown_table(frame), ""])
    return lines


def _dataset_rows(
    project_dir: Path,
    config: ProjectConfig,
    split_rows: dict[str, int],
) -> int:
    subset = project_dir / "data" / "modeling_subset.csv"
    if subset.exists():
        try:
            return max(0, sum(1 for _ in subset.open(encoding="utf-8")) - 1)
        except OSError:
            pass
    return int(
        split_rows["train"] + split_rows["validation"] + split_rows["test"]
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_classification_report(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if frame.empty:
        return frame
    unnamed = [column for column in frame.columns if str(column).startswith("Unnamed:")]
    if unnamed:
        frame = frame.rename(columns={unnamed[0]: "Class"})
    for column in ("generated_by", "version", "generated_on"):
        if column in frame:
            frame = frame.drop(columns=column)
    return frame


def _confusion_from_predictions(path: Path) -> pd.DataFrame:
    predictions = _read_csv(path)
    if predictions.empty or not {
        "actual_class",
        "predicted_class",
    }.issubset(predictions.columns):
        return pd.DataFrame()
    labels = sorted(
        {
            str(value)
            for value in pd.concat(
                [
                    predictions["actual_class"],
                    predictions["predicted_class"],
                ],
                ignore_index=True,
            ).dropna()
        }
    )
    actual = predictions["actual_class"].map(str)
    predicted = predictions["predicted_class"].map(str)
    return pd.crosstab(actual, predicted).reindex(
        index=labels,
        columns=labels,
        fill_value=0,
    )


def _progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback:
        callback(message)
