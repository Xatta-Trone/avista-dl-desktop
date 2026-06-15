"""Publication-quality matplotlib plots for saved model outputs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from app.branding import report_footer, report_footer_text


PUBLICATION_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
}


def plot_confusion_matrix_publication(
    matrix: np.ndarray,
    class_labels: list[Any],
    output_dir: str | Path,
    model_name: str,
    split_name: str,
    *,
    show_row_percentages: bool = True,
) -> None:
    """Save a labeled confusion matrix as PNG and PDF."""

    plt = _pyplot()
    matrix = np.asarray(matrix)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with _publication_style(plt):
        figure, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        labels = [str(label) for label in class_labels]
        axis.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted Class",
            ylabel="Actual Class",
            title=f"Confusion Matrix - {model_name} - {split_name.title()}",
        )
        plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix, dtype=float),
            where=row_totals != 0,
        )
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                text = str(int(matrix[row, column]))
                if show_row_percentages:
                    text += f"\n({normalized[row, column] * 100:.1f}%)"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                )
        _add_figure_footer(figure)
        _save_figure(figure, output_path / "confusion_matrix")
        plt.close(figure)


def plot_roc_curve_publication(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[Any],
    output_dir: str | Path,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Save one-vs-rest ROC curves, AUC labels, and curve data."""

    plt = _pyplot()
    curves = _classification_curves(y_true, probabilities, class_labels)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    with _publication_style(plt):
        figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        for class_name, binary_target, scores in curves:
            fpr, tpr, thresholds = roc_curve(binary_target, scores)
            area = float(auc(fpr, tpr))
            axis.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={area:.3f})")
            rows.extend(
                {
                    "class": class_name,
                    "fpr": float(x),
                    "tpr": float(y),
                    "threshold": float(threshold),
                    "auc": area,
                }
                for x, y, threshold in zip(fpr, tpr, thresholds)
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="No-skill")
        axis.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"ROC Curve - {model_name} - {split_name.title()}",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", frameon=True)
        _add_figure_footer(figure)
        _save_figure(figure, output_path / "roc_curve")
        plt.close(figure)
    frame = _add_report_columns(pd.DataFrame(rows))
    frame.to_csv(output_path / "roc_curve.csv", index=False)
    return frame


def plot_pr_curve_publication(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[Any],
    output_dir: str | Path,
    model_name: str,
    split_name: str,
) -> pd.DataFrame:
    """Save one-vs-rest precision-recall curves with AP labels."""

    plt = _pyplot()
    curves = _classification_curves(y_true, probabilities, class_labels)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows = []
    with _publication_style(plt):
        figure, axis = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        for class_name, binary_target, scores in curves:
            precision, recall, thresholds = precision_recall_curve(binary_target, scores)
            average_precision = float(average_precision_score(binary_target, scores))
            axis.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{class_name} (AP={average_precision:.3f})",
            )
            rows.extend(
                {
                    "class": class_name,
                    "recall": float(recall_value),
                    "precision": float(precision_value),
                    "threshold": (
                        float(thresholds[min(index, len(thresholds) - 1)])
                        if len(thresholds)
                        else None
                    ),
                    "average_precision": average_precision,
                }
                for index, (precision_value, recall_value) in enumerate(
                    zip(precision, recall)
                )
            )
        axis.set(
            xlabel="Recall",
            ylabel="Precision",
            title=f"Precision-Recall Curve - {model_name} - {split_name.title()}",
            xlim=(0, 1),
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower left", frameon=True)
        _add_figure_footer(figure)
        _save_figure(figure, output_path / "pr_curve")
        plt.close(figure)
    frame = _add_report_columns(pd.DataFrame(rows))
    frame.to_csv(output_path / "pr_curve.csv", index=False)
    return frame


def plot_feature_importance_publication(
    feature_names: list[str],
    importances: np.ndarray,
    output_dir: str | Path,
    model_name: str,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    """Save sorted feature importance data and horizontal bar plots."""

    plt = _pyplot()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    frame = _add_report_columns(pd.DataFrame(
        {"feature": feature_names, "importance": np.asarray(importances, dtype=float)}
    ).sort_values("importance", ascending=False))
    frame.to_csv(output_path / "feature_importance.csv", index=False)
    shown = frame.head(top_n).sort_values("importance")
    figure_height = max(4.5, 0.3 * len(shown) + 1.5)
    with _publication_style(plt):
        figure, axis = plt.subplots(
            figsize=(7.2, figure_height),
            constrained_layout=True,
        )
        axis.barh(shown["feature"], shown["importance"], color="#2f6f9f")
        axis.set(
            xlabel="Importance",
            ylabel="Feature",
            title=f"Feature Importance - {model_name}",
        )
        axis.grid(axis="x", alpha=0.25)
        _add_figure_footer(figure)
        _save_figure(figure, output_path / "feature_importance")
        plt.close(figure)
    return frame


def _classification_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_labels: list[Any],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    y = np.asarray(y_true)
    scores = np.asarray(probabilities)
    labels = list(class_labels)
    if len(labels) < 2:
        return []
    if len(labels) == 2:
        return [(str(labels[1]), (y == labels[1]).astype(int), scores[:, 1])]
    binary_targets = label_binarize(y, classes=labels)
    return [
        (str(class_name), binary_targets[:, index], scores[:, index])
        for index, class_name in enumerate(labels)
    ]


def _save_figure(figure: Any, base_path: Path) -> None:
    figure.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(base_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def _add_figure_footer(figure: Any) -> None:
    figure.text(
        0.01,
        0.01,
        report_footer_text().replace("\n", " | "),
        fontsize=7,
        color="0.35",
        ha="left",
        va="bottom",
    )


def _add_report_columns(frame: pd.DataFrame) -> pd.DataFrame:
    footer = report_footer()
    result = frame.copy()
    result["generated_by"] = footer["generated_by"]
    result["version"] = footer["version"]
    result["generated_on"] = footer["generated_on"]
    return result


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    return plt


@contextmanager
def _publication_style(plt: Any) -> Iterator[None]:
    with plt.rc_context(PUBLICATION_RCPARAMS):
        yield
