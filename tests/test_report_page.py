import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.project_config import ProjectConfig
from app.core.report_generator import (
    PERFORMANCE_COLUMNS,
    collect_model_performance,
    create_curve_comparison,
    create_deep_training_comparison,
    generate_project_report,
)


def _report_project(tmp_path: Path, *, include_outputs: bool = True) -> ProjectConfig:
    config = ProjectConfig(
        project_name="report-demo",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        target_column="target",
        feature_columns=["feature_a", "feature_b"],
        split_method="stratified",
        imbalance_method="none",
        selected_models=["decision_tree"],
        enable_cross_validation=True,
        cv_folds=3,
    )
    config.save()
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        {
            "feature_a": range(12),
            "feature_b": range(12, 24),
            "target": [0, 1] * 6,
        }
    ).to_csv(data_dir / "modeling_subset.csv", index=False)
    split_dir = tmp_path / "outputs" / "data_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "y_train_balanced.npy", np.array([0, 1] * 4))
    np.save(split_dir / "y_val.npy", np.array([0, 1]))
    np.save(split_dir / "y_test.npy", np.array([0, 1]))
    if include_outputs:
        _write_model_outputs(tmp_path)
    return config


def _write_model_outputs(tmp_path: Path) -> None:
    training_dir = tmp_path / "outputs" / "training"
    model_dir = training_dir / "DecisionTree"
    test_dir = model_dir / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model_name": "Decision Tree",
        "status": "trained",
        "train_metrics": {"accuracy": 0.95, "macro_f1": 0.94},
        "validation_metrics": {"accuracy": 0.90, "macro_f1": 0.89},
        "test_metrics": {"accuracy": 0.88, "macro_f1": 0.87, "roc_auc": 0.92},
        "cv_summary": {
            "accuracy": {"mean": 0.91, "std": 0.02},
            "macro_f1": {"mean": 0.90, "std": 0.03},
        },
        "saved": True,
    }
    (training_dir / "training_results.json").write_text(
        json.dumps({"results": [result]}),
        encoding="utf-8",
    )
    (model_dir / "model_config.json").write_text(
        json.dumps({"display_name": "Decision Tree"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "class": ["0", "0", "0", "1", "1", "1"],
            "fpr": [0.0, 0.2, 1.0, 0.0, 0.1, 1.0],
            "tpr": [0.0, 0.8, 1.0, 0.0, 0.9, 1.0],
            "auc": [0.9] * 6,
        }
    ).to_csv(test_dir / "roc_curve.csv", index=False)
    pd.DataFrame(
        {
            "class": ["0", "0", "0", "1", "1", "1"],
            "recall": [0.0, 0.8, 1.0, 0.0, 0.9, 1.0],
            "precision": [1.0, 0.9, 0.5, 1.0, 0.85, 0.5],
            "average_precision": [0.91] * 6,
        }
    ).to_csv(test_dir / "pr_curve.csv", index=False)
    for split_name, matrix, accuracy in (
        ("train", [[7, 1], [0, 8]], 0.94),
        ("validation", [[1, 0], [1, 1]], 0.67),
        ("test", [[4, 1], [0, 5]], 0.90),
    ):
        split_dir = model_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            matrix,
            index=["no", "yes"],
            columns=["no", "yes"],
        ).to_csv(split_dir / "confusion_matrix.csv")
        pd.DataFrame(
            {
                "class": ["no", "yes", "accuracy", "macro avg"],
                "precision": [0.9, 0.9, accuracy, 0.9],
                "recall": [0.9, 0.9, accuracy, 0.9],
                "f1-score": [0.9, 0.9, accuracy, 0.9],
                "support": [5, 5, 10, 10],
            }
        ).to_csv(split_dir / "classification_report.csv", index=False)
    pd.DataFrame(
        {"feature": ["feature_a", "feature_b"], "importance": [0.7, 0.3]}
    ).to_csv(model_dir / "feature_importance.csv", index=False)
    deep_dir = training_dir / "MambaAttention"
    deep_dir.mkdir()
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [0.9, 0.7, 0.5],
            "validation_accuracy": [0.55, 0.65, 0.75],
        }
    ).to_csv(deep_dir / "training_history.csv", index=False)


def test_report_page_loads_and_generate_button_exists(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    window = MainWindow(config)

    assert window.pages[-1][0] == "Report"
    assert window.report_page.generate_button.text() == "Generate Report"
    assert window.report_page.summary_card.objectName() == "reportSummaryCard"
    assert window.report_page.performance_card.objectName() == "reportPerformanceCard"
    assert not window.nav_buttons[-1].icon().isNull()
    window.close()
    assert app is not None


def test_report_generation_creates_markdown_pdf_and_expected_figures(tmp_path):
    config = _report_project(tmp_path)

    artifacts = generate_project_report(config)

    assert artifacts.markdown_path.exists()
    assert artifacts.pdf_path.exists()
    assert artifacts.pdf_path.stat().st_size > 1000
    assert artifacts.performance_csv.exists()
    for filename in (
        "roc_curve_comparison.png",
        "roc_curve_comparison.pdf",
        "pr_curve_comparison.png",
        "pr_curve_comparison.pdf",
        "deep_training_curves.png",
        "deep_training_curves.pdf",
    ):
        assert (artifacts.output_dir / filename).exists()
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "# AVISTA Model Report" in markdown
    assert "## Reproducibility Metadata" in markdown
    assert "## Classification Reports" in markdown
    assert (
        "Unless otherwise stated, model comparison figures and diagnostic "
        "tables are based on the test set."
    ) in markdown
    assert "Generated by AVISTA" in markdown


def test_report_generation_missing_model_outputs_does_not_crash(tmp_path):
    config = _report_project(tmp_path, include_outputs=False)

    artifacts = generate_project_report(config)

    assert artifacts.markdown_path.exists()
    assert artifacts.pdf_path.exists()
    assert artifacts.performance.empty
    assert "Not available" in artifacts.markdown_path.read_text(encoding="utf-8")


def test_report_model_performance_table_combines_saved_results(tmp_path):
    _report_project(tmp_path)

    frame = collect_model_performance(tmp_path / "outputs" / "training")

    assert list(frame.columns) == PERFORMANCE_COLUMNS
    assert len(frame) == 1
    assert frame.iloc[0]["Model"] == "Decision Tree"
    assert frame.iloc[0]["Test Macro-F1"] == 0.87
    assert frame.iloc[0]["CV Accuracy Std"] == 0.02


def test_report_roc_and_pr_comparisons_have_no_sd_band(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    _report_project(tmp_path)
    output_dir = tmp_path / "outputs" / "report"
    output_dir.mkdir(parents=True)
    monkeypatch.setattr(
        Axes,
        "fill_between",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Report comparison figures must not use SD bands.")
        ),
    )

    for curve_name in ("roc", "pr"):
        path = create_curve_comparison(
            tmp_path / "outputs" / "training",
            output_dir,
            curve_name=curve_name,
        )
        assert path == output_dir / f"{curve_name}_curve_comparison.png"
        assert path.exists()
        assert path.with_suffix(".pdf").exists()


def test_report_deep_curves_have_no_sd_band(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    _report_project(tmp_path)
    output_dir = tmp_path / "outputs" / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        Axes,
        "fill_between",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Deep report figures must not use SD bands.")
        ),
    )

    path = create_deep_training_comparison(
        tmp_path / "outputs" / "training",
        output_dir,
    )

    assert path is not None
    assert path.exists()


def test_report_deep_curves_handle_missing_history_files(tmp_path):
    training_dir = tmp_path / "outputs" / "training"
    training_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs" / "report"
    output_dir.mkdir(parents=True)

    assert create_deep_training_comparison(training_dir, output_dir) is None


def test_report_page_renders_combined_performance_table(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    window = MainWindow(config)
    page = window.report_page
    page.refresh()

    assert page.performance_table.rowCount() == 1
    assert page.performance_table.item(0, 0).text() == "Decision Tree"
    assert page.performance_table.item(0, 7).text() == "0.8700"
    window.close()
    assert app is not None


def test_report_preview_is_half_width_capped_and_preserves_full_aspect(tmp_path):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    artifacts = generate_project_report(config)
    image = QImage(str(artifacts.generated_files["roc"]))
    assert image.width() > 2500
    assert image.height() > 1500

    window = MainWindow(config)
    page = window.report_page
    window.resize(1500, 900)
    window.navigate_to(len(window.pages) - 1)
    window.show()
    app.processEvents()
    page._set_preview(page.roc_preview, artifacts.generated_files["roc"])
    page.preview_containers[page.roc_preview].update_preview_width()
    app.processEvents()
    source_ratio = image.width() / image.height()
    shown = page.roc_preview.pixmap()
    assert shown is not None and not shown.isNull()
    assert abs((shown.width() / shown.height()) - source_ratio) < 0.02
    container_width = page.preview_containers[page.roc_preview].width()
    assert page.roc_preview.width() <= 900
    assert page.roc_preview.width() <= int(container_width * 0.52)
    assert page.roc_preview.width() >= int(container_width * 0.45)
    assert page.roc_preview.x() > 0
    window.close()
    assert app is not None


def test_report_diagnostic_preview_uses_half_width_and_stacks_when_narrow(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    window = MainWindow(config)
    page = window.report_page
    window.resize(1500, 900)
    window.navigate_to(len(window.pages) - 1)
    window.show()
    app.processEvents()
    page._update_diagnostic_layout()
    app.processEvents()

    assert not page._diagnostic_stacked
    body_width = page.diagnostic_body.width()
    assert page.diagnostic_preview_container.width() <= int(body_width * 0.55)
    assert page.diagnostic_confusion.width() <= 720

    window.resize(820, 760)
    app.processEvents()
    page._update_diagnostic_layout()
    app.processEvents()

    assert page._diagnostic_stacked
    assert page.diagnostic_grid.itemAtPosition(0, 0).widget() is (
        page.diagnostic_preview_container
    )
    assert page.diagnostic_grid.itemAtPosition(1, 0).widget() is (
        page.diagnostic_report_panel
    )
    window.close()
    assert app is not None


def test_report_model_diagnostic_card_updates_model_and_split(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    window = MainWindow(config)
    page = window.report_page

    assert page.diagnostic_card.objectName() == "reportDiagnosticCard"
    assert page.split_selector.currentText() == "Test"
    assert page.model_selector.currentText() == "Decision Tree"
    assert page.diagnostic_table.rowCount() == 4
    assert page.diagnostic_table.item(2, 0).text() == "accuracy"
    test_pixmap_key = page.diagnostic_confusion.pixmap().cacheKey()

    page.split_selector.setCurrentText("Train")

    assert page.diagnostic_table.rowCount() == 4
    assert page.diagnostic_confusion.pixmap().cacheKey() != test_pixmap_key
    assert page.diagnostic_warning.isHidden()
    window.close()
    assert app is not None


def test_report_missing_selected_diagnostics_show_warning_safely(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    validation_dir = (
        tmp_path / "outputs" / "training" / "DecisionTree" / "validation"
    )
    (validation_dir / "confusion_matrix.csv").unlink()
    (validation_dir / "classification_report.csv").unlink()
    window = MainWindow(config)
    page = window.report_page

    page.split_selector.setCurrentText("Validation")

    assert not page.diagnostic_warning.isHidden()
    assert "Not available" in page.diagnostic_warning.text()
    assert page.diagnostic_table.rowCount() == 0
    assert page.diagnostic_confusion.text() == "Not available"
    window.close()
    assert app is not None


def test_report_page_output_folder_opens(tmp_path, monkeypatch):
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = _report_project(tmp_path)
    report_dir = tmp_path / "outputs" / "report"
    report_dir.mkdir(parents=True)
    opened = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    window = MainWindow(config)

    window.report_page.open_report_folder()

    assert [Path(path).resolve() for path in opened] == [report_dir.resolve()]
    window.close()
    assert app is not None


def test_report_page_success_notification_auto_dismisses(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow(_report_project(tmp_path))
    page = window.report_page

    page._show_notification("Report generated successfully.", "success")

    assert page.notification_timer.isActive()
    assert page.notification_timer.interval() == 5000
    page.notification_timer.timeout.emit()
    assert page.notification_card.isHidden()
    window.close()
    assert app is not None
