import numpy as np

from app.utils.plotting import (
    plot_confusion_matrix_publication,
    plot_feature_importance_publication,
    plot_pr_curve_publication,
    plot_roc_curve_publication,
)


def test_publication_plotting_functions_create_png_pdf_and_csv(tmp_path):
    y_true = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.70, 0.20, 0.10],
            [0.10, 0.80, 0.10],
            [0.20, 0.65, 0.15],
            [0.05, 0.15, 0.80],
            [0.10, 0.20, 0.70],
        ]
    )

    plot_confusion_matrix_publication(
        np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
        ["Class A", "Class B", "Class C"],
        tmp_path,
        "Random Forest",
        "test",
    )
    plot_roc_curve_publication(
        y_true,
        probabilities,
        [0, 1, 2],
        tmp_path,
        "Random Forest",
        "test",
    )
    plot_pr_curve_publication(
        y_true,
        probabilities,
        [0, 1, 2],
        tmp_path,
        "Random Forest",
        "test",
    )
    plot_feature_importance_publication(
        [f"feature_{index}" for index in range(25)],
        np.linspace(0.01, 0.25, 25),
        tmp_path,
        "Random Forest",
    )

    for stem in (
        "confusion_matrix",
        "roc_curve",
        "pr_curve",
        "feature_importance",
    ):
        assert (tmp_path / f"{stem}.png").exists()
        assert (tmp_path / f"{stem}.pdf").exists()
    assert (tmp_path / "roc_curve.csv").exists()
    assert (tmp_path / "pr_curve.csv").exists()
    assert (tmp_path / "feature_importance.csv").exists()
