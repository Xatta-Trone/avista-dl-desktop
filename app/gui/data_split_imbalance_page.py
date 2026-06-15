"""Train/validation/test split and imbalance configuration page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QSize, QTimer, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.imbalance import apply_imbalance_strategy
from app.core.edge_case_checker import selected_column_missing_counts
from app.core.preprocessing import build_preprocessing_pipeline, save_artifacts
from app.core.project_config import ProjectConfig
from app.core.splitter import (
    build_class_coverage_report,
    class_coverage_issues,
    split_data_three_way,
)
from app.core.target_encoding import (
    decode_target,
    encode_target,
    load_or_fit_target_encoder,
    save_target_encoder,
)
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon

SUCCESS_COLOR = "#16A34A"
WARNING_COLOR = "#D97706"
ERROR_COLOR = "#DC2626"
NEUTRAL_ICON = "#6B7280"


class SpinArrowButton(QToolButton):
    def __init__(
        self,
        icon_name: str,
        callback,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.icon_name = icon_name
        self.clicked.connect(callback)
        self.setObjectName("splitSpinArrowButton")
        self.setFixedSize(20, 17)
        self.setIconSize(QSize(11, 11))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_icon(NEUTRAL_ICON)

    def enterEvent(self, event) -> None:
        self._set_icon(PRIMARY)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_icon(NEUTRAL_ICON)
        super().leaveEvent(event)

    def _set_icon(self, color: str) -> None:
        self.setIcon(get_fa_icon(self.icon_name, color))


class ComboArrowButton(QToolButton):
    def __init__(self, combo: QComboBox, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = combo
        self.clicked.connect(combo.showPopup)
        self.setObjectName("splitComboArrowButton")
        self.setFixedSize(28, 36)
        self.setIconSize(QSize(13, 13))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_icon(NEUTRAL_ICON)

    def enterEvent(self, event) -> None:
        self._set_icon(PRIMARY)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_icon(NEUTRAL_ICON)
        super().leaveEvent(event)

    def _set_icon(self, color: str) -> None:
        self.setIcon(get_fa_icon("fa6s.angle-down", color))


RATIO_PRESETS = {
    "Light Balancing": 0.40,
    "Moderate Balancing": 0.50,
    "Strong Balancing": 0.70,
}

SPLIT_METHOD_HELP = {
    "random": "Randomly split rows into train/validation/test.",
    "stratified": "Preserve target class proportions across train/validation/test.",
    "group": "Keep all records from the same group together.",
    "stratified_group": "Preserve class proportions while keeping groups together.",
    "time": "Use chronological order. Older records for training, newer records for validation/test.",
}

BALANCING_PRESET_HELP = {
    "Light Balancing": "Increase minority classes slightly.",
    "Moderate Balancing": "Increase minority classes to a medium level.",
    "Strong Balancing": "Increase minority classes aggressively.",
    "Custom": "User defines target class sizes or ratios.",
}

SAVED_SPLIT_FILES = (
    "imbalance_config.json",
    "split_indices.json",
    "class_distribution_before.csv",
    "class_distribution_after.csv",
    "class_coverage_report.csv",
    "preprocessing_artifact.joblib",
    "X_train_balanced.npy",
    "y_train_balanced.npy",
    "X_val.npy",
    "y_val.npy",
    "X_test.npy",
    "y_test.npy",
)

CLASSIFICATION_TARGET_FILES = (
    "y_train_encoded.npy",
    "y_train_balanced_encoded.npy",
    "y_val_encoded.npy",
    "y_test_encoded.npy",
    "y_train_original.npy",
    "y_train_balanced_original.npy",
    "y_val_original.npy",
    "y_test_original.npy",
    "target_label_encoder.joblib",
    "target_label_mapping.json",
)


class DataSplitImbalancePage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self._last_config_signature: tuple[str | None, tuple[str, ...]] | None = None
        self.success_notification_timer = QTimer(self)
        self.success_notification_timer.setSingleShot(True)
        self.success_notification_timer.setInterval(5000)
        self.success_notification_timer.timeout.connect(self._dismiss_success_notification)

        self.current_target_label = QLabel(
            "No target column selected. Please confirm Column Configuration first."
        )
        self.current_target_label.setObjectName("splitTargetValue")
        self.train_percent = self._percentage_input(70)
        self.validation_percent = self._percentage_input(10)
        self.test_percent = self._percentage_input(20)
        for spin in (self.train_percent, self.validation_percent, self.test_percent):
            spin.valueChanged.connect(self._update_percentage_validator)
        self.random_seed = QSpinBox()
        self.random_seed.setRange(0, 2_147_483_647)
        self.random_seed.setValue(42)
        self.random_seed.setFixedWidth(128)
        self.split_method = QComboBox()
        for method, description in SPLIT_METHOD_HELP.items():
            self.split_method.addItem(method)
            self.split_method.setItemData(
                self.split_method.count() - 1,
                description,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.split_method.currentTextChanged.connect(self._update_split_tooltip)
        self._update_split_tooltip()
        self.split_method.setFixedWidth(220)

        self.imbalance_method = QComboBox()
        self.imbalance_method.addItems(
            ["none", "random_oversample", "random_undersample", "smote", "smote_nc"]
        )
        self.imbalance_method.currentTextChanged.connect(self._update_balancing_visibility)
        self.imbalance_method.setFixedWidth(220)
        self.use_class_weights = QCheckBox("Use class weights if supported by selected model")
        self.ratio_preset = QComboBox()
        for preset, description in BALANCING_PRESET_HELP.items():
            self.ratio_preset.addItem(preset)
            self.ratio_preset.setItemData(
                self.ratio_preset.count() - 1,
                description,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.ratio_preset.setCurrentText("Moderate Balancing")
        self.ratio_preset.currentTextChanged.connect(self._update_custom_visibility)
        self.ratio_preset.currentTextChanged.connect(self._update_preset_tooltip)
        self._update_preset_tooltip()
        self.ratio_preset.setFixedWidth(220)
        self.custom_ratio_input = QLineEdit()
        self.custom_ratio_input.setPlaceholderText('JSON, e.g. {"class_a": 0.5, "class_b": 120}')
        self.custom_ratio_input.setVisible(False)
        self.percent_validator_label = QLabel("")
        self.percent_validator_label.setObjectName("splitPercentValidator")
        self.percent_validator_label.setWordWrap(True)

        self.before_distribution_tables = {
            "Full Dataset": self._distribution_table(),
            "Train Set": self._distribution_table(),
            "Validation Set": self._distribution_table(),
            "Test Set": self._distribution_table(),
        }
        self.after_distribution_tables = {
            "Train Set (Balanced)": self._distribution_table(),
            "Validation Set": self._distribution_table(),
            "Test Set": self._distribution_table(),
        }
        self.class_coverage_table = self._class_coverage_table()

        self.balancing_preset_container = QWidget()
        preset_form = QFormLayout(self.balancing_preset_container)
        preset_form.setContentsMargins(0, 0, 0, 0)
        preset_form.addRow("Balancing preset", self._combobox_control(self.ratio_preset))
        preset_form.addRow("Custom class targets", self.custom_ratio_input)
        self._update_balancing_visibility()

        self.feedback_card = self._feedback_card()

        self.warning_card = QFrame()
        self.warning_card.setObjectName("splitWarningCard")
        self.warning_card.setVisible(False)
        warning_layout = QHBoxLayout(self.warning_card)
        warning_layout.setContentsMargins(12, 8, 12, 8)
        warning_layout.setSpacing(8)
        self.warning_icon_label = QLabel()
        self.warning_icon_label.setObjectName("dataSplitWarningIcon")
        self.warning_icon_label.setFixedSize(18, 18)
        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        warning_layout.addWidget(self.warning_icon_label)
        warning_layout.addWidget(self.warning_label, stretch=1)

        content = QWidget()
        content.setObjectName("dataSplitContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)
        title = QLabel("Data Split & Imbalance")
        title.setObjectName("dataSplitTitle")
        content_layout.addWidget(title)

        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)
        self.split_configuration_card = self._split_configuration_card()
        self.before_balancing_card = self._before_balancing_card()
        self.class_coverage_card = self._class_coverage_card()
        self.imbalance_handling_card = self._imbalance_handling_card()
        self.after_balancing_card = self._after_balancing_card()
        self.confirmation_status_card = self._confirmation_status_card()
        for card in (
            self.split_configuration_card,
            self.before_balancing_card,
            self.class_coverage_card,
            self.imbalance_handling_card,
            self.after_balancing_card,
            self.confirmation_status_card,
        ):
            controls_layout.addWidget(card)

        self.empty_state_card = self._empty_state_card()
        content_layout.addWidget(self.empty_state_card, alignment=Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.controls_widget)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self._update_percentage_validator()
        self._apply_style()

    def refresh(self) -> None:
        config = self._reload_latest_config()
        if not config:
            self.current_target_label.setText(
                "No target column selected. Please confirm Column Configuration first."
            )
            self._show_empty_state()
            self._clear_split_state()
            return

        signature = (config.target_column, tuple(config.feature_columns or []))
        config_changed = signature != self._last_config_signature
        self._last_config_signature = signature

        if config.target_column:
            self.current_target_label.setText(f"Current target column: {config.target_column}")
            self._show_controls()
        else:
            self.current_target_label.setText(
                "No target column selected. Please confirm Column Configuration first."
            )
            self._show_empty_state()

        if config_changed:
            self._clear_split_state()

        self.train_percent.setValue(int(config.train_percent))
        self.validation_percent.setValue(int(config.validation_percent))
        self.test_percent.setValue(int(config.test_percent))
        self.random_seed.setValue(int(getattr(config, "random_seed", 42)))
        self._update_percentage_validator()
        self._select_combo(self.split_method, config.split_method or "random")
        self._select_combo(self.imbalance_method, config.imbalance_method or "none")
        self.use_class_weights.setChecked(bool(config.use_class_weights))
        self._select_combo(self.ratio_preset, self._preset_label(config.smote_ratio_preset))
        self._update_custom_visibility()
        if not config.target_column:
            self._clear_split_state()
            return

        saved_status = self._load_saved_results(config)
        if saved_status == "loaded":
            return
        if saved_status == "target_changed":
            self._recompute_before_distributions()
            self._show_reload_message(
                "Target column changed. Please confirm split and imbalance again.",
                warning=True,
            )
            return
        self._recompute_before_distributions()

    def confirm_split_and_imbalance(self) -> None:
        config = self._reload_latest_config()
        df = self.main_window.dataframe
        error = self._prerequisite_error(config, df)
        if error:
            self._show_error(error)
            return

        train = self.train_percent.value()
        validation = self.validation_percent.value()
        test = self.test_percent.value()
        total_percent = train + validation + test
        if total_percent != 100:
            self._show_error(
                f"Train + validation + test must equal 100%. Current total: {total_percent}%."
            )
            return

        config.train_percent = float(train)
        config.validation_percent = float(validation)
        config.test_percent = float(test)
        config.random_seed = int(self.random_seed.value())
        config.split_method = self.split_method.currentText()
        config.imbalance_method = self.imbalance_method.currentText()
        config.use_class_weights = self.use_class_weights.isChecked()
        config.smote_ratio_preset = self._preset_key()

        try:
            X, y, artifacts = self._build_latest_xy(config, df)
            split = split_data_three_way(X, y, df, config)
            coverage = build_class_coverage_report(
                y,
                split["y_train"],
                split["y_val"],
                split["y_test"],
            )
            coverage_issues = class_coverage_issues(coverage)
            output_dir = Path(config.project_dir) / "outputs" / "data_split"
            classification_target = (
                str(config.task_type or "auto").strip().lower() != "regression"
            )
            target_encoder = (
                load_or_fit_target_encoder(y, output_dir, config.target_column)
                if classification_target
                else None
            )
            encoded_split = dict(split)
            if target_encoder is not None:
                for key in ("y_train", "y_val", "y_test"):
                    encoded_split[key] = encode_target(target_encoder, split[key])
            sampling_strategy = self._sampling_strategy(
                encoded_split["y_train"],
                target_encoder,
            )
            self._store_imbalance_options(config, sampling_strategy)
            balanced = apply_imbalance_strategy(
                split["X_train"], encoded_split["y_train"], artifacts, config
            )
            if not balanced["imbalance_info"]["success"]:
                error_message = balanced["imbalance_info"].get("error") or balanced["imbalance_info"]["message"]
                raise ValueError(f"Imbalance handling failed: {error_message}")
            balanced["imbalance_info"]["warnings"] = (
                [
                    issue["message"]
                    for issue in coverage_issues
                    if issue["level"] == "warning"
                ]
                + balanced["imbalance_info"].get("warnings", [])
            )
            balanced_original = (
                decode_target(target_encoder, balanced["y_resampled"])
                if target_encoder is not None
                else np.asarray(balanced["y_resampled"])
            )
            self._save_outputs(
                output_dir,
                split,
                encoded_split,
                balanced,
                balanced_original,
                y,
                config,
                sampling_strategy,
                coverage,
                coverage_issues,
                artifacts,
                target_encoder,
            )
            config_path = config.save()
        except Exception as exc:
            self._show_error(str(exc))
            return

        before = self._distribution_frame(
            {
                "Full Dataset": y,
                "Train Set": split["y_train"],
                "Validation Set": split["y_val"],
                "Test Set": split["y_test"],
            }
        )
        after = self._distribution_frame(
            {
                "Train Set (Balanced)": balanced_original,
                "Validation Set": split["y_val"],
                "Test Set": split["y_test"],
            }
        )
        self._populate_distribution_tables(before, self.before_distribution_tables)
        self._populate_distribution_tables(after, self.after_distribution_tables)
        self._populate_class_coverage_table(coverage)
        self._show_success(
            split=split,
            balanced_rows=len(balanced["y_resampled"]),
            output_dir=output_dir,
            config_path=config_path,
            sampling_strategy=sampling_strategy,
            issues=coverage_issues,
            warnings=balanced["imbalance_info"].get("warnings", []),
        )

    def _save_outputs(
        self,
        output_dir: Path,
        split: dict[str, Any],
        encoded_split: dict[str, Any],
        balanced: dict[str, Any],
        balanced_original: np.ndarray,
        full_y: pd.Series,
        config: Any,
        sampling_strategy: Any,
        coverage: pd.DataFrame,
        coverage_issues: list[dict[str, str]],
        preprocessing_artifacts: Any,
        target_encoder: Any,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        indices = {
            "target_column": config.target_column,
            "task_type": config.task_type,
            "train_index": split["train_index"],
            "validation_index": split["validation_index"],
            "test_index": split["test_index"],
            "split_info": split["split_info"],
        }
        (output_dir / "split_indices.json").write_text(
            json.dumps(indices, indent=2, default=_json_scalar), encoding="utf-8"
        )

        before = self._distribution_frame(
            {
                "Full Dataset": full_y,
                "Train Set": split["y_train"],
                "Validation Set": split["y_val"],
                "Test Set": split["y_test"],
            }
        )
        after = self._distribution_frame(
            {
                "Train Set (Balanced)": balanced_original,
                "Validation Set": split["y_val"],
                "Test Set": split["y_test"],
            }
        )
        before.to_csv(output_dir / "class_distribution_before.csv", index=False)
        after.to_csv(output_dir / "class_distribution_after.csv", index=False)
        coverage.to_csv(output_dir / "class_coverage_report.csv", index=False)
        save_artifacts(
            preprocessing_artifacts,
            output_dir / "preprocessing_artifact.joblib",
        )

        imbalance_config = {
            "target_column": config.target_column,
            "task_type": config.task_type,
            "original_train_distribution": _json_counts(split["y_train"]),
            "balanced_train_distribution": _json_counts(balanced_original),
            "imbalance_method": config.imbalance_method,
            "balancing_preset": config.smote_ratio_preset,
            "random_seed": config.random_seed,
            "sampling_strategy_used": sampling_strategy,
            "class_weights_enabled": config.use_class_weights,
            "warnings": [
                issue["message"] for issue in coverage_issues if issue["level"] == "warning"
            ]
            + balanced["imbalance_info"].get("warnings", []),
            "errors": [
                issue["message"] for issue in coverage_issues if issue["level"] == "fatal"
            ]
            + (
                [balanced["imbalance_info"]["error"]]
                if balanced["imbalance_info"].get("error")
                else []
            ),
            "imbalance_info": balanced["imbalance_info"],
        }
        (output_dir / "imbalance_config.json").write_text(
            json.dumps(imbalance_config, indent=2, default=_json_scalar), encoding="utf-8"
        )

        np.save(output_dir / "X_train_balanced.npy", np.asarray(balanced["X_resampled"]))
        np.save(output_dir / "y_train_balanced.npy", np.asarray(balanced_original))
        np.save(output_dir / "X_val.npy", np.asarray(split["X_val"]))
        np.save(output_dir / "y_val.npy", np.asarray(split["y_val"]))
        np.save(output_dir / "X_test.npy", np.asarray(split["X_test"]))
        np.save(output_dir / "y_test.npy", np.asarray(split["y_test"]))
        if target_encoder is not None:
            np.save(output_dir / "y_train_encoded.npy", np.asarray(encoded_split["y_train"]))
            np.save(
                output_dir / "y_train_balanced_encoded.npy",
                np.asarray(balanced["y_resampled"], dtype=np.int64),
            )
            np.save(output_dir / "y_val_encoded.npy", np.asarray(encoded_split["y_val"]))
            np.save(output_dir / "y_test_encoded.npy", np.asarray(encoded_split["y_test"]))
            np.save(output_dir / "y_train_original.npy", np.asarray(split["y_train"]))
            np.save(
                output_dir / "y_train_balanced_original.npy",
                np.asarray(balanced_original),
            )
            np.save(output_dir / "y_val_original.npy", np.asarray(split["y_val"]))
            np.save(output_dir / "y_test_original.npy", np.asarray(split["y_test"]))
            save_target_encoder(target_encoder, output_dir)

    def _sampling_strategy(self, y_train: pd.Series, target_encoder: Any = None) -> Any:
        method = self.imbalance_method.currentText()
        if method not in {
            "random_oversample",
            "random_undersample",
            "smote",
            "smote_nc",
        }:
            return "auto"

        counts = y_train.value_counts()
        if counts.empty or len(counts) < 2:
            return "auto"

        majority_count = int(counts.max())
        minority_count = int(counts.min())
        if method == "random_oversample":
            return {
                _json_scalar(class_name): majority_count
                for class_name, count in counts.items()
                if int(count) < majority_count
            }
        if method == "random_undersample":
            return {
                _json_scalar(class_name): minority_count
                for class_name, count in counts.items()
                if int(count) > minority_count
            }

        if self.ratio_preset.currentText() == "Custom":
            return self._custom_sampling_strategy(counts, target_encoder)

        ratio = RATIO_PRESETS[self.ratio_preset.currentText()]
        majority_class = counts.idxmax()
        strategy = {}
        for class_name, count in counts.items():
            if class_name == majority_class:
                continue
            target_count = max(int(count), int(majority_count * ratio))
            if target_count > int(count):
                strategy[_json_scalar(class_name)] = target_count
        return strategy

    def _custom_sampling_strategy(
        self,
        counts: pd.Series,
        target_encoder: Any = None,
    ) -> dict[Any, int]:
        text = self.custom_ratio_input.text().strip()
        if not text:
            raise ValueError("Enter custom per-class target ratios or counts as JSON.")
        try:
            values = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid custom ratio JSON: {exc}") from exc
        if not isinstance(values, dict) or not values:
            raise ValueError("Custom ratio must be a non-empty JSON object.")

        majority_count = int(counts.max())
        class_lookup = {}
        for class_name in counts.index:
            display_name = class_name
            if target_encoder is not None:
                display_name = decode_target(target_encoder, [class_name])[0]
            class_lookup[str(_json_scalar(display_name))] = class_name
        strategy = {}
        for class_label, target in values.items():
            if str(class_label) not in class_lookup:
                raise ValueError(f"Custom ratio class '{class_label}' is not present in the training target.")
            numeric_target = float(target)
            target_count = int(majority_count * numeric_target) if numeric_target <= 1 else int(numeric_target)
            original_class = class_lookup[str(class_label)]
            current_count = int(counts[original_class])
            target_count = max(current_count, target_count)
            if target_count > current_count:
                strategy[_json_scalar(original_class)] = target_count
        return strategy

    def _store_imbalance_options(self, config: Any, sampling_strategy: Any) -> None:
        options = dict(config.preprocessing_options or {})
        imbalance_options = dict(options.get("imbalance", {}) or {})
        imbalance_options["sampling_strategy"] = sampling_strategy
        options["imbalance"] = imbalance_options
        options["random_seed"] = int(self.random_seed.value())
        config.preprocessing_options = options

    def _distribution_frame(self, datasets: dict[str, pd.Series]) -> pd.DataFrame:
        rows = []
        for split_name, values in datasets.items():
            display_values = pd.Series(values).map(_display_class_label)
            counts = display_values.value_counts(dropna=False, sort=False)
            counts = counts.sort_index(key=lambda index: index.astype(str))
            total = int(counts.sum())
            for class_name, count in counts.items():
                rows.append(
                    {
                        "split": split_name,
                        "class": str(class_name),
                        "count": int(count),
                        "percent": float(count / total * 100) if total else 0.0,
                    }
                )
        return pd.DataFrame(rows, columns=["split", "class", "count", "percent"])

    def _populate_distribution_tables(
        self,
        distribution: pd.DataFrame,
        tables: dict[str, QTableWidget],
    ) -> None:
        for split_name, table in tables.items():
            rows = distribution[distribution["split"] == split_name]
            table.setRowCount(len(rows))
            for row_index, (_, row) in enumerate(rows.iterrows()):
                class_item = QTableWidgetItem(str(row["class"]))
                class_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                count_item = QTableWidgetItem(str(row["count"]))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                percent_item = QTableWidgetItem(f"{row['percent']:.1f}%")
                percent_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, 0, class_item)
                table.setItem(row_index, 1, count_item)
                table.setItem(row_index, 2, percent_item)
            table.resizeRowsToContents()

    def _prerequisite_error(self, config: Any, df: pd.DataFrame | None) -> str | None:
        if config is None:
            return "Load or create a project first."
        if df is None:
            return "Load a dataset first."
        if not config.feature_columns:
            return "Confirm modeling columns before configuring data splits."
        if not config.target_column:
            return "Confirm a target column before configuring data splits."
        missing = [
            column
            for column in list(config.feature_columns) + [config.target_column]
            if column not in df.columns
        ]
        if missing:
            return f"Configured columns are missing from the loaded dataset: {missing}"
        missing_values = selected_column_missing_counts(df, config)
        if missing_values:
            details = "; ".join(
                f"Column '{column}' contains {count} empty values ({percentage:.1f}%)."
                for column, (count, percentage) in missing_values.items()
            )
            return f"{details} Please clean selected modeling columns before splitting."
        return None

    def _reload_latest_config(self) -> ProjectConfig | None:
        config = self.main_window.config
        if config is None:
            return None

        config_path = config.project_file
        if config_path.exists():
            try:
                config = ProjectConfig.load(config_path)
                self.main_window.config = config
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        return config

    def _load_saved_results(self, config: ProjectConfig) -> str:
        output_dir = Path(config.project_dir) / "outputs" / "data_split"
        imbalance_path = output_dir / "imbalance_config.json"
        indices_path = output_dir / "split_indices.json"
        before_path = output_dir / "class_distribution_before.csv"
        after_path = output_dir / "class_distribution_after.csv"
        coverage_path = output_dir / "class_coverage_report.csv"

        if not indices_path.exists() and not imbalance_path.exists():
            return "missing"

        metadata = []
        for path in (imbalance_path, indices_path):
            if not path.exists():
                continue
            try:
                metadata.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                return "missing"

        saved_targets = {
            item.get("target_column")
            for item in metadata
            if item.get("target_column") is not None
        }
        if not saved_targets or saved_targets != {config.target_column}:
            self._clear_distribution_tables(self.after_distribution_tables)
            return "target_changed"

        required_files = list(SAVED_SPLIT_FILES)
        if str(config.task_type or "auto").strip().lower() != "regression":
            required_files.extend(CLASSIFICATION_TARGET_FILES)
        if not all((output_dir / name).exists() for name in required_files):
            return "missing"

        try:
            before = pd.read_csv(before_path)
            after = pd.read_csv(after_path)
            coverage = pd.read_csv(coverage_path)
        except (OSError, ValueError, pd.errors.ParserError):
            return "missing"

        required_columns = {"split", "class", "count", "percent"}
        if not required_columns.issubset(before.columns) or not required_columns.issubset(after.columns):
            return "missing"
        coverage_columns = {
            "Class",
            "Full count",
            "Train count",
            "Validation count",
            "Test count",
            "Status",
        }
        if not coverage_columns.issubset(coverage.columns):
            return "missing"

        self._clear_split_state()
        self._populate_distribution_tables(before, self.before_distribution_tables)
        self._populate_distribution_tables(after, self.after_distribution_tables)
        self._populate_class_coverage_table(coverage)
        self._show_reload_message(
            f"Saved split/imbalance data loaded for target column: {config.target_column}"
        )
        return "loaded"

    def _build_latest_xy(
        self,
        config: ProjectConfig,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, Any]:
        target = config.target_column
        features = [column for column in (config.feature_columns or []) if column != target]
        config.feature_columns = features
        X, processed_y, artifacts = build_preprocessing_pipeline(
            df,
            config,
            encode_classification_target=False,
        )
        if str(config.task_type or "").strip().lower() == "classification":
            y = df[target].copy()
        else:
            y = processed_y
        return X, _normalize_target_values(y), artifacts

    def _recompute_before_distributions(self) -> None:
        config = self.main_window.config
        df = self.main_window.dataframe
        error = self._prerequisite_error(config, df)
        if error:
            self._clear_distribution_tables(self.before_distribution_tables)
            self._clear_distribution_tables(self.after_distribution_tables)
            return

        try:
            X, y, _ = self._build_latest_xy(config, df)
            split = split_data_three_way(X, y, df, config)
        except Exception as exc:
            self._clear_distribution_tables(self.before_distribution_tables)
            self._clear_distribution_tables(self.after_distribution_tables)
            self._show_error(f"Could not refresh split distributions: {exc}")
            return

        before = self._distribution_frame(
            {
                "Full Dataset": y,
                "Train Set": split["y_train"],
                "Validation Set": split["y_val"],
                "Test Set": split["y_test"],
            }
        )
        self._populate_distribution_tables(before, self.before_distribution_tables)
        coverage = build_class_coverage_report(
            y,
            split["y_train"],
            split["y_val"],
            split["y_test"],
        )
        self._populate_class_coverage_table(coverage)
        self._clear_distribution_tables(self.after_distribution_tables)

    def _clear_split_state(self) -> None:
        self._clear_distribution_tables(self.before_distribution_tables)
        self._clear_distribution_tables(self.after_distribution_tables)
        self.class_coverage_table.clearContents()
        self.class_coverage_table.setRowCount(0)
        self.feedback_card.setVisible(False)
        self.feedback_label.clear()
        self.warning_card.setVisible(False)
        self.warning_label.clear()

    def _clear_distribution_tables(self, tables: dict[str, QTableWidget]) -> None:
        for table in tables.values():
            table.clearContents()
            table.setRowCount(0)

    def _show_reload_message(self, message: str, warning: bool = False) -> None:
        if warning:
            self.feedback_card.setVisible(False)
            self.feedback_label.clear()
            self._show_warnings([message])
            return

        self.warning_card.setVisible(False)
        self.warning_label.clear()
        self._show_success_notification([message])

    def _show_success(
        self,
        split: dict[str, Any],
        balanced_rows: int,
        output_dir: Path,
        config_path: Path,
        sampling_strategy: Any,
        issues: list[dict[str, str]],
        warnings: list[str],
    ) -> None:
        info = split["split_info"]
        lines = [
            "Split and imbalance configuration saved successfully.",
            f"Target column: {self.main_window.config.target_column}",
            f"Train rows before balancing: {info['train_rows']}",
            f"Train rows after balancing: {balanced_rows}",
            f"Output folder: {output_dir}",
        ]
        self._show_success_notification(lines)
        self._show_split_issues(issues, warnings)

    def _show_split_issues(
        self,
        issues: list[dict[str, str]],
        warnings: list[str],
    ) -> None:
        blocking = [issue["message"] for issue in issues if issue["level"] == "fatal"]
        warning_messages = [issue["message"] for issue in issues if issue["level"] == "warning"]
        warning_messages.extend(
            warning for warning in warnings if warning not in warning_messages
        )
        if blocking:
            self.warning_card.setStyleSheet(
                f"QFrame {{ border: 1px solid {ERROR_COLOR}; border-left: 3px solid {ERROR_COLOR}; "
                "border-radius: 7px; background: #FFF8F7; padding: 10px; }}"
                "QWidget { border: none; background: transparent; }"
            )
            self.warning_icon_label.setPixmap(
                get_fa_icon("fa6s.circle-exclamation", ERROR_COLOR).pixmap(16, 16)
            )
            lines = ["Blocking split issues:"]
            lines.extend(blocking)
            if warning_messages:
                lines.append("Warnings:")
                lines.extend(warning_messages)
            self.warning_label.setText("\n".join(lines))
            self.warning_card.setVisible(True)
            return
        self._show_warnings(warning_messages)

    def _show_warnings(self, warnings: list[str]) -> None:
        self.warning_card.setStyleSheet(
            f"QFrame {{ border: 1px solid {WARNING_COLOR}; border-radius: 7px; "
            f"background: #FFF8E6; border-left: 3px solid {WARNING_COLOR}; padding: 10px; }}"
            "QWidget { border: none; background: transparent; }"
        )
        if not warnings:
            self.warning_card.setVisible(False)
            self.warning_label.clear()
            self.warning_icon_label.clear()
            return
        self.warning_icon_label.setPixmap(
            get_fa_icon("fa6s.triangle-exclamation", WARNING_COLOR).pixmap(16, 16)
        )
        self.warning_label.setText("Warnings:\n" + "\n".join(warnings))
        self.warning_card.setVisible(True)

    def _show_error(self, message: str) -> None:
        self.warning_card.setVisible(False)
        self.warning_label.clear()
        self.success_notification_timer.stop()
        self.feedback_card.setStyleSheet(
            f"QFrame {{ border: 1px solid {ERROR_COLOR}; border-left: 3px solid {ERROR_COLOR}; "
            "border-radius: 7px; background: #fff8f7; padding: 10px; }"
            "QWidget { border: none; background: transparent; }"
        )
        self.feedback_label.setText(f"Error: {message}")
        for index, (icon_label, text_label) in enumerate(
            zip(self.feedback_icons, self.feedback_labels)
        ):
            visible = index == 0
            text_label.parentWidget().setVisible(visible)
            if visible:
                icon_label.setPixmap(
                    get_fa_icon("fa6s.circle-exclamation", ERROR_COLOR).pixmap(16, 16)
                )
                text_label.setText(f"Error: {message}")
        self.feedback_card.setVisible(True)

    def _split_configuration_card(self) -> QWidget:
        card, layout = self._card(
            "splitConfigurationCard",
            "Split Configuration",
            "Choose train, validation, and test proportions for the current target.",
            "fa6s.sliders",
        )
        layout.addLayout(self._split_controls())
        layout.addWidget(self.percent_validator_label)
        return card

    def _before_balancing_card(self) -> QWidget:
        card, layout = self._card(
            "beforeBalancingCard",
            "Before Balancing",
            "Review target distributions immediately after splitting.",
            "fa6s.table",
        )
        layout.addLayout(self._distribution_row(self.before_distribution_tables))
        return card

    def _class_coverage_card(self) -> QWidget:
        card, layout = self._card(
            "classCoverageCard",
            "Class Coverage After Splitting",
            "Check whether every class is represented across train, validation, and test sets.",
            "fa6s.list-check",
        )
        layout.addWidget(self.class_coverage_table)
        return card

    def _imbalance_handling_card(self) -> QWidget:
        card, layout = self._card(
            "imbalanceHandlingCard",
            "Imbalance Handling",
            "Configure optional train-set balancing. Validation and test rows remain untouched.",
            "fa6s.scale-balanced",
        )
        layout.addLayout(self._imbalance_controls())
        return card

    def _after_balancing_card(self) -> QWidget:
        card, layout = self._card(
            "afterBalancingCard",
            "After Balancing",
            "Compare the balanced training set with unchanged validation and test sets.",
            "fa6s.chart-column",
        )
        note = QLabel("Only the training set is modified by imbalance handling.")
        note.setObjectName("splitNote")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addLayout(self._distribution_row(self.after_distribution_tables))
        return card

    def _confirmation_status_card(self) -> QWidget:
        card, layout = self._card(
            "confirmationStatusCard",
            "Confirmation / Status",
            "Save the split, balancing metadata, and downstream training artifacts.",
            "fa6s.circle-check",
        )
        self.confirm_button = QPushButton("Confirm Split & Imbalance")
        self.confirm_button.setObjectName("primaryDataSplitButton")
        self.confirm_button.setIcon(get_fa_icon("fa6s.floppy-disk", "#FFFFFF"))
        self.confirm_button.setIconSize(QSize(16, 16))
        self.confirm_button.clicked.connect(self.confirm_split_and_imbalance)
        layout.addWidget(self.confirm_button)
        layout.addWidget(self.feedback_card)
        layout.addWidget(self.warning_card)
        return card

    def _card(
        self,
        object_name: str,
        title: str,
        subtitle: str,
        icon_name: str,
    ) -> tuple[QWidget, QVBoxLayout]:
        card = QWidget()
        card.setObjectName(object_name)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)

        header = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setObjectName(f"{object_name}Icon")
        icon_label.setPixmap(get_fa_icon(icon_name).pixmap(22, 22))
        title_label = QLabel(title)
        title_label.setObjectName("splitCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)

        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("splitCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return card, layout

    def _empty_state_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("dataSplitEmptyCard")
        card.setMaximumWidth(620)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)
        empty_icon = QLabel()
        empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_icon.setPixmap(get_fa_icon("fa6s.circle-info").pixmap(42, 42))
        message = QLabel("Please confirm Column Configuration first.")
        message.setObjectName("dataSplitEmptyMessage")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(empty_icon)
        layout.addWidget(message)
        return card

    def _feedback_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("dataSplitFeedbackCard")
        card.setMaximumHeight(158)
        card.hide()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        self.feedback_icons: list[QLabel] = []
        self.feedback_labels: list[QLabel] = []
        for _ in range(5):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            icon_label = QLabel()
            icon_label.setFixedSize(18, 18)
            text_label = QLabel("")
            text_label.setWordWrap(True)
            row_layout.addWidget(icon_label)
            row_layout.addWidget(text_label, stretch=1)
            layout.addWidget(row)
            self.feedback_icons.append(icon_label)
            self.feedback_labels.append(text_label)
        self.feedback_label = self.feedback_labels[0]
        return card

    def _show_success_notification(self, messages: list[str]) -> None:
        self.feedback_card.setStyleSheet(self._feedback_style("success"))
        for index, (icon_label, text_label) in enumerate(
            zip(self.feedback_icons, self.feedback_labels)
        ):
            visible = index < len(messages)
            text_label.parentWidget().setVisible(visible)
            if visible:
                icon_label.setPixmap(
                    get_fa_icon("fa6s.circle-check", SUCCESS_COLOR).pixmap(16, 16)
                )
                text_label.setText(messages[index])
            else:
                text_label.clear()
        self.feedback_card.show()
        self.success_notification_timer.stop()
        self.success_notification_timer.start()

    def _dismiss_success_notification(self) -> None:
        if "Error:" not in self.feedback_label.text():
            self.feedback_card.hide()

    def _feedback_style(self, level: str) -> str:
        success = level == "success"
        border = SUCCESS_COLOR if success else ERROR_COLOR
        background = "#F8FFF9" if success else "#FFF8F7"
        return (
            f"QFrame {{ background: {background}; border: 1px solid {BORDER}; "
            f"border-left: 3px solid {border}; border-radius: 7px; }}"
            "QWidget { border: none; background: transparent; }"
            f"QLabel {{ color: {TEXT}; border: none; background: transparent; }}"
        )

    def _split_controls(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        form.addRow("Current target column", self.current_target_label)
        percentages = QHBoxLayout()
        percentages.setSpacing(10)
        percentages.addWidget(self._field_label("Train"))
        percentages.addWidget(self._spinbox_control(self.train_percent))
        percentages.addWidget(self._field_label("Validation"))
        percentages.addWidget(self._spinbox_control(self.validation_percent))
        percentages.addWidget(self._field_label("Test"))
        percentages.addWidget(self._spinbox_control(self.test_percent))
        percentages.addStretch(1)
        form.addRow("Split percentages", percentages)
        form.addRow("Split method", self._combobox_control(self.split_method))
        form.addRow("Random seed", self._spinbox_control(self.random_seed, width=128))
        return form

    def _imbalance_controls(self) -> QFormLayout:
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)
        form.addRow("Imbalance method", self._combobox_control(self.imbalance_method))
        form.addRow("", self.use_class_weights)
        form.addRow("", self.balancing_preset_container)
        return form

    def _distribution_row(self, tables: dict[str, QTableWidget]) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(12)
        for name, table in tables.items():
            panel = QFrame()
            panel.setObjectName("splitInnerTableCard")
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(10, 10, 10, 10)
            panel_layout.setSpacing(8)
            heading = QLabel(name)
            heading.setObjectName("splitInnerTableTitle")
            panel_layout.addWidget(heading)
            panel_layout.addWidget(table)
            row.addWidget(panel, stretch=1)
        return row

    def _distribution_table(self) -> QTableWidget:
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Class", "Count", "Percent"])
        table.setMinimumHeight(132)
        table.setMaximumHeight(190)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._style_table(table)
        return table

    def _style_table(self, table: QTableWidget) -> None:
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(24)
        table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setShowGrid(True)
        table.setWordWrap(False)

    def _class_coverage_table(self) -> QTableWidget:
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(
            [
                "Class",
                "Full count",
                "Train count",
                "Validation count",
                "Test count",
                "Status",
            ]
        )
        table.setMinimumHeight(180)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._style_table(table)
        return table

    def _coverage_badge(self, status: str) -> tuple[str, str]:
        lowered = status.lower()
        if "missing in train" in lowered or "blocking" in lowered:
            return "Missing", "#CF222E"
        if lowered == "ok":
            return "OK", "#2DA44E"
        return "Warning", "#BF8700"

    def _populate_class_coverage_table(self, coverage: pd.DataFrame) -> None:
        self.class_coverage_table.setRowCount(len(coverage))
        columns = [
            "Class",
            "Full count",
            "Train count",
            "Validation count",
            "Test count",
            "Status",
        ]
        for row_index, (_, row) in enumerate(coverage.iterrows()):
            for column_index, column in enumerate(columns):
                text = str(row[column])
                item = QTableWidgetItem(text)
                if column == "Status":
                    badge, color = self._coverage_badge(text)
                    item.setText(badge)
                    item.setToolTip(text)
                    item.setForeground(QColor(color))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                elif column == "Class":
                    item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.class_coverage_table.setItem(
                    row_index,
                    column_index,
                    item,
                )
        self.class_coverage_table.resizeRowsToContents()

    def _percentage_input(self, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 100)
        spin.setSuffix("%")
        spin.setValue(value)
        spin.setFixedWidth(86)
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        return spin

    def _spinbox_control(self, spin: QSpinBox, width: int = 86) -> QWidget:
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setFixedWidth(width)
        frame = QFrame()
        frame.setObjectName("splitSpinBoxControl")
        frame.setFixedWidth(width + 28)
        frame.setFixedHeight(36)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        buttons = QWidget()
        buttons.setObjectName("splitSpinButtonColumn")
        button_layout = QVBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)
        up_button = SpinArrowButton("fa6s.angle-up", spin.stepUp, frame)
        down_button = SpinArrowButton("fa6s.angle-down", spin.stepDown, frame)
        button_layout.addWidget(up_button)
        button_layout.addWidget(down_button)
        layout.addWidget(spin)
        layout.addWidget(buttons)
        return frame

    def _combobox_control(self, combo: QComboBox, width: int = 220) -> QWidget:
        combo.setFixedWidth(width)
        frame = QFrame()
        frame.setObjectName("splitComboBoxControl")
        frame.setFixedWidth(width + 28)
        frame.setFixedHeight(36)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(combo)
        layout.addWidget(ComboArrowButton(combo, frame))
        return frame

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("splitInlineFieldLabel")
        return label

    def _update_custom_visibility(self, text: str | None = None) -> None:
        is_custom = (text or self.ratio_preset.currentText()) == "Custom"
        self.custom_ratio_input.setVisible(
            is_custom and self.imbalance_method.currentText() in {"smote", "smote_nc"}
        )

    def _preset_key(self) -> str:
        return {
            "Light Balancing": "light",
            "Moderate Balancing": "moderate",
            "Strong Balancing": "strong",
            "Custom": "custom",
        }[self.ratio_preset.currentText()]

    def _preset_label(self, key: str | None) -> str:
        return {
            "conservative": "Light Balancing",
            "light": "Light Balancing",
            "baseline": "Moderate Balancing",
            "moderate": "Moderate Balancing",
            "aggressive": "Strong Balancing",
            "strong": "Strong Balancing",
            "custom": "Custom",
        }.get(key or "", "Moderate Balancing")

    def _update_balancing_visibility(self, _text: str | None = None) -> None:
        visible = self.imbalance_method.currentText() in {"smote", "smote_nc"}
        self.balancing_preset_container.setVisible(visible)
        self._update_custom_visibility()

    def _update_split_tooltip(self, text: str | None = None) -> None:
        self.split_method.setToolTip(SPLIT_METHOD_HELP.get(text or self.split_method.currentText(), ""))

    def _update_preset_tooltip(self, text: str | None = None) -> None:
        self.ratio_preset.setToolTip(
            BALANCING_PRESET_HELP.get(text or self.ratio_preset.currentText(), "")
        )

    def _update_percentage_validator(self) -> None:
        total = (
            self.train_percent.value()
            + self.validation_percent.value()
            + self.test_percent.value()
        )
        if total == 100:
            self.percent_validator_label.setText("")
            self.percent_validator_label.hide()
            return
        self.percent_validator_label.setText(
            f"Train + validation + test must equal 100%. Current total: {total}%."
        )
        self.percent_validator_label.show()

    def _show_empty_state(self) -> None:
        self.controls_widget.hide()
        self.empty_state_card.show()

    def _show_controls(self) -> None:
        self.empty_state_card.hide()
        self.controls_widget.show()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#dataSplitContent {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#dataSplitTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#splitCardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#splitCardSubtitle,
            QLabel#splitNote {{
                color: #5B6573;
                font-size: 12px;
            }}
            QLabel#splitTargetValue {{
                color: {TEXT};
                font-weight: 600;
            }}
            QLabel#splitPercentValidator {{
                color: #CF222E;
                font-size: 12px;
                font-weight: 600;
            }}
            QWidget#splitConfigurationCard,
            QWidget#beforeBalancingCard,
            QWidget#classCoverageCard,
            QWidget#imbalanceHandlingCard,
            QWidget#afterBalancingCard,
            QWidget#confirmationStatusCard,
            QWidget#dataSplitEmptyCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QFrame#splitInnerTableCard {{
                background: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QLabel#splitInnerTableTitle {{
                color: {TEXT};
                font-weight: 700;
                font-size: 12px;
            }}
            QLabel#dataSplitEmptyMessage {{ color: #5B6573; font-size: 13px; }}
            QLabel#splitInlineFieldLabel {{
                color: {TEXT};
                font-size: 12px;
                font-weight: 600;
            }}
            QLineEdit {{
                background: #FFFFFF;
                color: {TEXT};
                border: 1px solid {BORDER};
                border-radius: 7px;
                min-height: 34px;
                max-height: 38px;
                padding: 0 10px;
                selection-background-color: #DCEBFA;
                selection-color: {TEXT};
            }}
            QLineEdit:hover {{
                border-color: {PRIMARY};
            }}
            QLineEdit:focus {{
                border: 1px solid {PRIMARY};
            }}
            QFrame#splitComboBoxControl {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 7px;
            }}
            QFrame#splitComboBoxControl:hover {{
                border-color: {PRIMARY};
            }}
            QFrame#splitComboBoxControl QComboBox {{
                background: transparent;
                color: {TEXT};
                border: none;
                min-height: 34px;
                max-height: 36px;
                padding: 0 10px;
                selection-background-color: #DCEBFA;
                selection-color: {TEXT};
            }}
            QFrame#splitComboBoxControl QComboBox::drop-down {{
                width: 0px;
                border: none;
            }}
            QToolButton#splitComboArrowButton {{
                background: #F8FAFC;
                border: none;
                border-left: 1px solid {BORDER};
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
                padding: 0;
                margin: 0;
            }}
            QToolButton#splitComboArrowButton:hover {{
                background: #EFF6FF;
                border-left-color: {PRIMARY};
            }}
            QFrame#splitSpinBoxControl {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 7px;
            }}
            QFrame#splitSpinBoxControl:hover {{
                border-color: {PRIMARY};
            }}
            QFrame#splitSpinBoxControl QSpinBox {{
                background: transparent;
                color: {TEXT};
                border: none;
                min-height: 34px;
                max-height: 36px;
                padding: 0 8px 0 10px;
                selection-background-color: #DCEBFA;
                selection-color: {TEXT};
            }}
            QWidget#splitSpinButtonColumn {{
                background: #F8FAFC;
                border-left: 1px solid {BORDER};
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }}
            QToolButton#splitSpinArrowButton {{
                background: transparent;
                border: none;
                padding: 0;
                margin: 0;
            }}
            QToolButton#splitSpinArrowButton:hover {{
                background: #EFF6FF;
            }}
            QCheckBox {{ color: {TEXT}; spacing: 8px; }}
            QTableWidget {{
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                border: 1px solid {BORDER};
                border-radius: 6px;
                gridline-color: #E5E7EB;
                color: {TEXT};
                selection-background-color: transparent;
                selection-color: {TEXT};
            }}
            QHeaderView::section {{
                background: #EEF3F8;
                color: {TEXT};
                font-weight: 700;
                border: none;
                border-right: 1px solid {BORDER};
                border-bottom: 1px solid {BORDER};
                padding: 5px 6px;
            }}
            QPushButton {{
                min-height: 36px;
                border-radius: 6px;
                padding: 0 12px;
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton:hover {{ background: #EFF6FF; border-color: {PRIMARY}; }}
            QPushButton#primaryDataSplitButton {{
                min-height: 44px;
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
                border-radius: 7px;
                font-weight: 600;
            }}
            QPushButton#primaryDataSplitButton:hover {{ background: #00A6A6; }}
            """
        )

    def _select_combo(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _json_counts(values: pd.Series) -> dict[str, int]:
    normalized = pd.Series(values).map(_display_class_label)
    return {str(key): int(count) for key, count in normalized.value_counts(sort=False).items()}


def _normalize_target_values(values: pd.Series) -> pd.Series:
    series = pd.Series(values, index=values.index, name=values.name)
    non_missing_types = {type(value) for value in series.dropna()}
    if pd.api.types.is_numeric_dtype(series) and len(non_missing_types) <= 1:
        return series
    return series.map(_display_class_label)


def _display_class_label(value: Any) -> str:
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return str(_json_scalar(value))


def get_fa_icon(name: str, color: str = PRIMARY):
    """Return a qtawesome Font Awesome icon used by this page."""

    return icon(name, color=color)
