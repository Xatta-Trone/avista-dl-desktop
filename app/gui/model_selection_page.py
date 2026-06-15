"""Model selection and parameter configuration page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, QTimer, Qt, QThread
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.dependency_manager import check_optional_packages
from app.core.model_registry import ModelSpec, get_available_models
from app.gui.icon_system import BACKGROUND, BORDER, PRIMARY, TEXT, icon
from app.gui.workers import DependencyInstallWorker


CATEGORY_ORDER = (
    "Linear Models",
    "Tree-Based Models",
    "Boosting Models",
    "Kernel/Distance Models",
    "Naive Bayes",
    "Tabular Models",
)

CATEGORY_ALIASES = {
    "Tabular Models": {"Deep Tabular Models", "Foundation Tabular Models"},
}

SUCCESS_COLOR = "#16A34A"
ERROR_COLOR = "#DC2626"
NEUTRAL_ICON = "#6B7280"


class _ArrowButton(QToolButton):
    def __init__(
        self,
        icon_name: str,
        callback,
        object_name: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.icon_name = icon_name
        self.clicked.connect(callback)
        self.setObjectName(object_name)
        self.setFixedSize(22, 18)
        self.setIconSize(QSize(12, 12))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._set_icon(NEUTRAL_ICON)

    def enterEvent(self, event) -> None:
        self._set_icon(PRIMARY)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._set_icon(NEUTRAL_ICON)
        super().leaveEvent(event)

    def _set_icon(self, color: str) -> None:
        self.setIcon(icon(self.icon_name, color))


class AvistaComboBox(QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("avistaModelComboBox")
        self.setFixedHeight(36)
        self.setMinimumWidth(180)
        arrow = _ArrowButton(
            "fa6s.angle-down",
            self.showPopup,
            "modelComboArrowButton",
            self,
        )
        arrow.setFixedSize(28, 34)
        arrow.setIconSize(QSize(13, 13))
        arrow.move(self.width() - arrow.width() - 1, 1)
        arrow.show()
        self.arrow_button = arrow

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.arrow_button.move(self.width() - self.arrow_button.width() - 1, 1)


class AvistaSpinBox(QSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self._init_arrows()

    def _init_arrows(self) -> None:
        self.setObjectName("avistaModelSpinBox")
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFixedHeight(36)
        self.setMinimumWidth(110)
        self.up_button = _ArrowButton(
            "fa6s.angle-up",
            self.stepUp,
            "modelSpinArrowButton",
            self,
        )
        self.down_button = _ArrowButton(
            "fa6s.angle-down",
            self.stepDown,
            "modelSpinArrowButton",
            self,
        )
        self._position_arrows()
        self.up_button.show()
        self.down_button.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_arrows()

    def _position_arrows(self) -> None:
        x = self.width() - 23
        self.up_button.move(x, 0)
        self.down_button.move(x, 18)


class AvistaDoubleSpinBox(QDoubleSpinBox):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("avistaModelSpinBox")
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setFixedHeight(36)
        self.setMinimumWidth(110)
        self.up_button = _ArrowButton(
            "fa6s.angle-up",
            self.stepUp,
            "modelSpinArrowButton",
            self,
        )
        self.down_button = _ArrowButton(
            "fa6s.angle-down",
            self.stepDown,
            "modelSpinArrowButton",
            self,
        )
        self._position_arrows()
        self.up_button.show()
        self.down_button.show()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_arrows()

    def _position_arrows(self) -> None:
        x = self.width() - 23
        self.up_button.move(x, 0)
        self.down_button.move(x, 18)


class ModelParameterPanel(QScrollArea):
    """Reusable padded, scrollable shell for model parameter pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("modelParameterPanelScroll")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.content = QWidget()
        self.content.setObjectName("modelParameterPanelContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(12)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.content)

DEEP_TRAINING_DEFAULTS = {
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 80,
    "warmup_epochs": 5,
    "patience": 10,
    "loss_function": "cross_entropy",
    "validation_metric": "macro_f1",
}

DEEP_MONITORING_DEFAULTS = {
    "save_training_loss": True,
    "save_validation_loss": True,
    "save_validation_metric": True,
    "save_best_checkpoint": True,
    "early_stopping_patience": 10,
}

MAMBA_ATTENTION_TRAINING_DEFAULTS = {
    # from reference train_model(...): lr=1e-3
    "learning_rate": 0.001,
    # from reference BATCH_SIZE = 128
    "batch_size": 128,
    # from reference NUM_EPOCHS = 80
    "epochs": 80,
    # from reference WARMUP_EPOCHS = 5
    "warmup_epochs": 5,
    # from reference torch.optim.AdamW(...)
    "optimizer": "adamw",
    # from reference AdamW(...): weight_decay=1e-3
    "weight_decay": 0.001,
    # from reference LinearLR followed by CosineAnnealingLR
    "scheduler": "linear_warmup_cosine_annealing",
    # from reference LinearLR(...): start_factor=0.1
    "warmup_start_factor": 0.1,
}

MAMBA_ATTENTION_LOSS_DEFAULTS = {
    # from reference FocalLoss(...)
    "loss_function": "focal_loss",
    # from reference MambaAttention train_model(...): gamma=1.0
    "focal_gamma": 1.0,
    # from reference FocalLoss(...): ls=0.05
    "label_smoothing": 0.05,
    # from reference FocalLoss(alpha=class_weights_tensor, ...)
    "use_class_weights": True,
}

MAMBA_ATTENTION_MONITORING_DEFAULTS = {
    # from reference f1_score(...): average='macro'
    "validation_metric": "macro_f1",
    # from reference train_model(...): PAT_LIM = 30
    "early_stopping_patience": 30,
    # from reference best_st and model.load_state_dict(best_st)
    "restore_best_weights": True,
    # from reference torch.save(mamba.state_dict(), ... 'mamba_attention.pt')
    "save_final_state_dict": True,
}

FT_TRANSFORMER_TRAINING_DEFAULTS = {
    **MAMBA_ATTENTION_TRAINING_DEFAULTS,
    # from reference FT-Transformer train_model(...): lr=1e-3
    "learning_rate": 0.001,
}

FT_TRANSFORMER_LOSS_DEFAULTS = {
    **MAMBA_ATTENTION_LOSS_DEFAULTS,
    # from reference FT-Transformer train_model(...): gamma=1.0
    "focal_gamma": 1.0,
}

FT_TRANSFORMER_MONITORING_DEFAULTS = dict(MAMBA_ATTENTION_MONITORING_DEFAULTS)

AUTOINT_TRAINING_DEFAULTS = {
    **MAMBA_ATTENTION_TRAINING_DEFAULTS,
    # from reference AutoInt train_model(...): lr=1e-3
    "learning_rate": 0.001,
}

AUTOINT_LOSS_DEFAULTS = {
    **MAMBA_ATTENTION_LOSS_DEFAULTS,
    # from reference AutoInt train_model(...): gamma=1.0
    "focal_gamma": 1.0,
}

AUTOINT_MONITORING_DEFAULTS = dict(MAMBA_ATTENTION_MONITORING_DEFAULTS)

TAB_RESNET_TRAINING_DEFAULTS = {
    **MAMBA_ATTENTION_TRAINING_DEFAULTS,
    # from reference TabResNet train_model(...): lr=1e-3
    "learning_rate": 0.001,
}

TAB_RESNET_LOSS_DEFAULTS = {
    **MAMBA_ATTENTION_LOSS_DEFAULTS,
    # from reference TabResNet train_model(...): gamma=1.0
    "focal_gamma": 1.0,
}

TAB_RESNET_MONITORING_DEFAULTS = dict(MAMBA_ATTENTION_MONITORING_DEFAULTS)

MAMBA_ATTENTION_UI_METADATA = {
    "learning_rate": {"type": "float", "label": "Learning Rate", "section": "Training Parameters"},
    "batch_size": {"type": "int", "label": "Batch Size", "section": "Training Parameters"},
    "epochs": {"type": "int", "label": "Epoch Count", "section": "Training Parameters"},
    "warmup_epochs": {"type": "int", "label": "Warmup Epoch Count", "section": "Training Parameters"},
    "optimizer": {
        "type": "select",
        "options": ["adamw"],
        "label": "Optimizer",
        "section": "Training Parameters",
    },
    "weight_decay": {"type": "float", "label": "Weight Decay", "section": "Training Parameters"},
    "scheduler": {
        "type": "select",
        "options": ["linear_warmup_cosine_annealing"],
        "label": "Scheduler",
        "section": "Training Parameters",
    },
    "warmup_start_factor": {
        "type": "float",
        "label": "Warmup Start Factor",
        "section": "Training Parameters",
    },
    "loss_function": {
        "type": "select",
        "options": ["focal_loss"],
        "label": "Loss Function",
        "section": "Loss Parameters",
    },
    "focal_gamma": {"type": "float", "label": "Focal Gamma", "section": "Loss Parameters"},
    "label_smoothing": {"type": "float", "label": "Label Smoothing", "section": "Loss Parameters"},
    "use_class_weights": {
        "type": "bool",
        "label": "Use Training Class Weights",
        "section": "Loss Parameters",
    },
    "validation_metric": {
        "type": "select",
        "options": ["macro_f1"],
        "label": "Validation Metric",
        "section": "Monitoring / Saving Options",
    },
    "early_stopping_patience": {
        "type": "int",
        "label": "Early Stopping Patience",
        "section": "Monitoring / Saving Options",
    },
    "restore_best_weights": {
        "type": "bool",
        "label": "Restore Best Weights",
        "section": "Monitoring / Saving Options",
    },
    "save_final_state_dict": {
        "type": "bool",
        "label": "Save Final State Dict",
        "section": "Monitoring / Saving Options",
    },
}

MAMBA_ATTENTION_SECTION_ORDER = (
    "Architecture Parameters",
    "Training Parameters",
    "Loss Parameters",
    "Monitoring / Saving Options",
)

INFERRED_TORCH_PARAMS = {"input_dim", "num_classes", "n_features", "n_classes"}


class OptionalNumberWidget(QWidget):
    """Select between None and a typed custom numeric value."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        super().__init__()
        self.selector = AvistaComboBox()
        self.selector.addItems(metadata.get("options", ["none", "custom"]))
        if metadata["type"] == "select_or_int":
            self.number_input = AvistaSpinBox()
            self.number_input.setRange(
                int(metadata.get("min", -2_147_483_648)),
                int(metadata.get("max", 2_147_483_647)),
            )
            self.number_input.setSingleStep(int(metadata.get("step", 1)))
        else:
            self.number_input = AvistaDoubleSpinBox()
            self.number_input.setRange(
                float(metadata.get("min", -1_000_000_000.0)),
                float(metadata.get("max", 1_000_000_000.0)),
            )
            self.number_input.setDecimals(6)
            self.number_input.setSingleStep(float(metadata.get("step", 0.1)))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.selector)
        layout.addWidget(self.number_input)
        self.selector.currentTextChanged.connect(self._update_enabled)
        default = metadata.get("custom_default", metadata.get("default"))
        self.setValue(None if metadata.get("default") == "none" else default)

    def value(self) -> int | float | None:
        if self.selector.currentText() == "none":
            return None
        return self.number_input.value()

    def setValue(self, value: Any) -> None:
        is_none = value is None or (isinstance(value, str) and value.casefold() == "none")
        self.selector.setCurrentText("none" if is_none else "custom")
        if not is_none:
            self.number_input.setValue(value)
        self._update_enabled()

    def _update_enabled(self) -> None:
        self.number_input.setEnabled(self.selector.currentText() == "custom")


class ModelSelectionPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.model_specs = get_available_models(task_type="classification")
        self.model_checkboxes: dict[str, QCheckBox] = {}
        self.parameter_panels: dict[str, QWidget] = {}
        self.parameter_widgets: dict[str, dict[str, QWidget]] = {}
        self.parameter_defaults: dict[str, dict[str, Any]] = {}
        self.parameter_metadata: dict[str, dict[str, dict[str, Any]]] = {}
        self.parameter_section_groups: dict[str, dict[str, QGroupBox]] = {}
        self.parameter_warning_labels: dict[str, QLabel] = {}
        self.restore_default_buttons: dict[str, QPushButton] = {}
        self.restore_feedback_labels: dict[str, QLabel] = {}
        self.dependency_labels: dict[str, QLabel] = {}
        self.dependency_install_buttons: dict[str, QPushButton] = {}
        self.dependency_status: dict[str, bool] = {}
        self.install_threads: dict[str, QThread] = {}
        self.install_workers: dict[str, DependencyInstallWorker] = {}
        self.category_cards: dict[str, QWidget] = {}

        self.parameter_stack = QStackedWidget()
        self.parameter_stack.addWidget(self._empty_parameter_panel())

        self.enable_cross_validation = QCheckBox("Enable Cross Validation")
        self.cv_folds = AvistaSpinBox()
        self.cv_folds.setRange(2, 100)
        self.cv_folds.setValue(5)
        self.random_state = AvistaSpinBox()
        self.random_state.setRange(0, 2_147_483_647)
        self.random_state.setValue(42)

        self.success_notification_timer = QTimer(self)
        self.success_notification_timer.setSingleShot(True)
        self.success_notification_timer.setInterval(5000)
        self.success_notification_timer.timeout.connect(self._dismiss_success_notification)
        self.feedback_card = self._feedback_card()

        content = QWidget()
        content.setObjectName("modelSelectionContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)
        title = QLabel("Model Selection")
        title.setObjectName("modelSelectionTitle")
        content_layout.addWidget(title)
        self.empty_state_card = self._empty_state_card()
        self.empty_state_card.hide()
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(16)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.top_cards_layout = QHBoxLayout()
        self.top_cards_layout.setSpacing(16)
        self.top_cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_library_card = self._model_library_card()
        self.model_parameters_card = self._model_parameters_card()
        self.model_library_card.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        self.model_parameters_card.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        self.top_cards_layout.addWidget(
            self.model_library_card,
            stretch=1,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        self.top_cards_layout.addWidget(
            self.model_parameters_card,
            stretch=2,
            alignment=Qt.AlignmentFlag.AlignTop,
        )
        controls_layout.addLayout(self.top_cards_layout)
        self.global_training_options_card = self._global_training_options_card()
        self.confirmation_status_card = self._confirmation_status_card()
        controls_layout.addWidget(self.global_training_options_card)
        controls_layout.addWidget(self.confirmation_status_card)

        content_layout.addWidget(self.empty_state_card, alignment=Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(self.controls_widget, alignment=Qt.AlignmentFlag.AlignTop)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        self._apply_style()

    def refresh(self) -> None:
        config = self.main_window.config
        if not self._has_modeling_config():
            self._show_empty_state()
        else:
            self._show_controls()
        selected = set(config.selected_models if config else [])
        saved_params = config.model_params if config else {}

        for spec in self.model_specs:
            checkbox = self.model_checkboxes[spec.name]
            checkbox.blockSignals(True)
            checkbox.setChecked(spec.name in selected or spec.display_name in selected)
            checkbox.blockSignals(False)
            values = saved_params.get(spec.name, saved_params.get(spec.display_name, {}))
            self._set_parameter_values(spec.name, values)

        self.refresh_dependency_status()
        self.enable_cross_validation.setChecked(
            bool(config.enable_cross_validation) if config else False
        )
        self.cv_folds.setValue(int(config.cv_folds) if config else 5)
        self.random_state.setValue(int(config.random_state) if config else 42)
        self._show_first_checked_panel()

    def _has_modeling_config(self) -> bool:
        config = self.main_window.config
        return bool(config and config.feature_columns and config.target_column)

    def _show_empty_state(self) -> None:
        self.controls_widget.hide()
        self.empty_state_card.show()

    def _show_controls(self) -> None:
        self.empty_state_card.hide()
        self.controls_widget.show()

    def _empty_state_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("modelSelectionEmptyCard")
        card.setMaximumWidth(640)
        card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(12)
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setPixmap(icon("fa6s.circle-info").pixmap(42, 42))
        message = QLabel("Please complete Column Configuration before selecting models.")
        message.setObjectName("modelSelectionEmptyMessage")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        layout.addWidget(icon_label)
        layout.addWidget(message)
        return card

    def _feedback_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("modelSelectionFeedbackCard")
        card.setMaximumHeight(118)
        card.hide()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)
        self.feedback_icons: list[QLabel] = []
        self.feedback_labels: list[QLabel] = []
        for _ in range(3):
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
                icon_label.setPixmap(icon("fa6s.circle-check", SUCCESS_COLOR).pixmap(16, 16))
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

    def confirm_model_selection(self) -> None:
        config = self.main_window.config
        if config is None:
            self._show_error("Load or create a project before selecting models.")
            return

        selected_specs = [
            spec for spec in self.model_specs if self.model_checkboxes[spec.name].isChecked()
        ]
        config.selected_models = [spec.name for spec in selected_specs]
        config.model_params = {
            spec.name: self._parameter_values(spec.name) for spec in selected_specs
        }
        config.enable_cross_validation = self.enable_cross_validation.isChecked()
        config.cv_folds = self.cv_folds.value()
        config.random_state = self.random_state.value()

        try:
            config_path = config.save()
        except Exception as exc:
            self._show_error(f"Could not save model selection: {exc}")
            return

        cv_state = (
            f"enabled, {config.cv_folds} folds"
            if config.enable_cross_validation
            else "disabled"
        )
        self._show_success_notification(
            [
                "Model selection saved successfully.",
                f"Selected models: {len(selected_specs)}",
                f"Cross-validation: {cv_state}",
            ]
        )

    def _model_library_card(self) -> QWidget:
        card, layout = self._card(
            "modelLibraryCard",
            "Model Library",
            "Choose one or more classification models to train.",
            "fa6s.brain",
        )
        self.model_library_list_scroll = QScrollArea()
        self.model_library_list_scroll.setObjectName("modelLibraryListScroll")
        self.model_library_list_scroll.setWidgetResizable(True)
        self.model_library_list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.model_library_list_scroll.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Maximum,
        )
        list_content = QWidget()
        list_content.setObjectName("modelLibraryListContent")
        list_layout = QVBoxLayout(list_content)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(8)
        list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        specs_by_category = self._specs_by_display_category()
        for category in CATEGORY_ORDER:
            group, group_layout = self._inner_card(category)
            self.category_cards[category] = group
            for spec in specs_by_category[category]:
                row = QWidget()
                row.setObjectName("modelSelectionModelRow")
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                checkbox = QCheckBox(spec.display_name)
                checkbox.setToolTip(spec.description)
                checkbox.toggled.connect(
                    lambda checked, model_name=spec.name: self._model_toggled(
                        model_name, checked
                    )
                )
                checkbox.clicked.connect(
                    lambda _checked, model_name=spec.name: self._show_parameter_panel(
                        model_name
                    )
                )
                self.model_checkboxes[spec.name] = checkbox
                row_layout.addWidget(checkbox, stretch=1)
                if spec.requires_optional_package:
                    dependency_label = QLabel("")
                    dependency_label.setObjectName("modelDependencyLabel")
                    install_button = QPushButton(
                        f"Install {spec.requires_optional_package}"
                    )
                    install_button.setObjectName("secondaryModelButton")
                    install_button.setVisible(False)
                    install_button.clicked.connect(
                        lambda _checked=False, package=spec.requires_optional_package:
                        self.confirm_dependency_install(package)
                    )
                    self.dependency_labels[spec.name] = dependency_label
                    self.dependency_install_buttons[spec.name] = install_button
                    row_layout.addWidget(dependency_label)
                    row_layout.addWidget(install_button)
                self._add_parameter_panel(spec)
                group_layout.addWidget(row)
            list_layout.addWidget(group)
        self.model_library_list_scroll.setWidget(list_content)
        layout.addWidget(self.model_library_list_scroll)
        return card

    def _model_parameters_card(self) -> QWidget:
        card, layout = self._card(
            "modelParametersCard",
            "Model Parameters",
            "Configure the active model's editable options.",
            "fa6s.sliders",
        )
        layout.addWidget(self.parameter_stack)
        return card

    def _global_training_options_card(self) -> QWidget:
        card, layout = self._card(
            "globalTrainingOptionsCard",
            "Global Training Options",
            "Apply shared training settings across selected models.",
            "fa6s.gear",
        )
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(6)
        form.addRow("", self.enable_cross_validation)
        form.addRow("Number of CV folds", self.cv_folds)
        form.addRow("Random State", self.random_state)
        layout.addLayout(form)
        return card

    def _confirmation_status_card(self) -> QWidget:
        card, layout = self._card(
            "modelConfirmationStatusCard",
            "Confirmation / Status",
            "Save selected models and parameters to the project.",
            "fa6s.circle-check",
        )
        self.confirm_button = QPushButton("Confirm Model Selection")
        self.confirm_button.setObjectName("primaryModelSelectionButton")
        self.confirm_button.setIcon(icon("fa6s.floppy-disk", "#FFFFFF"))
        self.confirm_button.setIconSize(self.confirm_button.iconSize())
        self.confirm_button.clicked.connect(self.confirm_model_selection)
        layout.addWidget(self.confirm_button)
        layout.addWidget(self.feedback_card)
        return card

    def _specs_by_display_category(self) -> dict[str, list[ModelSpec]]:
        grouped: dict[str, list[ModelSpec]] = {category: [] for category in CATEGORY_ORDER}
        for spec in self.model_specs:
            display_category = spec.category
            for category, aliases in CATEGORY_ALIASES.items():
                if spec.category in aliases:
                    display_category = category
                    break
            if display_category in grouped:
                grouped[display_category].append(spec)
        return grouped

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
        layout.setSpacing(8)
        header = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(icon(icon_name).pixmap(22, 22))
        title_label = QLabel(title)
        title_label.setObjectName("modelCardTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch(1)
        layout.addLayout(header)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("modelCardSubtitle")
        subtitle_label.setWordWrap(True)
        layout.addWidget(subtitle_label)
        return card, layout

    def _inner_card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("modelInnerCategoryCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(6)
        heading = QLabel(title)
        heading.setObjectName("modelInnerCardTitle")
        layout.addWidget(heading)
        return card, layout

    def refresh_dependency_status(self) -> None:
        packages = sorted(
            {
                spec.requires_optional_package
                for spec in self.model_specs
                if spec.requires_optional_package
            }
        )
        if not packages:
            return

        config = self.main_window.config
        project_dir = config.project_dir if config else str(self._app_root())
        environment_mode = config.environment_mode if config else "packaged_runtime"
        result = check_optional_packages(
            packages,
            project_dir=project_dir,
            environment_mode=environment_mode,
            app_root=self._app_root(),
        )
        for package in packages:
            self.dependency_status[package] = bool(
                result.get("packages", {}).get(package, False)
            )
        self._render_dependency_status(result.get("error"))

    def confirm_dependency_install(self, package_name: str) -> None:
        config = self.main_window.config
        if config is None:
            self._show_error("Load or create a project before installing dependencies.")
            return
        if package_name in self.install_threads:
            return

        answer = QMessageBox.question(
            self,
            f"Install {package_name}?",
            f"Install {package_name} into the active environment using pip?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self._set_package_installing(package_name, True)
        thread = QThread(self)
        worker = DependencyInstallWorker(
            package_name,
            project_dir=config.project_dir,
            environment_mode=config.environment_mode,
            app_root=str(self._app_root()),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._dependency_install_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda package=package_name: self._clear_install_thread(package)
        )
        self.install_threads[package_name] = thread
        self.install_workers[package_name] = worker
        thread.start()

    def _dependency_install_finished(self, result: dict[str, Any]) -> None:
        package_name = str(result.get("package", "package"))
        if result.get("success"):
            self.refresh_dependency_status()
            if self.dependency_status.get(package_name):
                self._show_success(
                    f"{package_name} installed successfully. Optional model availability refreshed."
                )
            else:
                self._show_error(
                    f"{package_name} installation completed, but the package is still unavailable "
                    "in the active environment. See install_log.txt for command output."
                )
        else:
            self.dependency_status[package_name] = False
            self._render_dependency_status()
            error = result.get("error") or "pip install failed"
            self._show_error(
                f"Could not install {package_name}: {error}\n"
                "See install_log.txt for command output."
            )
        self._set_package_installing(package_name, False)

    def _render_dependency_status(self, check_error: str | None = None) -> None:
        for spec in self.model_specs:
            package = spec.requires_optional_package
            if not package:
                continue
            installed = self.dependency_status.get(package, False)
            checkbox = self.model_checkboxes[spec.name]
            label = self.dependency_labels[spec.name]
            button = self.dependency_install_buttons[spec.name]
            checkbox.setEnabled(installed)
            if not installed:
                checkbox.setChecked(False)
                label.setText("Missing")
                label.setVisible(True)
                button.setVisible(True)
            else:
                label.clear()
                label.setVisible(False)
                button.setVisible(False)
        if check_error:
            self._show_error(f"Could not check optional dependencies: {check_error}")

    def _set_package_installing(self, package_name: str, installing: bool) -> None:
        for spec in self.model_specs:
            if spec.requires_optional_package != package_name:
                continue
            button = self.dependency_install_buttons[spec.name]
            label = self.dependency_labels[spec.name]
            button.setEnabled(not installing)
            if installing:
                label.setText(f"Installing {package_name}...")
                label.setVisible(True)

    def _clear_install_thread(self, package_name: str) -> None:
        self.install_threads.pop(package_name, None)
        self.install_workers.pop(package_name, None)

    def _app_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _add_parameter_panel(self, spec: ModelSpec) -> None:
        panel = ModelParameterPanel()
        layout = panel.content_layout
        heading = QLabel(spec.display_name)
        heading.setObjectName("modelParameterHeading")
        layout.addWidget(heading)

        defaults = self._editable_defaults(spec)
        metadata = self._editable_metadata(spec, defaults)
        self.parameter_defaults[spec.name] = defaults
        self.parameter_metadata[spec.name] = metadata
        self.parameter_widgets[spec.name] = {}
        self.parameter_section_groups[spec.name] = {}
        if not defaults:
            layout.addWidget(QLabel("No editable default parameters available."))
        else:
            forms: dict[str, QFormLayout] = {}
            if spec.name in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
            }:
                for section in MAMBA_ATTENTION_SECTION_ORDER:
                    group = QGroupBox(section)
                    group.setObjectName("modelParameterSection")
                    forms[section] = QFormLayout(group)
                    forms[section].setHorizontalSpacing(12)
                    forms[section].setVerticalSpacing(7)
                    forms[section].setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                    forms[section].setFieldGrowthPolicy(
                        QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
                    )
                    self.parameter_section_groups[spec.name][section] = group
                    layout.addWidget(group)
            else:
                form = QFormLayout()
                form.setHorizontalSpacing(12)
                form.setVerticalSpacing(7)
                form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
                form.setFieldGrowthPolicy(
                    QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
                )
            for name, value in defaults.items():
                parameter_metadata = metadata[name]
                widget = self._parameter_widget(parameter_metadata, value)
                help_text = parameter_metadata.get("help")
                if help_text:
                    widget.setToolTip(help_text)
                self.parameter_widgets[spec.name][name] = widget
                label = parameter_metadata.get("label", self._display_parameter_name(name))
                if spec.name in {
                    "mamba_attention",
                    "ft_transformer",
                    "autoint",
                    "tab_resnet",
                }:
                    forms[parameter_metadata["section"]].addRow(label, widget)
                else:
                    form.addRow(label, widget)
            if spec.name not in {
                "mamba_attention",
                "ft_transformer",
                "autoint",
                "tab_resnet",
            }:
                layout.addLayout(form)
            warning_label = QLabel("")
            warning_label.setWordWrap(True)
            warning_label.setVisible(False)
            warning_label.setStyleSheet(
                "QLabel { border: 1px solid #bf8700; border-radius: 5px; "
                "background: #fff8c5; color: #6e4c00; padding: 8px; }"
            )
            self.parameter_warning_labels[spec.name] = warning_label
            layout.addWidget(warning_label)
            for widget in self.parameter_widgets[spec.name].values():
                self._connect_parameter_change(widget, spec.name)
            self._update_model_ui(spec.name)

        restore_button = QPushButton("Restore Defaults")
        restore_button.setObjectName("secondaryModelButton")
        restore_button.clicked.connect(
            lambda _checked=False, model_name=spec.name: self._restore_model_defaults(
                model_name
            )
        )
        restore_feedback = QLabel("")
        restore_feedback.setVisible(False)
        restore_feedback.setStyleSheet("color: #1a7f37;")
        self.restore_default_buttons[spec.name] = restore_button
        self.restore_feedback_labels[spec.name] = restore_feedback
        layout.addSpacing(6)
        layout.addWidget(restore_button)
        layout.addWidget(restore_feedback)

        self.parameter_panels[spec.name] = panel
        self.parameter_stack.addWidget(panel)

    def _editable_defaults(self, spec: ModelSpec) -> dict[str, Any]:
        defaults = {
            key: value
            for key, value in spec.default_params.items()
            if key not in INFERRED_TORCH_PARAMS
        }
        if spec.name == "mamba_attention":
            defaults.update(MAMBA_ATTENTION_TRAINING_DEFAULTS)
            defaults.update(MAMBA_ATTENTION_LOSS_DEFAULTS)
            defaults.update(MAMBA_ATTENTION_MONITORING_DEFAULTS)
        elif spec.name == "ft_transformer":
            defaults.update(FT_TRANSFORMER_TRAINING_DEFAULTS)
            defaults.update(FT_TRANSFORMER_LOSS_DEFAULTS)
            defaults.update(FT_TRANSFORMER_MONITORING_DEFAULTS)
        elif spec.name == "autoint":
            defaults.update(AUTOINT_TRAINING_DEFAULTS)
            defaults.update(AUTOINT_LOSS_DEFAULTS)
            defaults.update(AUTOINT_MONITORING_DEFAULTS)
        elif spec.name == "tab_resnet":
            defaults.update(TAB_RESNET_TRAINING_DEFAULTS)
            defaults.update(TAB_RESNET_LOSS_DEFAULTS)
            defaults.update(TAB_RESNET_MONITORING_DEFAULTS)
        elif spec.model_family == "torch":
            defaults.update(DEEP_TRAINING_DEFAULTS)
            defaults.update(DEEP_MONITORING_DEFAULTS)
        return defaults

    def _editable_metadata(
        self, spec: ModelSpec, defaults: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        metadata = {
            name: dict(spec.parameter_metadata.get(name, {}))
            for name in defaults
        }
        if spec.name in {
            "mamba_attention",
            "ft_transformer",
            "autoint",
            "tab_resnet",
        }:
            for name, parameter_metadata in MAMBA_ATTENTION_UI_METADATA.items():
                if not metadata.get(name):
                    metadata[name] = dict(parameter_metadata)
        elif spec.model_family == "torch":
            for name, value in {**DEEP_TRAINING_DEFAULTS, **DEEP_MONITORING_DEFAULTS}.items():
                metadata.setdefault(name, self._inferred_metadata(value))
        for name, value in defaults.items():
            metadata[name] = {"default": value, **metadata.get(name, {})}
            metadata[name].setdefault("type", self._inferred_metadata(value)["type"])
        return metadata

    def _inferred_metadata(self, value: Any) -> dict[str, Any]:
        if isinstance(value, bool):
            return {"type": "bool", "default": value}
        if isinstance(value, int):
            return {"type": "int", "default": value}
        if isinstance(value, float):
            return {"type": "float", "default": value}
        return {"type": "text", "default": value}

    def _parameter_widget(self, metadata: dict[str, Any], value: Any) -> QWidget:
        parameter_type = metadata["type"]
        if parameter_type == "select":
            widget = AvistaComboBox()
            widget.addItems([str(option) for option in metadata.get("options", [])])
            widget.setCurrentText(self._display_select_value(value, metadata))
            return widget
        if parameter_type in {"select_or_int", "select_or_float"}:
            widget = OptionalNumberWidget(metadata)
            widget.setValue(value)
            return widget
        if parameter_type == "bool":
            widget = QCheckBox()
            widget.setChecked(bool(value))
            return widget
        if parameter_type == "int":
            widget = AvistaSpinBox()
            widget.setRange(
                int(metadata.get("min", -2_147_483_648)),
                int(metadata.get("max", 2_147_483_647)),
            )
            widget.setSingleStep(int(metadata.get("step", 1)))
            widget.setValue(int(value))
            return widget
        if parameter_type == "float":
            widget = AvistaDoubleSpinBox()
            widget.setRange(
                float(metadata.get("min", -1_000_000_000.0)),
                float(metadata.get("max", 1_000_000_000.0)),
            )
            widget.setDecimals(8)
            widget.setSingleStep(float(metadata.get("step", 0.1)))
            widget.setValue(float(value if value is not None else metadata.get("default", 0.0)))
            return widget
        widget = QLineEdit()
        widget.setText("" if value is None else value if isinstance(value, str) else json.dumps(value))
        return widget

    def _parameter_values(self, model_name: str) -> dict[str, Any]:
        metadata = self.parameter_metadata[model_name]
        values = {
            name: self._widget_value(widget, metadata[name])
            for name, widget in self.parameter_widgets[model_name].items()
        }
        for name, parameter_metadata in metadata.items():
            condition = parameter_metadata.get("enabled_when")
            if condition and not self._condition_matches(model_name, condition):
                if self.parameter_defaults[model_name][name] is None:
                    values[name] = None
        return values

    def _set_parameter_values(self, model_name: str, values: dict[str, Any]) -> None:
        defaults = self.parameter_defaults[model_name]
        for name, widget in self.parameter_widgets[model_name].items():
            if name in values:
                value = values[name]
            else:
                value = self.parameter_metadata[model_name][name].get(
                    "custom_default",
                    self.parameter_metadata[model_name][name].get("default", defaults[name]),
                )
            if (
                name == "random_state"
                and isinstance(value, int)
                and value == self.random_state.value()
                and "use_experiment_seed"
                in self.parameter_metadata[model_name][name].get("options", [])
            ):
                value = "use_experiment_seed"
            self._set_widget_value(widget, value)
        self._update_model_ui(model_name)

    def _restore_model_defaults(self, model_name: str) -> None:
        self._set_parameter_values(model_name, {})
        config = self.main_window.config
        if config is not None:
            config.model_params.pop(model_name, None)
            spec = next(spec for spec in self.model_specs if spec.name == model_name)
            config.model_params.pop(spec.display_name, None)
        display_name = next(
            spec.display_name for spec in self.model_specs if spec.name == model_name
        )
        feedback = self.restore_feedback_labels[model_name]
        feedback.setText(f"Defaults restored for {display_name}.")
        feedback.setVisible(True)

    def _widget_value(self, widget: QWidget, metadata: dict[str, Any]) -> Any:
        if isinstance(widget, OptionalNumberWidget):
            return widget.value()
        if isinstance(widget, QComboBox):
            value = widget.currentText()
            if value.casefold() == "none":
                return None
            if value.casefold() == "true":
                return True
            if value.casefold() == "false":
                return False
            if value == "use_experiment_seed":
                return self.random_state.value()
            if metadata.get("options") == ["none", "-1", "1", "2", "4", "8"]:
                return int(value)
            return value
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        text = widget.text().strip()
        if not text:
            return None
        default = metadata.get("default")
        if isinstance(default, str):
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def _set_widget_value(self, widget: QWidget, value: Any) -> None:
        if isinstance(widget, OptionalNumberWidget):
            widget.setValue(value)
        elif isinstance(widget, QComboBox):
            widget.setCurrentText("none" if value is None else str(value))
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            if value is not None:
                widget.setValue(value)
        else:
            widget.setText(
                "" if value is None else value if isinstance(value, str) else json.dumps(value)
            )

    def _display_select_value(
        self, value: Any, metadata: dict[str, Any]
    ) -> str:
        if value is None:
            return "none"
        text = str(value)
        return text if text in metadata.get("options", []) else str(metadata.get("default", text))

    def _connect_parameter_change(self, widget: QWidget, model_name: str) -> None:
        update = lambda *_args, name=model_name: self._update_model_ui(name)
        if isinstance(widget, OptionalNumberWidget):
            widget.selector.currentTextChanged.connect(update)
            widget.number_input.valueChanged.connect(update)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(update)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(update)
        elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.valueChanged.connect(update)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(update)

    def _update_model_ui(self, model_name: str) -> None:
        for name, metadata in self.parameter_metadata.get(model_name, {}).items():
            condition = metadata.get("enabled_when")
            if condition:
                self.parameter_widgets[model_name][name].setEnabled(
                    self._condition_matches(model_name, condition)
                )
        warning_label = self.parameter_warning_labels.get(model_name)
        if warning_label is None:
            return
        warnings = self._parameter_warnings(model_name)
        warning_label.setText("\n".join(f"Warning: {warning}" for warning in warnings))
        warning_label.setVisible(bool(warnings))

    def _condition_matches(
        self, model_name: str, condition: dict[str, Any]
    ) -> bool:
        dependency_name = condition["parameter"]
        dependency_widget = self.parameter_widgets[model_name][dependency_name]
        dependency_metadata = self.parameter_metadata[model_name][dependency_name]
        return self._widget_value(dependency_widget, dependency_metadata) == condition["equals"]

    def _parameter_warnings(self, model_name: str) -> list[str]:
        values = {
            name: self._widget_value(widget, self.parameter_metadata[model_name][name])
            for name, widget in self.parameter_widgets[model_name].items()
        }
        warnings: list[str] = []
        if model_name == "logistic_regression":
            penalty = values["penalty"]
            solver = values["solver"]
            if penalty == "elasticnet" and solver != "saga":
                warnings.append("Elasticnet penalty requires solver=saga.")
            if penalty == "l1" and solver not in {"liblinear", "saga"}:
                warnings.append("L1 penalty works only with solver=liblinear or solver=saga.")
            if penalty is None and solver == "liblinear":
                warnings.append("penalty=none is not supported with solver=liblinear.")
            if values["dual"] and not (penalty == "l2" and solver == "liblinear"):
                warnings.append("dual=True is only valid for l2 penalty with solver=liblinear.")
        if model_name in {"random_forest", "extra_trees"}:
            if values["oob_score"] and not values["bootstrap"]:
                warnings.append("oob_score=True requires bootstrap=True.")
            if values["max_samples"] is not None and not values["bootstrap"]:
                warnings.append("max_samples is only valid when bootstrap=True.")
        target_class_count = self._target_class_count()
        if model_name == "xgboost":
            if target_class_count is not None and target_class_count > 2:
                if values["objective"] != "multi:softprob":
                    warnings.append(
                        "Multiclass targets should use objective=multi:softprob."
                    )
                if values["scale_pos_weight"] != 1.0:
                    warnings.append(
                        "scale_pos_weight is mainly intended for binary classification."
                    )
            if target_class_count == 2 and values["objective"] != "binary:logistic":
                warnings.append(
                    "Binary targets may use objective=binary:logistic."
                )
            if values["enable_categorical"]:
                warnings.append(
                    "enable_categorical=True requires compatible category-preserving input."
                )
        if model_name == "gradient_boosting":
            if values["n_iter_no_change"] is None:
                warnings.append(
                    "validation_fraction is used only when n_iter_no_change is not None."
                )
            if (
                values["loss"] == "exponential"
                and target_class_count is not None
                and target_class_count > 2
            ):
                warnings.append(
                    "exponential loss is intended for binary classification."
                )
        if model_name == "hist_gradient_boosting":
            if values["categorical_features"] == "from_dtype":
                warnings.append(
                    "categorical_features=from_dtype requires preserved categorical dtypes; "
                    "use none for one-hot encoded data."
                )
        if model_name == "adaboost":
            warnings.append("Base estimator customization is not currently implemented.")
            if values["n_estimators"] >= 5000:
                warnings.append("Very large n_estimators may slow training.")
        monotonic_widget = self.parameter_widgets[model_name].get("monotonic_cst")
        if isinstance(monotonic_widget, QLineEdit) and monotonic_widget.text().strip():
            try:
                parsed = json.loads(monotonic_widget.text())
                if not isinstance(parsed, list) or not all(
                    isinstance(value, int) and value in {-1, 0, 1} for value in parsed
                ):
                    raise ValueError
            except (json.JSONDecodeError, ValueError):
                warnings.append(
                    "monotonic_cst must be a JSON list containing only -1, 0, and 1."
                )
        return warnings

    def _target_class_count(self) -> int | None:
        config = self.main_window.config
        dataframe = self.main_window.dataframe
        target = config.target_column if config is not None else None
        if dataframe is None or not target or target not in dataframe.columns:
            return None
        return int(dataframe[target].dropna().nunique())

    def _model_toggled(self, model_name: str, checked: bool) -> None:
        if checked:
            self._show_parameter_panel(model_name)
        elif self.parameter_stack.currentWidget() is self.parameter_panels[model_name]:
            self._show_first_checked_panel()

    def _show_parameter_panel(self, model_name: str) -> None:
        self.parameter_stack.setCurrentWidget(self.parameter_panels[model_name])

    def _show_first_checked_panel(self) -> None:
        for spec in self.model_specs:
            if self.model_checkboxes[spec.name].isChecked():
                self.parameter_stack.setCurrentWidget(self.parameter_panels[spec.name])
                return
        self.parameter_stack.setCurrentIndex(0)

    def _empty_parameter_panel(self) -> QWidget:
        panel = ModelParameterPanel()
        layout = panel.content_layout
        label = QLabel("Select a model to configure its parameters.")
        label.setObjectName("modelParameterEmptyMessage")
        label.setWordWrap(True)
        layout.addWidget(label)
        return panel

    def _global_options_group(self) -> QGroupBox:
        group = QGroupBox("Global Training Options")
        form = QFormLayout(group)
        form.addRow("", self.enable_cross_validation)
        form.addRow("Number of CV folds", self.cv_folds)
        form.addRow("Random State", self.random_state)
        return group

    def _show_error(self, message: str) -> None:
        self.success_notification_timer.stop()
        self.feedback_card.setStyleSheet(self._feedback_style("error"))
        self.feedback_label.setText(f"Error: {message}")
        for index, (icon_label, text_label) in enumerate(
            zip(self.feedback_icons, self.feedback_labels)
        ):
            visible = index == 0
            text_label.parentWidget().setVisible(visible)
            if visible:
                icon_label.setPixmap(icon("fa6s.circle-exclamation", ERROR_COLOR).pixmap(16, 16))
                text_label.setText(f"Error: {message}")
        self.feedback_card.show()

    def _show_success(self, message: str) -> None:
        self._show_success_notification([message])

    def _display_parameter_name(self, name: str) -> str:
        return name.replace("_", " ").title()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#modelSelectionContent {{ background: {BACKGROUND}; color: {TEXT}; }}
            QLabel#modelSelectionTitle {{ font-size: 24px; font-weight: 700; }}
            QLabel#modelCardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLabel#modelCardSubtitle,
            QLabel#modelParameterEmptyMessage {{
                color: #5B6573;
                font-size: 12px;
            }}
            QWidget#modelLibraryCard,
            QWidget#modelParametersCard,
            QWidget#globalTrainingOptionsCard,
            QWidget#modelConfirmationStatusCard,
            QWidget#modelSelectionEmptyCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QFrame#modelInnerCategoryCard {{
                background: #F7F9FC;
                border: 1px solid {BORDER};
                border-radius: 8px;
            }}
            QScrollArea#modelLibraryListScroll,
            QScrollArea#modelParameterPanelScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#modelLibraryListContent,
            QWidget#modelParameterPanelContent {{
                background: transparent;
                border: none;
            }}
            QLabel#modelInnerCardTitle,
            QLabel#modelParameterHeading {{
                color: {TEXT};
                font-weight: 700;
                font-size: 13px;
            }}
            QWidget#modelSelectionModelRow {{
                background: transparent;
                border: none;
            }}
            QLabel#modelDependencyLabel {{
                color: #B45309;
                font-size: 12px;
            }}
            QLabel#modelSelectionEmptyMessage {{
                color: #5B6573;
                font-size: 13px;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
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
            QComboBox, QSpinBox, QDoubleSpinBox {{
                padding-right: 26px;
            }}
            QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {PRIMARY};
            }}
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {PRIMARY};
            }}
            QComboBox::drop-down {{
                width: 0px;
                border: none;
            }}
            QComboBox::down-arrow {{
                width: 0px;
                height: 0px;
            }}
            QToolButton#modelComboArrowButton,
            QToolButton#modelSpinArrowButton {{
                background: #F8FAFC;
                border: none;
                border-left: 1px solid {BORDER};
                padding: 0;
                margin: 0;
            }}
            QToolButton#modelComboArrowButton {{
                border-top-right-radius: 7px;
                border-bottom-right-radius: 7px;
            }}
            QToolButton#modelSpinArrowButton:hover,
            QToolButton#modelComboArrowButton:hover {{
                background: #EFF6FF;
                border-left-color: {PRIMARY};
            }}
            QCheckBox {{
                color: {TEXT};
                spacing: 8px;
            }}
            QGroupBox#modelParameterSection {{
                border: 1px solid {BORDER};
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px 10px 10px 10px;
                font-weight: 700;
            }}
            QPushButton {{
                min-height: 34px;
                border-radius: 6px;
                padding: 0 12px;
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            QPushButton:hover {{
                background: #EFF6FF;
                border-color: {PRIMARY};
            }}
            QPushButton#primaryModelSelectionButton {{
                min-height: 44px;
                color: #FFFFFF;
                background: {PRIMARY};
                border: none;
                border-radius: 7px;
                font-weight: 600;
            }}
            QPushButton#primaryModelSelectionButton:hover {{ background: #00A6A6; }}
            QPushButton#secondaryModelButton {{
                color: {PRIMARY};
                background: #FFFFFF;
                border: 1px solid {BORDER};
            }}
            """
        )
