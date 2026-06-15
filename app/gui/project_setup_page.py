"""Project setup page with AVISTA create and open workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.dataset_manager import (
    DUPLICATE_APPEND_TIMESTAMP,
    DUPLICATE_CANCEL,
    DUPLICATE_OVERWRITE,
    copy_dataset_into_project,
)
from app.core.project_config import PROJECT_FILE_EXTENSION, ProjectConfig
from app.gui.icon_system import (
    ACCENT,
    BACKGROUND,
    BORDER,
    FEEDBACK_COLORS,
    FEEDBACK_ICONS,
    PRIMARY,
    TEXT,
    icon,
)


PROJECT_SUBDIRS = ["data", "outputs", "logs", "artifacts"]
PROJECT_FILE_FILTER = "AVISTA Project (*.avista)"
OPEN_PROJECT_FILE_FILTER = (
    f"{PROJECT_FILE_FILTER};;Legacy Project (*.xtab);;"
    "Legacy Project (project_config.json)"
)


class FeedbackCard(QWidget):
    """Reusable icon-led feedback card for project actions."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("feedbackCard")
        self.icon_label = QLabel()
        self.icon_label.setFixedWidth(24)
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        layout.addWidget(self.icon_label)
        layout.addWidget(self.message_label, stretch=1)
        self.hide()

    def show_message(self, message: str, level: str) -> None:
        border, background = FEEDBACK_COLORS[level]
        self.icon_label.setPixmap(icon(FEEDBACK_ICONS[level], border).pixmap(20, 20))
        self.message_label.setText(message)
        self.setStyleSheet(
            f"""
            QWidget#feedbackCard {{
                background: {background};
                border: 1px solid {border};
                border-radius: 8px;
            }}
            QLabel {{
                color: {TEXT};
                border: none;
                background: transparent;
            }}
            """
        )
        self.show()


class ProjectSetupPage(QWidget):
    def __init__(self, main_window) -> None:
        super().__init__()
        self.main_window = main_window
        self.setObjectName("projectSetupPage")

        self.project_name_input = self._line_edit("Project name")
        self.project_parent_input = self._line_edit("Choose a parent folder")
        self.project_dir_input = self.project_parent_input
        self.input_file_input = self._line_edit("Choose an initial tabular dataset")
        self.existing_project_file_input = self._line_edit("Choose an AVISTA project")
        self.existing_project_dir_input = self.existing_project_file_input

        self.feedback_card = FeedbackCard()
        self.status_label = self.feedback_card.message_label
        self.create_project_info_label = self.feedback_card.message_label
        self.load_project_info_label = self.feedback_card.message_label

        self.project_loaded_value = QLabel("No")
        self.current_project_name_value = QLabel("Not loaded")
        self.current_project_file_value = QLabel("Not available")
        self.current_dataset_value = QLabel("Not available")
        self.current_modified_value = QLabel("Not available")
        for value in self._status_values():
            value.setTextInteractionFlags(value.textInteractionFlags())
            value.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Project Setup")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Create a portable AVISTA workspace or open an existing project."
        )
        subtitle.setObjectName("pageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        action_row = QHBoxLayout()
        action_row.setSpacing(16)
        self.create_project_card = self._build_create_card()
        self.open_project_card = self._build_open_card()
        action_row.addWidget(self.create_project_card, stretch=1)
        action_row.addWidget(self.open_project_card, stretch=1)
        layout.addLayout(action_row)

        self.current_project_card = self._build_current_project_card()
        layout.addWidget(self.current_project_card)
        layout.addWidget(self.feedback_card)
        layout.addStretch(1)
        self._apply_page_style()
        self._refresh_current_project()

    def create_project(self) -> None:
        project_name = self.project_name_input.text().strip()
        parent_dir = self.project_parent_input.text().strip()
        input_file = self.input_file_input.text().strip()

        if not project_name:
            self._show_error("Project name is required.")
            return
        if not parent_dir:
            self._show_error("Parent folder is required.")
            return
        if (
            Path(project_name).name != project_name
            or project_name.casefold().endswith((".avista", ".xtab"))
        ):
            self._show_error(
                "Enter a project name without path separators or a project extension."
            )
            return
        if not input_file:
            self._show_error("Initial input dataset is required.")
            return
        if not Path(input_file).exists():
            self._show_error(f"Input data file does not exist: {input_file}")
            return

        project_path = Path(parent_dir) / project_name
        project_file = project_path / f"{project_name}{PROJECT_FILE_EXTENSION}"
        if project_file.exists():
            self._show_error(f"Project file already exists: {project_file}")
            return
        try:
            project_path.mkdir(parents=True, exist_ok=True)
            for subdir in PROJECT_SUBDIRS:
                (project_path / subdir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._show_error(f"Could not create project folders: {exc}")
            return

        try:
            dataset = copy_dataset_into_project(
                input_file,
                project_path,
                duplicate_policy=self._duplicate_policy(
                    project_path / "data" / Path(input_file).name
                ),
            )
        except FileExistsError:
            self._show_warning("Project creation cancelled.")
            return
        except Exception as exc:
            self._show_error(f"Could not copy dataset into project: {exc}")
            return

        config = ProjectConfig(
            project_name=project_name,
            project_dir=str(project_path),
            project_file_path=str(project_file),
            input_file=str(project_path / dataset["project_relative_path"]),
            output_dir=str(project_path / "outputs"),
            dataset=dataset,
        )
        try:
            config_path = config.save()
        except Exception as exc:
            self._show_error(f"Could not save project: {exc}")
            return
        self.main_window.set_config(config)
        self._populate_create_fields(config)
        self.existing_project_file_input.setText(str(config_path))
        self._show_saved(config)

    def load_project(self) -> None:
        project_file = self.existing_project_file_input.text().strip()
        if not project_file:
            self._show_error("Select an AVISTA project file.")
            return

        try:
            config = ProjectConfig.load(project_file)
        except Exception as exc:
            self._show_error(f"Could not open project: {exc}")
            return

        self.main_window.set_config(config)
        self._populate_create_fields(config)
        self.existing_project_file_input.setText(str(config.project_file))
        self._show_loaded(config)

    def _build_create_card(self) -> QWidget:
        card, content = self._card(
            "createProjectCard",
            "Create New Project",
            "Start a managed AVISTA project with its own data and output folders.",
            "fa6s.folder-plus",
        )
        form = self._form()
        form.addRow("Project name", self.project_name_input)
        form.addRow(
            "Parent folder",
            self._path_row(self.project_parent_input, self._select_project_location),
        )
        form.addRow(
            "Initial input data file",
            self._path_row(self.input_file_input, self._select_input_file),
        )
        content.addLayout(form)
        content.addStretch(1)
        self.create_project_button = self._primary_button(
            "Create Project", "fa6s.circle-plus", self.create_project
        )
        content.addWidget(self.create_project_button)
        return card

    def _build_open_card(self) -> QWidget:
        card, content = self._card(
            "openProjectCard",
            "Open Existing Project",
            "Continue from an AVISTA project or migrate a supported legacy file.",
            "fa6s.folder-open",
        )
        form = self._form()
        form.addRow(
            "Project file",
            self._path_row(
                self.existing_project_file_input,
                self._select_existing_project_file,
            ),
        )
        content.addLayout(form)
        content.addStretch(1)
        self.open_project_button = self._primary_button(
            "Open Project", "fa6s.folder-open", self.load_project
        )
        content.addWidget(self.open_project_button)
        return card

    def _build_current_project_card(self) -> QWidget:
        card, content = self._card(
            "currentProjectCard",
            "Current Project",
            "Active project identity and managed dataset location.",
            "fa6s.diagram-project",
        )
        form = self._form()
        form.addRow("Project loaded", self.project_loaded_value)
        form.addRow("Project name", self.current_project_name_value)
        form.addRow("Project file path", self.current_project_file_value)
        form.addRow("Dataset path", self.current_dataset_value)
        form.addRow("Last modified", self.current_modified_value)
        content.addLayout(form)
        return card

    def _card(
        self,
        object_name: str,
        title_text: str,
        description: str,
        icon_name: str,
    ) -> tuple[QWidget, QVBoxLayout]:
        card = QWidget()
        card.setObjectName(object_name)
        shadow = QGraphicsDropShadowEffect(card)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(31, 41, 55, 28))
        shadow.setOffset(0, 4)
        card.setGraphicsEffect(shadow)
        content = QVBoxLayout(card)
        content.setContentsMargins(22, 20, 22, 20)
        content.setSpacing(12)

        header = QHBoxLayout()
        header.setSpacing(10)
        icon_label = QLabel()
        icon_label.setPixmap(icon(icon_name).pixmap(24, 24))
        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        header.addWidget(icon_label)
        header.addWidget(title)
        header.addStretch(1)
        content.addLayout(header)

        description_label = QLabel(description)
        description_label.setObjectName("cardDescription")
        description_label.setWordWrap(True)
        content.addWidget(description_label)
        return card, content

    def _form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 4, 0, 0)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        return form

    def _line_edit(self, placeholder: str) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setMinimumHeight(38)
        return line_edit

    def _primary_button(self, text: str, icon_name: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("primaryButton")
        button.setIcon(icon(icon_name, "#FFFFFF"))
        button.setMinimumHeight(42)
        button.clicked.connect(callback)
        return button

    def _populate_create_fields(self, config: ProjectConfig) -> None:
        self.project_name_input.setText(config.project_name)
        self.project_parent_input.setText(str(Path(config.project_dir).parent))
        self.input_file_input.setText(config.input_file)

    def _show_saved(self, config: ProjectConfig) -> None:
        relative_dataset = config.dataset.get("project_relative_path", "")
        message = "\n".join(
            [
                "Project saved successfully.",
                f"Project file: {config.project_file}",
                "Dataset copied into project:",
                relative_dataset,
            ]
        )
        self.feedback_card.show_message(message, "success")
        self._refresh_current_project(config)

    def _show_loaded(self, config: ProjectConfig) -> None:
        message = "\n".join(
            [
                f"Project loaded: {config.project_name}",
                f"Project file: {config.project_file}",
            ]
        )
        self.feedback_card.show_message(message, "success")
        self._refresh_current_project(config)

    def _show_error(self, message: str) -> None:
        self.feedback_card.show_message(f"Error: {message}", "error")

    def _show_warning(self, message: str) -> None:
        self.feedback_card.show_message(f"Warning: {message}", "warning")

    def _refresh_current_project(self, config: ProjectConfig | None = None) -> None:
        config = config or self.main_window.config
        if config is None:
            self.project_loaded_value.setText("No")
            self.current_project_name_value.setText("Not loaded")
            self.current_project_file_value.setText("Not available")
            self.current_dataset_value.setText("Not available")
            self.current_modified_value.setText("Not available")
            return
        project_file = config.project_file
        modified = (
            datetime.fromtimestamp(project_file.stat().st_mtime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            if project_file.exists()
            else "Not available"
        )
        self.project_loaded_value.setText("Yes")
        self.current_project_name_value.setText(config.project_name)
        self.current_project_file_value.setText(str(project_file))
        self.current_dataset_value.setText(config.input_file or "Not configured")
        self.current_modified_value.setText(modified)

    def _status_values(self) -> list[QLabel]:
        return [
            self.project_loaded_value,
            self.current_project_name_value,
            self.current_project_file_value,
            self.current_dataset_value,
            self.current_modified_value,
        ]

    def _select_project_location(self) -> None:
        suggested_name = self.project_name_input.text().strip() or "MyProject"
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Create AVISTA Project",
            f"{suggested_name}{PROJECT_FILE_EXTENSION}",
            PROJECT_FILE_FILTER,
        )
        if selected:
            path = Path(selected)
            self.project_parent_input.setText(str(path.parent))
            self.project_name_input.setText(path.stem)

    def _select_existing_project_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Open AVISTA Project",
            "",
            OPEN_PROJECT_FILE_FILTER,
        )
        if selected:
            self.existing_project_file_input.setText(selected)

    def _select_input_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input data file",
            "",
            "Tabular files (*.csv *.xlsx *.parquet *.feather *.fst);;All files (*)",
        )
        if file_path:
            self.input_file_input.setText(file_path)

    def _duplicate_policy(self, destination: Path) -> str:
        if not destination.exists():
            return DUPLICATE_APPEND_TIMESTAMP
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Dataset Already Exists")
        dialog.setIconPixmap(icon(FEEDBACK_ICONS["warning"], "#BF6A02").pixmap(32, 32))
        dialog.setText(f"A dataset named '{destination.name}' already exists.")
        overwrite = dialog.addButton("Overwrite", QMessageBox.ButtonRole.DestructiveRole)
        keep_both = dialog.addButton("Keep Both", QMessageBox.ButtonRole.AcceptRole)
        cancel = dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.setDefaultButton(keep_both)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is overwrite:
            return DUPLICATE_OVERWRITE
        if clicked is cancel:
            return DUPLICATE_CANCEL
        return DUPLICATE_APPEND_TIMESTAMP

    def _path_row(self, line_edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(line_edit)
        button = QPushButton("Browse")
        button.setObjectName("browseButton")
        button.setIcon(icon("fa6s.folder"))
        button.setMinimumHeight(38)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return row

    def _apply_page_style(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget#projectSetupPage {{
                background: {BACKGROUND};
                color: {TEXT};
            }}
            QLabel#pageTitle {{
                color: {TEXT};
                font-size: 24px;
                font-weight: 700;
            }}
            QLabel#pageSubtitle, QLabel#cardDescription {{
                color: #5B6573;
                font-size: 12px;
            }}
            QWidget#createProjectCard,
            QWidget#openProjectCard,
            QWidget#currentProjectCard {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 10px;
            }}
            QLabel#cardTitle {{
                color: {TEXT};
                font-size: 16px;
                font-weight: 700;
            }}
            QLineEdit {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 0 10px;
                color: {TEXT};
            }}
            QLineEdit:focus {{
                border: 1px solid {PRIMARY};
            }}
            QPushButton#browseButton {{
                background: #FFFFFF;
                color: {PRIMARY};
                border: 1px solid {BORDER};
                border-radius: 6px;
                padding: 0 12px;
            }}
            QPushButton#browseButton:hover {{
                background: #EFF6FF;
                border-color: {PRIMARY};
            }}
            QPushButton#primaryButton {{
                background: {PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 7px;
                padding: 0 18px;
                font-weight: 600;
            }}
            QPushButton#primaryButton:hover {{
                background: {ACCENT};
            }}
            """
        )
