"""AVISTA project-file configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.__version__ import APP_NAME, __version__


PROJECT_FILE_EXTENSION = ".avista"
LEGACY_PROJECT_FILE_EXTENSION = ".xtab"
PROJECT_FILE_VERSION = "1.0"
LEGACY_PROJECT_FILENAME = "project_config.json"


@dataclass
class ProjectConfig:
    """User-selected settings stored in an AVISTA project file."""

    project_name: str
    project_dir: str
    input_file: str
    output_dir: str
    project_file_path: str = ""
    project_file_version: str = PROJECT_FILE_VERSION
    application: str = APP_NAME
    application_version: str = __version__
    dataset: dict[str, Any] = field(default_factory=dict)
    target_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    label_encoding_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    group_column: str | None = None
    date_column: str | None = None
    excluded_columns: list[str] = field(default_factory=list)
    subgroup_columns: list[str] = field(default_factory=list)
    task_type: str | None = None
    split_method: str | None = None
    train_percent: float = 70.0
    validation_percent: float = 10.0
    test_percent: float = 20.0
    random_seed: int = 42
    imbalance_method: str | None = None
    use_class_weights: bool = False
    smote_ratio_preset: str = "baseline"
    selected_models: list[str] = field(default_factory=list)
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    enable_cross_validation: bool = False
    cv_folds: int = 5
    random_state: int = 42
    preprocessing_options: dict[str, Any] = field(default_factory=dict)
    xai_options: dict[str, Any] = field(default_factory=dict)
    environment_mode: str = "packaged_runtime"

    @property
    def project_file(self) -> Path:
        """Return the canonical absolute project-file path."""

        if self.project_file_path:
            path = Path(self.project_file_path)
            if not path.is_absolute():
                path = Path(self.project_dir) / path
            return path.resolve()
        return (
            Path(self.project_dir)
            / f"{self.project_name}{PROJECT_FILE_EXTENSION}"
        ).resolve()

    def project_metadata(self) -> dict[str, str]:
        """Return project identity fields for generated metadata files."""

        return {
            "project_file": str(self.project_file),
            "project_name": self.project_name,
            "project_file_version": self.project_file_version,
            "application": self.application,
            "application_version": self.application_version,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """Save the project as JSON-formatted ``.avista`` and return its path."""

        output_path = Path(path) if path is not None else self.project_file
        if output_path.suffix.casefold() != PROJECT_FILE_EXTENSION:
            raise ValueError("AVISTA project files must use the .avista extension.")
        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.project_dir = str(output_path.parent)
        self.project_file_path = str(output_path)
        self.project_file_version = PROJECT_FILE_VERSION
        self.application = APP_NAME
        self.application_version = __version__
        output_path.write_text(
            json.dumps(self._serialized_data(output_path.parent), indent=2),
            encoding="utf-8",
        )
        log_metadata = output_path.parent / "logs" / "project_metadata.json"
        log_metadata.parent.mkdir(parents=True, exist_ok=True)
        log_metadata.write_text(
            json.dumps(self.project_metadata(), indent=2),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ProjectConfig":
        """Load AVISTA projects and migrate legacy project formats."""

        source_path = Path(path).resolve()
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Project file does not exist: {source_path}")
        if source_path.suffix.casefold() == PROJECT_FILE_EXTENSION:
            return cls._load_project_data(source_path)
        if source_path.suffix.casefold() == LEGACY_PROJECT_FILE_EXTENSION:
            config = cls._load_project_data(source_path, legacy=True)
            config.project_file_path = str(
                source_path.with_suffix(PROJECT_FILE_EXTENSION)
            )
            config.save()
            return config
        if source_path.name.casefold() == LEGACY_PROJECT_FILENAME:
            config = cls._load_project_data(source_path, legacy=True)
            config.project_file_path = str(
                source_path.parent
                / f"{config.project_name}{PROJECT_FILE_EXTENSION}"
            )
            config.save()
            return config
        raise ValueError(
            "Select an AVISTA project file with the .avista extension "
            "or a legacy .xtab project."
        )

    def save_json(self, path: str | Path | None = None) -> Path:
        """Backward-compatible alias for :meth:`save`."""

        return self.save(path)

    @classmethod
    def load_json(cls, path: str | Path) -> "ProjectConfig":
        """Backward-compatible alias for :meth:`load`."""

        return cls.load(path)

    def _serialized_data(self, project_dir: Path) -> dict[str, Any]:
        data = asdict(self)
        data["project_dir"] = "."
        data["project_file_path"] = self.project_file.name
        data["input_file"] = _portable_path(self.input_file, project_dir)
        data["output_dir"] = _portable_path(self.output_dir, project_dir)
        if data["dataset"].get("project_relative_path"):
            data["dataset"]["project_relative_path"] = Path(
                data["dataset"]["project_relative_path"]
            ).as_posix()
        if data["dataset"].get("copied_project_path"):
            data["dataset"]["copied_project_path"] = Path(
                data["dataset"]["copied_project_path"]
            ).as_posix()
        return data

    @classmethod
    def _load_project_data(
        cls,
        source_path: Path,
        *,
        legacy: bool = False,
    ) -> "ProjectConfig":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Project file must contain a JSON object.")
        project_dir = source_path.parent
        data["project_dir"] = str(project_dir)
        data["project_file_path"] = str(source_path)
        data["project_file_version"] = str(
            data.get("project_file_version", PROJECT_FILE_VERSION)
        )
        data["application"] = APP_NAME
        data["application_version"] = __version__
        dataset = dict(data.get("dataset", {}) or {})
        relative_dataset = str(dataset.get("project_relative_path", "")).strip()
        data["dataset"] = dataset
        data["input_file"] = _resolved_project_path(
            relative_dataset or data.get("input_file", ""),
            project_dir,
        )
        data["output_dir"] = _resolved_project_path(
            data.get("output_dir", "outputs"),
            project_dir,
        )
        if legacy:
            data["project_file_version"] = PROJECT_FILE_VERSION
            data["application"] = APP_NAME
            data["application_version"] = __version__
        return cls(**data)


def _portable_path(value: str, project_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _resolved_project_path(value: str, project_dir: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        path = project_dir / path
    return str(path.resolve())
