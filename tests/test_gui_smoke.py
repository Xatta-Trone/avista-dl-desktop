import importlib.util
import json

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("PySide6") is None, reason="PySide6 is not installed.")


def _table_counts(table):
    return {
        table.item(row, 0).text(): int(table.item(row, 1).text())
        for row in range(table.rowCount())
    }


def test_main_window_smoke():
    from PySide6.QtWidgets import QApplication, QLabel

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()

    assert window.windowTitle() == "AVISTA"
    assert window.stack.count() == 9
    assert [label for label, _, _ in window.pages][5] == "Model Selection"
    assert all(not button.icon().isNull() for button in window.nav_buttons)
    assert all(button.objectName() == "sidebarButton" for button in window.nav_buttons)
    assert not window.windowIcon().isNull()
    sidebar_logo = window.findChild(QLabel, "sidebarLogo")
    assert sidebar_logo is not None
    assert sidebar_logo.pixmap() is not None
    assert not sidebar_logo.pixmap().isNull()
    window.close()
    assert app is not None


def test_qtawesome_icons_load():
    from PySide6.QtWidgets import QApplication

    from app.gui.icon_system import FEEDBACK_ICONS, PAGE_ICONS, icon

    app = QApplication.instance() or QApplication([])

    assert all(not icon(name).isNull() for name in PAGE_ICONS.values())
    assert all(not icon(name).isNull() for name in FEEDBACK_ICONS.values())
    assert app is not None


def test_project_setup_renders_cards_and_hides_environment_mode():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.project_setup_page

    assert page.create_project_card.objectName() == "createProjectCard"
    assert page.open_project_card.objectName() == "openProjectCard"
    assert page.current_project_card.objectName() == "currentProjectCard"
    assert page.create_project_card.parent() is page
    assert page.open_project_card.parent() is page
    assert not hasattr(page, "environment_mode_input")
    assert page.create_project_button.icon().isNull() is False
    assert page.open_project_button.icon().isNull() is False
    assert page.project_loaded_value.text() == "No"
    window.close()
    assert app is not None


def test_about_dialog_uses_logo_branding_and_clickable_profiles(monkeypatch):
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication

    from app.__version__ import APP_NAME, __version__
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    opened_urls = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        lambda url: opened_urls.append(url.toString()) or True,
    )
    window = MainWindow()
    dialog = window.create_about_dialog()
    dialog.show()
    app.processEvents()

    dialog.github_label.linkActivated.emit("https://github.com/xatta-trone")
    dialog.github_label.linkActivated.emit("https://github.com/shriyanksomvanshi")

    assert dialog.windowTitle() == f"About {APP_NAME}"
    assert dialog.isVisible()
    assert dialog.title_label.text() == APP_NAME
    assert dialog.version_label.text() == f"Version {__version__}"
    assert dialog.description_label.text() == (
        "Automated Vehicle Infrastructure-Sensitive Tabular Analysis"
    )
    assert dialog.logo_label.pixmap() is not None
    assert not dialog.logo_label.pixmap().isNull()
    assert not dialog.windowIcon().isNull()
    assert "Md Monzurul Islam (Xatta Trone)" in dialog.developers_label.text()
    assert "Shriyank Somvanshi" in dialog.developers_label.text()
    assert 'href="https://github.com/xatta-trone"' in dialog.github_label.text()
    assert 'href="https://github.com/shriyanksomvanshi"' in dialog.github_label.text()
    assert dialog.github_label.openExternalLinks() is False
    assert opened_urls == [
        "https://github.com/xatta-trone",
        "https://github.com/shriyanksomvanshi",
    ]
    dialog.close()
    window.close()
    assert app is not None


def test_environment_gpu_status_text():
    from app.gui.environment_page import _gpu_status_text

    assert _gpu_status_text({"cuda_available": True, "tensor_test_passed": True}) == "Ready"
    assert (
        _gpu_status_text({"cuda_available": True, "tensor_test_passed": False})
        == "CUDA Detected but Validation Failed"
    )
    assert (
        _gpu_status_text({"cuda_available": False, "nvidia_gpu_detected": True})
        == "NVIDIA GPU Found, CUDA PyTorch Not Active"
    )
    assert _gpu_status_text({"cuda_available": False, "nvidia_gpu_detected": False}) == "CPU Mode"


def test_environment_page_loads_cpu_gpu_and_memory_cards():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.environment_page

    assert page.cpu_card.values["cpu_name"].text() != "Unknown"
    assert page.cpu_card.values["architecture"].text() != "Unknown"
    assert page.memory_card.values["ram_total_bytes"].text() != "Unknown"
    assert page.gpu_card.values["gpu_name"].text() == "Unknown"
    assert page.gpu_card.icon_label.pixmap() is not None
    assert not page.gpu_card.icon_label.pixmap().isNull()
    assert page.run_gpu_button.text() == "Run GPU Check"
    assert page.refresh_system_button.text() == "Refresh System Info"
    assert page.repair_gpu_button.text() == "Repair GPU Runtime"
    assert page.repair_gpu_button.isHidden()
    assert not hasattr(page, "mode_label")
    assert page.cpu_card is not page.gpu_card
    assert page.gpu_card is not page.memory_card
    assert page.cpu_card is not page.memory_card
    assert page.cpu_card.objectName() == "cpuEnvironmentCard"
    assert page.gpu_card.objectName() == "gpuEnvironmentCard"
    assert page.memory_card.objectName() == "memoryEnvironmentCard"
    assert page.cpu_card.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert page.gpu_card.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert page.memory_card.testAttribute(Qt.WidgetAttribute.WA_StyledBackground)
    assert page.cpu_card.graphicsEffect() is not None
    assert page.gpu_card.graphicsEffect() is not None
    assert page.memory_card.graphicsEffect() is not None
    assert page.refresh_timer.interval() == 30_000
    assert page.refresh_timer.isActive()
    window.close()
    assert app is not None


def test_environment_refresh_system_info_updates_cpu_and_memory(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.environment_page
    monkeypatch.setattr(
        "app.gui.environment_page.collect_environment_info",
        lambda **_kwargs: {
            "psutil_available": True,
            "cpu_name": "Refreshed CPU",
            "physical_cores": 8,
            "logical_cores": 16,
            "cpu_usage_percent": 42.5,
            "architecture": "AMD64",
            "ram_total_bytes": 32 * 1024**3,
            "ram_available_bytes": 20 * 1024**3,
            "ram_used_percent": 37.5,
            "disk_free_bytes": 100 * 1024**3,
            "system_info_error": None,
        },
    )

    page.refresh_system_info()

    assert page.cpu_card.values["cpu_name"].text() == "Refreshed CPU"
    assert page.cpu_card.values["cpu_usage_percent"].text() == "42.5%"
    assert page.memory_card.values["ram_total_bytes"].text() == "32.0 GB"
    assert page.memory_card.values["ram_used_percent"].text() == "37.5%"
    assert page.status_label.text() == "System information refreshed."
    window.close()
    assert app is not None


def test_environment_gpu_card_repair_visibility_matches_gpu_state():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.environment_page

    page._show_gpu_info(
        {
            "cuda_available": True,
            "tensor_test_passed": True,
            "nvidia_gpu_detected": True,
            "gpu_count": 1,
        }
    )
    assert page.gpu_card.badge.text() == "OK"
    assert page.repair_gpu_button.isHidden()

    page._show_gpu_info(
        {
            "cuda_available": False,
            "tensor_test_passed": False,
            "nvidia_gpu_detected": True,
            "gpu_count": 0,
        }
    )
    assert page.gpu_card.badge.text() == "Warning"
    assert not page.repair_gpu_button.isHidden()
    assert page.gpu_card.message_label.text() == (
        "NVIDIA GPU detected, but CUDA PyTorch is not active."
    )

    page._show_gpu_info(
        {
            "cuda_available": False,
            "tensor_test_passed": False,
            "nvidia_gpu_detected": False,
            "gpu_count": 0,
        }
    )
    assert page.gpu_card.badge.text() == "Not available"
    assert page.repair_gpu_button.isHidden()
    assert page.gpu_card.message_label.text() == "No NVIDIA GPU detected."
    window.close()
    assert app is not None


def test_environment_gpu_repair_uses_existing_function_in_background(
    tmp_path, monkeypatch
):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="environment-repair",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )
    config.save()
    window.config = config
    calls = []

    def fake_repair(project_dir):
        calls.append(project_dir)
        return {"success": True, "message": "Existing repair completed."}

    monkeypatch.setattr("app.gui.workers.repair_gpu_torch", fake_repair)
    monkeypatch.setattr(
        "app.gui.workers.check_gpu",
        lambda: {
            "torch_installed": True,
            "cuda_available": True,
            "gpu_count": 1,
            "gpu_name": "Repaired GPU",
            "torch_cuda_version": "12.6",
            "cudnn_version": 90100,
            "tensor_test_passed": True,
            "nvidia_smi_available": True,
            "nvidia_gpu_detected": True,
            "nvidia_smi_gpu_name": "Repaired GPU",
            "driver_version": "555.99",
            "gpu_memory_total_mb": 8192,
            "gpu_memory_used_mb": 1024,
            "gpu_memory_free_mb": 7168,
            "error": None,
        },
    )
    page = window.environment_page
    page._show_gpu_info(
        {
            "cuda_available": False,
            "tensor_test_passed": False,
            "nvidia_gpu_detected": True,
        }
    )

    page.repair_gpu_runtime()

    assert page.thread is not None
    assert not page.repair_gpu_button.isEnabled()
    assert page.status_label.text() == "Repairing GPU runtime..."
    assert window.isEnabled()

    for _ in range(100):
        app.processEvents()
        if page.repair_gpu_button.isEnabled():
            break
        QTest.qWait(10)

    assert calls == [str(tmp_path)]
    assert page.gpu_card.badge.text() == "OK"
    assert page.gpu_card.values["gpu_name"].text() == "Repaired GPU"
    assert page.repair_gpu_button.isHidden()
    assert "Existing repair completed." in page.status_label.text()
    while page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()
    assert app is not None


def test_environment_gpu_worker_is_non_blocking_and_updates_success(
    tmp_path, monkeypatch
):
    import time

    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="environment-success",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )
    config.save()
    window.config = config

    def fake_check_gpu():
        time.sleep(0.05)
        return {
            "torch_installed": True,
            "cuda_available": True,
            "gpu_count": 1,
            "gpu_name": "Test GPU",
            "torch_cuda_version": "12.6",
            "cudnn_version": 90100,
            "tensor_test_passed": True,
            "nvidia_smi_available": True,
            "nvidia_gpu_detected": True,
            "nvidia_smi_gpu_name": "Test GPU",
            "driver_version": "555.99",
            "gpu_memory_total_mb": 8192,
            "gpu_memory_used_mb": 1024,
            "gpu_memory_free_mb": 7168,
            "error": None,
        }

    monkeypatch.setattr("app.gui.workers.check_gpu", fake_check_gpu)
    page = window.environment_page

    page.run_gpu_check()

    assert page.thread is not None
    assert not page.run_gpu_button.isEnabled()
    assert not page.progress.isHidden()
    assert page.status_label.text() == "Checking GPU..."
    assert window.isEnabled()

    for _ in range(100):
        app.processEvents()
        if page.run_gpu_button.isEnabled():
            break
        QTest.qWait(10)

    assert page.run_gpu_button.isEnabled()
    assert page.gpu_card.values["gpu_name"].text() == "Test GPU"
    assert page.gpu_card.values["cuda_available"].text() == "Yes"
    assert page.gpu_card.badge.text() == "OK"
    saved = tmp_path / "logs" / "environment_info.json"
    assert saved.exists()
    assert "GPU check complete: Ready" in page.status_label.text()
    while page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()
    assert app is not None


def test_environment_gpu_worker_failure_shows_error_without_crashing(
    tmp_path, monkeypatch
):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="environment-failure",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
    )

    def failed_check():
        raise RuntimeError("GPU probe failed")

    monkeypatch.setattr("app.gui.workers.check_gpu", failed_check)
    page = window.environment_page
    page.run_gpu_check()

    for _ in range(100):
        app.processEvents()
        if page.run_gpu_button.isEnabled():
            break
        QTest.qWait(10)

    assert page.run_gpu_button.isEnabled()
    assert page.gpu_card.badge.text() == "Error"
    assert "GPU check failed: GPU probe failed" in page.status_label.text()
    saved = tmp_path / "logs" / "environment_info.json"
    assert saved.exists()
    assert "GPU probe failed" in saved.read_text(encoding="utf-8")
    while page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()
    assert app is not None


def test_startup_environment_check_starts_after_main_window_show(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    calls = []
    monkeypatch.setattr(
        window.environment_page,
        "start_startup_environment_check",
        lambda: calls.append("started"),
    )

    assert calls == []
    window.show()
    assert calls == []
    app.processEvents()

    assert calls == ["started"]
    window.close()


def test_startup_environment_check_is_non_blocking_and_disables_manual_check(
    tmp_path, monkeypatch
):
    import threading

    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    worker_started = threading.Event()
    release_worker = threading.Event()

    def delayed_check():
        worker_started.set()
        release_worker.wait(timeout=2)
        return {
            "torch_installed": True,
            "cuda_available": True,
            "gpu_count": 1,
            "gpu_name": "Startup GPU",
            "tensor_test_passed": True,
            "nvidia_smi_available": True,
            "nvidia_gpu_detected": True,
            "error": None,
        }

    monkeypatch.setattr("app.gui.workers.check_gpu", delayed_check)
    window = MainWindow()

    window.show()
    app.processEvents()

    page = window.environment_page
    assert worker_started.wait(timeout=1)
    assert window.isEnabled()
    assert page.thread is not None
    assert not page.run_gpu_button.isEnabled()
    assert page.status_label.text() == "Environment check running..."

    release_worker.set()
    for _ in range(100):
        app.processEvents()
        if page.run_gpu_button.isEnabled():
            break
        QTest.qWait(10)

    assert page.run_gpu_button.isEnabled()
    assert window.environment_info["gpu_name"] == "Startup GPU"
    assert page.gpu_card.values["gpu_name"].text() == "Startup GPU"
    assert (tmp_path / "logs" / "environment_info.json").exists()
    while page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()


def test_environment_page_loads_completed_startup_result(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    startup_info = {
        "psutil_available": True,
        "cpu_name": "Cached CPU",
        "architecture": "AMD64",
        "ram_total_bytes": 16 * 1024**3,
        "cuda_available": True,
        "tensor_test_passed": True,
        "gpu_count": 1,
        "gpu_name": "Cached GPU",
    }
    window.environment_info = startup_info

    window.environment_page.refresh()

    assert window.environment_page.cpu_card.values["cpu_name"].text() == "Cached CPU"
    assert window.environment_page.gpu_card.values["gpu_name"].text() == "Cached GPU"
    assert window.environment_page.gpu_card.badge.text() == "OK"
    window.close()
    assert app is not None


def test_startup_environment_check_never_triggers_gpu_repair(
    tmp_path, monkeypatch
):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    monkeypatch.chdir(tmp_path)
    repair_calls = []
    monkeypatch.setattr(
        "app.gui.workers.repair_gpu_torch",
        lambda project_dir: repair_calls.append(project_dir),
    )
    monkeypatch.setattr(
        "app.gui.workers.check_gpu",
        lambda: {
            "torch_installed": True,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": "NVIDIA Test GPU",
            "tensor_test_passed": False,
            "nvidia_smi_available": True,
            "nvidia_gpu_detected": True,
            "repair_recommended": True,
            "error": None,
        },
    )
    window = MainWindow()
    window.show()

    for _ in range(100):
        app.processEvents()
        if window.environment_page.run_gpu_button.isEnabled():
            break
        QTest.qWait(10)

    assert repair_calls == []
    assert not window.environment_page.repair_gpu_button.isHidden()
    while window.environment_page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()


def test_project_setup_load_existing_project(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    input_file = tmp_path / "data" / "data.csv"
    input_file.parent.mkdir()
    input_file.write_text("x,y\n1,0\n", encoding="utf-8")
    config = ProjectConfig(
        project_name="loaded-demo",
        project_dir=str(tmp_path),
        input_file=str(input_file),
        output_dir=str(tmp_path / "outputs"),
        dataset={
            "project_relative_path": "data/data.csv",
            "original_source_path": "D:/source/data.csv",
            "copied_project_path": "data/data.csv",
            "copied_to_project": True,
            "file_size": input_file.stat().st_size,
            "copy_timestamp": "2026-07-12T15:30:00",
        },
        target_column="y",
        feature_columns=["x"],
        environment_mode="shared_cpu_env",
    )
    project_file = config.save()

    window = MainWindow()
    page = window.project_setup_page
    page.existing_project_file_input.setText(str(project_file))
    page.load_project()

    assert window.config.project_name == "loaded-demo"
    assert page.project_name_input.text() == "loaded-demo"
    assert page.input_file_input.text() == str(input_file)
    assert window.config.environment_mode == "shared_cpu_env"
    assert window.dataframe is not None
    assert window.dataframe.columns.tolist() == ["x", "y"]
    assert page.main_window.data_import_page.load_button.isHidden()
    assert page.main_window.data_import_page.replace_dataset_button.isHidden()
    assert not page.main_window.data_import_page.preview_card.isHidden()
    assert "Project loaded: loaded-demo" in page.status_label.text()
    assert f"Project file: {project_file}" in page.status_label.text()
    assert "Project loaded: loaded-demo" in page.load_project_info_label.text()
    assert page.project_loaded_value.text() == "Yes"
    assert page.current_project_name_value.text() == "loaded-demo"
    assert page.current_project_file_value.text() == str(project_file)
    assert page.current_dataset_value.text() == str(input_file)
    assert page.current_modified_value.text() != "Not available"
    assert "project_config.json" not in page.status_label.text()
    window.close()
    assert app is not None


def test_project_setup_creates_avista_project_structure(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.project_setup_page
    page.project_name_input.setText("MyProject")
    page.project_parent_input.setText(str(tmp_path))
    source = tmp_path / "source" / "input.csv"
    source.parent.mkdir()
    source.write_text("x,y\n1,0\n", encoding="utf-8")
    page.input_file_input.setText(str(source))

    page.create_project()

    project_dir = tmp_path / "MyProject"
    project_file = project_dir / "MyProject.avista"
    assert window.config is not None
    assert window.config.project_file == project_file.resolve()
    assert project_file.exists()
    copied_dataset = project_dir / "data" / "input.csv"
    assert copied_dataset.exists()
    stored = json.loads(project_file.read_text(encoding="utf-8"))
    assert stored["input_file"] == "data/input.csv"
    assert stored["dataset"]["project_relative_path"] == "data/input.csv"
    assert stored["dataset"]["original_source_path"] == str(source.resolve())
    assert window.dataframe is not None
    for folder_name in ("data", "outputs", "logs", "artifacts"):
        assert (project_dir / folder_name).is_dir()
    assert "Project saved successfully." in page.status_label.text()
    assert f"Project file: {project_file.resolve()}" in page.status_label.text()
    assert "Dataset copied into project:\ndata/input.csv" in page.status_label.text()
    assert page.project_loaded_value.text() == "Yes"
    assert page.current_project_name_value.text() == "MyProject"
    assert "project_config.json" not in page.status_label.text()
    window.close()
    assert app is not None


def test_project_setup_project_file_filters():
    from app.gui.project_setup_page import (
        OPEN_PROJECT_FILE_FILTER,
        PROJECT_FILE_FILTER,
    )

    assert PROJECT_FILE_FILTER == "AVISTA Project (*.avista)"
    assert OPEN_PROJECT_FILE_FILTER.startswith("AVISTA Project (*.avista)")
    assert "Legacy Project (*.xtab)" in OPEN_PROJECT_FILE_FILTER


def test_data_import_no_project_shows_info_and_hides_loaded_controls():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_import_page

    assert not page.empty_state_area.isHidden()
    assert not page.empty_state_card.isHidden()
    assert page.empty_state_card.maximumWidth() == 600
    assert page.empty_state_card.minimumWidth() == 420
    assert page.empty_state_title.text() == "No project selected"
    assert page.empty_state_message.text() == (
        "Please create a new AVISTA project or open an existing project "
        "before importing data."
    )
    assert page.empty_state_icon.pixmap() is not None
    assert not page.empty_state_icon.pixmap().isNull()
    assert page.info_card.isHidden()
    assert page.cards_widget.isHidden()
    assert page.preview_card.isHidden()
    assert page.pagination_widget.isHidden()
    assert page.table.isHidden()
    assert page.load_button.isHidden()
    assert page.replace_dataset_button.isHidden()
    window.close()
    assert app is not None


def test_data_import_empty_state_navigates_to_project_setup():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.navigate_to(2)
    page = window.data_import_page

    page.go_to_project_setup_button.click()

    assert window.stack.currentIndex() == 0
    assert window.nav_buttons[0].isChecked()
    window.close()
    assert app is not None


def test_open_project_warns_when_dataset_is_missing(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    config = ProjectConfig(
        project_name="missing-data",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data" / "missing.csv"),
        output_dir=str(tmp_path / "outputs"),
        dataset={
            "project_relative_path": "data/missing.csv",
            "original_source_path": "D:/source/missing.csv",
            "copied_project_path": "data/missing.csv",
            "copied_to_project": True,
        },
    )
    project_file = config.save()

    window = MainWindow(initial_config=ProjectConfig.load(project_file))

    assert window.dataframe is None
    page = window.data_import_page
    assert page.status_label.text() == (
        "Project dataset could not be found. "
        "Please replace the dataset from Project Setup."
    )
    assert not page.info_card.isHidden()
    assert page.cards_widget.isHidden()
    assert page.preview_card.isHidden()
    assert page.pagination_widget.isHidden()
    assert page.table.isHidden()
    assert page.load_button.isHidden()
    assert page.replace_dataset_button.isHidden()
    window.close()
    assert app is not None


def test_replace_dataset_copies_updates_and_loads(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    config = ProjectConfig(
        project_name="replace-data",
        project_dir=str(project_dir),
        input_file="",
        output_dir=str(project_dir / "outputs"),
    )
    config.save()
    replacement = tmp_path / "external" / "replacement.csv"
    replacement.parent.mkdir()
    replacement.write_text("feature,target\n10,1\n20,0\n", encoding="utf-8")
    monkeypatch.setattr(
        "app.gui.data_import_page.QFileDialog.getOpenFileName",
        lambda *_args, **_kwargs: (str(replacement), ""),
    )
    window = MainWindow(initial_config=config)

    window.data_import_page.replace_dataset()

    copied = project_dir / "data" / "replacement.csv"
    stored = json.loads((project_dir / "replace-data.avista").read_text(encoding="utf-8"))
    assert copied.exists()
    assert window.config.input_file == str(copied)
    assert window.config.dataset["project_relative_path"] == "data/replacement.csv"
    assert stored["input_file"] == "data/replacement.csv"
    assert stored["dataset"]["project_relative_path"] == "data/replacement.csv"
    assert stored["dataset"]["original_source_path"] == str(replacement.resolve())
    assert window.dataframe is not None
    assert len(window.dataframe) == 2
    assert "Dataset copied into project:" in window.data_import_page.status_label.text()
    window.close()
    assert app is not None


def test_data_import_table_value_text_shows_null():
    import pandas as pd

    from app.gui.data_import_page import _table_value_text

    assert _table_value_text(None) == "null"
    assert _table_value_text(float("nan")) == "null"
    assert _table_value_text(pd.NA) == "null"
    assert _table_value_text("") == ""


def test_data_import_preview_paginates_large_dataframe(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication, QTableView

    from app.core.data_loader import summarize_dataframe
    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="large-preview",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "large.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    df = pd.DataFrame({"value": range(100_000), "label": ["x"] * 100_000})
    window.dataframe = df

    page = window.data_import_page
    page.summary = summarize_dataframe(df)
    page._populate_cards(page.summary, df)
    page.render_preview_page()

    assert isinstance(page.table, QTableView)
    assert not page.cards_widget.isHidden()
    assert not page.preview_card.isHidden()
    assert not page.pagination_widget.isHidden()
    assert not page.table.isHidden()
    assert page.load_button.isHidden()
    assert page.replace_dataset_button.isHidden()
    assert page.cards_layout.count() == 7
    assert page.preview_model.rowCount() == 50
    assert len(page.get_current_page_slice()) == 50
    assert page.page_label.text() == "Page 1 of 2000"

    page.next_page()
    assert page.current_page == 1
    assert page.get_current_page_slice().index[0] == 50

    page.rows_per_page_combo.setCurrentText("200")
    assert page.preview_model.rowCount() == 200
    assert len(page.get_current_page_slice()) <= 200
    window.close()
    assert app is not None


def test_column_config_confirms_modeling_subset(tmp_path):
    import json

    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="column-demo",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        group_column="site",
    )
    window.config = config
    window.dataframe = pd.DataFrame(
        {
            "city": ["Austin", "Dallas", "Houston"],
            "target": [0, 1, 0],
            "age": [20, 30, 40],
        }
    )

    page = window.column_config_page
    page.refresh()
    assert [
        page.available_columns_list.item(index).text()
        for index in range(page.available_columns_list.count())
    ] == ["age", "city", "target"]
    assert page.target_input.count() == 1
    for column in ["city", "age", "target"]:
        matches = page.available_columns_list.findItems(column, Qt.MatchFlag.MatchExactly)
        matches[0].setSelected(True)
        page.add_selected()
    assert [page.target_input.itemText(index) for index in range(1, page.target_input.count())] == [
        "age",
        "city",
        "target",
    ]
    assert [
        page.selected_columns_list.item(index).text()
        for index in range(page.selected_columns_list.count())
    ] == ["age", "city", "target"]
    page.target_input.setCurrentText("target")
    assert page.target_plot.target_column == "target"
    assert page.target_plot.figure.axes[0].get_title() == "Target Distribution: target"
    assert page.target_plot.figure.axes[0].get_xlabel() == "Class"
    assert page.target_plot.figure.axes[0].get_ylabel() == "Count"
    assert [
        page.label_encoding_columns_list.item(index).text()
        for index in range(page.label_encoding_columns_list.count())
    ] == ["city"]
    encoding_item = page.label_encoding_columns_list.item(0)
    encoding_item.setCheckState(Qt.CheckState.Checked)

    subset_path = tmp_path / "data" / "modeling_subset.csv"
    assert not subset_path.exists()
    page.confirm_modeling_columns()

    assert config.feature_columns == ["age", "city"]
    assert config.target_column == "target"
    assert config.group_column == "site"
    assert config.label_encoding_columns == ["city"]
    assert subset_path.exists()
    assert pd.read_csv(subset_path).columns.tolist() == ["age", "city", "target"]
    saved_config = json.loads((tmp_path / "column-demo.avista").read_text(encoding="utf-8"))
    assert saved_config["group_column"] == "site"
    assert saved_config["label_encoding_columns"] == ["city"]
    metadata = saved_config["preprocessing_options"]["label_encoding_metadata"]
    assert metadata["city"]["unique_count"] == 3
    assert metadata["city"]["missing_count"] == 0
    assert "Modeling configuration saved successfully." in page.feedback_label.text()
    assert [label.text() for label in page.feedback_labels] == [
        "Modeling configuration saved successfully.",
        "Selected features: 2",
        "Target column: target",
        "Label-encoded columns: 1",
    ]
    assert all(not icon.pixmap().isNull() for icon in page.feedback_icons)
    assert not page.feedback_card.isHidden()
    assert page.feedback_card.maximumHeight() == 136
    feedback_margins = page.feedback_card.layout().contentsMargins()
    assert feedback_margins.top() == 8
    assert feedback_margins.bottom() == 8
    assert "#f8fff9" in page.feedback_card.styleSheet().lower()
    assert "#2da44e" in page.feedback_card.styleSheet().lower()
    assert not page.status_card.isHidden()
    assert not page.status_icon.pixmap().isNull()
    assert page.status_icon.objectName() == "successLineIcon"
    assert page.status_label.text() == f"Modeling subset saved to: {subset_path}"
    assert "#2da44e" in page.status_card.styleSheet().lower()
    assert page.success_notification_timer.isSingleShot()
    assert page.success_notification_timer.interval() == 5000
    assert page.success_notification_timer.isActive()
    page.success_notification_timer.timeout.emit()
    assert page.feedback_card.isHidden()
    assert page.status_card.isHidden()
    window.close()
    assert app is not None


def test_column_config_restores_only_eligible_label_encoding_columns(tmp_path):
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="encoding-restore",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["age", "city", "active"],
        target_column="target",
        label_encoding_columns=["city", "age", "missing"],
    )
    window.dataframe = pd.DataFrame(
        {
            "age": [20, 30],
            "city": ["Austin", "Dallas"],
            "active": [True, False],
            "target": ["yes", "no"],
        }
    )

    page = window.column_config_page
    page.refresh()

    assert [
        page.label_encoding_columns_list.item(index).text()
        for index in range(page.label_encoding_columns_list.count())
    ] == ["active", "city"]
    assert [
        page.label_encoding_columns_list.item(index).text()
        for index in range(page.label_encoding_columns_list.count())
        if page.label_encoding_columns_list.item(index).checkState()
        == Qt.CheckState.Checked
    ] == [
        "city",
    ]
    window.close()
    assert app is not None


def test_column_config_label_click_previews_without_toggling_checkbox(tmp_path):
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="encoding-preview",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["category"],
        target_column="target",
    )
    window.dataframe = pd.DataFrame(
        {
            "category": ["A", "B", "A", None],
            "target": [0, 1, 0, 1],
        }
    )

    page = window.column_config_page
    page.refresh()
    item = page.label_encoding_columns_list.item(0)
    assert item.text() == "category"
    assert item.checkState() == Qt.CheckState.Unchecked

    page.label_encoding_columns_list.itemClicked.emit(item)

    assert item.checkState() == Qt.CheckState.Unchecked
    assert page.unique_column_name.text() == "category"
    assert "Unique values: 2" in page.unique_summary_label.text()
    assert "Missing values: 1" in page.unique_summary_label.text()
    assert [
        page.unique_values_list.item(index).text()
        for index in range(page.unique_values_list.count())
    ] == [
        "A \u2014 2 rows (50.0%)",
        "B \u2014 1 rows (25.0%)",
        "Missing/Null \u2014 1 rows (25.0%)",
    ]

    item.setCheckState(Qt.CheckState.Checked)
    assert item.checkState() == Qt.CheckState.Checked
    window.close()
    assert app is not None


def test_column_config_empty_state_hides_configuration_controls():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.column_config_page
    page.refresh()

    assert not page.empty_state_card.isHidden()
    assert page.controls_widget.isHidden()
    assert page.empty_state_card.maximumWidth() == 620
    window.close()
    assert app is not None


def test_model_selection_page_populates_registry_models():
    from PySide6.QtWidgets import QApplication

    from app.core.model_registry import get_available_models
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    specs = get_available_models(task_type="classification")

    assert set(page.model_checkboxes) == {spec.name for spec in specs}
    assert len(page.model_checkboxes) == 16
    common_deep_parameters = {
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "patience",
        "loss_function",
        "validation_metric",
        "save_training_loss",
        "save_validation_loss",
        "save_validation_metric",
        "save_best_checkpoint",
        "early_stopping_patience",
    }
    assert set(page.parameter_widgets["mamba_attention"]) == {
        "hidden_dim",
        "dropout",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_start_factor",
        "loss_function",
        "focal_gamma",
        "label_smoothing",
        "use_class_weights",
        "validation_metric",
        "early_stopping_patience",
        "restore_best_weights",
        "save_final_state_dict",
    }
    assert "input_dim" not in page.parameter_widgets["mamba_attention"]
    assert "num_classes" not in page.parameter_widgets["mamba_attention"]
    assert "patience" not in page.parameter_widgets["mamba_attention"]
    assert set(page.parameter_section_groups["mamba_attention"]) == {
        "Architecture Parameters",
        "Training Parameters",
        "Loss Parameters",
        "Monitoring / Saving Options",
    }
    assert set(page.parameter_widgets["ft_transformer"]) == {
        "dropout",
        "d_token",
        "n_heads",
        "n_layers",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_start_factor",
        "loss_function",
        "focal_gamma",
        "label_smoothing",
        "use_class_weights",
        "validation_metric",
        "early_stopping_patience",
        "restore_best_weights",
        "save_final_state_dict",
    }
    assert "n_features" not in page.parameter_widgets["ft_transformer"]
    assert "n_classes" not in page.parameter_widgets["ft_transformer"]
    assert set(page.parameter_section_groups["ft_transformer"]) == {
        "Architecture Parameters",
        "Training Parameters",
        "Loss Parameters",
        "Monitoring / Saving Options",
    }
    assert set(page.parameter_widgets["autoint"]) == {
        "dropout",
        "d",
        "n_heads",
        "n_layers",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_start_factor",
        "loss_function",
        "focal_gamma",
        "label_smoothing",
        "use_class_weights",
        "validation_metric",
        "early_stopping_patience",
        "restore_best_weights",
        "save_final_state_dict",
    }
    assert "n_features" not in page.parameter_widgets["autoint"]
    assert "n_classes" not in page.parameter_widgets["autoint"]
    assert set(page.parameter_section_groups["autoint"]) == {
        "Architecture Parameters",
        "Training Parameters",
        "Loss Parameters",
        "Monitoring / Saving Options",
    }
    assert set(page.parameter_widgets["tab_resnet"]) == {
        "dropout",
        "hidden",
        "n_blocks",
        "learning_rate",
        "batch_size",
        "epochs",
        "warmup_epochs",
        "optimizer",
        "weight_decay",
        "scheduler",
        "warmup_start_factor",
        "loss_function",
        "focal_gamma",
        "label_smoothing",
        "use_class_weights",
        "validation_metric",
        "early_stopping_patience",
        "restore_best_weights",
        "save_final_state_dict",
    }
    assert "input_dim" not in page.parameter_widgets["tab_resnet"]
    assert "n_classes" not in page.parameter_widgets["tab_resnet"]
    assert set(page.parameter_section_groups["tab_resnet"]) == {
        "Architecture Parameters",
        "Training Parameters",
        "Loss Parameters",
        "Monitoring / Saving Options",
    }
    assert page.parameter_widgets["mamba_attention"]["hidden_dim"].value() == 256
    assert page.parameter_widgets["mamba_attention"]["dropout"].value() == 0.3
    assert page.parameter_widgets["mamba_attention"]["early_stopping_patience"].value() == 30
    assert page.parameter_widgets["ft_transformer"]["d_token"].value() == 128
    assert page.parameter_widgets["ft_transformer"]["n_heads"].value() == 8
    assert page.parameter_widgets["ft_transformer"]["n_layers"].value() == 3
    assert page.parameter_widgets["ft_transformer"]["dropout"].value() == 0.1
    assert page.parameter_widgets["autoint"]["d"].value() == 64
    assert page.parameter_widgets["autoint"]["n_heads"].value() == 4
    assert page.parameter_widgets["autoint"]["n_layers"].value() == 3
    assert page.parameter_widgets["autoint"]["dropout"].value() == 0.1
    assert page.parameter_widgets["tab_resnet"]["hidden"].value() == 256
    assert page.parameter_widgets["tab_resnet"]["n_blocks"].value() == 6
    assert page.parameter_widgets["tab_resnet"]["dropout"].value() == 0.2
    assert set(page.parameter_widgets["tabpfn"]) == {"n_estimators"}
    assert page.parameter_widgets["tabpfn"]["n_estimators"].value() == 8
    removed_tabpfn_fields = {
        "maximum_training_samples",
        "tabpfn_max_samples",
        "cv_estimators",
        "cv_n_estimators",
        "final_estimators",
        "final_n_estimators",
        "prediction_batch_size",
        "random_state",
    }
    assert removed_tabpfn_fields.isdisjoint(page.parameter_widgets["tabpfn"])
    assert page.parameter_widgets["gaussian_nb"] == {}
    window.close()
    assert app is not None


def test_model_selection_shows_missing_optional_dependencies(monkeypatch, tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    monkeypatch.setattr(
        "app.gui.model_selection_page.check_optional_packages",
        lambda *_args, **_kwargs: {
            "success": False,
            "packages": {"xgboost": False, "tabpfn": True, "torch": False},
            "missing": ["xgboost", "torch"],
            "error": None,
        },
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="dependencies",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    page = window.model_selection_page
    page.refresh()

    assert not page.model_checkboxes["xgboost"].isEnabled()
    assert page.dependency_labels["xgboost"].text() == "Missing"
    assert not page.dependency_install_buttons["xgboost"].isHidden()
    assert page.dependency_install_buttons["xgboost"].text() == "Install xgboost"
    assert page.model_checkboxes["tabpfn"].isEnabled()
    assert page.dependency_labels["tabpfn"].text() == ""
    assert page.dependency_labels["tabpfn"].isHidden()
    assert page.dependency_install_buttons["tabpfn"].isHidden()
    assert not page.model_checkboxes["mamba_attention"].isEnabled()
    assert page.dependency_install_buttons["mamba_attention"].text() == "Install torch"
    window.close()
    assert app is not None


def test_model_selection_dependency_install_result_refreshes_status(
    monkeypatch, tmp_path
):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    statuses = {"xgboost": False, "tabpfn": True, "torch": True}

    def fake_check(*_args, **_kwargs):
        return {
            "success": all(statuses.values()),
            "packages": dict(statuses),
            "missing": [name for name, installed in statuses.items() if not installed],
            "error": None,
        }

    monkeypatch.setattr(
        "app.gui.model_selection_page.check_optional_packages", fake_check
    )
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="dependency-result",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    page = window.model_selection_page
    page.refresh()
    statuses["xgboost"] = True

    page._dependency_install_finished(
        {"success": True, "package": "xgboost", "error": None}
    )

    assert page.model_checkboxes["xgboost"].isEnabled()
    assert page.dependency_labels["xgboost"].text() == ""
    assert page.dependency_labels["xgboost"].isHidden()
    assert page.dependency_install_buttons["xgboost"].isHidden()
    assert "installed successfully" in page.feedback_label.text()
    window.close()
    assert app is not None


def test_model_selection_page_renders_card_layout(tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QFrame, QSizePolicy, QToolButton

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="model-card-layout",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    page = window.model_selection_page
    page.refresh()

    assert page.model_library_card.objectName() == "modelLibraryCard"
    assert page.model_parameters_card.objectName() == "modelParametersCard"
    assert page.global_training_options_card.objectName() == "globalTrainingOptionsCard"
    assert page.confirmation_status_card.objectName() == "modelConfirmationStatusCard"
    assert page.model_library_card.layout().spacing() <= 8
    assert page.top_cards_layout.alignment() & Qt.AlignmentFlag.AlignTop
    assert set(page.category_cards) == {
        "Linear Models",
        "Tree-Based Models",
        "Boosting Models",
        "Kernel/Distance Models",
        "Naive Bayes",
        "Tabular Models",
    }
    assert all(card.layout().spacing() <= 6 for card in page.category_cards.values())
    assert (
        page.model_library_card.sizePolicy().verticalPolicy()
        == QSizePolicy.Policy.Maximum
    )
    assert (
        page.model_parameters_card.sizePolicy().verticalPolicy()
        == QSizePolicy.Policy.Maximum
    )
    assert page.model_library_list_scroll.widgetResizable()
    assert page.model_library_list_scroll.frameShape() == QFrame.Shape.NoFrame
    combo_arrow_buttons = page.findChildren(QToolButton, "modelComboArrowButton")
    spin_arrow_buttons = page.findChildren(QToolButton, "modelSpinArrowButton")
    assert combo_arrow_buttons
    assert spin_arrow_buttons
    assert all(button.icon().isNull() is False for button in combo_arrow_buttons)
    assert all(button.icon().isNull() is False for button in spin_arrow_buttons)
    assert "QComboBox::drop-down" in page.styleSheet()
    assert "QToolButton#modelComboArrowButton" in page.styleSheet()
    assert "QToolButton#modelSpinArrowButton" in page.styleSheet()
    assert not page.controls_widget.isHidden()
    assert page.empty_state_card.isHidden()
    window.close()
    assert app is not None


def test_model_selection_no_config_state_hides_controls():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    page.refresh()

    assert not page.empty_state_card.isHidden()
    assert page.controls_widget.isHidden()
    window.close()
    assert app is not None


def test_model_selection_confirm_button_is_primary():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page

    assert page.confirm_button.objectName() == "primaryModelSelectionButton"
    assert page.confirm_button.icon().isNull() is False
    assert "QPushButton#primaryModelSelectionButton" in page.styleSheet()
    window.close()
    assert app is not None


def test_model_selection_success_notification_auto_dismisses(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="model-success-dismiss",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    page = window.model_selection_page
    page.refresh()
    page.model_checkboxes["logistic_regression"].setChecked(True)
    page.confirm_model_selection()

    assert not page.feedback_card.isHidden()
    assert "Model selection saved successfully." in page.feedback_label.text()

    page.success_notification_timer.timeout.emit()

    assert page.feedback_card.isHidden()
    window.close()
    assert app is not None


def test_model_selection_checkbox_click_always_shows_model_parameters():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    checkbox = page.model_checkboxes["logistic_regression"]
    parameter_panel = page.parameter_panels["logistic_regression"]

    checkbox.click()

    assert checkbox.isChecked()
    assert page.parameter_stack.currentWidget() is parameter_panel

    checkbox.click()

    assert not checkbox.isChecked()
    assert page.parameter_stack.currentWidget() is parameter_panel
    window.close()
    assert app is not None


def test_model_selection_renders_metadata_widgets_and_converts_values():
    from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox

    from app.gui.main_window import MainWindow
    from app.gui.model_selection_page import OptionalNumberWidget

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    logistic_widgets = page.parameter_widgets["logistic_regression"]
    forest_widgets = page.parameter_widgets["random_forest"]

    assert isinstance(logistic_widgets["penalty"], QComboBox)
    assert isinstance(logistic_widgets["C"], QDoubleSpinBox)
    assert isinstance(logistic_widgets["max_iter"], QSpinBox)
    assert isinstance(forest_widgets["max_depth"], OptionalNumberWidget)
    assert isinstance(forest_widgets["max_samples"], OptionalNumberWidget)

    logistic_widgets["class_weight"].setCurrentText("none")
    logistic_widgets["n_jobs"].setCurrentText("-1")
    forest_widgets["max_depth"].setValue(12)
    forest_widgets["max_samples"].setValue(None)

    logistic_values = page._parameter_values("logistic_regression")
    forest_values = page._parameter_values("random_forest")

    assert logistic_values["class_weight"] is None
    assert logistic_values["n_jobs"] == -1
    assert isinstance(logistic_values["n_jobs"], int)
    assert forest_values["max_depth"] == 12
    assert isinstance(forest_values["max_depth"], int)
    assert forest_values["max_samples"] is None
    window.close()
    assert app is not None


def test_model_selection_renders_and_converts_boosting_parameters():
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QSpinBox,
    )

    from app.gui.main_window import MainWindow
    from app.gui.model_selection_page import OptionalNumberWidget

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    xgb = page.parameter_widgets["xgboost"]
    gradient = page.parameter_widgets["gradient_boosting"]
    hist = page.parameter_widgets["hist_gradient_boosting"]
    adaboost = page.parameter_widgets["adaboost"]

    assert isinstance(xgb["objective"], QComboBox)
    assert isinstance(xgb["n_estimators"], QSpinBox)
    assert isinstance(xgb["learning_rate"], QDoubleSpinBox)
    assert isinstance(xgb["enable_categorical"], QCheckBox)
    assert isinstance(gradient["max_depth"], OptionalNumberWidget)
    assert isinstance(hist["early_stopping"], QComboBox)
    assert isinstance(adaboost["estimator"], QComboBox)

    page.random_state.setValue(73)
    xgb["random_state"].setCurrentText("use_experiment_seed")
    xgb["n_jobs"].setCurrentText("-1")
    hist["early_stopping"].setCurrentText("true")
    hist["categorical_features"].setCurrentText("none")

    xgb_values = page._parameter_values("xgboost")
    hist_values = page._parameter_values("hist_gradient_boosting")

    assert xgb_values["random_state"] == 73
    assert xgb_values["n_jobs"] == -1
    assert hist_values["early_stopping"] is True
    assert hist_values["categorical_features"] is None
    window.close()
    assert app is not None


def test_model_selection_boosting_warnings_do_not_crash():
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="warnings",
        project_dir=".",
        input_file="data.csv",
        output_dir="outputs",
        target_column="target",
    )
    window.dataframe = pd.DataFrame({"target": ["a", "b", "c"]})
    page = window.model_selection_page

    xgb = page.parameter_widgets["xgboost"]
    xgb["objective"].setCurrentText("binary:logistic")
    xgb["scale_pos_weight"].setValue(2.0)
    xgb["enable_categorical"].setChecked(True)
    xgb_warning = page.parameter_warning_labels["xgboost"].text()
    assert "Multiclass targets should use objective=multi:softprob" in xgb_warning
    assert "scale_pos_weight is mainly intended for binary" in xgb_warning
    assert "requires compatible category-preserving input" in xgb_warning

    gradient = page.parameter_widgets["gradient_boosting"]
    gradient["loss"].setCurrentText("exponential")
    gradient_warning = page.parameter_warning_labels["gradient_boosting"].text()
    assert "validation_fraction is used only" in gradient_warning
    assert "exponential loss is intended for binary" in gradient_warning

    assert "preserved categorical dtypes" in page.parameter_warning_labels[
        "hist_gradient_boosting"
    ].text()
    assert "customization is not currently implemented" in page.parameter_warning_labels[
        "adaboost"
    ].text()
    window.close()
    assert app is not None


def test_model_selection_parameter_dependencies_and_warnings_do_not_crash():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.model_selection_page
    logistic_widgets = page.parameter_widgets["logistic_regression"]
    forest_widgets = page.parameter_widgets["random_forest"]

    assert not logistic_widgets["l1_ratio"].isEnabled()
    logistic_widgets["penalty"].setCurrentText("elasticnet")
    logistic_widgets["solver"].setCurrentText("lbfgs")

    assert logistic_widgets["l1_ratio"].isEnabled()
    assert "requires solver=saga" in page.parameter_warning_labels[
        "logistic_regression"
    ].text()

    forest_widgets["bootstrap"].setChecked(False)
    forest_widgets["oob_score"].setChecked(True)
    forest_widgets["max_samples"].setValue(0.5)
    forest_widgets["monotonic_cst"].setText("not-a-list")

    warning_text = page.parameter_warning_labels["random_forest"].text()
    assert "oob_score=True requires bootstrap=True" in warning_text
    assert "max_samples is only valid when bootstrap=True" in warning_text
    assert "monotonic_cst must be a JSON list" in warning_text
    window.close()
    assert app is not None


def test_model_selection_restore_defaults_affects_only_active_model(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="restore-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
        selected_models=["logistic_regression", "random_forest"],
        model_params={
            "logistic_regression": {"max_iter": 750},
            "random_forest": {"n_estimators": 500},
        },
        enable_cross_validation=True,
        cv_folds=7,
        random_state=99,
    )
    window.config = config
    page = window.model_selection_page
    page.refresh()

    logistic_checkbox = page.model_checkboxes["logistic_regression"]
    forest_checkbox = page.model_checkboxes["random_forest"]
    page.parameter_widgets["logistic_regression"]["max_iter"].setValue(900)
    page.parameter_widgets["random_forest"]["n_estimators"].setValue(600)
    page._show_parameter_panel("logistic_regression")
    parameter_panel = page.parameter_panels["logistic_regression"]
    parameter_layout = parameter_panel.content_layout

    assert parameter_panel.widgetResizable()
    assert parameter_layout.contentsMargins().left() == 16
    assert parameter_layout.contentsMargins().top() == 16
    assert parameter_layout.contentsMargins().right() == 16
    assert parameter_layout.contentsMargins().bottom() == 16
    assert 10 <= parameter_layout.spacing() <= 12

    assert page.restore_default_buttons["logistic_regression"].isVisibleTo(page)
    assert page.restore_default_buttons["logistic_regression"].objectName() == "secondaryModelButton"
    assert "QPushButton#secondaryModelButton" in page.styleSheet()

    page.restore_default_buttons["logistic_regression"].click()

    assert page.parameter_widgets["logistic_regression"]["max_iter"].value() == 100
    assert page.parameter_widgets["random_forest"]["n_estimators"].value() == 600
    assert "logistic_regression" not in config.model_params
    assert config.model_params["random_forest"] == {"n_estimators": 500}
    assert logistic_checkbox.isChecked()
    assert forest_checkbox.isChecked()
    assert page.enable_cross_validation.isChecked()
    assert page.cv_folds.value() == 7
    assert page.random_state.value() == 99
    assert (
        page.restore_feedback_labels["logistic_regression"].text()
        == "Defaults restored for Logistic Regression."
    )
    window.close()
    assert app is not None


def test_mamba_attention_restore_defaults_uses_reference_values(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="mamba-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        model_params={"mamba_attention": {"hidden_dim": 64, "epochs": 12}},
    )
    page = window.model_selection_page
    page.refresh()

    page.parameter_widgets["mamba_attention"]["hidden_dim"].setValue(64)
    page.parameter_widgets["mamba_attention"]["epochs"].setValue(12)
    page.parameter_widgets["mamba_attention"]["focal_gamma"].setValue(2.0)
    page.parameter_widgets["mamba_attention"]["early_stopping_patience"].setValue(4)
    page.restore_default_buttons["mamba_attention"].click()

    values = page._parameter_values("mamba_attention")
    assert values["hidden_dim"] == 256
    assert values["dropout"] == 0.3
    assert values["learning_rate"] == 0.001
    assert values["batch_size"] == 128
    assert values["epochs"] == 80
    assert values["warmup_epochs"] == 5
    assert values["optimizer"] == "adamw"
    assert values["weight_decay"] == 0.001
    assert values["scheduler"] == "linear_warmup_cosine_annealing"
    assert values["warmup_start_factor"] == 0.1
    assert values["loss_function"] == "focal_loss"
    assert values["focal_gamma"] == 1.0
    assert values["label_smoothing"] == 0.05
    assert values["use_class_weights"] is True
    assert values["validation_metric"] == "macro_f1"
    assert values["early_stopping_patience"] == 30
    assert values["restore_best_weights"] is True
    assert values["save_final_state_dict"] is True
    assert "mamba_attention" not in window.config.model_params
    window.close()
    assert app is not None


def test_ft_transformer_restore_defaults_uses_reference_values(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="ft-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        model_params={"ft_transformer": {"d_token": 64, "epochs": 12}},
    )
    page = window.model_selection_page
    page.refresh()

    page.parameter_widgets["ft_transformer"]["d_token"].setValue(64)
    page.parameter_widgets["ft_transformer"]["epochs"].setValue(12)
    page.parameter_widgets["ft_transformer"]["focal_gamma"].setValue(2.0)
    page.restore_default_buttons["ft_transformer"].click()

    values = page._parameter_values("ft_transformer")
    assert values["d_token"] == 128
    assert values["n_heads"] == 8
    assert values["n_layers"] == 3
    assert values["dropout"] == 0.1
    assert values["learning_rate"] == 0.001
    assert values["focal_gamma"] == 1.0
    assert "n_features" not in values
    assert "n_classes" not in values
    window.close()
    assert app is not None


def test_autoint_restore_defaults_uses_reference_values(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="autoint-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        model_params={"autoint": {"d": 32, "epochs": 12}},
    )
    page = window.model_selection_page
    page.refresh()

    page.parameter_widgets["autoint"]["d"].setValue(32)
    page.parameter_widgets["autoint"]["epochs"].setValue(12)
    page.parameter_widgets["autoint"]["focal_gamma"].setValue(2.0)
    page.restore_default_buttons["autoint"].click()

    values = page._parameter_values("autoint")
    assert values["d"] == 64
    assert values["n_heads"] == 4
    assert values["n_layers"] == 3
    assert values["dropout"] == 0.1
    assert values["learning_rate"] == 0.001
    assert values["focal_gamma"] == 1.0
    assert "n_features" not in values
    assert "n_classes" not in values
    window.close()
    assert app is not None


def test_tab_resnet_restore_defaults_uses_reference_values(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="tabresnet-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        model_params={"tab_resnet": {"hidden": 32, "epochs": 12}},
    )
    page = window.model_selection_page
    page.refresh()

    page.parameter_widgets["tab_resnet"]["hidden"].setValue(32)
    page.parameter_widgets["tab_resnet"]["epochs"].setValue(12)
    page.parameter_widgets["tab_resnet"]["focal_gamma"].setValue(2.0)
    page.restore_default_buttons["tab_resnet"].click()

    values = page._parameter_values("tab_resnet")
    assert values["hidden"] == 256
    assert values["n_blocks"] == 6
    assert values["dropout"] == 0.2
    assert values["learning_rate"] == 0.001
    assert values["focal_gamma"] == 1.0
    assert "input_dim" not in values
    assert "n_classes" not in values
    window.close()
    assert app is not None


def test_tabpfn_restore_defaults_resets_only_estimators(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="tabpfn-defaults",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        model_params={
            "tabpfn": {
                "n_estimators": 32,
                "tabpfn_max_samples": 100,
                "prediction_batch_size": 10,
            }
        },
    )
    page = window.model_selection_page
    page.refresh()

    assert set(page.parameter_widgets["tabpfn"]) == {"n_estimators"}
    page.parameter_widgets["tabpfn"]["n_estimators"].setValue(32)
    page.restore_default_buttons["tabpfn"].click()

    assert page._parameter_values("tabpfn") == {"n_estimators": 8}
    window.close()
    assert app is not None


def test_model_selection_confirm_saves_config_fields(tmp_path):
    import json

    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="model-selection",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    window.config = config
    page = window.model_selection_page
    page.refresh()

    page.model_checkboxes["logistic_regression"].setChecked(True)
    page.model_checkboxes["tab_resnet"].setChecked(True)
    page.parameter_widgets["logistic_regression"]["max_iter"].setValue(750)
    page.parameter_widgets["logistic_regression"]["class_weight"].setCurrentText("none")
    page.parameter_widgets["logistic_regression"]["n_jobs"].setCurrentText("-1")
    page.parameter_widgets["tab_resnet"]["epochs"].setValue(25)
    page.parameter_widgets["tab_resnet"]["focal_gamma"].setValue(1.5)
    page.enable_cross_validation.setChecked(True)
    page.cv_folds.setValue(6)
    page.random_state.setValue(99)
    page.confirm_model_selection()

    assert config.selected_models == ["logistic_regression", "tab_resnet"]
    assert config.model_params["logistic_regression"]["max_iter"] == 750
    assert config.model_params["logistic_regression"]["class_weight"] is None
    assert config.model_params["logistic_regression"]["n_jobs"] == -1
    assert config.model_params["tab_resnet"]["epochs"] == 25
    assert config.model_params["tab_resnet"]["focal_gamma"] == 1.5
    assert config.enable_cross_validation is True
    assert config.cv_folds == 6
    assert config.random_state == 99

    saved = json.loads((tmp_path / "model-selection.avista").read_text(encoding="utf-8"))
    assert saved["selected_models"] == ["logistic_regression", "tab_resnet"]
    assert saved["model_params"]["logistic_regression"]["class_weight"] is None
    assert saved["model_params"]["logistic_regression"]["n_jobs"] == -1
    assert saved["model_params"]["tab_resnet"]["epochs"] == 25
    assert saved["model_params"]["tab_resnet"]["focal_gamma"] == 1.5
    assert saved["enable_cross_validation"] is True
    assert saved["cv_folds"] == 6
    assert saved["random_state"] == 99
    assert "Model selection saved successfully." in page.feedback_label.text()
    assert page.feedback_labels[1].text() == "Selected models: 2"
    assert page.feedback_labels[2].text() == "Cross-validation: enabled, 6 folds"
    window.close()
    assert app is not None


def test_data_split_page_saves_three_way_artifacts(tmp_path):
    import json

    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="split-page-demo",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.config = config
    window.dataframe = pd.DataFrame(
        {
            "feature": range(100),
            "target": [0, 1] * 50,
        }
    )

    page = window.data_split_imbalance_page
    page.refresh()
    page.split_method.setCurrentText("stratified")
    page.random_seed.setValue(123)
    page.imbalance_method.setCurrentText("none")
    assert page.balancing_preset_container.isHidden()
    assert "Preserve target class proportions" in page.split_method.toolTip()
    page.confirm_split_and_imbalance()

    output_dir = tmp_path / "outputs" / "data_split"
    expected = [
        "split_indices.json",
        "class_distribution_before.csv",
        "class_distribution_after.csv",
        "class_coverage_report.csv",
        "imbalance_config.json",
        "X_train_balanced.npy",
        "y_train_balanced.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ]
    assert all((output_dir / name).exists() for name in expected)
    assert config.train_percent == 70
    assert config.validation_percent == 10
    assert config.test_percent == 20
    assert config.random_seed == 123
    assert page.before_distribution_tables["Train Set"].rowCount() > 0
    assert page.after_distribution_tables["Train Set (Balanced)"].rowCount() > 0
    assert page.class_coverage_table.rowCount() == 2
    assert "Split and imbalance configuration saved successfully." in page.feedback_label.text()
    assert not page.feedback_card.isHidden()
    split_metadata = json.loads((output_dir / "split_indices.json").read_text())
    imbalance_metadata = json.loads((output_dir / "imbalance_config.json").read_text())
    assert split_metadata["target_column"] == "target"
    assert imbalance_metadata["target_column"] == "target"
    window.close()
    assert app is not None


def test_data_split_page_encodes_string_targets_and_saves_mapping(tmp_path):
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="string-target",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame(
        {
            "feature": range(90),
            "target": [
                "Advanced_Automation",
                "Assisted_Driving",
                "Partial_Automation",
            ]
            * 30,
        }
    )

    page = window.data_split_imbalance_page
    page.split_method.setCurrentText("stratified")
    page.confirm_split_and_imbalance()

    output_dir = tmp_path / "outputs" / "data_split"
    mapping = json.loads(
        (output_dir / "target_label_mapping.json").read_text(encoding="utf-8")
    )
    encoder = joblib.load(output_dir / "target_label_encoder.joblib")
    encoded = np.load(output_dir / "y_train_balanced_encoded.npy")
    original = np.load(
        output_dir / "y_train_balanced_original.npy",
        allow_pickle=True,
    )

    assert mapping == {
        "0": "Advanced_Automation",
        "1": "Assisted_Driving",
        "2": "Partial_Automation",
    }
    assert encoded.dtype.kind in {"i", "u"}
    assert np.array_equal(encoder.inverse_transform(encoded), original)
    assert set(original) == set(mapping.values())
    window.close()
    assert app is not None


def test_data_split_page_encodes_mixed_type_targets(tmp_path):
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="mixed-target",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
        split_method="random",
    )
    window.dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "target": ["A", 1] * 30,
        }
    )

    page = window.data_split_imbalance_page
    page.confirm_split_and_imbalance()

    output_dir = tmp_path / "outputs" / "data_split"
    mapping = json.loads(
        (output_dir / "target_label_mapping.json").read_text(encoding="utf-8")
    )
    encoder = joblib.load(output_dir / "target_label_encoder.joblib")
    encoded = np.load(output_dir / "y_train_balanced_encoded.npy")
    original = np.load(
        output_dir / "y_train_balanced_original.npy",
        allow_pickle=True,
    )

    assert mapping == {"0": "1", "1": "A"}
    assert encoder.classes_.tolist() == ["1", "A"]
    assert encoded.dtype.kind in {"i", "u"}
    assert np.array_equal(encoder.inverse_transform(encoded), original)
    assert "Error:" not in page.feedback_label.text()
    window.close()
    assert app is not None


def test_column_target_change_invalidates_saved_encoder_and_split(tmp_path):
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="target-change",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="old_target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "old_target": ["A", "B"] * 30,
            "new_target": ["X", "Y", "Z"] * 20,
        }
    )
    window.data_split_imbalance_page.confirm_split_and_imbalance()
    output_dir = tmp_path / "outputs" / "data_split"
    assert (output_dir / "target_label_encoder.joblib").exists()

    page = window.column_config_page
    page.refresh()
    new_target = page.selected_columns_list.findItems(
        "new_target",
        Qt.MatchFlag.MatchExactly,
    )
    if not new_target:
        available = page.available_columns_list.findItems(
            "new_target",
            Qt.MatchFlag.MatchExactly,
        )[0]
        available.setSelected(True)
        page.add_selected()
    page.target_input.setCurrentText("new_target")
    page.confirm_modeling_columns()

    assert not (output_dir / "target_label_encoder.joblib").exists()
    assert not (output_dir / "split_indices.json").exists()
    assert window.config.target_column == "new_target"
    window.close()
    assert app is not None


def test_data_split_page_loads_saved_results_for_current_target(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="saved-split",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
        split_method="stratified",
    )
    config.save_json()
    window.config = config
    window.dataframe = pd.DataFrame(
        {"feature": range(100), "target": [0] * 80 + [1] * 20}
    )

    page = window.data_split_imbalance_page
    page.imbalance_method.setCurrentText("random_oversample")
    page.confirm_split_and_imbalance()
    expected_after_rows = page.after_distribution_tables["Train Set (Balanced)"].rowCount()

    page._clear_split_state()
    page.refresh()

    assert page.before_distribution_tables["Full Dataset"].rowCount() == 2
    assert (
        page.after_distribution_tables["Train Set (Balanced)"].rowCount()
        == expected_after_rows
    )
    assert (
        page.feedback_label.text()
        == "Saved split/imbalance data loaded for target column: target"
    )
    assert "#F8FFF9" in page.feedback_card.styleSheet()
    window.close()
    assert app is not None


def test_data_split_page_renders_section_cards(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="split-card-layout",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

    page = window.data_split_imbalance_page
    page.refresh()

    assert page.split_configuration_card.objectName() == "splitConfigurationCard"
    assert page.before_balancing_card.objectName() == "beforeBalancingCard"
    assert page.class_coverage_card.objectName() == "classCoverageCard"
    assert page.imbalance_handling_card.objectName() == "imbalanceHandlingCard"
    assert page.after_balancing_card.objectName() == "afterBalancingCard"
    assert page.confirmation_status_card.objectName() == "confirmationStatusCard"
    assert not page.controls_widget.isHidden()
    assert page.empty_state_card.isHidden()
    window.close()
    assert app is not None


def test_data_split_page_no_target_state_hides_tables(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="split-no-target",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column=None,
        task_type="classification",
    )
    window.dataframe = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    page = window.data_split_imbalance_page
    page.refresh()

    assert not page.empty_state_card.isHidden()
    assert page.controls_widget.isHidden()
    labels = page.empty_state_card.findChildren(type(page.current_target_label))
    assert any(
        "Please confirm Column Configuration first." in label.text()
        for label in labels
    )
    assert all(table.rowCount() == 0 for table in page.before_distribution_tables.values())
    assert all(table.rowCount() == 0 for table in page.after_distribution_tables.values())
    window.close()
    assert app is not None


def test_data_split_page_success_notification_auto_dismisses(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="split-success-dismiss",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame({"feature": range(40), "target": [0, 1] * 20})

    page = window.data_split_imbalance_page
    page.confirm_split_and_imbalance()
    assert not page.feedback_card.isHidden()

    page.success_notification_timer.timeout.emit()

    assert page.feedback_card.isHidden()
    window.close()
    assert app is not None


def test_data_split_page_tables_use_improved_styling(tmp_path):
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QAbstractItemView, QHeaderView

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="split-table-style",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame({"feature": range(30), "target": [0, 1, 2] * 10})

    page = window.data_split_imbalance_page
    page.refresh()
    table = page.before_distribution_tables["Full Dataset"]

    assert table.alternatingRowColors()
    assert table.selectionMode() == QAbstractItemView.SelectionMode.NoSelection
    assert table.horizontalHeader().sectionResizeMode(0) == QHeaderView.ResizeMode.Stretch
    assert "QHeaderView::section" in page.styleSheet()
    assert table.item(0, 0).textAlignment() & Qt.AlignmentFlag.AlignLeft
    assert table.item(0, 1).textAlignment() & Qt.AlignmentFlag.AlignCenter
    window.close()
    assert app is not None


def test_data_split_page_confirm_button_is_primary(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    assert page.confirm_button.objectName() == "primaryDataSplitButton"
    assert page.confirm_button.icon().isNull() is False
    assert "QPushButton#primaryDataSplitButton" in page.styleSheet()
    window.close()
    assert app is not None


def test_data_split_page_loads_with_safe_icons(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.data_split_imbalance_page import get_fa_icon
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    for name in (
        "fa6s.sliders",
        "fa6s.circle-exclamation",
        "fa6s.circle-check",
        "fa6s.triangle-exclamation",
        "fa6s.circle-info",
        "fa6s.floppy-disk",
    ):
        assert get_fa_icon(name).pixmap(16, 16).isNull() is False
    assert page is not None
    window.close()
    assert app is not None


def test_data_split_error_notification_icon_renders(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    page._show_error("Invalid split settings.")

    assert page.feedback_label.text() == "Error: Invalid split settings."
    assert page.feedback_icons[0].pixmap().isNull() is False
    assert "#DC2626" in page.feedback_card.styleSheet()
    window.close()
    assert app is not None


def test_data_split_success_notification_icon_renders(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    page._show_success_notification(["Saved."])

    assert page.feedback_label.text() == "Saved."
    assert page.feedback_icons[0].pixmap().isNull() is False
    assert "#16A34A" in page.feedback_card.styleSheet()
    window.close()
    assert app is not None


def test_data_split_configuration_card_icon_renders(tmp_path):
    from PySide6.QtWidgets import QApplication, QLabel

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    icon_label = page.split_configuration_card.findChild(
        QLabel,
        "splitConfigurationCardIcon",
    )

    assert icon_label is not None
    assert icon_label.pixmap().isNull() is False
    window.close()
    assert app is not None


def test_data_split_controls_use_polished_input_style(tmp_path):
    from PySide6.QtWidgets import QApplication, QToolButton

    from app.gui.data_split_imbalance_page import get_fa_icon
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.data_split_imbalance_page

    stylesheet = page.styleSheet()
    assert "QFrame#splitComboBoxControl" in stylesheet
    assert "QFrame#splitSpinBoxControl" in stylesheet
    assert "QComboBox::drop-down" in stylesheet
    assert "image:" not in stylesheet
    assert "QSpinBox::up-button" not in stylesheet
    assert "QSpinBox::down-button" not in stylesheet
    assert "::up-arrow" not in stylesheet
    assert "::down-arrow" not in stylesheet
    assert get_fa_icon("fa6s.angle-up").pixmap(12, 12).isNull() is False
    assert get_fa_icon("fa6s.angle-down").pixmap(12, 12).isNull() is False
    spin_arrow_buttons = page.findChildren(QToolButton, "splitSpinArrowButton")
    assert len(spin_arrow_buttons) == 8
    assert all(button.icon().isNull() is False for button in spin_arrow_buttons)
    combo_arrow_buttons = page.findChildren(QToolButton, "splitComboArrowButton")
    assert len(combo_arrow_buttons) == 3
    assert all(button.icon().isNull() is False for button in combo_arrow_buttons)
    assert page.train_percent.width() <= 100
    assert page.validation_percent.width() <= 100
    assert page.test_percent.width() <= 100
    assert page.random_seed.width() <= 140
    assert page.split_method.width() <= 230
    assert page.imbalance_method.width() <= 230
    assert page.ratio_preset.width() <= 230
    assert page.train_percent.text() == "70%"
    assert page.validation_percent.text() == "10%"
    assert page.test_percent.text() == "20%"
    spin_arrow_buttons[0].click()
    assert page.train_percent.value() == 71
    spin_arrow_buttons[1].click()
    assert page.train_percent.value() == 70
    window.close()
    assert app is not None


def test_data_split_page_blocks_invalid_percentage_total(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="invalid-split",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame({"feature": range(20), "target": [0, 1] * 10})

    page = window.data_split_imbalance_page
    page.train_percent.setValue(65)
    page.validation_percent.setValue(10)
    page.test_percent.setValue(20)
    page.confirm_split_and_imbalance()

    assert "Current total: 95%" in page.feedback_label.text()
    assert not (tmp_path / "outputs" / "data_split" / "split_indices.json").exists()
    window.close()
    assert app is not None


def test_data_split_page_after_table_uses_balanced_training_target(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="balanced-split",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
    )
    window.dataframe = pd.DataFrame(
        {"feature": range(100), "target": [0] * 80 + [1] * 20}
    )

    page = window.data_split_imbalance_page
    page.split_method.setCurrentText("stratified")
    page.imbalance_method.setCurrentText("random_oversample")
    page.confirm_split_and_imbalance()

    before_table = page.before_distribution_tables["Train Set"]
    after_table = page.after_distribution_tables["Train Set (Balanced)"]
    before_counts = sorted(int(before_table.item(row, 1).text()) for row in range(before_table.rowCount()))
    after_counts = sorted(int(after_table.item(row, 1).text()) for row in range(after_table.rowCount()))
    assert before_counts != after_counts
    assert len(set(after_counts)) == 1
    window.close()
    assert app is not None


def test_data_split_balancing_uses_only_training_class_set(tmp_path):
    import json

    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.config = ProjectConfig(
        project_name="training-class-set",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        date_column="event_date",
        task_type="classification",
        split_method="time",
    )
    window.dataframe = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0, 0, 0, 0, 1, 1, 1, 0, 0, 2],
            "event_date": pd.date_range("2026-01-01", periods=10, freq="D"),
        }
    )

    page = window.data_split_imbalance_page
    page.refresh()
    page.imbalance_method.setCurrentText("random_oversample")
    page.confirm_split_and_imbalance()

    before_train = _table_counts(page.before_distribution_tables["Train Set"])
    before_validation = _table_counts(page.before_distribution_tables["Validation Set"])
    before_test = _table_counts(page.before_distribution_tables["Test Set"])
    after_train = _table_counts(page.after_distribution_tables["Train Set (Balanced)"])
    after_validation = _table_counts(page.after_distribution_tables["Validation Set"])
    after_test = _table_counts(page.after_distribution_tables["Test Set"])

    assert before_train == {"0": 4, "1": 3}
    assert after_train == {"0": 4, "1": 4}
    assert "2" not in after_train
    assert after_validation == before_validation == {"0": 1}
    assert after_test == before_test == {"0": 1, "2": 1}
    assert (
        "Class '2' appears in validation/test but not training. "
        "The model cannot predict this class."
        in page.warning_label.text()
    )
    assert "Class '1' is absent from the validation set" in page.warning_label.text()
    assert "Class '1' is absent from the test set" in page.warning_label.text()
    assert "Warnings:" not in page.feedback_label.text()
    assert "Split and imbalance configuration saved successfully." in page.feedback_label.text()
    assert "#16A34A" in page.feedback_card.styleSheet()
    assert "#F8FFF9" in page.feedback_card.styleSheet()
    assert not page.warning_card.isHidden()
    assert "#DC2626" in page.warning_card.styleSheet()
    assert "#FFF8F7" in page.warning_card.styleSheet()
    coverage = pd.read_csv(
        tmp_path / "outputs" / "data_split" / "class_coverage_report.csv"
    )
    class_two = coverage[coverage["Class"] == 2].iloc[0]
    assert class_two["Status"] == "Missing in train - blocking"

    metadata = json.loads(
        (tmp_path / "outputs" / "data_split" / "imbalance_config.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["sampling_strategy_used"] == {"1": 4}
    assert metadata["balanced_train_distribution"] == {"0": 4, "1": 4}
    assert "2" not in metadata["balanced_train_distribution"]
    window.close()
    assert app is not None


def test_data_split_page_refreshes_when_saved_target_changes(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="target-refresh",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="mixed_target",
        task_type=None,
        split_method="random",
    )
    config.save_json()
    window.config = config
    window.dataframe = pd.DataFrame(
        {
            "feature": range(40),
            "mixed_target": ["A", 1.0] * 20,
            "new_target": [0, 1] * 20,
        }
    )

    page = window.data_split_imbalance_page
    page.refresh()
    assert page.current_target_label.text() == "Current target column: mixed_target"
    assert page.before_distribution_tables["Full Dataset"].rowCount() == 2

    config = ProjectConfig.load(tmp_path / "target-refresh.avista")
    config.target_column = "new_target"
    config.save_json()
    page.after_distribution_tables["Train Set (Balanced)"].setRowCount(1)

    page.refresh()

    assert window.config.target_column == "new_target"
    assert page.current_target_label.text() == "Current target column: new_target"
    assert page.before_distribution_tables["Full Dataset"].rowCount() == 2
    assert page.after_distribution_tables["Train Set (Balanced)"].rowCount() == 0
    window.close()
    assert app is not None


def test_data_split_page_rejects_saved_results_for_previous_target(tmp_path):
    import pandas as pd
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="changed-saved-target",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="old_target",
        task_type="classification",
        split_method="stratified",
    )
    config.save_json()
    window.config = config
    window.dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "old_target": [0, 1] * 30,
            "new_target": ["A", "B", "C"] * 20,
        }
    )

    page = window.data_split_imbalance_page
    page.confirm_split_and_imbalance()

    updated = ProjectConfig.load(tmp_path / "changed-saved-target.avista")
    updated.target_column = "new_target"
    updated.save_json()
    page.refresh()

    assert page.before_distribution_tables["Full Dataset"].rowCount() == 3
    assert page.after_distribution_tables["Train Set (Balanced)"].rowCount() == 0
    assert page.feedback_label.text() == ""
    assert (
        page.warning_label.text()
        == "Warnings:\nTarget column changed. Please confirm split and imbalance again."
    )
    assert not page.warning_card.isHidden()
    window.close()
    assert app is not None


def test_edge_case_page_uses_confirmed_split_artifacts(tmp_path):
    import pandas as pd
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="edge-page",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
        split_method="stratified",
        imbalance_method="none",
    )
    config.save_json()
    window.config = config
    window.dataframe = pd.DataFrame(
        {"feature": range(100), "target": [0, 1] * 50, "unused": [None] * 100}
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    window.dataframe[["feature", "target"]].to_csv(
        data_dir / "modeling_subset.csv",
        index=False,
    )
    window.data_split_imbalance_page.confirm_split_and_imbalance()

    page = window.edge_case_report_page
    page.refresh()
    page.run_checks()

    report_path = tmp_path / "outputs" / "edge_cases" / "edge_case_report.json"
    for _ in range(200):
        app.processEvents()
        if report_path.exists() and page.run_button.isEnabled():
            break
        QTest.qWait(10)
    assert report_path.exists()
    assert "Current target column: target" == page.target_label.text()
    assert "Selected feature columns: 1" == page.feature_count_label.text()
    assert "Data Split & Imbalance confirmed: Yes" == page.split_status_label.text()
    assert "Imbalance method: none" == page.imbalance_label.text()
    assert "Can continue: True" in page.status_label.text()
    assert page.content_stack.currentWidget() is page.report_content
    assert page.status_tiles["ready"][1].text() == "YES"
    assert not page.issue_empty_states["fatal"].isHidden()
    assert not page.issue_empty_states["error"].isHidden()
    assert page.overall_status.text() == "READY FOR TRAINING"
    while page.thread is not None:
        app.processEvents()
        QTest.qWait(5)
    window.close()
    assert app is not None


def test_edge_case_page_empty_state_and_dynamic_issue_tables(tmp_path):
    from PySide6.QtWidgets import QApplication

    from app.core.error_handler import EdgeCaseReport
    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="edge-layout",
        project_dir=str(tmp_path),
        input_file="",
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        imbalance_method="smote",
    )
    config.save()
    window.config = config
    page = window.edge_case_report_page
    page.refresh()

    assert page.content_stack.currentWidget() is page.empty_state
    assert page.empty_run_button.icon().isNull() is False
    assert page.run_button.objectName() == "edgeCasePrimaryButton"
    assert page.issue_tables["warning"].isSortingEnabled()
    assert page.notification_timer.isSingleShot()

    report = EdgeCaseReport(
        context={
            **config.project_metadata(),
            "target_column": "target",
            "feature_count": 1,
            "column_configuration_confirmed": True,
            "split_confirmed": True,
            "imbalance_method": "smote",
        }
    )
    report.add(
        "fatal",
        "artifacts",
        "Saved target artifact is missing.",
        "Confirm Data Split & Imbalance again.",
    )
    report.add(
        "error",
        "target",
        "Column 'target' contains missing values.",
        "Clean the target values.",
        affected_column="target",
    )
    report.add(
        "warning",
        "features",
        "Column 'feature' has rare values.",
        "Review the rare values.",
        affected_column="feature",
    )
    report_path = (
        tmp_path / "outputs" / "edge_cases" / "edge_case_report.json"
    )
    report.save_json(report_path)
    page.report_metadata = json.loads(report_path.read_text(encoding="utf-8"))
    page._render_report(report, report_path, config)

    assert page.content_stack.currentWidget() is page.report_content
    assert page.issue_tables["fatal"].rowCount() == 1
    assert page.issue_tables["error"].rowCount() == 1
    assert page.issue_tables["warning"].rowCount() == 1
    assert page.issue_tables["error"].item(0, 2).text() == "target"
    assert page.issue_tables["warning"].item(0, 2).text() == "feature"
    assert page.status_tiles["fatals"][1].text() == "1"
    assert page.status_tiles["errors"][1].text() == "1"
    assert page.status_tiles["warnings"][1].text() == "1"
    assert page.overall_status.text() == "NOT READY FOR TRAINING"
    window.close()
    assert app is not None


def test_training_page_shows_ready_confirmed_status(tmp_path):
    import json

    import pandas as pd
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="training-page",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        feature_columns=["feature"],
        target_column="target",
        task_type="classification",
        split_method="stratified",
        imbalance_method="none",
        selected_models=["logistic_regression"],
        enable_cross_validation=True,
        cv_folds=3,
    )
    config.save_json()
    window.config = config
    window.dataframe = pd.DataFrame({"feature": range(100), "target": [0, 1] * 50})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    window.dataframe.to_csv(data_dir / "modeling_subset.csv", index=False)
    window.data_split_imbalance_page.confirm_split_and_imbalance()
    window.edge_case_report_page.run_checks()
    edge_path = tmp_path / "outputs" / "edge_cases" / "edge_case_report.json"
    for _ in range(200):
        app.processEvents()
        if edge_path.exists() and window.edge_case_report_page.run_button.isEnabled():
            break
        QTest.qWait(10)

    page = window.training_page
    page.refresh()

    assert page.status_values["column_confirmed"].text() == "Yes"
    assert page.status_values["split_confirmed"].text() == "Yes"
    assert page.status_values["edge_passed"].text() == "Yes"
    assert page.status_values["target"].text() == "target"
    assert page.status_values["feature_count"].text() == "1"
    assert page.status_values["cv_enabled"].text() == "3 folds"
    assert page.status_values["cv_folds"].text() == "3"
    assert page.readiness_card.objectName() == "trainingReadinessCard"
    assert page.controls_card.objectName() == "trainingControlsCard"
    assert page.progress_card.objectName() == "trainingProgressCard"
    assert page.curves_card.objectName() == "trainingCurvesCard"
    assert page.results_card.objectName() == "trainingResultsCard"
    assert page.outputs_card.objectName() == "trainingOutputsCard"
    assert page.readiness_value.text() == "Ready To Train"
    assert page.start_button.objectName() == "primaryTrainingButton"
    assert page.stop_button.objectName() == "dangerTrainingButton"
    assert not page.start_button.icon().isNull()
    assert page.start_button.isEnabled()

    page._on_model_finished(
        "Logistic Regression",
        {
            "model_name": "Logistic Regression",
            "status": "trained",
            "train_metrics": {"accuracy": 1.0, "macro_f1": 1.0},
            "validation_metrics": {"accuracy": 0.9, "macro_f1": 0.8},
            "test_metrics": {"accuracy": 0.85, "macro_f1": 0.84, "roc_auc": 0.9},
            "cv_summary": {
                "accuracy": {"mean": 0.88, "std": 0.02},
                "macro_f1": {"mean": 0.87, "std": 0.03},
            },
            "saved": True,
        },
    )
    assert page.result_table.rowCount() == 1
    assert page.result_table.item(0, 0).text() == "Logistic Regression"
    assert page.result_table.item(0, 13).text() == "Yes"
    window.close()
    assert app is not None


def test_training_page_keeps_thread_references_until_qt_destroys_thread():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page
    thread_marker = object()
    worker_marker = object()
    page.thread = thread_marker
    page.worker = worker_marker

    page._thread_finished()

    assert page.thread is thread_marker
    assert page.worker is worker_marker

    page._clear_thread_references()

    assert page.thread is None
    assert page.worker is None
    window.close()
    assert app is not None


def test_training_page_upserts_running_failed_and_final_result_without_duplicates():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_model_started("Decision Tree")
    assert page.result_table.rowCount() == 1
    assert page.result_table.item(0, 0).text() == "Decision Tree"
    assert page.result_table.item(0, 1).text() == "Running"

    failed = {
        "model_name": "Decision Tree",
        "status": "failed",
        "error": "fit failed",
        "saved": False,
    }
    page._on_model_result_ready(failed)
    assert page.result_table.rowCount() == 1
    assert page.result_table.item(0, 1).text() == "failed"
    assert page.result_table.item(0, 1).toolTip() == "fit failed"

    page._on_training_finished({"results": [failed]})
    assert page.result_table.rowCount() == 1
    window.close()
    assert app is not None


def test_training_page_epoch_progress_updates_live_curve():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "MambaAttention",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )
    page._on_progress(
        {
            "event": "epoch_progress",
            "model": "MambaAttention",
            "epoch": 1,
            "total_epochs": 80,
            "train_loss": 0.95,
            "train_accuracy": 0.57,
            "validation_loss": 0.88,
            "validation_macro_f1": 0.41,
            "validation_accuracy": 0.52,
            "fold": 0,
            "total_folds": 0,
            "step": "epoch 1",
            "percent": 1,
        }
    )

    assert page.curve_epochs == [1]
    assert page.curve_train_loss == [0.95]
    assert page.curve_train_accuracy == [0.57]
    assert page.curve_validation_loss == [0.88]
    assert page.curve_validation_macro_f1 == [0.41]
    assert page.curve_validation_accuracy == [0.52]
    assert len(page.curve_figure.axes) == 2
    assert page.curve_figure.axes[0].get_title() == "Accuracy"
    assert page.curve_figure.axes[1].get_title() == "Loss"
    assert page.curve_canvas.isVisible() is False
    assert page.curve_canvas.isHidden() is False
    log_text = page.log_text.toPlainText()
    assert "Live training curve started for MambaAttention" in log_text
    assert "Training curves updated: epoch 1" in log_text
    window.close()
    assert app is not None


def test_training_page_sklearn_progress_keeps_curve_message():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "Decision Tree",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )

    assert page.curve_epochs == []
    assert page.curve_message.text() == "No deep learning model currently training."
    assert page.curve_canvas.isHidden()
    window.close()
    assert app is not None


def test_training_page_ft_transformer_progress_updates_live_curve():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "FT-Transformer",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )
    page._on_progress(
        {
            "event": "epoch_progress",
            "model": "FT-Transformer",
            "epoch": 1,
            "total_epochs": 80,
            "train_loss": 0.75,
            "train_accuracy": 0.64,
            "validation_loss": 0.68,
            "validation_macro_f1": 0.55,
            "validation_accuracy": 0.6,
            "fold": 0,
            "total_folds": 0,
            "step": "epoch 1",
            "percent": 1,
        }
    )

    assert page.curve_model_name == "FT-Transformer"
    assert page.curve_epochs == [1]
    assert page.curve_train_loss == [0.75]
    assert page.curve_train_accuracy == [0.64]
    assert page.curve_validation_loss == [0.68]
    assert page.curve_validation_macro_f1 == [0.55]
    assert page.curve_validation_accuracy == [0.6]
    assert len(page.curve_figure.axes) == 2
    window.close()
    assert app is not None


def test_training_page_autoint_progress_updates_live_curve():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "AutoInt",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )
    page._on_progress(
        {
            "event": "epoch_progress",
            "model": "AutoInt",
            "epoch": 1,
            "total_epochs": 80,
            "train_loss": 0.72,
            "train_accuracy": 0.67,
            "validation_loss": 0.65,
            "validation_macro_f1": 0.58,
            "validation_accuracy": 0.63,
            "fold": 0,
            "total_folds": 0,
            "step": "epoch 1",
            "percent": 1,
        }
    )

    assert page.curve_model_name == "AutoInt"
    assert page.curve_epochs == [1]
    assert page.curve_train_loss == [0.72]
    assert page.curve_train_accuracy == [0.67]
    assert page.curve_validation_loss == [0.65]
    assert page.curve_validation_macro_f1 == [0.58]
    assert page.curve_validation_accuracy == [0.63]
    assert len(page.curve_figure.axes) == 2
    window.close()
    assert app is not None


def test_training_page_tab_resnet_progress_updates_live_curve():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "TabResNet",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )
    page._on_progress(
        {
            "event": "epoch_progress",
            "model": "TabResNet",
            "epoch": 1,
            "total_epochs": 80,
            "train_loss": 0.7,
            "train_accuracy": 0.69,
            "validation_loss": 0.62,
            "validation_macro_f1": 0.6,
            "validation_accuracy": 0.65,
            "fold": 0,
            "total_folds": 0,
            "step": "epoch 1",
            "percent": 1,
        }
    )

    assert page.curve_model_name == "TabResNet"
    assert page.curve_epochs == [1]
    assert page.curve_train_loss == [0.7]
    assert page.curve_train_accuracy == [0.69]
    assert page.curve_validation_loss == [0.62]
    assert page.curve_validation_macro_f1 == [0.6]
    assert page.curve_validation_accuracy == [0.65]
    assert len(page.curve_figure.axes) == 2
    window.close()
    assert app is not None


def test_training_page_tabpfn_does_not_show_live_curve():
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    page._on_progress(
        {
            "model": "TabPFN 2.5",
            "fold": 0,
            "total_folds": 0,
            "step": "started",
            "percent": 0,
        }
    )

    assert page.curve_model_name == ""
    assert page.curve_canvas.isHidden()
    assert page.curve_message.text() == "No deep learning model currently training."
    window.close()
    assert app is not None


def test_training_page_notifications_and_output_actions(tmp_path):
    import json

    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="training-outputs",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    config.save_json()
    window.config = config
    output_dir = tmp_path / "outputs" / "training"
    output_dir.mkdir(parents=True)
    (output_dir / "training_results.csv").write_text(
        "model,status\nDecision Tree,trained\n",
        encoding="utf-8",
    )
    (output_dir / "training_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "model_name": "Decision Tree",
                        "status": "trained",
                        "saved": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    page = window.training_page
    page.refresh()

    assert page.open_results_button.isEnabled()
    assert page.open_report_button.isEnabled()
    assert page.models_saved_value.text() == "1"
    assert int(page.reports_generated_value.text()) >= 2

    page._show_notification("Training started", "success")
    assert page.notification_card.isHidden() is False
    assert page.notification_timer.interval() == 5000
    page._show_notification("Review warning", "warning")
    assert page.notification_timer.interval() == 8000
    page._show_notification("Training failed", "error")
    assert not page.notification_timer.isActive()
    assert page.notification_card.isHidden() is False

    window.close()
    assert app is not None


def test_training_worker_writes_aggregate_results_csv(tmp_path):
    import csv

    from app.gui.workers import _write_training_results_csv

    path = tmp_path / "training_results.csv"
    _write_training_results_csv(
        path,
        [
            {
                "model_name": "Random Forest",
                "status": "trained",
                "train_metrics": {"accuracy": 0.95, "macro_f1": 0.94},
                "validation_metrics": {"accuracy": 0.9, "macro_f1": 0.89},
                "test_metrics": {
                    "accuracy": 0.88,
                    "macro_f1": 0.87,
                    "roc_auc": 0.92,
                },
                "cv_summary": {
                    "accuracy": {"mean": 0.91, "std": 0.02},
                    "macro_f1": {"mean": 0.9, "std": 0.03},
                },
                "saved": True,
            }
        ],
    )

    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    assert rows[0]["model"] == "Random Forest"
    assert rows[0]["validation_macro_f1"] == "0.89"
    assert rows[0]["saved"] == "True"


def test_training_subprocess_progress_includes_train_accuracy():
    from app.gui.workers import _subprocess_progress

    progress = _subprocess_progress(
        {
            "event": "epoch_progress",
            "model": "TabResNet",
            "epoch": 2,
            "total_epochs": 80,
            "train_loss": 0.6,
            "train_accuracy": 0.72,
            "validation_loss": 0.64,
            "validation_macro_f1": 0.61,
            "validation_accuracy": 0.66,
            "percent": 2,
        }
    )

    assert progress["event"] == "epoch_progress"
    assert progress["train_accuracy"] == 0.72


def test_training_page_icons_use_avista_primary_and_button_text_colors():
    from PySide6.QtWidgets import QApplication, QLabel

    from app.gui.icon_system import PRIMARY
    from app.gui.main_window import MainWindow
    from app.gui.training_page import DISABLED_COLOR

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page

    for card_name in (
        "trainingReadinessCard",
        "trainingControlsCard",
        "trainingProgressCard",
        "trainingCurvesCard",
        "trainingResultsCard",
        "trainingOutputsCard",
    ):
        header_icon = page.findChild(QLabel, f"{card_name}Icon")
        assert header_icon is not None
        assert header_icon.property("iconColor") == PRIMARY

    tile_icons = page.findChildren(QLabel, "trainingStatusTileIcon")
    assert len(tile_icons) == 8
    assert all(label.property("iconColor") == PRIMARY for label in tile_icons)

    assert page.start_button.property("iconColor") == "#FFFFFF"
    assert page.stop_button.property("iconColor") == "#FFFFFF"
    for button in (
        page.open_output_button,
        page.open_folder_button,
        page.open_results_button,
        page.open_report_button,
    ):
        assert button.property("iconColor") == PRIMARY
        assert button.property("disabledIconColor") == DISABLED_COLOR
    assert page.stop_button.property("disabledIconColor") == DISABLED_COLOR

    window.close()
    assert app is not None


def test_training_page_running_state_animates_and_resets_all_exit_paths(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    page = window.training_page
    ready_status = page._empty_status([])
    ready_status["ready"] = True
    monkeypatch.setattr(page, "_latest_config", lambda: None)
    monkeypatch.setattr(page, "_preflight_status", lambda _config: ready_status)

    page._set_running(True)
    assert page.start_button.text() == "Training..."
    assert not page.start_button.isEnabled()
    assert page.stop_button.isEnabled()
    assert page.spinner_timer.isActive()
    first_spinner_key = page.start_button.icon().cacheKey()
    page.spinner_timer.timeout.emit()
    assert page.start_button.icon().cacheKey() != first_spinner_key

    page._on_training_finished({"results": []})
    assert page.start_button.text() == "Start Training"
    assert page.start_button.isEnabled()
    assert not page.stop_button.isEnabled()
    assert not page.spinner_timer.isActive()

    page._set_running(True)
    page._on_training_failed("worker crashed")
    assert page.start_button.text() == "Start Training"
    assert page.start_button.isEnabled()
    assert not page.stop_button.isEnabled()
    assert not page.spinner_timer.isActive()

    page._set_running(True)
    page._on_training_cancelled()
    assert page.start_button.text() == "Start Training"
    assert page.start_button.isEnabled()
    assert not page.stop_button.isEnabled()
    assert not page.spinner_timer.isActive()

    page._set_running(True)
    page._thread_finished()
    assert page.start_button.text() == "Start Training"
    assert page.start_button.isEnabled()
    assert not page.stop_button.isEnabled()
    assert not page.spinner_timer.isActive()

    window.close()
    assert app is not None


def test_training_page_output_folder_stays_enabled_while_running(
    tmp_path,
    monkeypatch,
):
    from PySide6.QtWidgets import QApplication

    from app.core.project_config import ProjectConfig
    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    config = ProjectConfig(
        project_name="running-output",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
    )
    config.save_json()
    output_dir = tmp_path / "outputs" / "training"
    output_dir.mkdir(parents=True)
    window.config = config
    page = window.training_page

    page.refresh()
    ready_status = page._empty_status([])
    ready_status["ready"] = True
    monkeypatch.setattr(page, "_preflight_status", lambda _config: ready_status)
    page._set_running(True)

    assert page.open_output_button.isEnabled()
    assert page.open_folder_button.isEnabled()
    assert page.stop_button.isEnabled()

    page._set_running(False)
    window.close()
    assert app is not None
