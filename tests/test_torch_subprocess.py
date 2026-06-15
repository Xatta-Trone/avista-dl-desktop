import io
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from app.core.project_config import ProjectConfig
from app.gui.workers import TrainingWorker, build_torch_subprocess_command
from app.training import run_torch_model
from tests.test_trainer_evaluator import save_encoded_string_training_bundle


def make_config(tmp_path, selected_models=None):
    config = ProjectConfig(
        project_name="subprocess-test",
        project_dir=str(tmp_path),
        input_file=str(tmp_path / "data.csv"),
        output_dir=str(tmp_path / "outputs"),
        target_column="target",
        feature_columns=["feature"],
        task_type="classification",
        selected_models=selected_models or ["mamba_attention"],
    )
    config.save_json()
    return config


def test_torch_subprocess_command_contains_required_arguments(tmp_path):
    config = make_config(tmp_path)

    command = build_torch_subprocess_command(config, "mamba_attention")

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
    ]
    assert command[command.index("--project-dir") + 1] == str(tmp_path.resolve())
    assert command[command.index("--config") + 1] == str(
        (tmp_path / "subprocess-test.avista").resolve()
    )
    assert command[command.index("--model") + 1] == "MambaAttention"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs\\training\\MambaAttention"
    )


def test_worker_emits_first_model_result_before_second_finishes(
    tmp_path,
    monkeypatch,
):
    config = make_config(
        tmp_path,
        selected_models=["logistic_regression", "decision_tree"],
    )
    worker = TrainingWorker(config, save_outputs=False)
    received = []
    worker.model_result_ready.connect(received.append)
    calls = []

    def fake_train_saved_models(model_config, **_kwargs):
        model_name = model_config.selected_models[0]
        calls.append(model_name)
        if model_name == "decision_tree":
            assert received[0]["model_name"] == "Logistic Regression"
        display_name = (
            "Logistic Regression"
            if model_name == "logistic_regression"
            else "Decision Tree"
        )
        return {
            "results": [
                {
                    "model_name": display_name,
                    "status": "trained",
                    "saved": False,
                }
            ]
        }

    monkeypatch.setattr(
        "app.gui.workers.train_saved_models",
        fake_train_saved_models,
    )

    summary = worker._run_selected_models()

    assert calls == ["logistic_regression", "decision_tree"]
    assert [result["model_name"] for result in received] == [
        "Logistic Regression",
        "Decision Tree",
    ]
    assert len(summary["results"]) == 2


def test_ft_transformer_subprocess_command_uses_model_output_folder(tmp_path):
    config = make_config(tmp_path, selected_models=["ft_transformer"])

    command = build_torch_subprocess_command(config, "ft_transformer")

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
    ]
    assert command[command.index("--model") + 1] == "FT-Transformer"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs\\training\\FT-Transformer"
    )


def test_autoint_subprocess_command_uses_model_output_folder(tmp_path):
    config = make_config(tmp_path, selected_models=["autoint"])

    command = build_torch_subprocess_command(config, "autoint")

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
    ]
    assert command[command.index("--model") + 1] == "AutoInt"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs\\training\\AutoInt"
    )


def test_tab_resnet_subprocess_command_uses_model_output_folder(tmp_path):
    config = make_config(tmp_path, selected_models=["tab_resnet"])

    command = build_torch_subprocess_command(config, "tab_resnet")

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
    ]
    assert command[command.index("--model") + 1] == "TabResNet"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs\\training\\TabResNet"
    )


def test_tabpfn_subprocess_command_uses_safe_output_folder(tmp_path):
    config = make_config(tmp_path, selected_models=["tabpfn"])

    command = build_torch_subprocess_command(config, "tabpfn")

    assert command[:4] == [
        sys.executable,
        "-u",
        "-m",
        "app.training.run_torch_model",
    ]
    assert command[command.index("--model") + 1] == "TabPFN 2.5"
    assert command[command.index("--output-dir") + 1].endswith(
        "outputs\\training\\TabPFN_2_5"
    )


def test_torch_subprocess_entrypoint_saves_failure_reason(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    output_dir = tmp_path / "outputs" / "training" / "MambaAttention"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_torch_model",
            "--project-dir",
            str(tmp_path),
            "--config",
            str(tmp_path / "subprocess-test.avista"),
            "--model",
            "UnsupportedTorchModel",
            "--output-dir",
            str(output_dir),
        ],
    )

    return_code = run_torch_model.main()

    assert return_code == 1
    failure = json.loads(
        (output_dir / "failure_reason.json").read_text(encoding="utf-8")
    )
    assert "Unsupported torch model" in failure["error"]
    assert failure["source"] == "torch_subprocess"


def test_worker_contains_nonzero_torch_subprocess_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"started","model":"MambaAttention"}\n'
            )
            self.stderr = io.StringIO("native heap corruption\n")
            self.returncode = -1073740940

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)

    result = worker._run_torch_subprocess("mamba_attention")

    assert result["status"] == "failed"
    assert result["return_code"] == -1073740940
    assert result["error"] == (
        "MambaAttention failed in subprocess. GUI remained stable."
    )
    assert "native heap corruption" in result["stderr_tail"]
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "MambaAttention"
        / "failure_reason.json"
    )
    assert failure_path.exists()


def test_worker_parses_epoch_progress_before_process_completion(
    tmp_path,
    monkeypatch,
):
    config = make_config(tmp_path)
    received = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"epoch_progress","model":"MambaAttention",'
                '"epoch":1,"total_epochs":2,"train_loss":0.9,'
                '"validation_loss":0.8,"validation_macro_f1":0.4,'
                '"validation_accuracy":0.5}\n'
                '{"event":"result","result":{"model_name":"MambaAttention",'
                '"status":"trained","saved":false}}\n'
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            assert received and received[0]["epoch"] == 1
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)
    monkeypatch.setattr(worker, "_on_progress", received.append)

    result = worker._run_torch_subprocess("mamba_attention")

    assert result["status"] == "trained"
    assert received[0]["validation_accuracy"] == 0.5


def test_worker_parses_ft_transformer_live_curve_event(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["ft_transformer"])
    received = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"started","model":"FT-Transformer"}\n'
                '{"event":"epoch_progress","model":"FT-Transformer",'
                '"epoch":1,"total_epochs":2,"train_loss":0.7,'
                '"validation_loss":0.6,"validation_macro_f1":0.5,'
                '"validation_accuracy":0.75}\n'
                '{"event":"result","result":{"model_name":"FT-Transformer",'
                '"status":"trained","saved":false}}\n'
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)
    monkeypatch.setattr(worker, "_on_progress", received.append)

    result = worker._run_torch_subprocess("ft_transformer")

    epoch_event = next(item for item in received if item.get("event") == "epoch_progress")
    assert result["status"] == "trained"
    assert epoch_event["model"] == "FT-Transformer"
    assert epoch_event["validation_accuracy"] == 0.75


def test_worker_saves_exact_ft_transformer_child_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["ft_transformer"])

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"failed","model_name":"FT-Transformer",'
                '"error":"d_token must be divisible by n_heads"}\n'
            )
            self.stderr = io.StringIO("d_token must be divisible by n_heads\n")
            self.returncode = 1

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)

    result = worker._run_torch_subprocess("ft_transformer")

    assert result["status"] == "failed"
    assert result["error"] == "d_token must be divisible by n_heads"
    failure_path = (
        tmp_path
        / "outputs"
        / "training"
        / "FT-Transformer"
        / "failure_reason.json"
    )
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["error"] == result["error"]


def test_worker_parses_autoint_live_curve_event(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["autoint"])
    received = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"started","model":"AutoInt"}\n'
                '{"event":"epoch_progress","model":"AutoInt",'
                '"epoch":1,"total_epochs":2,"train_loss":0.7,'
                '"validation_loss":0.6,"validation_macro_f1":0.5,'
                '"validation_accuracy":0.75}\n'
                '{"event":"result","result":{"model_name":"AutoInt",'
                '"status":"trained","saved":false}}\n'
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)
    monkeypatch.setattr(worker, "_on_progress", received.append)

    result = worker._run_torch_subprocess("autoint")

    epoch_event = next(item for item in received if item.get("event") == "epoch_progress")
    assert result["status"] == "trained"
    assert epoch_event["model"] == "AutoInt"
    assert epoch_event["validation_accuracy"] == 0.75


def test_worker_parses_tab_resnet_live_curve_event(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["tab_resnet"])
    received = []

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                '{"event":"started","model":"TabResNet"}\n'
                '{"event":"epoch_progress","model":"TabResNet",'
                '"epoch":1,"total_epochs":2,"train_loss":0.7,'
                '"validation_loss":0.6,"validation_macro_f1":0.5,'
                '"validation_accuracy":0.75}\n'
                '{"event":"result","result":{"model_name":"TabResNet",'
                '"status":"trained","saved":false}}\n'
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)
    monkeypatch.setattr(worker, "_on_progress", received.append)

    result = worker._run_torch_subprocess("tab_resnet")

    epoch_event = next(item for item in received if item.get("event") == "epoch_progress")
    assert result["status"] == "trained"
    assert epoch_event["model"] == "TabResNet"
    assert epoch_event["validation_accuracy"] == 0.75


def test_worker_preserves_tabpfn_skipped_subprocess_result(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["tabpfn"])
    reason = "TabPFN skipped: package tabpfn is not installed."

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO(
                "\n".join(
                    [
                        json.dumps(
                            {"event": "started", "model": "TabPFN 2.5"}
                        ),
                        json.dumps(
                            {
                                "event": "result",
                                "result": {
                                    "model_name": "TabPFN 2.5",
                                    "status": "skipped",
                                    "reason": reason,
                                    "saved": False,
                                },
                            }
                        ),
                        json.dumps(
                            {"event": "complete", "model": "TabPFN 2.5"}
                        ),
                    ]
                )
                + "\n"
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self):
            return self.returncode

    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: FakeProcess(),
    )
    worker = TrainingWorker(config)

    result = worker._run_torch_subprocess("tabpfn")

    assert result["status"] == "skipped"
    assert result["reason"] == reason


def test_worker_keeps_sklearn_training_in_existing_path(tmp_path, monkeypatch):
    config = make_config(tmp_path, selected_models=["decision_tree"])
    calls = []

    def fake_train_saved_models(config_arg, **_kwargs):
        calls.append(list(config_arg.selected_models))
        return {
            "status": "completed",
            "results": [
                {
                    "model_name": "Decision Tree",
                    "status": "trained",
                    "saved": False,
                }
            ],
        }

    monkeypatch.setattr(
        "app.gui.workers.train_saved_models",
        fake_train_saved_models,
    )
    monkeypatch.setattr(
        "app.gui.workers.subprocess.Popen",
        lambda *_args, **_kwargs: pytest.fail("Popen must not run for sklearn."),
    )
    worker = TrainingWorker(config, save_outputs=False)

    summary = worker._run_selected_models()

    assert calls == [["decision_tree"]]
    assert summary["results"][0]["status"] == "trained"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_torch_subprocess_success_smoke(tmp_path):
    config = make_config(tmp_path)
    config.task_type = "auto"
    config.feature_columns = ["x1", "x2", "cat"]
    config.model_params = {
        "mamba_attention": {
            "hidden_dim": 8,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "warmup_epochs": 1,
            "early_stopping_patience": 1,
        }
    }
    config.save_json()
    save_encoded_string_training_bundle(tmp_path, config)
    command = build_torch_subprocess_command(config, "mamba_attention")
    environment = {
        **__import__("os").environ,
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    epoch_events = [
        event
        for event in events
        if event.get("event") == "epoch_progress"
    ]
    assert epoch_events
    assert "train_loss" in epoch_events[0]
    assert "validation_loss" in epoch_events[0]
    assert "validation_macro_f1" in epoch_events[0]
    assert "validation_accuracy" in epoch_events[0]
    assert any(event.get("event") == "result" for event in events)
    output_dir = (
        tmp_path
        / "outputs"
        / "training"
        / "MambaAttention"
    )
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "training_curves.pdf").exists()
    history = pd.read_csv(output_dir / "training_history.csv")
    assert history.columns.tolist() == [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_macro_f1",
        "validation_accuracy",
    ]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_ft_transformer_subprocess_success_smoke(tmp_path):
    config = make_config(tmp_path, selected_models=["ft_transformer"])
    config.task_type = "auto"
    config.feature_columns = ["x1", "x2", "cat"]
    config.model_params = {
        "ft_transformer": {
            "d_token": 8,
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "warmup_epochs": 1,
            "early_stopping_patience": 1,
        }
    }
    config.save_json()
    save_encoded_string_training_bundle(tmp_path, config)
    command = build_torch_subprocess_command(config, "ft_transformer")
    environment = {
        **__import__("os").environ,
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    epoch_event = next(
        event for event in events if event.get("event") == "epoch_progress"
    )
    assert epoch_event["model"] == "FT-Transformer"
    assert "validation_accuracy" in epoch_event
    output_dir = tmp_path / "outputs" / "training" / "FT-Transformer"
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "training_curves.pdf").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_autoint_subprocess_success_smoke(tmp_path):
    config = make_config(tmp_path, selected_models=["autoint"])
    config.task_type = "auto"
    config.feature_columns = ["x1", "x2", "cat"]
    config.model_params = {
        "autoint": {
            "d": 8,
            "n_heads": 2,
            "n_layers": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "warmup_epochs": 1,
            "early_stopping_patience": 1,
        }
    }
    config.save_json()
    save_encoded_string_training_bundle(tmp_path, config)
    command = build_torch_subprocess_command(config, "autoint")
    environment = {
        **__import__("os").environ,
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    epoch_event = next(
        event for event in events if event.get("event") == "epoch_progress"
    )
    assert epoch_event["model"] == "AutoInt"
    assert "validation_accuracy" in epoch_event
    output_dir = tmp_path / "outputs" / "training" / "AutoInt"
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "training_curves.pdf").exists()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is not installed",
)
def test_tab_resnet_subprocess_success_smoke(tmp_path):
    config = make_config(tmp_path, selected_models=["tab_resnet"])
    config.task_type = "auto"
    config.feature_columns = ["x1", "x2", "cat"]
    config.model_params = {
        "tab_resnet": {
            "hidden": 8,
            "n_blocks": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 1,
            "warmup_epochs": 1,
            "early_stopping_patience": 1,
        }
    }
    config.save_json()
    save_encoded_string_training_bundle(tmp_path, config)
    command = build_torch_subprocess_command(config, "tab_resnet")
    environment = {
        **__import__("os").environ,
        "PYTHONFAULTHANDLER": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    epoch_event = next(
        event for event in events if event.get("event") == "epoch_progress"
    )
    assert epoch_event["model"] == "TabResNet"
    assert "validation_accuracy" in epoch_event
    output_dir = tmp_path / "outputs" / "training" / "TabResNet"
    assert (output_dir / "trained_model.pt").exists()
    assert (output_dir / "training_history.csv").exists()
    assert (output_dir / "training_curves.png").exists()
    assert (output_dir / "training_curves.pdf").exists()
