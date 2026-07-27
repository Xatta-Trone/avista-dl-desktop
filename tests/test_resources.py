from pathlib import Path

import app.utils.resources as resources
from app.utils.resources import (
    get_app_resource_path,
    resolve_tabpfn_checkpoint,
)


def test_app_resource_path_resolves_development_checkpoint():
    path = get_app_resource_path(
        "app/assets/tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )

    assert path.is_file()
    assert path.name == "tabpfn-v2.5-classifier-v2.5_default.ckpt"


def test_app_resource_path_resolves_development_logo():
    path = get_app_resource_path("app/assets/logo.png")

    assert path.is_file()
    assert path.name == "logo.png"
    assert path.stat().st_size > 0


def test_app_resource_path_resolves_pyinstaller_bundle(tmp_path, monkeypatch):
    resource = tmp_path / "reference" / "checkpoint.ckpt"
    resource.parent.mkdir()
    resource.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

    resolved = get_app_resource_path("reference/checkpoint.ckpt")

    assert resolved == resource.resolve()


def test_app_resource_path_resolves_packaged_standalone_asset(tmp_path, monkeypatch):
    executable = tmp_path / "AVISTA.exe"
    executable.write_bytes(b"exe")
    asset = tmp_path / "app" / "assets" / "logo.png"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"logo")
    monkeypatch.setattr("sys.executable", str(executable))
    monkeypatch.setitem(resources.__dict__, "__compiled__", object())

    resolved = get_app_resource_path("app/assets/logo.png")

    assert resources.is_packaged_application()
    assert resolved == asset.resolve()


def test_tabpfn_checkpoint_resolver_supports_pyinstaller_internal_root(
    tmp_path,
    monkeypatch,
):
    checkpoint = (
        tmp_path
        / "app"
        / "assets"
        / "tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr("sys._MEIPASS", str(tmp_path), raising=False)

    assert resolve_tabpfn_checkpoint() == checkpoint.resolve()


def test_tabpfn_checkpoint_resolver_reports_checked_paths(
    tmp_path,
    monkeypatch,
):
    executable = tmp_path / "AVISTADeepWorker.exe"
    executable.write_bytes(b"worker")
    monkeypatch.delattr("sys._MEIPASS", raising=False)
    monkeypatch.setattr("sys.executable", str(executable))
    monkeypatch.setitem(resources.__dict__, "__compiled__", object())
    monkeypatch.setitem(
        resources.__dict__,
        "__file__",
        str(tmp_path / "_internal" / "app" / "utils" / "resources.py"),
    )

    try:
        resolve_tabpfn_checkpoint()
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("Missing checkpoint must raise FileNotFoundError.")

    assert "app\\assets\\tabpfn-v2.5-classifier-v2.5_default.ckpt" in message
    assert "assets\\tabpfn-v2.5-classifier-v2.5_default.ckpt" in message
