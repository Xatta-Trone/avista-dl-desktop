"""Central classification model factory."""

from __future__ import annotations

from typing import Any

from app.core.model_registry import get_model_spec
from app.models.sklearn_models import create_sklearn_model


def create_model(
    model_name: str,
    task_type: str = "classification",
    params: dict[str, Any] | None = None,
    class_weights: dict[Any, float] | str | None = None,
    device: str | None = None,
) -> Any:
    """Instantiate a registered classification model."""

    normalized_task = task_type.strip().lower()
    if normalized_task != "classification":
        raise ValueError("The model factory currently supports classification models only.")

    spec = get_model_spec(model_name)
    model_params = dict(spec.default_params)
    if params:
        model_params.update(params)

    if spec.model_family in {"sklearn", "xgboost"}:
        return create_sklearn_model(
            spec.name,
            normalized_task,
            params=model_params,
            class_weights=class_weights,
        )
    if spec.model_family == "torch":
        return _create_torch_model(spec.name, model_params, device)
    if spec.model_family == "tabpfn":
        return _create_tabpfn_model(model_params, device)
    raise ValueError(f"Model family '{spec.model_family}' is not supported.")


def _create_torch_model(
    model_name: str,
    params: dict[str, Any],
    device: str | None,
) -> Any:
    try:
        from app.models.torch_tabular_models import (
            AutoIntClassifier,
            FTTransformerClassifier,
            MambaAttentionClassifier,
            TabResNet,
        )
    except ImportError as exc:
        raise ImportError(
            "Optional package 'torch' is required for deep tabular models. "
            "Install PyTorch to use this model."
        ) from exc

    model_classes = {
        "mamba_attention": MambaAttentionClassifier,
        "ft_transformer": FTTransformerClassifier,
        "autoint": AutoIntClassifier,
        "tab_resnet": TabResNet,
    }
    model = model_classes[model_name](**params)
    return model.to(device) if device is not None else model


def _create_tabpfn_model(params: dict[str, Any], device: str | None) -> Any:
    try:
        from tabpfn import TabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "Optional package 'tabpfn' is required for TabPFN. "
            "Install tabpfn to use this model."
        ) from exc

    from app.utils.resources import get_app_resource_path

    checkpoint_path = get_app_resource_path(
        "app/assets/tabpfn-v2.5-classifier-v2.5_default.ckpt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Bundled TabPFN checkpoint not found in app/assets."
        )
    model_params = {
        "n_estimators": int(params.get("n_estimators", 8)),
        "model_path": str(checkpoint_path),
    }
    if device is not None:
        model_params["device"] = device
    return TabPFNClassifier(**model_params)
