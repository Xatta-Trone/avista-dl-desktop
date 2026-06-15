"""Central target label encoding for classification artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


TARGET_ENCODER_FILE = "target_label_encoder.joblib"
TARGET_MAPPING_FILE = "target_label_mapping.json"


def load_or_fit_target_encoder(
    target: pd.Series,
    output_dir: str | Path,
    target_column: str,
) -> LabelEncoder:
    """Reuse a matching saved encoder or fit one on the confirmed target."""

    output_path = Path(output_dir)
    encoder_path = output_path / TARGET_ENCODER_FILE
    metadata_path = output_path / "split_indices.json"
    encoder = None
    if encoder_path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("target_column") == target_column:
                encoder = joblib.load(encoder_path)
                encoder.transform(_normalized_target_values(target))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            encoder = None

    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(_normalized_target_values(target))
    return encoder


def encode_target(encoder: LabelEncoder, target: pd.Series | np.ndarray) -> pd.Series:
    """Encode target values while preserving pandas alignment metadata."""

    values = encoder.transform(_normalized_target_values(target))
    if isinstance(target, pd.Series):
        return pd.Series(values, index=target.index, name=target.name, dtype=np.int64)
    return pd.Series(values, dtype=np.int64)


def decode_target(encoder: LabelEncoder, target: Any) -> np.ndarray:
    """Decode integer class ids to original labels."""

    return encoder.inverse_transform(np.asarray(target, dtype=int).reshape(-1))


def save_target_encoder(
    encoder: LabelEncoder,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist the encoder and a JSON-safe encoded-to-original mapping."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    encoder_path = output_path / TARGET_ENCODER_FILE
    mapping_path = output_path / TARGET_MAPPING_FILE
    joblib.dump(encoder, encoder_path)
    mapping = {
        str(index): _json_scalar(class_name)
        for index, class_name in enumerate(encoder.classes_)
    }
    mapping_path.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return encoder_path, mapping_path


def invalidate_target_artifacts(
    output_dir: str | Path,
    previous_target: str | None,
    current_target: str | None,
) -> None:
    """Discard saved split artifacts when the confirmed target changes."""

    if not previous_target or previous_target == current_target:
        return
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    for path in output_path.iterdir():
        if path.is_file():
            path.unlink()


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def _normalized_target_values(target: pd.Series | np.ndarray) -> np.ndarray:
    """Return string labels accepted by sklearn encoders for mixed-type targets."""

    return pd.Series(target).map(_display_label).to_numpy(dtype=str)


def _display_label(value: Any) -> str:
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return str(_json_scalar(value))
