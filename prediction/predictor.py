"""Prediction helpers for CLI and desktop log-input interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tensorflow.keras.models import load_model


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.json"


def _resolve_path(path_value: str) -> Path:
    """Resolve artifact paths from config across common relative layouts."""
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        BASE_DIR / raw_path,
        BASE_DIR.parent / raw_path,
        BASE_DIR / "saved_models" / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found. Run training first with project/train.py")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _encode_sequence(event_sequence: List[str], label_encoder, sequence_length: int):
    if len(event_sequence) != sequence_length:
        raise ValueError(f"Input sequence length must be exactly {sequence_length}")

    known = set(label_encoder.classes_)
    encoded = []
    unknown = []

    for event_id in event_sequence:
        if event_id in known:
            encoded.append(int(label_encoder.transform([event_id])[0]))
        else:
            encoded.append(0)
            unknown.append(event_id)

    return np.array(encoded, dtype=np.int32).reshape(1, sequence_length), unknown


def predict_failure(event_sequence: List[str]) -> Dict[str, object]:
    """Predict anomaly probability and class for a given EventId sequence."""
    config = _load_config()
    threshold = float(config.get("decision_threshold", 0.5))
    sequence_length = int(config["sequence_length"])

    model_path = _resolve_path(str(config["model_path"]))
    encoder_path = _resolve_path(str(config["encoder_path"]))

    model = load_model(model_path)
    label_encoder = joblib.load(encoder_path)

    x_input, unknown = _encode_sequence(
        event_sequence=event_sequence,
        label_encoder=label_encoder,
        sequence_length=sequence_length,
    )

    probability = float(model.predict(x_input, verbose=0)[0][0])
    predicted_failure = bool(probability >= threshold or len(unknown) > 0)

    return {
        "input_sequence": event_sequence,
        "anomaly_probability": probability,
        "decision_threshold": threshold,
        "predicted_failure": predicted_failure,
        "unknown_event_ids": unknown,
    }
