"""Prediction helpers for CLI and desktop log-input interfaces."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from project.model.lstm_architecture import AttentionPooling
from project.self_learning import append_sequence_record, maybe_trigger_retraining


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.json"
DATASET_NPZ_PATH = BASE_DIR.parent / "dataset" / "HDFS.npz"


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


@lru_cache(maxsize=1)
def _load_event_failure_stats() -> Dict[str, Dict[str, float]]:
    """Build per-event normal/failure frequencies from the HDFS NPZ traces."""
    if not DATASET_NPZ_PATH.exists():
        return {}

    payload = np.load(DATASET_NPZ_PATH, allow_pickle=True)
    traces = payload["x_data"]
    labels = payload["y_data"]

    stats: Dict[str, Dict[str, float]] = {}
    for trace, label in zip(traces, labels):
        target_key = "failure_count" if int(label) == 1 else "normal_count"
        for event_id in trace:
            key = str(event_id)
            if key not in stats:
                stats[key] = {
                    "failure_count": 0.0,
                    "normal_count": 0.0,
                    "failure_ratio": 0.0,
                }
            stats[key][target_key] += 1.0

    for values in stats.values():
        values["failure_ratio"] = values["failure_count"] / (values["normal_count"] + 1.0)

    return stats


def _alert_level(probability: float) -> str:
    """Map anomaly probability to severity level for monitoring-style output."""
    if probability < 0.40:
        return "NORMAL"
    if probability <= 0.70:
        return "WARNING"
    return "CRITICAL FAILURE"


def _infer_root_cause_event(event_sequence: List[str], predicted_failure: bool) -> tuple[str | None, str]:
    """Infer most suspicious event in a sequence using failure-frequency heuristic."""
    if not predicted_failure:
        return None, "No root cause event for normal predictions."

    stats = _load_event_failure_stats()
    if not stats:
        return None, "Root cause statistics unavailable because dataset traces could not be loaded."

    counts_in_sequence: Dict[str, int] = {}
    for event_id in event_sequence:
        counts_in_sequence[event_id] = counts_in_sequence.get(event_id, 0) + 1

    best_event = None
    best_score = -1.0
    best_failure_count = 0.0
    best_ratio = 0.0

    for event_id, count in counts_in_sequence.items():
        event_stats = stats.get(event_id)
        if event_stats is None:
            continue

        ratio = float(event_stats["failure_ratio"])
        failure_count = float(event_stats["failure_count"])
        score = ratio * float(count)

        if score > best_score or (score == best_score and failure_count > best_failure_count):
            best_event = event_id
            best_score = score
            best_failure_count = failure_count
            best_ratio = ratio

    if best_event is None:
        return None, "No known event in this sequence has historical failure statistics."

    explanation = (
        f"Event {best_event} shows elevated failure association "
        f"(failure_ratio={best_ratio:.2f}) in historical traces."
    )
    return best_event, explanation


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


def predict_failure(event_sequence: List[str], enable_self_learning: bool = False) -> Dict[str, object]:
    """Predict anomaly probability and class for a given EventId sequence."""
    config = _load_config()
    threshold = float(config.get("decision_threshold", 0.5))
    sequence_length = int(config["sequence_length"])

    model_path = _resolve_path(str(config["model_path"]))
    encoder_path = _resolve_path(str(config["encoder_path"]))

    model = load_model(model_path, custom_objects={"AttentionPooling": AttentionPooling})
    label_encoder = joblib.load(encoder_path)

    x_input, unknown = _encode_sequence(
        event_sequence=event_sequence,
        label_encoder=label_encoder,
        sequence_length=sequence_length,
    )

    probability = float(model.predict(x_input, verbose=0)[0][0])
    predicted_failure = bool(probability >= threshold)
    predicted_label = int(probability >= threshold)
    alert_level = _alert_level(probability)
    root_cause_event, root_cause_explanation = _infer_root_cause_event(
        event_sequence=event_sequence,
        predicted_failure=predicted_failure,
    )

    if enable_self_learning:
        try:
            append_sequence_record(event_sequence=event_sequence, label=predicted_label)
            maybe_trigger_retraining()
        except Exception as exc:
            print(f"Self-learning skipped: {exc}")

    return {
        "input_sequence": event_sequence,
        "anomaly_probability": probability,
        "decision_threshold": threshold,
        "predicted_failure": predicted_failure,
        "alert_level": alert_level,
        "root_cause_event": root_cause_event,
        "root_cause_explanation": root_cause_explanation,
        "unknown_event_ids": unknown,
        "unknown_event_warning": (
            "Some event IDs were unseen during training and were encoded as 0. "
            "Prediction still follows the model probability threshold."
            if unknown
            else None
        ),
    }
