"""Self-learning module for incremental retraining from online log sequences."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from project.model.lstm_architecture import build_lstm_model
from project.preprocessing.data_preprocessing import encode_event_ids, generate_sequences, load_dataset


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent
CONFIG_PATH = BASE_DIR / "config.json"
DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "dataset" / "Event_traces.csv"
DEFAULT_ONLINE_DATASET_PATH = BASE_DIR / "online_dataset.csv"
DEFAULT_STATE_PATH = BASE_DIR / "saved_models" / "self_learning_state.json"
ONLINE_COLUMNS = ["Event1", "Event2", "Event3", "Event4", "Event5", "Label"]

_FILE_LOCK = threading.Lock()


def _resolve_path(path_value: str, fallback: Path) -> Path:
    raw = Path(path_value)
    if raw.is_absolute():
        return raw

    candidates = [
        BASE_DIR / raw,
        WORKSPACE_ROOT / raw,
        fallback,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found. Run training first using python -m project.train")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _get_self_learning_config(config: Dict[str, object]) -> Dict[str, object]:
    payload = config.get("self_learning", {})
    if not isinstance(payload, dict):
        payload = {}
    return payload


def _get_online_dataset_path(config: Dict[str, object]) -> Path:
    sl = _get_self_learning_config(config)
    configured = str(sl.get("online_dataset_path", DEFAULT_ONLINE_DATASET_PATH))
    return _resolve_path(configured, fallback=DEFAULT_ONLINE_DATASET_PATH)


def _get_state_path(config: Dict[str, object]) -> Path:
    sl = _get_self_learning_config(config)
    configured = str(sl.get("state_path", DEFAULT_STATE_PATH))
    return _resolve_path(configured, fallback=DEFAULT_STATE_PATH)


def ensure_online_dataset(config: Dict[str, object] | None = None) -> Path:
    """Create online dataset file with header if missing."""
    active_config = config or _load_config()
    csv_path = _get_online_dataset_path(active_config)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        csv_path.write_text(",".join(ONLINE_COLUMNS) + "\n", encoding="utf-8")
    return csv_path


def append_sequence_record(event_sequence: List[str], label: int) -> None:
    """Append latest predicted sequence and pseudo-label to online dataset."""
    config = _load_config()
    sequence_length = int(config.get("sequence_length", 5))
    if len(event_sequence) != sequence_length:
        raise ValueError(f"Expected sequence length {sequence_length}, got {len(event_sequence)}")

    row = [str(token) for token in event_sequence] + [int(label)]

    with _FILE_LOCK:
        csv_path = ensure_online_dataset(config)
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(row)


def _read_online_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=ONLINE_COLUMNS)

    df = pd.read_csv(csv_path)
    for col in ONLINE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"online_dataset.csv missing column: {col}")

    # Keep rows that have all events and a binary label.
    frame = df[ONLINE_COLUMNS].dropna().copy()
    frame["Label"] = frame["Label"].astype(int)
    frame = frame[frame["Label"].isin([0, 1])]
    for col in ONLINE_COLUMNS[:-1]:
        frame[col] = frame[col].astype(str).str.strip()
    frame = frame[(frame[ONLINE_COLUMNS[:-1]] != "").all(axis=1)]
    return frame.reset_index(drop=True)


def _online_to_training_frame(online_df: pd.DataFrame) -> pd.DataFrame:
    event_cols = ONLINE_COLUMNS[:-1]
    features = online_df[event_cols].astype(str).agg(" ".join, axis=1)
    labels = np.where(online_df["Label"].astype(int).to_numpy() == 1, "anomaly", "success")
    return pd.DataFrame({"Label": labels, "Features": features})


def _balance_training_set(x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    y_int = y_train.astype(int)
    counts = np.bincount(y_int, minlength=2)
    if counts[0] == 0 or counts[1] == 0:
        return x_train, y_train

    majority = int(np.argmax(counts))
    minority = 1 - majority
    if counts[minority] >= counts[majority]:
        return x_train, y_train

    rng = np.random.default_rng(seed=random_state)
    minority_idx = np.where(y_int == minority)[0]
    extra_idx = rng.choice(minority_idx, size=int(counts[majority] - counts[minority]), replace=True)

    x_balanced = np.concatenate([x_train, x_train[extra_idx]], axis=0)
    y_balanced = np.concatenate([y_train, y_train[extra_idx]], axis=0)
    shuffle_idx = rng.permutation(len(y_balanced))
    return x_balanced[shuffle_idx], y_balanced[shuffle_idx]


def _find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true.astype(int), y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1_values = (2 * precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
    return float(thresholds[int(np.argmax(f1_values))])


def _load_state(state_path: Path) -> Dict[str, float]:
    if not state_path.exists():
        return {"last_retrained_total": 0.0, "last_retrain_timestamp": 0.0}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return {
            "last_retrained_total": float(payload.get("last_retrained_total", 0.0)),
            "last_retrain_timestamp": float(payload.get("last_retrain_timestamp", 0.0)),
        }
    except json.JSONDecodeError:
        return {"last_retrained_total": 0.0, "last_retrain_timestamp": 0.0}


def _save_state(state_path: Path, total_rows: int) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_retrained_total": int(total_rows),
        "last_retrain_timestamp": float(time.time()),
    }
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def retrain_model() -> Dict[str, object]:
    """Retrain LSTM model on original + online datasets and persist updated artifacts."""
    config = _load_config()
    sl = _get_self_learning_config(config)

    online_path = ensure_online_dataset(config)
    state_path = _get_state_path(config)

    online_df = _read_online_dataframe(online_path)
    if online_df.empty:
        return {"updated": False, "reason": "No new online sequences available."}

    base_df = load_dataset(DEFAULT_DATASET_PATH)
    merged_df = pd.concat([base_df[["Label", "Features"]], _online_to_training_frame(online_df)], ignore_index=True)

    sequence_length = int(config.get("sequence_length", 5))
    step_size = int(config.get("step_size", 1))
    random_state = int(sl.get("random_state", 42))
    epochs = int(sl.get("epochs", 3))
    batch_size = int(sl.get("batch_size", 64))

    encoded_traces, labels, encoder = encode_event_ids(merged_df)
    x, y, meta = generate_sequences(
        encoded_traces=encoded_traces,
        trace_labels=labels,
        sequence_length=sequence_length,
        step_size=step_size,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    x_train_main, x_val, y_train_main, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )

    x_train_balanced, y_train_balanced = _balance_training_set(
        x_train=x_train_main,
        y_train=y_train_main,
        random_state=random_state,
    )

    model = build_lstm_model(vocab_size=len(encoder.classes_), sequence_length=sequence_length)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    model.fit(
        x_train_balanced,
        y_train_balanced,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    val_prob = model.predict(x_val, verbose=0).flatten()
    decision_threshold = _find_best_threshold(y_true=y_val, y_prob=val_prob)

    model_path = _resolve_path(str(config["model_path"]), fallback=BASE_DIR / "saved_models" / "lstm_failure_predictor.keras")
    encoder_path = _resolve_path(str(config["encoder_path"]), fallback=BASE_DIR / "saved_models" / "label_encoder.joblib")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    encoder_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    joblib.dump(encoder, encoder_path)

    config["decision_threshold"] = float(decision_threshold)
    config["model_path"] = str(model_path)
    config["encoder_path"] = str(encoder_path)
    if "self_learning" not in config or not isinstance(config["self_learning"], dict):
        config["self_learning"] = {}
    config["self_learning"].update(
        {
            "enabled": bool(config["self_learning"].get("enabled", True)),
            "online_dataset_path": str(online_path),
            "state_path": str(state_path),
            "retrain_threshold": int(config["self_learning"].get("retrain_threshold", 500)),
            "retrain_interval_seconds": int(config["self_learning"].get("retrain_interval_seconds", 600)),
            "epochs": epochs,
            "batch_size": batch_size,
            "last_retrained_window_count": int(meta["samples"]),
        }
    )
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    _save_state(state_path=state_path, total_rows=len(online_df))

    return {
        "updated": True,
        "online_rows": int(len(online_df)),
        "window_count": int(meta["samples"]),
        "decision_threshold": float(decision_threshold),
        "model_path": str(model_path),
        "encoder_path": str(encoder_path),
    }


def maybe_trigger_retraining() -> bool:
    """Trigger retraining when enough new rows arrive or interval is reached."""
    config = _load_config()
    sl = _get_self_learning_config(config)
    if not bool(sl.get("enabled", True)):
        return False

    retrain_threshold = int(sl.get("retrain_threshold", 500))
    retrain_interval_seconds = int(sl.get("retrain_interval_seconds", 600))

    with _FILE_LOCK:
        online_path = ensure_online_dataset(config)
        online_df = _read_online_dataframe(online_path)

        state_path = _get_state_path(config)
        state = _load_state(state_path)

        total_rows = len(online_df)
        previous_total = int(state.get("last_retrained_total", 0))
        new_sequences = max(0, total_rows - previous_total)

        last_ts = float(state.get("last_retrain_timestamp", 0.0))
        interval_reached = last_ts > 0 and (time.time() - last_ts) >= retrain_interval_seconds
        threshold_reached = new_sequences >= retrain_threshold

        if not threshold_reached and not (new_sequences > 0 and interval_reached):
            return False

    print(f"New sequences collected: {new_sequences}")
    print("Triggering model retraining...")

    result = retrain_model()
    if not bool(result.get("updated", False)):
        print(f"Retraining skipped: {result.get('reason', 'No update.')}")
        return False

    print("Model retrained successfully")
    print("Updated model saved")
    return True
