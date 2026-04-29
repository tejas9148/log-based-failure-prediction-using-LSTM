"""Training orchestration for the submission-ready project pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from project.evaluation.analysis import build_results_analysis, save_results_analysis
from project.evaluation.visualizations import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_event_transition_comparison,
    plot_loss_curves,
    plot_roc_curve,
)
from project.model.lstm_architecture import build_lstm_model
from project.preprocessing.data_preprocessing import encode_event_ids, generate_sequences, load_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
DEFAULT_DATASET_PATH = WORKSPACE_ROOT / "dataset" / "Event_traces.csv"
DEFAULT_NPZ_PATH = WORKSPACE_ROOT / "dataset" / "HDFS.npz"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
PLOTS_DIR = BASE_DIR / "plots"
CONFIG_PATH = BASE_DIR / "config.json"
MODEL_PATH = SAVED_MODELS_DIR / "lstm_failure_predictor.keras"
ENCODER_PATH = SAVED_MODELS_DIR / "label_encoder.joblib"
ANALYSIS_PATH = BASE_DIR / "evaluation" / "results_analysis.md"


def _sample_balanced_subset(
    df,
    max_records: int,
    random_state: int,
    normal_to_anomaly_ratio: float = 1.1,
):
    """Create a class-balanced subset with a slight normal-class majority when requested."""
    if max_records <= 0:
        raise ValueError("max_records must be a positive number.")
    if normal_to_anomaly_ratio < 1.0:
        raise ValueError("normal_to_anomaly_ratio must be at least 1.0.")

    labels = df["Label"].astype(str).str.strip().str.lower()
    normal_idx = labels[labels.isin(["success", "normal"])].index
    anomaly_idx = labels[labels.isin(["fail", "anomaly"])].index

    anomaly_count = int(round(max_records / (normal_to_anomaly_ratio + 1.0)))
    normal_count = int(max_records - anomaly_count)

    if anomaly_count <= 0 or normal_count <= 0:
        raise ValueError("Requested sampling ratio produced an empty class.")
    if len(normal_idx) < normal_count or len(anomaly_idx) < anomaly_count:
        raise ValueError(
            f"Not enough rows for balanced subset={max_records} with ratio={normal_to_anomaly_ratio}. "
            f"normal/success={len(normal_idx)}, anomaly/fail={len(anomaly_idx)}"
        )

    rng = np.random.default_rng(seed=random_state)
    sampled_normal = rng.choice(normal_idx.to_numpy(), size=normal_count, replace=False)
    sampled_anomaly = rng.choice(anomaly_idx.to_numpy(), size=anomaly_count, replace=False)
    selected_idx = np.concatenate([sampled_normal, sampled_anomaly])
    rng.shuffle(selected_idx)
    return df.loc[selected_idx].reset_index(drop=True)


def _sample_exact_class_subset(
    df,
    normal_count: int,
    anomaly_count: int,
    random_state: int,
):
    """Create a subset with exact normal and anomaly counts."""
    if normal_count <= 0 or anomaly_count <= 0:
        raise ValueError("normal_count and anomaly_count must be positive numbers.")

    labels = df["Label"].astype(str).str.strip().str.lower()
    normal_idx = labels[labels.isin(["success", "normal"])].index
    anomaly_idx = labels[labels.isin(["fail", "anomaly"])].index

    if len(normal_idx) < normal_count or len(anomaly_idx) < anomaly_count:
        raise ValueError(
            f"Not enough rows for exact subset normal={normal_count}, anomaly={anomaly_count}. "
            f"normal/success={len(normal_idx)}, anomaly/fail={len(anomaly_idx)}"
        )

    rng = np.random.default_rng(seed=random_state)
    sampled_normal = rng.choice(normal_idx.to_numpy(), size=normal_count, replace=False)
    sampled_anomaly = rng.choice(anomaly_idx.to_numpy(), size=anomaly_count, replace=False)
    selected_idx = np.concatenate([sampled_normal, sampled_anomaly])
    rng.shuffle(selected_idx)
    return df.loc[selected_idx].reset_index(drop=True)


def _balance_training_set(x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    """Oversample minority class in training set to reduce imbalance."""
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


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_precision: float | None,
) -> float:
    """Choose threshold by precision target, else by validation accuracy."""
    precisions, recalls, thresholds = precision_recall_curve(y_true.astype(int), y_prob)
    if len(thresholds) == 0:
        return 0.5

    if target_precision is not None:
        indices = np.where(precisions[:-1] >= target_precision)[0]
        if len(indices) > 0:
            best_idx = int(indices[np.argmax(recalls[:-1][indices])])
            return float(thresholds[best_idx])

    best_accuracy = -1.0
    best_threshold = 0.5
    y_true_int = y_true.astype(int)
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        accuracy = float(np.mean(y_pred == y_true_int))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold


def _save_config(sequence_length: int, step_size: int, threshold: float) -> None:
    """Persist config used by prediction system and demo."""
    existing: Dict[str, object] = {}
    if CONFIG_PATH.exists():
        try:
            existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}

    self_learning = existing.get("self_learning", {})
    if not isinstance(self_learning, dict):
        self_learning = {}

    payload = {
        "dataset_type": "HDFS_Event_traces",
        "sequence_length": sequence_length,
        "step_size": step_size,
        "decision_threshold": threshold,
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "self_learning": {
            "enabled": bool(self_learning.get("enabled", True)),
            "online_dataset_path": str(self_learning.get("online_dataset_path", BASE_DIR / "online_dataset.csv")),
            "state_path": str(self_learning.get("state_path", BASE_DIR / "saved_models" / "self_learning_state.json")),
            "retrain_threshold": int(self_learning.get("retrain_threshold", 500)),
            "retrain_interval_seconds": int(self_learning.get("retrain_interval_seconds", 600)),
            "epochs": int(self_learning.get("epochs", 3)),
            "batch_size": int(self_learning.get("batch_size", 64)),
        },
    }
    CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_pipeline(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    sequence_length: int = 5,
    step_size: int = 1,
    max_records: int | None = 10000,
    balanced_subset: bool = True,
    normal_to_anomaly_ratio: float = 1.1,
    normal_count: int | None = None,
    anomaly_count: int | None = None,
    oversample_training: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    epochs: int = 6,
    batch_size: int = 64,
    tune_threshold: bool = True,
    target_precision: float | None = None,
    use_class_weight: bool = True,
) -> Dict[str, object]:
    """Execute preprocessing, model training, evaluation, and artifact generation."""
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    original_count = len(df)

    if balanced_subset and normal_count is not None and anomaly_count is not None:
        df = _sample_exact_class_subset(
            df,
            normal_count=normal_count,
            anomaly_count=anomaly_count,
            random_state=random_state,
        )
    elif balanced_subset and max_records is not None:
        df = _sample_balanced_subset(
            df,
            max_records=max_records,
            random_state=random_state,
            normal_to_anomaly_ratio=normal_to_anomaly_ratio,
        )
    elif max_records is not None:
        df = df.sample(n=min(max_records, len(df)), random_state=random_state).reset_index(drop=True)

    encoded_traces, labels, label_encoder = encode_event_ids(df)
    x, y, meta = generate_sequences(
        encoded_traces=encoded_traces,
        trace_labels=labels,
        sequence_length=sequence_length,
        step_size=step_size,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
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

    if oversample_training:
        x_train_balanced, y_train_balanced = _balance_training_set(
            x_train=x_train_main,
            y_train=y_train_main,
            random_state=random_state,
        )
    else:
        x_train_balanced, y_train_balanced = x_train_main, y_train_main

    model = build_lstm_model(vocab_size=len(label_encoder.classes_), sequence_length=sequence_length)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]

    class_weight = None
    if use_class_weight:
        classes = np.unique(y_train_balanced.astype(int))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_balanced.astype(int))
        class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    history = model.fit(
        x_train_balanced,
        y_train_balanced,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    test_loss, raw_test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    decision_threshold = 0.5
    if tune_threshold:
        val_prob = model.predict(x_val, verbose=0).flatten()
        decision_threshold = _find_best_threshold(
            y_true=y_val,
            y_prob=val_prob,
            target_precision=target_precision,
        )

    y_prob = model.predict(x_test, verbose=0).flatten()
    y_pred = (y_prob >= decision_threshold).astype(int)
    threshold_test_accuracy = float(np.mean(y_pred == y_test.astype(int)))

    cm = confusion_matrix(y_test.astype(int), y_pred, labels=[0, 1])
    precision = float(precision_score(y_test.astype(int), y_pred, zero_division=0))
    recall = float(recall_score(y_test.astype(int), y_pred, zero_division=0))
    f1 = float(f1_score(y_test.astype(int), y_pred, zero_division=0))

    model.save(MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    _save_config(sequence_length=sequence_length, step_size=step_size, threshold=float(decision_threshold))

    confusion_path = PLOTS_DIR / "confusion_matrix.png"
    roc_path = PLOTS_DIR / "roc_curve.png"
    acc_path = PLOTS_DIR / "training_validation_accuracy.png"
    loss_path = PLOTS_DIR / "training_validation_loss.png"
    transitions_path = PLOTS_DIR / "event_transition_comparison.png"

    plot_confusion_matrix(cm=cm, output_path=confusion_path)
    roc_auc = plot_roc_curve(y_true=y_test, y_prob=y_prob, output_path=roc_path)
    plot_accuracy_curves(history=history, output_path=acc_path)
    plot_loss_curves(history=history, output_path=loss_path)
    transition_meta = plot_event_transition_comparison(
        npz_path=DEFAULT_NPZ_PATH,
        output_path=transitions_path,
    )

    analysis_text = build_results_analysis(
        cm=cm,
        precision=precision,
        recall=recall,
        f1_score=f1,
        threshold=float(decision_threshold),
        roc_auc=roc_auc,
    )
    save_results_analysis(content=analysis_text, output_path=ANALYSIS_PATH)

    return {
        "dataset_path": str(dataset_path),
        "original_trace_count": int(original_count),
        "selected_trace_count": int(len(df)),
        "window_count": int(meta["samples"]),
        "test_loss": float(test_loss),
        "raw_test_accuracy": float(raw_test_accuracy),
        "test_accuracy": float(threshold_test_accuracy),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "decision_threshold": float(decision_threshold),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm.tolist(),
        "model_path": str(MODEL_PATH),
        "encoder_path": str(ENCODER_PATH),
        "config_path": str(CONFIG_PATH),
        "confusion_matrix_plot": str(confusion_path),
        "roc_plot": str(roc_path),
        "accuracy_plot": str(acc_path),
        "loss_plot": str(loss_path),
        "event_transition_plot": str(transitions_path),
        "event_transition_meta": transition_meta,
        "analysis_path": str(ANALYSIS_PATH),
    }
