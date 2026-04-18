"""Evaluation plotting helpers for project submission visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """Save confusion matrix heatmap."""
    _ensure_dir(output_path)

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(cm, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Normal (0)", "Anomaly (1)"])
    axis.set_yticklabels(["Normal (0)", "Anomaly (1)"])

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            axis.text(col, row, str(int(cm[row, col])), ha="center", va="center", color="black")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path) -> float:
    """Save ROC curve and return AUC score."""
    _ensure_dir(output_path)

    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)
    roc_auc = float(auc(fpr, tpr))

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", color="#1f77b4")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set_title("Receiver Operating Characteristic")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend(loc="lower right")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return roc_auc


def plot_accuracy_curves(history, output_path: Path) -> None:
    """Save training vs validation accuracy plot."""
    _ensure_dir(output_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history.history.get("accuracy", []), label="Train Accuracy", color="#2ca02c")
    axis.plot(history.history.get("val_accuracy", []), label="Val Accuracy", color="#ff7f0e")
    axis.set_title("Training vs Validation Accuracy")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_loss_curves(history, output_path: Path) -> None:
    """Save training vs validation loss plot."""
    _ensure_dir(output_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history.history.get("loss", []), label="Train Loss", color="#d62728")
    axis.plot(history.history.get("val_loss", []), label="Val Loss", color="#9467bd")
    axis.set_title("Training vs Validation Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
