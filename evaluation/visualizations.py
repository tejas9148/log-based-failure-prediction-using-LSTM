"""Evaluation plotting helpers for project submission visuals."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
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


def _extract_transition_counters(traces, labels) -> tuple[Counter, Counter]:
    """Count adjacent event transitions separately for normal and anomaly traces."""
    normal_counter: Counter = Counter()
    anomaly_counter: Counter = Counter()

    for trace, label in zip(traces, labels):
        tokens = [str(token) for token in trace]
        if len(tokens) < 2:
            continue

        transitions = zip(tokens[:-1], tokens[1:])
        if int(label) == 1:
            anomaly_counter.update(transitions)
        else:
            normal_counter.update(transitions)

    return normal_counter, anomaly_counter


def _draw_transition_graph(axis, transitions: list[tuple[str, str, float]], title: str, edge_color: str) -> None:
    """Render a weighted directed transition graph on a matplotlib axis."""
    graph = nx.DiGraph()
    for src, dst, weight in transitions:
        graph.add_edge(src, dst, weight=weight)

    if graph.number_of_nodes() == 0:
        axis.set_title(title)
        axis.axis("off")
        axis.text(0.5, 0.5, "No transitions available", ha="center", va="center", fontsize=11)
        return

    layout = nx.spring_layout(graph, seed=42, k=0.9)
    weights = [float(graph[u][v]["weight"]) for u, v in graph.edges()]
    max_weight = max(weights) if weights else 1.0
    widths = [1.2 + (2.8 * w / max_weight) for w in weights]

    nx.draw_networkx_nodes(graph, pos=layout, node_color="#f5f5f5", edgecolors="#222", node_size=1000, ax=axis)
    nx.draw_networkx_labels(graph, pos=layout, font_size=9, ax=axis)
    nx.draw_networkx_edges(
        graph,
        pos=layout,
        edge_color=edge_color,
        width=widths,
        arrows=True,
        arrowsize=14,
        alpha=0.85,
        connectionstyle="arc3,rad=0.12",
        ax=axis,
    )

    axis.set_title(title)
    axis.axis("off")


def plot_event_transition_comparison(
    npz_path: Path,
    output_path: Path,
    top_k: int = 35,
    min_count: int = 3,
) -> dict[str, int]:
    """Save side-by-side normal vs anomaly event transition graphs from HDFS.npz."""
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ dataset not found: {npz_path}")

    _ensure_dir(output_path)

    payload = np.load(npz_path, allow_pickle=True)
    traces = payload["x_data"]
    labels = payload["y_data"]

    normal_counter, anomaly_counter = _extract_transition_counters(traces=traces, labels=labels)

    normal_top = [
        (src, dst, float(count))
        for (src, dst), count in normal_counter.most_common(top_k)
        if int(count) >= min_count
    ]
    anomaly_top = [
        (src, dst, float(count))
        for (src, dst), count in anomaly_counter.most_common(top_k)
        if int(count) >= min_count
    ]

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    _draw_transition_graph(
        axis=axes[0],
        transitions=normal_top,
        title="Normal Event Transition Pattern",
        edge_color="#2f855a",
    )
    _draw_transition_graph(
        axis=axes[1],
        transitions=anomaly_top,
        title="Anomaly Event Transition Pattern (Highlighted)",
        edge_color="#c53030",
    )
    figure.suptitle("Event Transition Comparison: Normal vs Anomaly", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    return {
        "normal_unique_transitions": int(len(normal_counter)),
        "anomaly_unique_transitions": int(len(anomaly_counter)),
        "normal_drawn_transitions": int(len(normal_top)),
        "anomaly_drawn_transitions": int(len(anomaly_top)),
    }
