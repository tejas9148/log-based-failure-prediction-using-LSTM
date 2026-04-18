"""Generate plain-language result interpretation for report and viva."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def build_results_analysis(
    cm: np.ndarray,
    precision: float,
    recall: float,
    f1_score: float,
    threshold: float,
    roc_auc: float,
) -> str:
    """Build report-friendly interpretation text."""
    tn, fp, fn, tp = cm.ravel().tolist()

    specificity = _safe_div(tn, tn + fp)
    false_alarm_rate = _safe_div(fp, tn + fp)
    miss_rate = _safe_div(fn, fn + tp)

    lines = [
        "# Results Analysis",
        "",
        "## Confusion Matrix Explanation",
        f"- True Negatives (TN): {tn} normal traces correctly predicted as normal.",
        f"- False Positives (FP): {fp} normal traces incorrectly flagged as anomalies.",
        f"- False Negatives (FN): {fn} anomaly traces missed by the model.",
        f"- True Positives (TP): {tp} anomaly traces correctly detected.",
        "",
        "## Metric Interpretation",
        f"- Precision: {precision:.4f}. This shows how trustworthy anomaly alerts are.",
        f"- Recall: {recall:.4f}. This shows how many true anomalies are captured.",
        f"- F1 Score: {f1_score:.4f}. This balances precision and recall.",
        f"- ROC-AUC: {roc_auc:.4f}. This reflects overall separability across thresholds.",
        f"- Specificity: {specificity:.4f}. This indicates normal-event recognition quality.",
        f"- False Alarm Rate: {false_alarm_rate:.4f}. Lower values reduce unnecessary alerts.",
        f"- Miss Rate: {miss_rate:.4f}. Lower values mean fewer missed anomalies.",
        "",
        "## Why Threshold Tuning Helps",
        (
            "The default threshold 0.5 is not always optimal for imbalanced anomaly datasets. "
            f"Using a tuned threshold ({threshold:.4f}) allows the classifier to prioritize "
            "the desired balance between precision and recall."
        ),
        (
            "In this project, threshold optimization supports better anomaly detection behavior "
            "for real-world logs where failure patterns are less frequent than normal patterns."
        ),
    ]
    return "\n".join(lines)


def save_results_analysis(content: str, output_path: Path) -> None:
    """Write analysis markdown to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
