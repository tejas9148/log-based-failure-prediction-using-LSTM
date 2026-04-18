"""Train and evaluate the submission-ready LSTM pipeline."""

from __future__ import annotations

from project.training.trainer import train_pipeline


def main() -> None:
    results = train_pipeline()

    print("Training complete")
    print(f"Dataset: {results['dataset_path']}")
    print(f"Original Trace Count: {results['original_trace_count']}")
    print(f"Selected Trace Count: {results['selected_trace_count']}")
    print(f"Generated Windows: {results['window_count']}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Decision Threshold: {results['decision_threshold']:.4f}")
    print(f"Confusion Matrix: {results['confusion_matrix']}")
    print(f"Saved Model: {results['model_path']}")
    print(f"Saved Encoder: {results['encoder_path']}")
    print(f"Saved Config: {results['config_path']}")
    print(f"Confusion Matrix Plot: {results['confusion_matrix_plot']}")
    print(f"ROC Curve Plot: {results['roc_plot']}")
    print(f"Train/Val Accuracy Plot: {results['accuracy_plot']}")
    print(f"Train/Val Loss Plot: {results['loss_plot']}")
    print(f"Results Analysis: {results['analysis_path']}")


if __name__ == "__main__":
    main()
