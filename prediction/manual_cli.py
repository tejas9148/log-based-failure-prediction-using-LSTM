"""Manual terminal interface for anomaly prediction demo."""

from __future__ import annotations

from project.prediction.predictor import predict_failure


def run_manual_cli() -> None:
    print("Manual Log Failure Prediction")
    print("Enter sequence as space-separated EventIds. Example: E1 E2 E3 E4 E5")

    raw = input("Sequence: ").strip()
    if not raw:
        print("No input provided.")
        return

    sequence = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]

    try:
        result = predict_failure(sequence)
    except Exception as exc:
        print(f"Prediction failed: {exc}")
        return

    print("\nPrediction Result")
    print(f"Input Sequence: {result['input_sequence']}")
    print(f"Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"Decision Threshold: {result['decision_threshold']:.4f}")
    print(f"Predicted Class: {'Failure/Anomaly' if result['predicted_failure'] else 'Normal/Success'}")
    if result["unknown_event_ids"]:
        print(f"Unknown EventIds: {result['unknown_event_ids']}")


if __name__ == "__main__":
    run_manual_cli()
