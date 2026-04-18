"""CLI prediction entry point for submission demo."""

from __future__ import annotations

import argparse

from project.prediction.predictor import predict_failure


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict log anomaly from a sequence of EventIds")
    parser.add_argument("sequence", nargs="*", help="Event sequence, e.g. E1 E2 E3 E4 E5")
    args = parser.parse_args()

    if args.sequence:
        sequence = [token.strip() for token in args.sequence if token.strip()]
    else:
        raw = input("Enter sequence (space-separated): ").strip()
        sequence = [token.strip() for token in raw.replace(",", " ").split() if token.strip()]

    result = predict_failure(sequence)

    print("Prediction Result")
    print(f"Input Sequence: {result['input_sequence']}")
    print(f"Anomaly Probability: {result['anomaly_probability']:.4f}")
    print(f"Decision Threshold: {result['decision_threshold']:.4f}")
    print(f"Predicted Class: {'Failure/Anomaly' if result['predicted_failure'] else 'Normal/Success'}")
    if result["unknown_event_ids"]:
        print(f"Unknown EventIds: {result['unknown_event_ids']}")


if __name__ == "__main__":
    main()
