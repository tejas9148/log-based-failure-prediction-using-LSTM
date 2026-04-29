"""Simple real-time log monitor that predicts anomalies from a growing log file."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

from project.prediction.predictor import predict_failure
from project.prediction.template_matcher import TemplateMatcher


BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.json"


def _load_sequence_length() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found. Run training first using python -m project.train")
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return int(payload["sequence_length"])


def monitor_log_file(log_file: Path, poll_seconds: float = 1.0, start_from_end: bool = True) -> None:
    """Watch appended log lines, map to EventIds, and run sequence-level predictions."""
    if not log_file.exists():
        raise FileNotFoundError(f"Log file does not exist: {log_file}")

    matcher = TemplateMatcher()
    sequence_length = _load_sequence_length()
    window: deque[str] = deque(maxlen=sequence_length)

    print(f"Monitoring started for: {log_file}")
    print(f"Sliding window length: {sequence_length}")

    with log_file.open("r", encoding="utf-8", errors="ignore") as handle:
        if start_from_end:
            handle.seek(0, 2)

        while True:
            position = handle.tell()
            line = handle.readline()

            if not line:
                handle.seek(position)
                time.sleep(max(0.1, poll_seconds))
                continue

            matched = matcher.match(line)
            if matched is None:
                continue

            window.append(matched.event_id)
            if len(window) < sequence_length:
                continue

            sequence = list(window)
            result = predict_failure(sequence, enable_self_learning=False)
            status = "FAILURE/ANOMALY" if result["predicted_failure"] else "NORMAL/SUCCESS"
            print(
                f"Sequence={sequence} | Prob={result['anomaly_probability']:.4f} "
                f"| Threshold={result['decision_threshold']:.4f} | Prediction={status}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time log monitor for LSTM anomaly prediction")
    parser.add_argument("log_file", help="Path to a continuously growing log file")
    parser.add_argument("--poll-seconds", type=float, default=1.0, help="Polling interval for new lines")
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Read existing file content first instead of starting at end",
    )
    args = parser.parse_args()

    monitor_log_file(
        log_file=Path(args.log_file),
        poll_seconds=args.poll_seconds,
        start_from_end=not args.from_start,
    )


if __name__ == "__main__":
    main()
