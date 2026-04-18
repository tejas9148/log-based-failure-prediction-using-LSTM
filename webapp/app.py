"""Flask web application for raw HDFS log input and LSTM anomaly prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.prediction.predictor import predict_failure
from project.prediction.template_matcher import MatchResult, TemplateMatcher, parse_logs


CONFIG_PATH = BASE_DIR / "config.json"

app = Flask(__name__, template_folder="templates", static_folder="static")
matcher = TemplateMatcher()


def _load_sequence_length() -> int:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("config.json not found. Run training first using python -m project.train")
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return int(data["sequence_length"])


@app.route("/", methods=["GET", "POST"])
def index():
    sequence_length = _load_sequence_length()
    raw_logs = ""
    mapping_rows = []
    prediction_data = None
    error_message = None
    success_message = None

    if request.method == "POST":
        raw_logs = request.form.get("raw_logs", "")
        lines = parse_logs(raw_logs)

        if not lines:
            error_message = "Please paste log lines before predicting."
        else:
            event_ids = []
            for index, line in enumerate(lines, start=1):
                matched = matcher.match(line)
                if matched is None:
                    event_id = "UNKNOWN_EVENT"
                    template = "No matching template"
                else:
                    event_id = matched.event_id
                    template = matched.template
                    event_ids.append(event_id)

                mapping_rows.append(
                    {
                        "line_no": index,
                        "raw_line": line,
                        "event_id": event_id,
                        "template": template,
                    }
                )

            if len(event_ids) < sequence_length:
                error_message = (
                    f"Need at least {sequence_length} matched logs to predict. "
                    f"Received {len(event_ids)} matched events."
                )
            else:
                sequence = event_ids[-sequence_length:]
                try:
                    prediction_data = predict_failure(sequence)
                    success_message = "Prediction completed successfully."
                except Exception as exc:
                    error_message = str(exc)

    return render_template(
        "index.html",
        raw_logs=raw_logs,
        mapping_rows=mapping_rows,
        prediction_data=prediction_data,
        sequence_length=sequence_length,
        error_message=error_message,
        success_message=success_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
