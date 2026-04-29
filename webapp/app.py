"""Flask web application for raw HDFS log input and LSTM anomaly prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from collections import Counter
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from project.prediction.predictor import predict_failure
from project.prediction.template_matcher import (
	MatchResult,
	TemplateMatcher,
	find_event_line_numbers,
	parse_logs,
)
from project.evaluation.visualizations import plot_event_transition_comparison
from project.prediction.log_streamer import DEFAULT_LOG_FILE, load_log_lines, resolve_log_file


CONFIG_PATH = BASE_DIR / "config.json"
DATASET_NPZ_PATH = PROJECT_ROOT / "dataset" / "HDFS.npz"
TRANSITION_IMAGE_PATH = BASE_DIR / "webapp" / "static" / "event_transition_comparison.png"

app = Flask(__name__, template_folder="templates", static_folder="static")
matcher = TemplateMatcher()


# Simple in-memory stream state used by the realtime monitor endpoints
STREAM_STATE: dict = {
	"active": False,
	"position": 0,
	"window": [],
	"processed": 0,
	"anomalies": 0,
	"probabilities": [],
	"event_counts": Counter(),
	"records": [],
	"log_file": str(DEFAULT_LOG_FILE),
}


def _load_sequence_length() -> int:
	if not CONFIG_PATH.exists():
		raise FileNotFoundError("config.json not found. Run training first using python -m project.train")
	data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
	return int(data["sequence_length"])


def _ensure_transition_plot() -> str | None:
	"""Generate comparison graph once and return static URL path if available."""
	try:
		if not TRANSITION_IMAGE_PATH.exists():
			plot_event_transition_comparison(
				npz_path=DATASET_NPZ_PATH,
				output_path=TRANSITION_IMAGE_PATH,
			)
		return "event_transition_comparison.png"
	except Exception:
		return None


@app.route("/", methods=["GET"])
def home():
	return render_template("home.html")


@app.route("/manual", methods=["GET", "POST"])
def manual():
	sequence_length = _load_sequence_length()
	transition_plot = _ensure_transition_plot()
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
					root_event = prediction_data.get("root_cause_event")
					if root_event:
						prediction_data["root_cause_line_numbers"] = find_event_line_numbers(
							mapping_rows=mapping_rows,
							event_id=str(root_event),
						)
					success_message = "Prediction completed successfully."
				except Exception as exc:
					error_message = str(exc)

	return render_template(
		"index.html",
		raw_logs=raw_logs,
		mapping_rows=mapping_rows,
		prediction_data=prediction_data,
		transition_plot=transition_plot,
		sequence_length=sequence_length,
		error_message=error_message,
		success_message=success_message,
	)


@app.route("/realtime", methods=["GET"]) 
def realtime_page():
	sequence_length = _load_sequence_length()
	return render_template("realtime.html", sequence_length=sequence_length, stream_state=STREAM_STATE)


@app.route("/start_stream", methods=["POST"])
def start_stream():
	data = request.json or {}
	log_file_text = data.get("log_file", "dataset/HDFS.log")
	try:
		log_file = resolve_log_file(log_file_text)
		if not log_file.exists():
			return jsonify({"error": f"Log file not found: {log_file}"}), 400
	except Exception as e:
		return jsonify({"error": str(e)}), 400
	STREAM_STATE.update(
		{
			"active": True,
			"position": 0,
			"window": [],
			"processed": 0,
			"anomalies": 0,
			"probabilities": [],
			"event_counts": Counter(),
			"records": [],
			"log_file": str(log_file),
		}
	)
	return jsonify({"status": "started", "log_file": str(log_file)})


@app.route("/stop_stream", methods=["POST"])
def stop_stream():
	STREAM_STATE["active"] = False
	return jsonify({"status": "stopped"})


def _process_one_stream_step():
	seq_len = _load_sequence_length()
	log_file_str = STREAM_STATE.get("log_file", "dataset/HDFS.log")
	try:
		lines = load_log_lines(log_file_str)
	except FileNotFoundError as e:
		STREAM_STATE["active"] = False
		return {"active": False, "error": str(e)}

	if STREAM_STATE["position"] >= len(lines):
		STREAM_STATE["active"] = False
		return {"active": False, "message": "end_of_file"}

	current_line = lines[STREAM_STATE["position"]]
	matched = matcher.match(current_line)
	event_id = matched.event_id if matched else "UNKNOWN_EVENT"

	STREAM_STATE["position"] += 1
	STREAM_STATE["processed"] += 1
	STREAM_STATE["window"].append(event_id)
	if len(STREAM_STATE["window"]) > seq_len:
		STREAM_STATE["window"] = STREAM_STATE["window"][-seq_len:]
	STREAM_STATE["event_counts"][event_id] += 1

	record = {
		"log_line": current_line,
		"event_id": event_id,
		"window": list(STREAM_STATE["window"]),
		"prediction": None,
		"probability": None,
		"alert_level": None,
	}

	if len(STREAM_STATE["window"]) == seq_len:
		result = predict_failure(list(STREAM_STATE["window"]), enable_self_learning=False)
		record["prediction"] = "Failure / Anomaly" if result["predicted_failure"] else "Normal / Success"
		record["probability"] = float(result["anomaly_probability"])
		record["alert_level"] = str(result.get("alert_level"))
		STREAM_STATE["probabilities"].append(float(result["anomaly_probability"]))
		if result["predicted_failure"]:
			STREAM_STATE["anomalies"] += 1
	STREAM_STATE["records"].append(record)
	return {"active": True, "record": record, "processed": STREAM_STATE["processed"], "anomalies": STREAM_STATE["anomalies"]}


@app.route("/stream_step", methods=["POST"])
def stream_step():
	if not STREAM_STATE.get("active"):
		return jsonify({"active": False})
	step = _process_one_stream_step()
	return jsonify(step)


@app.route("/dashboard", methods=["GET"])
def dashboard_page():
	# Provide simple dashboard; charts read data from /dashboard_data via JS
	transition_plot = _ensure_transition_plot()
	return render_template("dashboard.html", transition_plot=transition_plot, stream_state=STREAM_STATE)


@app.route("/dashboard_data", methods=["GET"])
def dashboard_data():
	return jsonify(
		{
			"processed": STREAM_STATE.get("processed", 0),
			"anomalies": STREAM_STATE.get("anomalies", 0),
			"anomaly_percentage": (STREAM_STATE.get("anomalies", 0) / max(1, STREAM_STATE.get("processed", 0))) * 100.0,
			"probabilities": STREAM_STATE.get("probabilities", []),
			"event_counts": dict(STREAM_STATE.get("event_counts", {})),
		}
	)


if __name__ == "__main__":
	app.run(debug=True)
