"""Desktop application for raw-log to EventId mapping and anomaly prediction."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from tkinter import END, BOTH, LEFT, RIGHT, VERTICAL, Button, Frame, Label, Tk, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter.ttk import Scrollbar, Treeview

from project.prediction.predictor import predict_failure


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
TEMPLATE_CSV_PATH = WORKSPACE_ROOT / "dataset" / "HDFS.log_templates.csv"
CONFIG_PATH = BASE_DIR / "config.json"


@dataclass
class MatchResult:
    event_id: str
    template: str


class TemplateMatcher:
    """Convert HDFS templates to regex and match incoming logs to EventIds."""

    def __init__(self, template_path: Path) -> None:
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        self._patterns: list[tuple[re.Pattern[str], str, str, int]] = []
        with template_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                event_id = str(row["EventId"]).strip()
                template = str(row["EventTemplate"]).strip()
                pattern_text = self._template_to_regex(template)
                score = self._specificity_score(template)
                self._patterns.append((re.compile(pattern_text), event_id, template, score))

    @staticmethod
    def _template_to_regex(template: str) -> str:
        escaped = re.escape(template)
        escaped = escaped.replace(r"\[\*\]", r".*")
        escaped = escaped.replace(r"\*", r".*")
        return r"^" + escaped + r"$"

    @staticmethod
    def _specificity_score(template: str) -> int:
        wildcards = template.count("[*]") + template.count("*")
        return max(0, len(template) - (wildcards * 2))

    def match(self, log_line: str) -> MatchResult | None:
        line = log_line.strip()
        if not line:
            return None

        best: MatchResult | None = None
        best_score = -1
        for pattern, event_id, template, score in self._patterns:
            if pattern.match(line):
                if score > best_score:
                    best = MatchResult(event_id=event_id, template=template)
                    best_score = score
        return best


def load_sequence_length(config_path: Path) -> int:
    if not config_path.exists():
        raise FileNotFoundError("config.json not found. Run training first using python -m project.train")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return int(data["sequence_length"])


def parse_logs(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def run_app() -> None:
    matcher = TemplateMatcher(TEMPLATE_CSV_PATH)
    sequence_length = load_sequence_length(CONFIG_PATH)

    root = Tk()
    root.title("Log Failure Prediction - Raw Log Input")
    root.geometry("1200x760")

    top_frame = Frame(root)
    top_frame.pack(fill=BOTH, padx=10, pady=8)

    Label(
        top_frame,
        text=(
            "Paste actual logs below, then click Process and Predict. "
            "The app maps each log line to template EventId before prediction."
        ),
        anchor="w",
        justify=LEFT,
    ).pack(fill=BOTH)

    input_box = ScrolledText(root, wrap="word", height=14)
    input_box.pack(fill=BOTH, padx=10, pady=8)

    table_frame = Frame(root)
    table_frame.pack(fill=BOTH, expand=True, padx=10, pady=8)

    columns = ("line_no", "event_id", "template")
    mapping_table = Treeview(table_frame, columns=columns, show="headings", height=14)
    mapping_table.heading("line_no", text="Line")
    mapping_table.heading("event_id", text="Mapped EventId")
    mapping_table.heading("template", text="Matched Template")
    mapping_table.column("line_no", width=80, anchor="center")
    mapping_table.column("event_id", width=160, anchor="center")
    mapping_table.column("template", width=860, anchor="w")

    table_scroll = Scrollbar(table_frame, orient=VERTICAL, command=mapping_table.yview)
    mapping_table.configure(yscrollcommand=table_scroll.set)
    mapping_table.pack(side=LEFT, fill=BOTH, expand=True)
    table_scroll.pack(side=RIGHT, fill="y")

    result_box = ScrolledText(root, wrap="word", height=10)
    result_box.pack(fill=BOTH, padx=10, pady=8)

    def process_and_predict() -> None:
        raw_logs = input_box.get("1.0", END)
        lines = parse_logs(raw_logs)
        if not lines:
            messagebox.showwarning("No input", "Please paste log lines before prediction.")
            return

        for item in mapping_table.get_children():
            mapping_table.delete(item)

        event_ids: list[str] = []
        for idx, line in enumerate(lines, start=1):
            matched = matcher.match(line)
            if matched is None:
                event_id = "UNKNOWN_EVENT"
                template = "No matching template"
            else:
                event_id = matched.event_id
                template = matched.template

            mapping_table.insert("", END, values=(idx, event_id, template))
            event_ids.append(event_id)

        if len(event_ids) < sequence_length:
            messagebox.showerror(
                "Insufficient logs",
                (
                    f"Need at least {sequence_length} logs to predict. "
                    f"Received {len(event_ids)}."
                ),
            )
            return

        sequence = event_ids[-sequence_length:]

        try:
            prediction = predict_failure(sequence)
        except Exception as exc:
            messagebox.showerror("Prediction error", str(exc))
            return

        decision = "Failure/Anomaly" if prediction["predicted_failure"] else "Normal/Success"

        result_box.delete("1.0", END)
        result_box.insert(END, "Prediction Summary\n")
        result_box.insert(END, "===================\n")
        result_box.insert(END, f"Required Sequence Length: {sequence_length}\n")
        result_box.insert(END, f"Input Log Count: {len(event_ids)}\n")
        result_box.insert(END, f"Used EventId Sequence: {sequence}\n")
        result_box.insert(END, f"Anomaly Probability: {prediction['anomaly_probability']:.4f}\n")
        result_box.insert(END, f"Decision Threshold: {prediction['decision_threshold']:.4f}\n")
        result_box.insert(END, f"Final Prediction: {decision}\n")
        result_box.insert(END, f"Alert Level: {prediction['alert_level']}\\n")
        if prediction["root_cause_event"]:
            result_box.insert(END, f"Root Cause Event: {prediction['root_cause_event']}\\n")
            result_box.insert(END, f"Root Cause Explanation: {prediction['root_cause_explanation']}\\n")
        if prediction["unknown_event_ids"]:
            result_box.insert(END, f"Unknown EventIds in Sequence: {prediction['unknown_event_ids']}\n")

    def load_example() -> None:
        sample = """INFO Receiving block blk_120 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
INFO Received block blk_120 src: /10.250.19.102:54106 dest: /10.250.19.102:50010 of size 67108864
INFO PacketResponder 1 for block blk_120 terminating
INFO BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811111030_0001_m_000145_0/part-00145. blk_12345
INFO BLOCK* ask 10.251.111.130:50010 to replicate blk_12345 to datanode(s) 10.250.10.10:50010"""
        input_box.delete("1.0", END)
        input_box.insert(END, sample)

    button_frame = Frame(root)
    button_frame.pack(fill=BOTH, padx=10, pady=8)

    Button(button_frame, text="Load Example Logs", command=load_example, width=20).pack(side=LEFT, padx=6)
    Button(button_frame, text="Process and Predict", command=process_and_predict, width=20).pack(
        side=LEFT, padx=6
    )

    root.mainloop()


if __name__ == "__main__":
    run_app()
