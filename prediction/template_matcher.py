"""Template matching utilities for mapping raw HDFS logs to EventIds."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
TEMPLATE_CSV_PATH = WORKSPACE_ROOT / "dataset" / "HDFS.log_templates.csv"


@dataclass
class MatchResult:
    event_id: str
    template: str


class TemplateMatcher:
    """Convert HDFS templates to regex and match incoming logs to EventIds."""

    def __init__(self, template_path: Path = TEMPLATE_CSV_PATH) -> None:
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
            if pattern.match(line) and score > best_score:
                best = MatchResult(event_id=event_id, template=template)
                best_score = score
        return best


def parse_logs(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]
