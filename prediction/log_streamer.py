"""Helpers for reading and streaming log files line-by-line."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator


BASE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = BASE_DIR.parent
DEFAULT_LOG_FILE = WORKSPACE_ROOT / "hdfs_logs.txt"


def resolve_log_file(log_file: str | Path | None = None) -> Path:
    """Resolve a log-file path against the common project locations."""
    if log_file is None:
        candidates = [DEFAULT_LOG_FILE, BASE_DIR / "hdfs_logs.txt", BASE_DIR / "dataset" / "hdfs_logs.txt"]
    else:
        raw = Path(log_file)
        candidates = [raw]
        if not raw.is_absolute():
            candidates.extend(
                [
                    BASE_DIR / raw,
                    WORKSPACE_ROOT / raw,
                ]
            )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def load_log_lines(log_file: str | Path) -> list[str]:
    """Load stripped log lines from a file."""
    path = resolve_log_file(log_file)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def stream_log_lines(
    log_file: str | Path,
    start_index: int = 0,
    delay_seconds: float = 1.0,
) -> Iterator[tuple[int, str]]:
    """Yield log lines one by one with an optional delay between items."""
    lines = load_log_lines(log_file)
    for index in range(start_index, len(lines)):
        yield index, lines[index]
        if delay_seconds > 0:
            time.sleep(delay_seconds)