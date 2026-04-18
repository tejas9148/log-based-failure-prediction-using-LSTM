"""Data loading and preprocessing utilities for log failure prediction."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


REQUIRED_COLUMNS = ["Label", "Features"]
EVENT_PATTERN = re.compile(r"E\d+")


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset CSV and validate required columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def parse_feature_sequence(feature_text: str) -> List[str]:
    """Extract event ids in the form E1, E2, ... from feature text."""
    if not isinstance(feature_text, str):
        return []
    return EVENT_PATTERN.findall(feature_text)


def encode_event_ids(df: pd.DataFrame) -> Tuple[List[np.ndarray], np.ndarray, LabelEncoder]:
    """Map labels to binary and encode EventIds into integer tokens."""
    frame = df.copy()
    frame["Label"] = frame["Label"].astype(str).str.strip().str.lower()

    label_map = {
        "success": 0,
        "normal": 0,
        "fail": 1,
        "anomaly": 1,
    }
    if not frame["Label"].isin(label_map).all():
        unknown = frame.loc[~frame["Label"].isin(label_map), "Label"].unique().tolist()
        raise ValueError(f"Unsupported labels in dataset: {unknown}")

    labels = frame["Label"].map(label_map).to_numpy(dtype=np.float32)

    token_series = frame["Features"].apply(parse_feature_sequence)
    token_series = token_series[token_series.apply(len) > 0]
    valid_indices = token_series.index.to_numpy()
    labels = labels[valid_indices]

    all_tokens = [token for tokens in token_series for token in tokens]
    if not all_tokens:
        raise ValueError("No EventIds found in Features column.")

    encoder = LabelEncoder()
    encoder.fit(all_tokens)

    encoded_traces: List[np.ndarray] = []
    for tokens in token_series:
        encoded_traces.append(encoder.transform(tokens).astype(np.int32))

    return encoded_traces, labels, encoder


def generate_sequences(
    encoded_traces: list[np.ndarray],
    trace_labels: np.ndarray,
    sequence_length: int = 5,
    step_size: int = 1,
) -> Tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Generate fixed-length sliding windows from encoded traces."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")
    if step_size <= 0:
        raise ValueError("step_size must be greater than 0")

    x_data = []
    y_data = []

    for trace, label in zip(encoded_traces, trace_labels):
        if len(trace) < sequence_length:
            continue

        for index in range(0, len(trace) - sequence_length + 1, step_size):
            x_data.append(trace[index : index + sequence_length])
            y_data.append(int(label))

    if not x_data:
        raise ValueError("No windows generated. Check sequence_length and dataset contents.")

    x = np.array(x_data, dtype=np.int32)
    y = np.array(y_data, dtype=np.float32)

    return x, y, {
        "samples": int(len(x)),
        "normal_samples": int(np.sum(y == 0)),
        "anomaly_samples": int(np.sum(y == 1)),
    }
