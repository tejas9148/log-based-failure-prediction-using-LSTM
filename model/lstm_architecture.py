"""LSTM architecture for sequence-based log anomaly classification."""

from __future__ import annotations

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM


def build_lstm_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 64,
    lstm_units: int = 128,
):
    """Build and compile an LSTM binary classifier."""
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
            LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
