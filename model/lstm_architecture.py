"""LSTM architecture for sequence-based log anomaly classification."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(package="project")
class AttentionPooling(Layer):
    """Simple attention pooling over a sequence of hidden states."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_layer = Dense(1, activation="tanh")

    def build(self, input_shape):
        self.score_layer.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score_layer(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)
        return weighted_sum

    def get_config(self):
        return super().get_config()


def build_lstm_model(
    vocab_size: int,
    sequence_length: int,
    embedding_dim: int = 128,
    lstm_units: int = 128,
):
    """Build and compile an attention-based BiLSTM binary classifier."""
    inputs = tf.keras.Input(shape=(sequence_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length)(inputs)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.0))(x)
    x = Dropout(0.2)(x)
    x = AttentionPooling()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=5e-4), loss="binary_crossentropy", metrics=["accuracy"])
    return model
