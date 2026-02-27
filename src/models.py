"""
models.py
---------
Defines three Keras model builders:

    build_rnn(input_shape)              → SimpleRNN(64) → Dense(1)
    build_lstm(input_shape)             → LSTM(64)      → Dense(1)
    build_lstm_attention(input_shape)   → LSTM(64, return_sequences=True)
                                          → AttentionLayer → Dense(1)

All models use:
    • Adam optimiser
    • MSE loss
    • Identical hidden units (64) and output layer for fair comparison
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.models   import Sequential, Model
from tensorflow.keras.layers   import SimpleRNN, LSTM, Dense, Input, Layer
import tensorflow.keras.backend as K


# ── Simple RNN ──────────────────────────────────────────────────────────────────
def build_rnn(input_shape: tuple[int, int]) -> Sequential:
    """
    Parameters
    ----------
    input_shape : (window_size, n_features)  e.g. (10, 1)
    """
    model = Sequential(
        [
            Input(shape=input_shape, name="Input"),
            SimpleRNN(
                64,
                activation="tanh",
                return_sequences=False,
                name="SimpleRNN_64",
            ),
            Dense(1, name="Output"),
        ],
        name="SimpleRNN_Model",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# ── LSTM ────────────────────────────────────────────────────────────────────────
def build_lstm(input_shape: tuple[int, int]) -> Sequential:
    """
    Drop-in replacement for SimpleRNN — same hyperparameters, LSTM cell.
    """
    model = Sequential(
        [
            Input(shape=input_shape, name="Input"),
            LSTM(
                64,
                return_sequences=False,
                name="LSTM_64",
            ),
            Dense(1, name="Output"),
        ],
        name="LSTM_Model",
    )
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Attention Layer ─────────────────────────────────────────────────────────────
class AttentionLayer(Layer):
    """
    Bahdanau-style (additive) soft attention over the LSTM output sequence.

    Forward pass
    ------------
    Input  : (batch, timesteps, units)  — full LSTM hidden-state sequence
    Output : (batch, units)             — weighted context vector

    The scalar energy for each timestep t is:
        e_t = tanh( h_t · W  +  b_t )
    Attention weights:
        a   = softmax(e)
    Context vector:
        c   = Σ_t  a_t * h_t
    """

    def build(self, input_shape):
        units = input_shape[-1]
        timesteps = input_shape[1]

        self.W = self.add_weight(
            name="attn_W",
            shape=(units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(timesteps, 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # Energy: (batch, timesteps, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Softmax over time axis: (batch, timesteps, 1)
        a = K.softmax(e, axis=1)
        # Context vector: (batch, units)
        context = K.sum(x * a, axis=1)
        return context

    def get_config(self):
        return super().get_config()


# ── LSTM + Attention ────────────────────────────────────────────────────────────
def build_lstm_attention(input_shape: tuple[int, int]) -> Model:
    """
    LSTM with return_sequences=True feeds all hidden states into AttentionLayer,
    which produces a single context vector passed to the Dense output.
    """
    inp = Input(shape=input_shape, name="Input")
    x   = LSTM(64, return_sequences=True, name="LSTM_64")(inp)
    ctx = AttentionLayer(name="Attention")(x)
    out = Dense(1, name="Output")(ctx)

    model = Model(inputs=inp, outputs=out, name="LSTM_Attention_Model")
    model.compile(optimizer="adam", loss="mse")
    return model
