"""
preprocessor.py
---------------
Normalises the Close price and builds sliding-window sequences for
supervised learning.

    X[i] = scaled_prices[i : i + window_size]   → shape (window_size, 1)
    y[i] = scaled_prices[i + window_size]        → shape (1,)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ── Public API ─────────────────────────────────────────────────────────────────
def preprocess(
    df: pd.DataFrame,
    window_size: int = 10,
    split: float     = 0.80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, np.ndarray]:
    """
    Parameters
    ----------
    df          : DataFrame with a 'Close' and 'Date' column.
    window_size : Number of past days used as input features.
    split       : Fraction of data used for training.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                           : fitted MinMaxScaler (needed for inverse transform)
    test_dates                       : date array aligned with the test predictions
    """
    # 1. Scale
    close = df["Close"].values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    # 2. Build sequences
    X, y = _create_sequences(scaled, window_size)

    # 3. Split
    n_train = int(len(X) * split)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # 4. Dates aligned to test set
    test_dates = df["Date"].values[n_train + window_size:]

    _print_samples(X, y, scaler, window_size, n_samples=3)

    return X_train, X_test, y_train, y_test, scaler, test_dates


# ── Helpers ─────────────────────────────────────────────────────────────────────
def _create_sequences(
    data: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window across the data and return (X, y) arrays."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])          # (window_size, 1)
        y.append(data[i + window_size])               # (1,)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _print_samples(
    X: np.ndarray,
    y: np.ndarray,
    scaler: MinMaxScaler,
    window_size: int,
    n_samples: int = 3,
) -> None:
    """Pretty-print a few labelled input-output samples."""
    print(f"\n[preprocessor] Sample input-output sequences (window={window_size}):")
    print("-" * 62)
    for i in range(n_samples):
        input_prices  = scaler.inverse_transform(X[i]).flatten()
        target_price  = scaler.inverse_transform(y[i].reshape(-1, 1))[0][0]
        print(f"  Sample {i + 1}:")
        print(f"    Input  ({window_size} days) : {np.round(input_prices, 2).tolist()}")
        print(f"    Target (next day) : ${target_price:.2f}")
        print("-" * 62)
