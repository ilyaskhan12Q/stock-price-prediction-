"""
visualize.py
------------
All matplotlib plotting functions used by train.py.

Each function:
    • Creates a clearly labelled figure
    • Saves it to `save_path` (Path or str)
    • Calls plt.show() so it also renders in Jupyter
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = "#1565C0"
RED    = "#EF5350"
GREEN  = "#00897B"
ORANGE = "#FF8C42"
PURPLE = "#8E24AA"
GREY   = "#90CAF9"


# ── 1. Close Price History ──────────────────────────────────────────────────────
def plot_close_price(df: pd.DataFrame, save_path: Path | str = "plots/close_price.png") -> None:
    """Line chart of closing price + volume bar chart."""
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle("Stock Price History", fontsize=16, fontweight="bold")

    ax1 = axes[0]
    ax1.plot(df["Date"], df["Close"], color=BLUE, linewidth=1.2, label="Close Price")
    ax1.fill_between(df["Date"], df["Close"], alpha=0.12, color=BLUE)
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.set_title("Closing Price Over Time", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    ax2.bar(df["Date"], df["Volume"], color=GREY, width=1, alpha=0.7, label="Volume")
    ax2.set_ylabel("Volume", fontsize=11)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    _save(save_path)


# ── 2. Training Loss ────────────────────────────────────────────────────────────
def plot_loss(history, title: str = "Training Loss", save_path: Path | str = "plots/loss.png") -> None:
    """Train vs validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history.history["loss"],     label="Train Loss", color=BLUE,  linewidth=2)
    ax.plot(history.history["val_loss"], label="Val Loss",   color=RED,   linewidth=2, linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(save_path)


# ── 3. Single-model predictions vs actuals ─────────────────────────────────────
def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    save_path: Path | str = "plots/predictions.png",
) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, y_true, color=BLUE,   linewidth=1.8, label="Actual Price")
    ax.plot(dates, y_pred, color=ORANGE, linewidth=1.8, label="Predicted",  linestyle="--")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save(save_path)


# ── 4. All models overlay ──────────────────────────────────────────────────────
def plot_all_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    rnn_pred: np.ndarray,
    lstm_pred: np.ndarray,
    attn_pred: np.ndarray | None = None,
    save_path: Path | str = "plots/all_predictions.png",
) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, y_true,    color=BLUE,   linewidth=2,   label="Actual Price")
    ax.plot(dates, rnn_pred,  color=RED,    linewidth=1.5, label="Simple RNN",       linestyle="--")
    ax.plot(dates, lstm_pred, color=GREEN,  linewidth=1.5, label="LSTM",             linestyle="-.")
    if attn_pred is not None:
        ax.plot(dates, attn_pred, color=ORANGE, linewidth=1.5, label="LSTM + Attention", linestyle=":")
    ax.set_title("All Models — Predictions vs Actual", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save(save_path)


# ── 5. Comparison bar chart ────────────────────────────────────────────────────
def plot_comparison_bar(
    results: dict[str, dict],
    save_path: Path | str = "plots/model_comparison.png",
) -> None:
    """Side-by-side RMSE and MAE bar chart for all models."""
    models  = list(results.keys())
    rmse    = [results[m]["rmse"] for m in models]
    mae     = [results[m]["mae"]  for m in models]
    colours = [RED, GREEN, ORANGE][: len(models)]
    x       = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")

    for ax, values, metric in zip(axes, [rmse, mae], ["RMSE ($)", "MAE ($)"]):
        bars = ax.bar(x, values, color=colours, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel("Error (USD)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + h * 0.01,
                f"${h:.2f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    plt.tight_layout()
    _save(save_path)


# ── Helper ─────────────────────────────────────────────────────────────────────
def _save(path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[visualize] Saved → {path}")
