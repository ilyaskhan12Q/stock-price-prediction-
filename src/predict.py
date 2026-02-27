"""
predict.py
----------
Load a saved Keras model and run inference on new / unseen data.

Usage:
    python src/predict.py --model_path saved_models/lstm_model.keras --ticker AAPL --days 30
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from data_loader  import load_data
from preprocessor import _create_sequences


def parse_args():
    parser = argparse.ArgumentParser(description="Stock price inference with a saved model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved .keras model")
    parser.add_argument("--ticker",     type=str, default="AAPL")
    parser.add_argument("--start",      type=str, default="2020-01-01")
    parser.add_argument("--end",        type=str, default="2024-12-31")
    parser.add_argument("--csv",        type=str, default=None)
    parser.add_argument("--window",     type=int, default=10)
    parser.add_argument("--days",       type=int, default=30, help="Number of future days to forecast")
    parser.add_argument("--plots_dir",  type=str, default="plots")
    return parser.parse_args()


def recursive_forecast(model, last_window: np.ndarray, scaler: MinMaxScaler, n_days: int) -> np.ndarray:
    """
    Autoregressively predicts `n_days` into the future.
    Each prediction is fed back as input for the next step.
    """
    window = last_window.copy()   # (window_size, 1)
    preds  = []
    for _ in range(n_days):
        x    = window.reshape(1, len(window), 1)
        pred = model.predict(x, verbose=0)[0, 0]
        preds.append(pred)
        window = np.append(window[1:], [[pred]], axis=0)

    preds_usd = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds_usd.flatten()


def main():
    args = parse_args()

    # Load model
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"[predict] Loaded model from {model_path}")

    # Load and scale data
    df = load_data(ticker=args.ticker, start=args.start, end=args.end, csv_path=args.csv)
    close = df["Close"].values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)

    # Last window for forecasting
    last_window = scaled[-args.window:]

    # Recursive future forecast
    future_prices = recursive_forecast(model, last_window, scaler, args.days)

    # Build future date index
    last_date    = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=args.days)

    # Print forecast
    print(f"\n[predict] {args.days}-day forecast for {args.ticker}:")
    print("-" * 40)
    for date, price in zip(future_dates, future_prices):
        print(f"  {date.date()}  →  ${price:.2f}")

    # Plot
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    history_n = min(180, len(df))
    ax.plot(df["Date"].iloc[-history_n:], df["Close"].iloc[-history_n:],
            color="#1565C0", linewidth=1.8, label="Historical Price")
    ax.plot(future_dates, future_prices,
            color="#FF8C42", linewidth=2, linestyle="--", marker="o",
            markersize=4, label=f"Forecast ({args.days} days)")
    ax.axvline(x=last_date, color="grey", linestyle=":", linewidth=1.5, label="Forecast Start")
    ax.set_title(f"{args.ticker} — {args.days}-Day Price Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    out = plots_dir / "forecast.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[predict] Plot saved → {out}")


if __name__ == "__main__":
    main()
