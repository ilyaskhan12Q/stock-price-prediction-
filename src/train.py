"""
train.py
--------
Entry point: loads data, preprocesses, trains all models, saves results.

Usage:
    python src/train.py
    python src/train.py --ticker TSLA --epochs 20 --window 20
"""

import argparse
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader   import load_data
from preprocessor  import preprocess
from models        import build_rnn, build_lstm, build_lstm_attention
from evaluate      import evaluate_model, print_comparison_table
from visualize     import (
    plot_close_price,
    plot_loss,
    plot_predictions,
    plot_all_predictions,
    plot_comparison_bar,
)


# ── CLI arguments ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Stock Price Prediction: RNN / LSTM / Attention")
    parser.add_argument("--ticker",     type=str,   default="AAPL",  help="Yahoo Finance ticker symbol")
    parser.add_argument("--start",      type=str,   default="2015-01-01")
    parser.add_argument("--end",        type=str,   default="2024-12-31")
    parser.add_argument("--csv",        type=str,   default=None,    help="Path to local CSV file (overrides ticker)")
    parser.add_argument("--window",     type=int,   default=10,      help="Sliding window size (days)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--split",      type=float, default=0.80,    help="Train/test split ratio")
    parser.add_argument("--plots_dir",  type=str,   default="plots")
    parser.add_argument("--no_attention", action="store_true",       help="Skip LSTM+Attention model")
    return parser.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Stock Price Prediction — RNN / LSTM / LSTM+Attention")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df = load_data(ticker=args.ticker, start=args.start, end=args.end, csv_path=args.csv)
    print(f"      Shape       : {df.shape}")
    print(f"      Date range  : {df['Date'].min().date()} → {df['Date'].max().date()}")
    print(f"      Close range : ${df['Close'].min():.2f} – ${df['Close'].max():.2f}")
    plot_close_price(df, save_path=plots_dir / "close_price.png")

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing...")
    X_train, X_test, y_train, y_test, scaler, test_dates = preprocess(
        df, window_size=args.window, split=args.split
    )
    print(f"      Window size  : {args.window} days")
    print(f"      Train samples: {X_train.shape[0]}")
    print(f"      Test  samples: {X_test.shape[0]}")

    input_shape = (args.window, 1)
    results     = {}

    # ── 3. Simple RNN ─────────────────────────────────────────────────────────
    print("\n[3/5] Training Simple RNN...")
    rnn_model = build_rnn(input_shape)
    t0 = time.time()
    rnn_history = rnn_model.fit(
        X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_split=0.1, verbose=1, shuffle=False
    )
    rnn_time = time.time() - t0

    rnn_rmse, rnn_mae, rnn_pred, y_true = evaluate_model(rnn_model, X_test, y_test, scaler, "Simple RNN")
    results["Simple RNN"] = dict(rmse=rnn_rmse, mae=rnn_mae, time=rnn_time)

    plot_loss(rnn_history,       "Simple RNN — Training Loss",      plots_dir / "rnn_loss.png")
    plot_predictions(test_dates, y_true, rnn_pred,
                     "Simple RNN — Predictions vs Actual",          plots_dir / "rnn_predictions.png")

    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    rnn_model.save(saved_models_dir / "rnn_model.keras")
    print(f"✅ RNN model saved → {saved_models_dir / 'rnn_model.keras'}")

    # ── 4. LSTM ───────────────────────────────────────────────────────────────
    print("\n[4/5] Training LSTM...")
    lstm_model = build_lstm(input_shape)
    t0 = time.time()
    lstm_history = lstm_model.fit(
        X_train, y_train,
        epochs=args.epochs, batch_size=args.batch_size,
        validation_split=0.1, verbose=1, shuffle=False
    )
    lstm_time = time.time() - t0

    lstm_rmse, lstm_mae, lstm_pred, _ = evaluate_model(lstm_model, X_test, y_test, scaler, "LSTM")
    results["LSTM"] = dict(rmse=lstm_rmse, mae=lstm_mae, time=lstm_time)

    plot_loss(lstm_history,      "LSTM — Training Loss",            plots_dir / "lstm_loss.png")
    plot_predictions(test_dates, y_true, lstm_pred,
                     "LSTM — Predictions vs Actual",                plots_dir / "lstm_predictions.png")

    lstm_model.save(saved_models_dir / "lstm_model.keras")
    print(f"✅ LSTM model saved → {saved_models_dir / 'lstm_model.keras'}")

    # ── 5. LSTM + Attention ───────────────────────────────────────────────────
    attn_pred = None
    if not args.no_attention:
        print("\n[5/5] Training LSTM + Attention...")
        attn_model = build_lstm_attention(input_shape)
        t0 = time.time()
        attn_history = attn_model.fit(
            X_train, y_train,
            epochs=args.epochs, batch_size=args.batch_size,
            validation_split=0.1, verbose=1, shuffle=False
        )
        attn_time = time.time() - t0

        attn_rmse, attn_mae, attn_pred, _ = evaluate_model(
            attn_model, X_test, y_test, scaler, "LSTM + Attention")
        results["LSTM + Attention"] = dict(rmse=attn_rmse, mae=attn_mae, time=attn_time)

        plot_loss(attn_history,      "LSTM + Attention — Training Loss",  plots_dir / "attn_loss.png")
        plot_predictions(test_dates, y_true, attn_pred,
                         "LSTM + Attention — Predictions vs Actual",      plots_dir / "attn_predictions.png")

        attn_model.save(saved_models_dir / "lstm_attention_model.keras")
        print(f"✅ Attention model saved → {saved_models_dir / 'lstm_attention_model.keras'}")
    else:
        print("\n[5/5] Skipping LSTM + Attention (--no_attention flag set)")

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_comparison_table(results)
    plot_comparison_bar(results, save_path=plots_dir / "model_comparison.png")
    plot_all_predictions(
        test_dates, y_true, rnn_pred, lstm_pred, attn_pred,
        save_path=plots_dir / "all_predictions.png"
    )

    # Save metrics to JSON
    metrics_path = plots_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Metrics saved → {metrics_path}")
    print(f"✅ All plots saved → {plots_dir}/")
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()
