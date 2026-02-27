"""
data_loader.py
--------------
Loads historical stock price data from Yahoo Finance or a local CSV.
Falls back to synthetic Geometric Brownian Motion data if both fail.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ── Public API ─────────────────────────────────────────────────────────────────
def load_data(
    ticker: str = "AAPL",
    start: str  = "2015-01-01",
    end: str    = "2024-12-31",
    csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Returns a clean DataFrame with at least columns:
        Date (datetime64), Open, High, Low, Close, Volume

    Priority:
        1. Local CSV  (if csv_path is provided)
        2. yfinance   (downloads from Yahoo Finance)
        3. Synthetic  (GBM fallback — useful for offline testing)
    """
    if csv_path:
        return _load_csv(csv_path)

    try:
        return _load_yfinance(ticker, start, end)
    except Exception as exc:
        print(f"[data_loader] yfinance failed ({exc}). Generating synthetic data...")
        return _generate_synthetic(n=2500, start=start)


# ── Private helpers ─────────────────────────────────────────────────────────────
def _load_csv(path: str) -> pd.DataFrame:
    """Load and validate a local CSV file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(p, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    _validate(df)
    print(f"[data_loader] Loaded CSV: {path}")
    return df


def _load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    import yfinance as yf  # optional dependency

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned empty DataFrame for ticker '{ticker}'")

    # Flatten MultiIndex columns (yfinance >= 0.2 may return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    _validate(df)
    print(f"[data_loader] Downloaded {ticker} ({start} → {end})  rows={len(df)}")
    return df


def _generate_synthetic(n: int = 2500, start: str = "2015-01-01") -> pd.DataFrame:
    """
    Generate a synthetic stock price series using Geometric Brownian Motion.
    Useful for unit tests and offline development.
    """
    np.random.seed(42)
    mu    = 0.0003          # daily drift
    sigma = 0.015           # daily volatility
    S0    = 130.0           # initial price

    dates  = pd.date_range(start=start, periods=n, freq="B")
    log_returns = (mu - 0.5 * sigma**2) + sigma * np.random.randn(n)
    prices = S0 * np.exp(np.cumsum(log_returns))

    df = pd.DataFrame({
        "Date"  : dates,
        "Open"  : prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High"  : prices * (1 + np.random.uniform(0.000, 0.015, n)),
        "Low"   : prices * (1 - np.random.uniform(0.000, 0.015, n)),
        "Close" : prices,
        "Volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    })
    print(f"[data_loader] Synthetic data generated  rows={len(df)}")
    return df


def _validate(df: pd.DataFrame) -> None:
    """Raise informative errors if the DataFrame doesn't meet requirements."""
    if "Close" not in df.columns:
        raise ValueError("Dataset must contain a 'Close' column.")
    if len(df) < 1000:
        raise ValueError(f"Dataset must have ≥ 1000 rows (found {len(df)}).")
    if df["Close"].isnull().any():
        missing = df["Close"].isnull().sum()
        print(f"[data_loader] Warning: {missing} missing Close values — forward-filling.")
        df["Close"].ffill(inplace=True)
