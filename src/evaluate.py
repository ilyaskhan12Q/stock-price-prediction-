"""
evaluate.py
-----------
Model evaluation utilities:
    - evaluate_model()          : computes RMSE and MAE in original USD scale
    - print_comparison_table()  : pretty-prints the summary table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model


# ── Public API ─────────────────────────────────────────────────────────────────
def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    label: str = "Model",
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Generates predictions, inverse-transforms them to USD, and computes RMSE/MAE.

    Returns
    -------
    rmse    : float
    mae     : float
    y_pred  : np.ndarray — predicted prices in USD  (n, 1)
    y_true  : np.ndarray — actual   prices in USD   (n, 1)
    """
    y_pred_scaled = model.predict(X_test, verbose=0)

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    _print_metrics(label, rmse, mae)
    return rmse, mae, y_pred, y_true


def print_comparison_table(results: dict[str, dict]) -> None:
    """
    Prints a formatted comparison table.

    Parameters
    ----------
    results : { "Model Name": {"rmse": float, "mae": float, "time": float} }
    """
    rows = [
        {
            "Model"         : name,
            "RMSE ($)"      : round(v["rmse"], 4),
            "MAE ($)"       : round(v["mae"],  4),
            "Train Time (s)": round(v["time"], 2),
        }
        for name, v in results.items()
    ]
    df = pd.DataFrame(rows)

    best_rmse_idx = df["RMSE ($)"].idxmin()
    best_mae_idx  = df["MAE ($)"].idxmin()

    print("\n" + "=" * 62)
    print("                 MODEL COMPARISON TABLE")
    print("=" * 62)
    print(df.to_string(index=False))
    print("=" * 62)
    print(f"  Best RMSE → {df.loc[best_rmse_idx, 'Model']}")
    print(f"  Best MAE  → {df.loc[best_mae_idx,  'Model']}")

    # Improvement %
    names = df["Model"].tolist()
    if len(names) >= 2:
        r0, r1 = df.iloc[0]["RMSE ($)"], df.iloc[1]["RMSE ($)"]
        pct = (r0 - r1) / r0 * 100
        print(f"\n  {names[0]} → {names[1]} RMSE improvement : {pct:+.2f}%")
    if len(names) >= 3:
        r1, r2 = df.iloc[1]["RMSE ($)"], df.iloc[2]["RMSE ($)"]
        pct = (r1 - r2) / r1 * 100
        print(f"  {names[1]} → {names[2]} RMSE improvement : {pct:+.2f}%")
    print("=" * 62)


# ── Private helpers ─────────────────────────────────────────────────────────────
def _print_metrics(label: str, rmse: float, mae: float) -> None:
    width = max(len(label) + 4, 40)
    print("\n" + "─" * width)
    print(f"  {label}")
    print("─" * width)
    print(f"  RMSE : ${rmse:.4f}")
    print(f"  MAE  : ${mae:.4f}")
    print("─" * width)
