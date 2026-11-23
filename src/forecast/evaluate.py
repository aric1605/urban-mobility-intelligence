"""Evaluation metrics and baseline forecasting utilities."""

import numpy as np


def calc_mae(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return float(np.mean(np.abs(actual - pred)))


def calc_smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 200%]."""
    denom = (np.abs(actual) + np.abs(pred)) / 2.0 + 1e-8
    return float(np.mean(np.abs(actual - pred) / denom) * 100.0)


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season_lag: int = 168) -> np.ndarray:
    """Generate seasonal-naive forecast by repeating values from season_lag steps ago.
    Falls back to lag 24 if history < 168.
    """
    if len(history) < season_lag:
        season_lag = 24 if len(history) >= 24 else 1

    tile_count = int(np.ceil(horizon / season_lag))
    repeated = np.tile(history[-season_lag:], tile_count)
    return repeated[:horizon]
