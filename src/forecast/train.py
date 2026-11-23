"""Precompute SARIMAX forecasts and honest holdout evaluation for city and top-20 zones."""

from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.forecast.evaluate import calc_mae, calc_smape, seasonal_naive_forecast


DB_PATH = Path("data/urban.db")
HOLDOUT_HOURS = 168  # 7 days
HORIZON_HOURS = 48   # 48h ahead


def fit_and_forecast_single_series(
    zone_id: int,
    zone_name: str,
    df_series: pd.DataFrame,
) -> Tuple[Dict[str, any], List[Dict[str, any]]]:
    """Fit SARIMAX and evaluate against Seasonal Naive baseline on last 7 days holdout.
    Then forecast 48 hours ahead and detect anomalies.
    """
    # Ensure complete time series
    full_idx = pd.date_range(
        start=df_series["hour_ts"].min(),
        end=df_series["hour_ts"].max(),
        freq="h",
    )
    df_reindexed = df_series.set_index("hour_ts").reindex(full_idx)
    df_reindexed["congestion_index"] = df_reindexed["congestion_index"].interpolate(method="linear").bfill().ffill()
    values = df_reindexed["congestion_index"].values
    ts_strings = [ts.strftime("%Y-%m-%d %H:00:00") for ts in full_idx]

    # Holdout split
    train_vals = values[:-HOLDOUT_HOURS]
    test_vals = values[-HOLDOUT_HOURS:]

    # Baseline: seasonal naive (same hour last week, lag 168)
    base_pred = seasonal_naive_forecast(train_vals, HOLDOUT_HOURS, season_lag=168)
    baseline_mae = calc_mae(test_vals, base_pred)
    baseline_smape = calc_smape(test_vals, base_pred)

    # SARIMAX model fit with timeout / fallback
    model_mae = baseline_mae
    model_smape = baseline_smape
    used_model = "Seasonal Baseline"
    full_fitted = values.copy()
    full_resid_std = 0.05
    future_pred = base_pred[:HORIZON_HOURS]
    lower_ci = np.clip(future_pred - 1.96 * full_resid_std, 0.0, 1.0)
    upper_ci = np.clip(future_pred + 1.96 * full_resid_std, 0.0, 1.0)

    try:
        t0 = time.time()
        mod = SARIMAX(
            train_vals,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=35)
        pred_holdout = res.forecast(HOLDOUT_HOURS)
        model_mae = calc_mae(test_vals, pred_holdout)
        model_smape = calc_smape(test_vals, pred_holdout)

        # Fit on full series for 48h future forecast
        full_mod = SARIMAX(
            values,
            order=(1, 0, 1),
            seasonal_order=(1, 1, 0, 24),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        full_res = full_mod.fit(disp=False, maxiter=35)
        full_fitted = full_res.fittedvalues
        residuals = values - full_fitted
        full_resid_std = float(np.std(residuals)) if len(residuals) > 0 else 0.05

        forecast_res = full_res.get_forecast(steps=HORIZON_HOURS)
        future_pred = np.clip(forecast_res.predicted_mean, 0.0, 1.0)
        ci = forecast_res.conf_int(alpha=0.05)
        lower_ci = np.clip(ci[:, 0], 0.0, 1.0)
        upper_ci = np.clip(ci[:, 1], 0.0, 1.0)
        used_model = "SARIMAX"
        print(f"  [{zone_name[:20]}] Fit {time.time()-t0:.1f}s | Model MAE: {model_mae:.4f} vs Base MAE: {baseline_mae:.4f}")
    except Exception as e:
        print(f"  [{zone_name[:20]}] Fallback to baseline due to: {e}")
        future_pred = seasonal_naive_forecast(values, HORIZON_HOURS, season_lag=168)
        lower_ci = np.clip(future_pred - 0.1, 0.0, 1.0)
        upper_ci = np.clip(future_pred + 0.1, 0.0, 1.0)

    winner = "SARIMAX" if model_mae <= baseline_mae else "Baseline"

    eval_record = {
        "zone_id": zone_id,
        "zone_name": zone_name,
        "model_mae": round(float(model_mae), 4),
        "model_smape": round(float(model_smape), 2),
        "baseline_mae": round(float(baseline_mae), 4),
        "baseline_smape": round(float(baseline_smape), 2),
        "winner": winner,
    }

    # Anomaly detection on history: actual > fitted + 2 * resid_std
    forecast_rows: List[Dict[str, any]] = []
    threshold = 2.0 * full_resid_std

    # Historical records (last 336 hours = 14 days kept in forecasts table for charting)
    chart_history_len = min(len(values), 336)
    hist_start = len(values) - chart_history_len

    for i in range(hist_start, len(values)):
        act_val = float(values[i])
        fit_val = float(full_fitted[i]) if i < len(full_fitted) else act_val
        is_anom = 1 if (act_val - fit_val) > threshold else 0
        forecast_rows.append({
            "zone_id": zone_id,
            "hour_ts": ts_strings[i],
            "actual_congestion": round(act_val, 4),
            "forecast_congestion": round(fit_val, 4),
            "lower_ci": None,
            "upper_ci": None,
            "is_anomaly": is_anom,
            "is_future": 0,
        })

    # Future 48h records
    last_dt = full_idx[-1]
    for h in range(1, HORIZON_HOURS + 1):
        fut_dt = last_dt + timedelta(hours=h)
        fut_ts = fut_dt.strftime("%Y-%m-%d %H:00:00")
        f_val = float(future_pred[h - 1])
        l_val = float(lower_ci[h - 1])
        u_val = float(upper_ci[h - 1])
        forecast_rows.append({
            "zone_id": zone_id,
            "hour_ts": fut_ts,
            "actual_congestion": None,
            "forecast_congestion": round(f_val, 4),
            "lower_ci": round(l_val, 4),
            "upper_ci": round(u_val, 4),
            "is_anomaly": 0,
            "is_future": 1,
        })

    return eval_record, forecast_rows


def run_forecast_pipeline() -> pd.DataFrame:
    """Precompute forecasts for City-Wide (zone 0) and Top 20 Busiest Zones."""
    print("=" * 60)
    print("STARTING SARIMAX FORECASTING & EVALUATION PIPELINE")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)

    # 1. City-Wide
    print("[FORECAST] Processing City-Wide series...")
    city_df = pd.read_sql(
        "SELECT hour_ts, congestion_index FROM city_hourly ORDER BY hour_ts",
        conn,
    )
    city_df["hour_ts"] = pd.to_datetime(city_df["hour_ts"])

    all_evals = []
    all_forecast_rows = []

    city_eval, city_forecasts = fit_and_forecast_single_series(0, "City-Wide (All Zones)", city_df)
    all_evals.append(city_eval)
    all_forecast_rows.extend(city_forecasts)

    # 2. Top 20 Busiest Zones
    print("[FORECAST] Fetching Top 20 Busiest Zones...")
    top_zones_query = """
    SELECT zh.zone_id, z.zone_name, z.borough, SUM(zh.trips) as trips
    FROM zone_hourly zh
    JOIN zones z ON zh.zone_id = z.zone_id
    GROUP BY zh.zone_id, z.zone_name, z.borough
    ORDER BY trips DESC
    LIMIT 20
    """
    top_zones = pd.read_sql(top_zones_query, conn)

    for _, row in top_zones.iterrows():
        zid = int(row["zone_id"])
        zname = f"{row['zone_name']} ({row['borough']})"
        z_df = pd.read_sql(
            "SELECT hour_ts, congestion_index FROM zone_hourly WHERE zone_id = ? ORDER BY hour_ts",
            conn,
            params=(zid,),
        )
        z_df["hour_ts"] = pd.to_datetime(z_df["hour_ts"])
        z_eval, z_forecasts = fit_and_forecast_single_series(zid, zname, z_df)
        all_evals.append(z_eval)
        all_forecast_rows.extend(z_forecasts)

    eval_df = pd.DataFrame(all_evals)
    forecast_df = pd.DataFrame(all_forecast_rows)

    # Store into SQLite
    print(f"[SQLITE] Writing {len(eval_df)} eval rows and {len(forecast_df)} forecast rows...")
    conn.execute("DELETE FROM forecast_eval")
    conn.execute("DELETE FROM forecasts")
    conn.commit()

    eval_df.to_sql("forecast_eval", conn, if_exists="append", index=False)
    forecast_df.to_sql("forecasts", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("=" * 60)
    print("HONEST FORECAST EVALUATION METRICS TABLE (LAST 7 DAYS HOLDOUT)")
    print("=" * 60)
    print(eval_df.to_string(index=False))
    print("=" * 60)

    return eval_df


if __name__ == "__main__":
    run_forecast_pipeline()
