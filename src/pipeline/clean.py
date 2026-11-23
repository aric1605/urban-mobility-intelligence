"""Cleaning and aggregation routines for taxi trips, baselines, and spatial joins."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import pyarrow.parquet as pq


REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "trip_distance",
    "passenger_count",
]


def clean_trip_records(
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Filter raw taxi trip records according to pipeline rules:
    - Drop nulls in required columns
    - Pickup datetime strictly within [start_ts, end_ts]
    - Duration between 1 and 180 minutes
    - Distance > 0.1 miles and <= 100 miles
    - Implied speed > 0 and <= 80 mph
    """
    if df.empty:
        return pd.DataFrame(columns=["zone_id", "hour_ts", "speed"])

    # Drop nulls
    df = df.dropna(subset=REQUIRED_COLUMNS)

    # Date range filter
    dt_mask = (df["tpep_pickup_datetime"] >= start_ts) & (df["tpep_pickup_datetime"] <= end_ts)
    df = df[dt_mask]
    if df.empty:
        return pd.DataFrame(columns=["zone_id", "hour_ts", "speed"])

    # Duration in minutes
    duration_secs = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()
    valid_dur = (duration_secs >= 60.0) & (duration_secs <= 180.0 * 60.0)

    # Distance filter
    valid_dist = (df["trip_distance"] > 0.1) & (df["trip_distance"] <= 100.0)

    df_filtered = df[valid_dur & valid_dist].copy()
    if df_filtered.empty:
        return pd.DataFrame(columns=["zone_id", "hour_ts", "speed"])

    duration_hours = (df_filtered["tpep_dropoff_datetime"] - df_filtered["tpep_pickup_datetime"]).dt.total_seconds() / 3600.0
    speeds = df_filtered["trip_distance"] / duration_hours
    valid_speed = (speeds > 0.0) & (speeds <= 80.0)

    df_clean = pd.DataFrame({
        "zone_id": df_filtered.loc[valid_speed, "PULocationID"].astype("int16"),
        "hour_ts": df_filtered.loc[valid_speed, "tpep_pickup_datetime"].dt.floor("h"),
        "speed": speeds[valid_speed].astype("float32"),
    })
    return df_clean


def process_parquet_file(
    file_path: Path,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    chunk_size: int = 250_000,
) -> pd.DataFrame:
    """Read parquet in row groups / batches to keep memory bounded, returning cleaned trips."""
    pq_file = pq.ParquetFile(file_path)
    total_rows = pq_file.metadata.num_rows
    print(f"[PROCESS] Reading {file_path.name} ({total_rows:,} total rows) in chunks of {chunk_size:,}...")

    cleaned_chunks: List[pd.DataFrame] = []
    processed_count = 0

    for batch in pq_file.iter_batches(batch_size=chunk_size, columns=REQUIRED_COLUMNS):
        batch_df = batch.to_pandas()
        clean_df = clean_trip_records(batch_df, start_ts, end_ts)
        if not clean_df.empty:
            cleaned_chunks.append(clean_df)
        processed_count += len(batch_df)
        del batch_df
        gc.collect()

    if not cleaned_chunks:
        return pd.DataFrame(columns=["zone_id", "hour_ts", "speed"])

    combined = pd.concat(cleaned_chunks, ignore_index=True)
    del cleaned_chunks
    gc.collect()
    print(f"[PROCESS] Finished {file_path.name}: retained {len(combined):,} valid trips from {processed_count:,} rows")
    return combined


def compute_hourly_aggregates(trips_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute:
    1. zone_baseline: free_flow_speed per zone (90th percentile of avg speeds in off-peak 22:00-05:00)
    2. zone_hourly: trips, avg_speed, median_speed, p10_speed, congestion_index
    3. city_hourly: trips, avg_speed, congestion_index
    """
    print("[AGG] Computing (zone, hour) speed metrics...")
    # Group by (zone_id, hour_ts)
    grouped = trips_df.groupby(["zone_id", "hour_ts"])["speed"]
    agg_df = grouped.agg(
        trips="count",
        avg_speed="mean",
        median_speed="median",
        p10_speed=lambda s: s.quantile(0.10),
    ).reset_index()

    agg_df["avg_speed"] = agg_df["avg_speed"].round(2)
    agg_df["median_speed"] = agg_df["median_speed"].round(2)
    agg_df["p10_speed"] = agg_df["p10_speed"].round(2)

    # Determine free_flow_speed per zone
    # Off-peak hours: 22, 23, 0, 1, 2, 3, 4, 5
    print("[AGG] Calculating free_flow_speed per zone (off-peak 22:00-05:00 90th percentile)...")
    hour_nums = agg_df["hour_ts"].dt.hour
    off_peak_mask = hour_nums.isin([22, 23, 0, 1, 2, 3, 4, 5])
    off_peak_df = agg_df[off_peak_mask]

    # Global fallback off-peak 90th percentile
    city_offpeak_ffs = float(off_peak_df["avg_speed"].quantile(0.90)) if not off_peak_df.empty else 20.0

    zone_ffs = off_peak_df.groupby("zone_id")["avg_speed"].quantile(0.90).to_dict()

    # Create baseline DataFrame for all observed zones
    all_zones = agg_df["zone_id"].unique()
    baseline_records = []
    for zid in all_zones:
        ffs = zone_ffs.get(zid)
        if ffs is None or np.isnan(ffs) or ffs <= 0:
            # Fallback to zone's own 90th percentile, or city-wide
            zone_speeds = agg_df.loc[agg_df["zone_id"] == zid, "avg_speed"]
            ffs = float(zone_speeds.quantile(0.90)) if not zone_speeds.empty else city_offpeak_ffs
        if ffs <= 0:
            ffs = city_offpeak_ffs
        baseline_records.append({"zone_id": int(zid), "free_flow_speed": round(float(ffs), 2)})

    baseline_df = pd.DataFrame(baseline_records)
    ffs_map = baseline_df.set_index("zone_id")["free_flow_speed"].to_dict()

    # Compute congestion_index: clip(1 - avg_speed / free_flow_speed, 0, 1) where trips >= 10 else None
    print("[AGG] Computing congestion_index (trips >= 10 threshold)...")
    zone_ffs_series = agg_df["zone_id"].map(ffs_map)
    raw_congestion = 1.0 - (agg_df["avg_speed"] / zone_ffs_series)
    clipped_congestion = np.clip(raw_congestion, 0.0, 1.0).round(4)

    agg_df["congestion_index"] = np.where(agg_df["trips"] >= 10, clipped_congestion, np.nan)

    # Format hour_ts as ISO string
    agg_df["hour_ts"] = agg_df["hour_ts"].dt.strftime("%Y-%m-%d %H:00:00")

    # City-wide hourly aggregate
    print("[AGG] Computing city-wide hourly series...")
    city_grouped = agg_df.groupby("hour_ts")
    city_hourly = city_grouped.apply(
        lambda g: pd.Series({
            "trips": int(g["trips"].sum()),
            "avg_speed": round(float((g["avg_speed"] * g["trips"]).sum() / g["trips"].sum()), 2),
            "congestion_index": round(float(g["congestion_index"].dropna().mean()), 4) if not g["congestion_index"].dropna().empty else 0.0,
        }),
        include_groups=False,
    ).reset_index()

    return baseline_df, agg_df, city_hourly


def spatial_join_crashes(
    crashes_path: Path,
    zones_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Perform spatial join between collision coordinates and TLC taxi zone polygons."""
    print(f"[SPATIAL] Loading crashes from {crashes_path}...")
    df = pd.read_csv(crashes_path)
    df = df.dropna(subset=["latitude", "longitude", "crash_date", "crash_time"])

    # Convert coordinates and filter NYC bounding box
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    valid_coords = (
        (df["latitude"] >= 40.40) & (df["latitude"] <= 41.10) &
        (df["longitude"] >= -74.40) & (df["longitude"] <= -73.60)
    )
    df = df[valid_coords].copy()

    # Create GeoDataFrame
    points = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    crashes_gdf = gpd.GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    # Spatial join
    print(f"[SPATIAL] Joining {len(crashes_gdf):,} crashes with {len(zones_gdf)} zones...")
    zones_subset = zones_gdf[["LocationID", "geometry"]].rename(columns={"LocationID": "zone_id"})
    joined = gpd.sjoin(crashes_gdf, zones_subset, how="left", predicate="within")

    # Build clean output DataFrame
    # Format crash_ts
    crash_time_cleaned = joined["crash_time"].astype(str).str.strip()
    crash_date_cleaned = pd.to_datetime(joined["crash_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    crash_ts = crash_date_cleaned + " " + crash_time_cleaned + ":00"

    out_df = pd.DataFrame({
        "crash_id": joined["collision_id"].astype(str),
        "crash_ts": crash_ts,
        "lat": joined["latitude"].round(6),
        "lon": joined["longitude"].round(6),
        "zone_id": joined["zone_id"].fillna(-1).astype(int),
        "borough": joined["borough"].fillna("Unknown").astype(str),
        "injured": pd.to_numeric(joined["number_of_persons_injured"], errors="coerce").fillna(0).astype(int),
        "killed": pd.to_numeric(joined["number_of_persons_killed"], errors="coerce").fillna(0).astype(int),
        "factor": joined["contributing_factor_vehicle_1"].fillna("Unspecified").astype(str),
    })

    # Drop any remaining null crash_ts
    out_df = out_df.dropna(subset=["crash_id", "crash_ts"])
    print(f"[SPATIAL] Mapped {len(out_df):,} crashes. Zone matched: {(out_df['zone_id'] > 0).sum():,}")
    return out_df
