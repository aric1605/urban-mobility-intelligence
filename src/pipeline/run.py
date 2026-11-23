"""Main pipeline orchestrator to download, clean, aggregate, and store urban mobility data."""

from datetime import datetime
import os
from pathlib import Path
import sqlite3
import time
import zipfile
import geopandas as gpd
import pandas as pd

from src.pipeline.download import (
    find_consecutive_months,
    download_taxi_data,
    download_taxi_zones,
    download_crashes,
)
from src.pipeline.clean import (
    process_parquet_file,
    compute_hourly_aggregates,
    spatial_join_crashes,
)


DB_PATH = Path("data/urban.db")
SCHEMA_PATH = Path("sql/schema.sql")
GEOJSON_PATH = Path("data/taxi_zones.geojson")


def build_pipeline() -> None:
    """Run the complete end-to-end data pipeline."""
    start_wall_time = time.time()
    print("=" * 60)
    print("STARTING URBAN MOBILITY DATA PIPELINE")
    print("=" * 60)

    # 1. Identify 2 consecutive months
    months = find_consecutive_months(num_months=2)
    (y1, m1), (y2, m2) = months[0], months[1]
    start_date_str = f"{y1}-{m1:02d}-01"
    # End date: last day of month 2
    if m2 == 12:
        end_date_str = f"{y2}-12-31"
    else:
        # Use pandas Period to get end of month
        end_date_str = pd.Period(f"{y2}-{m2:02d}").end_time.strftime("%Y-%m-%d")

    start_ts = pd.Timestamp(f"{start_date_str} 00:00:00")
    end_ts = pd.Timestamp(f"{end_date_str} 23:59:59")
    print(f"[PIPELINE] Target Date Window: {start_date_str} to {end_date_str}")

    # 2. Download taxi data
    taxi_paths = download_taxi_data(months)

    # 3. Download & prepare taxi zones
    zones_zip_path = download_taxi_zones()
    extract_dir = Path("data/raw/taxi_zones")
    if not (extract_dir / "taxi_zones" / "taxi_zones.shp").exists():
        print("[PIPELINE] Extracting taxi_zones.zip...")
        with zipfile.ZipFile(zones_zip_path) as z:
            z.extractall(extract_dir)

    shp_path = extract_dir / "taxi_zones" / "taxi_zones.shp"
    print(f"[PIPELINE] Loading zones shapefile: {shp_path}")
    zones_gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
    zones_gdf["LocationID"] = zones_gdf["LocationID"].astype(int)
    zones_gdf["centroid_lat"] = zones_gdf.geometry.centroid.y.round(6)
    zones_gdf["centroid_lon"] = zones_gdf.geometry.centroid.x.round(6)

    # Export simplified GeoJSON for fast map rendering in Streamlit
    simplified_gdf = zones_gdf.copy()
    simplified_gdf["geometry"] = simplified_gdf["geometry"].simplify(tolerance=0.0003, preserve_topology=True)
    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    simplified_gdf[["LocationID", "zone", "borough", "centroid_lat", "centroid_lon", "geometry"]].to_file(
        GEOJSON_PATH, driver="GeoJSON"
    )
    print(f"[PIPELINE] Saved simplified GeoJSON to {GEOJSON_PATH} ({GEOJSON_PATH.stat().st_size:,} bytes)")

    # 4. Download crashes
    crashes_csv_path = download_crashes(start_date_str, end_date_str)

    # 5. Clean & process taxi parquet files
    all_clean_trips = []
    for p in taxi_paths:
        clean_df = process_parquet_file(p, start_ts, end_ts, chunk_size=300_000)
        all_clean_trips.append(clean_df)

    combined_trips = pd.concat(all_clean_trips, ignore_index=True)
    del all_clean_trips
    total_trips = len(combined_trips)
    print(f"[PIPELINE] Combined cleaned trips across 2 months: {total_trips:,}")

    # 6. Hourly Aggregations
    baseline_df, zone_hourly_df, city_hourly_df = compute_hourly_aggregates(combined_trips)
    del combined_trips

    # 7. Spatial join on crashes
    crashes_df = spatial_join_crashes(crashes_csv_path, zones_gdf)

    # Prepare zones table
    zones_table = pd.DataFrame({
        "zone_id": zones_gdf["LocationID"],
        "zone_name": zones_gdf["zone"].astype(str),
        "borough": zones_gdf["borough"].astype(str),
        "centroid_lat": zones_gdf["centroid_lat"],
        "centroid_lon": zones_gdf["centroid_lon"],
        "geometry_geojson": None,
    })

    # 8. Initialize SQLite Database
    print(f"[SQLITE] Creating database at {DB_PATH}...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)

    # 9. Load tables into SQLite
    print("[SQLITE] Inserting zones...")
    zones_table.to_sql("zones", conn, if_exists="append", index=False)

    print("[SQLITE] Inserting zone baselines...")
    baseline_df.to_sql("zone_baseline", conn, if_exists="append", index=False)

    print(f"[SQLITE] Inserting zone_hourly ({len(zone_hourly_df):,} rows)...")
    zone_hourly_df.to_sql("zone_hourly", conn, if_exists="append", index=False)

    print(f"[SQLITE] Inserting city_hourly ({len(city_hourly_df):,} rows)...")
    city_hourly_df.to_sql("city_hourly", conn, if_exists="append", index=False)

    print(f"[SQLITE] Inserting crashes ({len(crashes_df):,} rows)...")
    crashes_df.to_sql("crashes", conn, if_exists="append", index=False)

    print("[SQLITE] Optimizing database (VACUUM & ANALYZE)...")
    conn.execute("ANALYZE;")
    conn.commit()
    conn.close()

    # VACUUM in separate connection
    conn2 = sqlite3.connect(DB_PATH)
    conn2.execute("VACUUM;")
    conn2.close()

    elapsed = time.time() - start_wall_time
    db_size_mb = DB_PATH.stat().st_size / (1024 * 1024)

    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print(f"Elapsed Time:       {elapsed:.1f}s")
    print(f"Total Trips:        {total_trips:,}")
    print(f"Zones:              {len(zones_table):,}")
    print(f"Zone Hourly Rows:   {len(zone_hourly_df):,}")
    print(f"City Hourly Rows:   {len(city_hourly_df):,}")
    print(f"Crashes Mapped:     {len(crashes_df):,}")
    print(f"SQLite DB Size:     {db_size_mb:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    build_pipeline()
