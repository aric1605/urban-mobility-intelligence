"""Core unit tests with zero-network in-memory fixtures for Urban Mobility System."""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pytest

from src.pipeline.clean import clean_trip_records
from src.forecast.evaluate import calc_mae, calc_smape


def test_cleaning_rules():
    """Verify that cleaning rules correctly drop trips with invalid duration, distance, and speed."""
    base_ts = pd.Timestamp("2026-08-15 12:00:00")
    start_ts = pd.Timestamp("2026-08-01 00:00:00")
    end_ts = pd.Timestamp("2026-08-31 23:59:59")

    # 6-row DataFrame:
    # 0: Valid (15 min, 3.0 mi -> 12 mph)
    # 1: Duration too short (< 1 min)
    # 2: Duration too long (> 180 min)
    # 3: Distance too short (<= 0.1 mi)
    # 4: Distance too long (> 100 mi)
    # 5: Speed too high (> 80 mph)
    df = pd.DataFrame({
        "tpep_pickup_datetime": [base_ts] * 6,
        "tpep_dropoff_datetime": [
            base_ts + pd.Timedelta(minutes=15),
            base_ts + pd.Timedelta(seconds=30),
            base_ts + pd.Timedelta(minutes=200),
            base_ts + pd.Timedelta(minutes=5),
            base_ts + pd.Timedelta(minutes=60),
            base_ts + pd.Timedelta(minutes=2),
        ],
        "PULocationID": [101, 102, 103, 104, 105, 106],
        "trip_distance": [3.0, 1.0, 5.0, 0.05, 150.0, 10.0],
        "passenger_count": [1, 1, 1, 1, 1, 1],
    })

    cleaned = clean_trip_records(df, start_ts, end_ts)
    assert len(cleaned) == 1, f"Expected exactly 1 valid trip, got {len(cleaned)}"
    assert cleaned.iloc[0]["zone_id"] == 101
    assert round(cleaned.iloc[0]["speed"], 1) == 12.0


def test_congestion_index_math():
    """Verify congestion index clipping to [0, 1] and thresholding (trips >= 10)."""
    free_flow_speed = 20.0
    speeds = np.array([10.0, 25.0, 0.0, 10.0])
    trips = np.array([15, 12, 10, 5])

    raw_congestion = 1.0 - (speeds / free_flow_speed)
    clipped = np.clip(raw_congestion, 0.0, 1.0)
    congestion_index = np.where(trips >= 10, clipped, np.nan)

    assert round(congestion_index[0], 2) == 0.50   # 1 - 10/20 = 0.5
    assert round(congestion_index[1], 2) == 0.00   # 1 - 25/20 = -0.25 -> 0.0
    assert round(congestion_index[2], 2) == 1.00   # 1 - 0/20 = 1.0
    assert np.isnan(congestion_index[3])           # trips < 10 -> NaN


def test_sql_top_hotspots_in_memory():
    """Verify top_hotspots query ranking with RANK() OVER against in-memory SQLite fixture."""
    conn = sqlite3.connect(":memory:")

    conn.execute("""
        CREATE TABLE zones (
            zone_id INTEGER PRIMARY KEY,
            zone_name TEXT,
            borough TEXT
        );
    """)
    conn.execute("""
        CREATE TABLE zone_hourly (
            zone_id INTEGER,
            hour_ts TEXT,
            trips INTEGER,
            avg_speed REAL,
            congestion_index REAL
        );
    """)

    conn.execute("INSERT INTO zones VALUES (1, 'Midtown', 'Manhattan'), (2, 'Bushwick', 'Brooklyn');")
    conn.execute("INSERT INTO zone_hourly VALUES (1, '2026-08-01 08:00:00', 50, 6.0, 0.65);")
    conn.execute("INSERT INTO zone_hourly VALUES (1, '2026-08-01 09:00:00', 60, 5.5, 0.70);")
    conn.execute("INSERT INTO zone_hourly VALUES (2, '2026-08-01 08:00:00', 40, 15.0, 0.25);")
    conn.execute("INSERT INTO zone_hourly VALUES (2, '2026-08-01 09:00:00', 40, 14.0, 0.30);")
    conn.commit()

    query_path = Path("sql/queries/top_hotspots.sql")
    with open(query_path, "r", encoding="utf-8") as f:
        sql = f.read()

    res = pd.read_sql_query(sql, conn, params={"start_hour": 8, "end_hour": 9, "min_trips": 50})
    conn.close()

    assert len(res) == 2
    assert res.iloc[0]["zone_name"] == "Midtown"
    assert res.iloc[0]["congestion_rank"] == 1
    assert res.iloc[1]["zone_name"] == "Bushwick"
    assert res.iloc[1]["congestion_rank"] == 2


def test_forecast_evaluation_metrics():
    """Verify MAE and sMAPE evaluation functions against synthetic arrays."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 18.0, 33.0])

    # MAE = (2 + 2 + 3) / 3 = 7/3 = 2.3333...
    mae = calc_mae(actual, pred)
    assert pytest.approx(mae, 0.001) == 7.0 / 3.0

    # sMAPE = mean( 2*|a-p| / (|a| + |p|) ) * 100
    # term1: 2*2 / 22 = 4/22
    # term2: 2*2 / 38 = 4/38
    # term3: 2*3 / 63 = 6/63
    expected_smape = float(np.mean([4.0 / 22.0, 4.0 / 38.0, 6.0 / 63.0]) * 100.0)
    smape = calc_smape(actual, pred)
    assert pytest.approx(smape, 0.01) == expected_smape


def test_spatial_join_toy_polygons():
    """Verify points-in-polygons spatial join mapping between coordinates and zones."""
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    zones_gdf = gpd.GeoDataFrame({
        "LocationID": [10, 20],
        "zone": ["Zone Alpha", "Zone Beta"],
        "geometry": [poly1, poly2],
    }, crs="EPSG:4326")

    p1 = Point(0.5, 0.5)
    p2 = Point(2.5, 2.5)
    points_gdf = gpd.GeoDataFrame({
        "point_id": [1, 2],
        "geometry": [p1, p2],
    }, crs="EPSG:4326")

    joined = gpd.sjoin(points_gdf, zones_gdf, how="left", predicate="within")
    assert len(joined) == 2
    assert joined.loc[joined["point_id"] == 1, "LocationID"].values[0] == 10
    assert joined.loc[joined["point_id"] == 2, "LocationID"].values[0] == 20
