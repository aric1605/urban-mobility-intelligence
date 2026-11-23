-- Schema for Urban Mobility & Traffic Intelligence System
-- SQLite 3 database: data/urban.db

CREATE TABLE IF NOT EXISTS zones (
    zone_id INTEGER PRIMARY KEY,
    zone_name TEXT NOT NULL,
    borough TEXT NOT NULL,
    centroid_lat REAL,
    centroid_lon REAL,
    geometry_geojson TEXT
);

CREATE TABLE IF NOT EXISTS zone_baseline (
    zone_id INTEGER PRIMARY KEY,
    free_flow_speed REAL NOT NULL,
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

CREATE TABLE IF NOT EXISTS zone_hourly (
    zone_id INTEGER NOT NULL,
    hour_ts TEXT NOT NULL,
    trips INTEGER NOT NULL,
    avg_speed REAL NOT NULL,
    median_speed REAL NOT NULL,
    p10_speed REAL NOT NULL,
    congestion_index REAL,
    PRIMARY KEY (zone_id, hour_ts),
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

CREATE TABLE IF NOT EXISTS city_hourly (
    hour_ts TEXT PRIMARY KEY,
    trips INTEGER NOT NULL,
    avg_speed REAL NOT NULL,
    congestion_index REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS crashes (
    crash_id TEXT PRIMARY KEY,
    crash_ts TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    zone_id INTEGER,
    borough TEXT,
    injured INTEGER DEFAULT 0,
    killed INTEGER DEFAULT 0,
    factor TEXT,
    FOREIGN KEY (zone_id) REFERENCES zones(zone_id)
);

CREATE TABLE IF NOT EXISTS forecasts (
    zone_id INTEGER NOT NULL,
    hour_ts TEXT NOT NULL,
    actual_congestion REAL,
    forecast_congestion REAL NOT NULL,
    lower_ci REAL,
    upper_ci REAL,
    is_anomaly INTEGER DEFAULT 0,
    is_future INTEGER DEFAULT 0,
    PRIMARY KEY (zone_id, hour_ts)
);

CREATE TABLE IF NOT EXISTS forecast_eval (
    zone_id INTEGER PRIMARY KEY,
    zone_name TEXT NOT NULL,
    model_mae REAL NOT NULL,
    model_smape REAL NOT NULL,
    baseline_mae REAL NOT NULL,
    baseline_smape REAL NOT NULL,
    winner TEXT NOT NULL
);

-- Common filter indexes
CREATE INDEX IF NOT EXISTS idx_zone_hourly_ts ON zone_hourly(hour_ts);
CREATE INDEX IF NOT EXISTS idx_zone_hourly_zone ON zone_hourly(zone_id);
CREATE INDEX IF NOT EXISTS idx_crashes_zone ON crashes(zone_id);
CREATE INDEX IF NOT EXISTS idx_crashes_ts ON crashes(crash_ts);
CREATE INDEX IF NOT EXISTS idx_forecasts_zone_ts ON forecasts(zone_id, hour_ts);
