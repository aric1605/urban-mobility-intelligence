"""Database access layer and query execution for Urban Mobility & Traffic Intelligence System."""

from pathlib import Path
import sqlite3
from typing import Any, Dict, Optional
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = ROOT_DIR / "data" / "urban.db"
SQL_DIR = ROOT_DIR / "sql" / "queries"


def get_connection(db_path: Path = DB_PATH, read_only: bool = True) -> sqlite3.Connection:
    """Obtain a SQLite database connection, read-only by default for safe concurrent access."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Run pipeline first.")
    if read_only:
        uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _load_query(filename: str) -> str:
    """Read a query file from sql/queries/."""
    file_path = SQL_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Query file {file_path} does not exist.")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def query_top_hotspots(
    conn: sqlite3.Connection,
    start_hour: int = 0,
    end_hour: int = 23,
    min_trips: int = 100,
) -> pd.DataFrame:
    """Rank zones by mean congestion index within an hour range."""
    sql = _load_query("top_hotspots.sql")
    params = {"start_hour": start_hour, "end_hour": end_hour, "min_trips": min_trips}
    return pd.read_sql_query(sql, conn, params=params)


def query_peak_hours(conn: sqlite3.Connection, zone_id: int = -1) -> pd.DataFrame:
    """Get hour-of-day x day-of-week mean congestion and worst hour rankings."""
    sql = _load_query("peak_hours.sql")
    params = {"zone_id": zone_id}
    return pd.read_sql_query(sql, conn, params=params)


def query_rolling_7day_avg(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute 7-day (168-hour) rolling average of city-wide congestion and speed."""
    sql = _load_query("rolling_7day_avg.sql")
    return pd.read_sql_query(sql, conn)


def query_crash_density(conn: sqlite3.Connection, min_crashes: int = 5) -> pd.DataFrame:
    """Aggregate crashes and casualties by zone with rankings."""
    sql = _load_query("crash_density_by_zone_and_hour.sql")
    params = {"min_crashes": min_crashes}
    return pd.read_sql_query(sql, conn, params=params)


def query_zones_getting_worse(
    conn: sqlite3.Connection,
    midpoint_ts: str = "2026-09-01 00:00:00",
    min_trips: int = 100,
) -> pd.DataFrame:
    """Compare first-half vs second-half mean congestion per zone."""
    sql = _load_query("zones_getting_worse.sql")
    params = {"midpoint_ts": midpoint_ts, "min_trips": min_trips}
    return pd.read_sql_query(sql, conn, params=params)


def get_kpis(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Retrieve key computed summary metrics across the dataset."""
    total_trips = conn.execute("SELECT SUM(trips) FROM city_hourly").fetchone()[0] or 0
    city_mean_cong = conn.execute("SELECT ROUND(AVG(congestion_index), 4) FROM city_hourly").fetchone()[0] or 0.0
    crash_count = conn.execute("SELECT COUNT(*) FROM crashes").fetchone()[0] or 0
    injured_count = conn.execute("SELECT SUM(injured) FROM crashes").fetchone()[0] or 0
    killed_count = conn.execute("SELECT SUM(killed) FROM crashes").fetchone()[0] or 0

    worst_zone_row = conn.execute(
        """
        SELECT z.zone_name, z.borough, ROUND(AVG(zh.congestion_index), 4) as cong
        FROM zone_hourly zh
        JOIN zones z ON zh.zone_id = z.zone_id
        WHERE zh.congestion_index IS NOT NULL
        GROUP BY zh.zone_id
        HAVING SUM(zh.trips) >= 500
        ORDER BY cong DESC LIMIT 1
        """
    ).fetchone()
    worst_zone = f"{worst_zone_row['zone_name']} ({worst_zone_row['borough']}) - {worst_zone_row['cong']:.1%}" if worst_zone_row else "N/A"

    worst_hour_row = conn.execute(
        """
        SELECT CAST(strftime('%H', hour_ts) AS INTEGER) as hr, ROUND(AVG(congestion_index), 4) as cong
        FROM city_hourly
        GROUP BY hr
        ORDER BY cong DESC LIMIT 1
        """
    ).fetchone()
    worst_hour = f"{worst_hour_row['hr']:02d}:00 ({worst_hour_row['cong']:.1%})" if worst_hour_row else "N/A"

    return {
        "total_trips": int(total_trips),
        "city_mean_congestion": float(city_mean_cong),
        "total_crashes": int(crash_count),
        "total_injured": int(injured_count),
        "total_killed": int(killed_count),
        "worst_zone": worst_zone,
        "worst_hour": worst_hour,
    }


def get_all_zones(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch zone metadata."""
    return pd.read_sql_query(
        "SELECT zone_id, zone_name, borough, centroid_lat, centroid_lon FROM zones ORDER BY borough, zone_name",
        conn,
    )


def get_top_zones(conn: sqlite3.Connection, limit: int = 20) -> pd.DataFrame:
    """Fetch the top N busiest zones by total trips."""
    sql = """
    SELECT zh.zone_id, z.zone_name, z.borough, SUM(zh.trips) as total_trips
    FROM zone_hourly zh
    JOIN zones z ON zh.zone_id = z.zone_id
    GROUP BY zh.zone_id, z.zone_name, z.borough
    ORDER BY total_trips DESC
    LIMIT :limit
    """
    return pd.read_sql_query(sql, conn, params={"limit": limit})
