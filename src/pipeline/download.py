"""Data acquisition module for NYC TLC Yellow Taxi, Taxi Zones, and Crashes."""

from pathlib import Path
import time
from typing import List, Tuple
import requests


RAW_DATA_DIR = Path("data/raw")
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
TAXI_ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
CRASHES_BASE_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.csv"


def download_file(url: str, dest_path: Path, max_attempts: int = 2, timeout: int = 300) -> Path:
    """Download a file with retry logic and caching to disk."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[CACHE] Using existing file: {dest_path} ({dest_path.stat().st_size:,} bytes)")
        return dest_path

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[DOWNLOAD] Attempt {attempt}/{max_attempts}: {url} -> {dest_path}")
            start_time = time.time()
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            elapsed = time.time() - start_time
            file_size = dest_path.stat().st_size
            print(f"[DONE] Downloaded {file_size:,} bytes in {elapsed:.1f}s")
            return dest_path
        except Exception as e:
            last_error = e
            print(f"[ERROR] Attempt {attempt} failed: {e}")
            if dest_path.exists():
                dest_path.unlink()
            if attempt < max_attempts:
                time.sleep(2)

    raise RuntimeError(f"Failed to download {url} after {max_attempts} attempts: {last_error}")


def find_consecutive_months(num_months: int = 2, max_check_attempts: int = 3) -> List[Tuple[int, int]]:
    """Probe CloudFront to find the newest consecutive available months."""
    print("[PROBE] Checking newest available TLC Yellow Taxi months...")
    candidate_months: List[Tuple[int, int]] = []
    # Probe from late 2026 downwards
    years_to_check = [2026, 2025, 2024]
    consecutive_found: List[Tuple[int, int]] = []

    for year in years_to_check:
        for month in range(12, 0, -1):
            url = TLC_BASE_URL.format(year=year, month=month)
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    candidate_months.append((year, month))
                    if len(candidate_months) >= 2:
                        # Check if the last two are consecutive
                        m1_y, m1_m = candidate_months[-2]
                        m2_y, m2_m = candidate_months[-1]
                        # m2 is previous month of m1
                        expected_prev_y = m1_y if m1_m > 1 else m1_y - 1
                        expected_prev_m = m1_m - 1 if m1_m > 1 else 12
                        if (m2_y, m2_m) == (expected_prev_y, expected_prev_m):
                            consecutive_found = [(m2_y, m2_m), (m1_y, m1_m)]  # chronological order
                            print(f"[PROBE] Found 2 consecutive months: {consecutive_found[0]} and {consecutive_found[1]}")
                            return consecutive_found
            except Exception as e:
                print(f"[PROBE] Check failed for {year}-{month:02d}: {e}")
                continue

    if not consecutive_found:
        raise RuntimeError("Could not find 2 consecutive available TLC parquet files.")
    return consecutive_found


def download_taxi_data(months: List[Tuple[int, int]], dest_dir: Path = RAW_DATA_DIR) -> List[Path]:
    """Download yellow taxi trip records for specified months."""
    paths: List[Path] = []
    for y, m in months:
        url = TLC_BASE_URL.format(year=y, month=m)
        dest_path = dest_dir / f"yellow_tripdata_{y}-{m:02d}.parquet"
        downloaded = download_file(url, dest_path, max_attempts=2, timeout=300)
        paths.append(downloaded)
    return paths


def download_taxi_zones(dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download the official NYC taxi zones shapefile zip."""
    dest_path = dest_dir / "taxi_zones.zip"
    return download_file(TAXI_ZONES_URL, dest_path, max_attempts=2, timeout=300)


def download_crashes(start_date: str, end_date: str, dest_dir: Path = RAW_DATA_DIR) -> Path:
    """Download NYC Motor Vehicle Collisions CSV for the given date range (YYYY-MM-DD)."""
    dest_path = dest_dir / f"crashes_{start_date}_{end_date}.csv"
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[CACHE] Using existing crash data: {dest_path}")
        return dest_path

    # Construct Socrata URL query
    # Date in Socrata formatted e.g. 2026-08-01T00:00:00
    params = {
        "$select": "collision_id,crash_date,crash_time,latitude,longitude,borough,number_of_persons_injured,number_of_persons_killed,contributing_factor_vehicle_1",
        "$where": f"latitude IS NOT NULL AND longitude IS NOT NULL AND crash_date >= '{start_date}T00:00:00' AND crash_date <= '{end_date}T23:59:59'",
        "$limit": "50000",
    }
    encoded_url = requests.Request("GET", CRASHES_BASE_URL, params=params).prepare().url

    return download_file(encoded_url, dest_path, max_attempts=2, timeout=300)
