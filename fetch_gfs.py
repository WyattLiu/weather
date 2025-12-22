#!/usr/bin/env python3
"""
Fetch GFS (Global Forecast System) data from NOAA NOMADS.
GFS provides forecasts up to 16 days (384 hours).
Data is available 4 times daily: 00z, 06z, 12z, 18z

Source: https://nomads.ncep.noaa.gov/
"""

import os
import sys
import requests
from datetime import datetime, timedelta
import time

# Configuration
OUTPUT_DIR = "/home/wyatt/weather"
BASE_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"

# US Continental bounding box (same as ECMWF analysis)
US_LAT_MIN, US_LAT_MAX = 25, 50
US_LON_MIN, US_LON_MAX = 235, 295  # 0-360 format: -125 to -65 = 235 to 295

# Forecast hours to download (every 6 hours up to 384h = 16 days)
# Using key hours: 0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384
FORECAST_HOURS = list(range(0, 385, 24))  # Daily snapshots for HDD calculation


def get_latest_gfs_cycle():
    """Determine the latest available GFS cycle."""
    now = datetime.utcnow()

    # GFS cycles: 00, 06, 12, 18 UTC
    # Data typically available ~4-5 hours after cycle time
    cycles = [0, 6, 12, 18]

    for hours_back in range(0, 24, 6):
        check_time = now - timedelta(hours=hours_back)
        cycle = max(c for c in cycles if c <= check_time.hour)
        cycle_time = check_time.replace(hour=cycle, minute=0, second=0, microsecond=0)

        # Check if this cycle is at least 5 hours old (data should be available)
        if (now - cycle_time).total_seconds() >= 5 * 3600:
            return cycle_time

    # Fallback to previous day's 18z
    yesterday = now - timedelta(days=1)
    return yesterday.replace(hour=18, minute=0, second=0, microsecond=0)


def download_gfs_data(cycle_time, forecast_hour):
    """
    Download GFS 2m temperature data for a specific forecast hour.
    Uses NOMADS filter to get only the variables and region we need.
    """
    date_str = cycle_time.strftime("%Y%m%d")
    cycle_str = f"{cycle_time.hour:02d}"

    # Construct the filter URL
    # file: gfs.t{HH}z.pgrb2.0p25.f{FFF}
    file_name = f"gfs.t{cycle_str}z.pgrb2.0p25.f{forecast_hour:03d}"

    params = {
        'file': file_name,
        'lev_2_m_above_ground': 'on',  # 2m level
        'var_TMP': 'on',  # Temperature
        'subregion': '',
        'leftlon': US_LON_MIN,
        'rightlon': US_LON_MAX,
        'toplat': US_LAT_MAX,
        'bottomlat': US_LAT_MIN,
        'dir': f'/gfs.{date_str}/{cycle_str}/atmos'
    }

    # Build URL
    url = BASE_URL + '?' + '&'.join(f"{k}={v}" for k, v in params.items())

    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200 and len(response.content) > 1000:
            return response.content
        else:
            print(f"  Warning: Got status {response.status_code}, size {len(response.content)} bytes")
            return None
    except Exception as e:
        print(f"  Error downloading f{forecast_hour:03d}: {e}")
        return None


def fetch_gfs_forecast(cycle_time=None):
    """Fetch complete GFS forecast for HDD analysis."""
    if cycle_time is None:
        cycle_time = get_latest_gfs_cycle()

    date_str = cycle_time.strftime("%Y%m%d")
    cycle_str = f"{cycle_time.hour:02d}"

    print(f"Fetching GFS forecast: {date_str} {cycle_str}z")
    print(f"Forecast hours: {FORECAST_HOURS[0]} to {FORECAST_HOURS[-1]} ({len(FORECAST_HOURS)} steps)")
    print(f"Region: {US_LAT_MIN}-{US_LAT_MAX}N, {360-US_LON_MAX}W-{360-US_LON_MIN}W")
    print()

    # Output file
    output_file = os.path.join(OUTPUT_DIR, f"gfs_{date_str}_{cycle_str}z.grib2")

    # Check if already exists
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return output_file

    # Download each forecast hour and concatenate
    all_data = b''
    successful_hours = []

    for fhr in FORECAST_HOURS:
        print(f"  Downloading f{fhr:03d}h...", end=' ', flush=True)
        data = download_gfs_data(cycle_time, fhr)

        if data:
            all_data += data
            successful_hours.append(fhr)
            print(f"OK ({len(data)} bytes)")
        else:
            print("FAILED")

        # Rate limiting
        time.sleep(0.5)

    if successful_hours:
        # Save combined file
        with open(output_file, 'wb') as f:
            f.write(all_data)

        print(f"\nSaved: {output_file}")
        print(f"Successfully downloaded {len(successful_hours)}/{len(FORECAST_HOURS)} forecast hours")
        print(f"Forecast range: {successful_hours[0]}h to {successful_hours[-1]}h = {successful_hours[-1]/24:.0f} days")
        return output_file
    else:
        print("\nNo data downloaded!")
        return None


def main():
    """Main entry point."""
    print("=" * 60)
    print("GFS Forecast Fetcher (NOAA NOMADS)")
    print("16-day forecast for US Continental HDD Analysis")
    print("=" * 60)
    print()

    # Get specific cycle if provided
    cycle_time = None
    if len(sys.argv) > 1:
        try:
            # Parse YYYYMMDD_HH format
            arg = sys.argv[1]
            if '_' in arg:
                date_part, hour_part = arg.split('_')
                cycle_time = datetime.strptime(date_part, "%Y%m%d")
                cycle_time = cycle_time.replace(hour=int(hour_part))
            else:
                cycle_time = datetime.strptime(arg, "%Y%m%d")
        except ValueError:
            print(f"Invalid date format: {sys.argv[1]}")
            print("Use: YYYYMMDD or YYYYMMDD_HH")
            sys.exit(1)

    result = fetch_gfs_forecast(cycle_time)

    if result:
        print("\nDone!")
        sys.exit(0)
    else:
        print("\nFailed to fetch GFS data")
        sys.exit(1)


if __name__ == "__main__":
    main()
