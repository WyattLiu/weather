#!/usr/bin/env python3
"""
Fetch historical GFS data from AWS Open Data.
AWS has GFS data from 2021 to present.

Source: https://noaa-gfs-bdp-pds.s3.amazonaws.com/
"""

import os
import sys
import requests
from datetime import datetime, timedelta
import time

# Configuration
OUTPUT_DIR = "/home/wyatt/weather"
AWS_BASE_URL = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"

# US Continental bounding box
US_LAT_MIN, US_LAT_MAX = 25, 50
US_LON_MIN, US_LON_MAX = 235, 295  # 0-360 format

# For historical, just get daily (00z) and fewer forecast hours to save space
FORECAST_HOURS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384]


def download_gfs_from_aws(date, cycle='00', forecast_hour=0):
    """
    Download GFS data from AWS S3.
    File pattern: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF
    """
    date_str = date.strftime("%Y%m%d")

    # AWS path structure
    file_name = f"gfs.t{cycle}z.pgrb2.0p25.f{forecast_hour:03d}"
    url = f"{AWS_BASE_URL}/gfs.{date_str}/{cycle}/atmos/{file_name}"

    try:
        response = requests.get(url, timeout=120, stream=True)
        if response.status_code == 200:
            return response.content
        else:
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def fetch_historical_day(date, cycle='00'):
    """Fetch all forecast hours for a single day."""
    date_str = date.strftime("%Y%m%d")
    output_file = os.path.join(OUTPUT_DIR, f"gfs_historical_{date_str}_{cycle}z.grib2")

    # Skip if already exists
    if os.path.exists(output_file):
        print(f"  Already exists: {output_file}")
        return output_file

    print(f"  Fetching {date_str} {cycle}z...")

    all_data = b''
    successful = 0

    for fhr in FORECAST_HOURS:
        data = download_gfs_from_aws(date, cycle, fhr)
        if data and len(data) > 1000:
            all_data += data
            successful += 1
            print(f"    f{fhr:03d}: OK", end=' ', flush=True)
        else:
            print(f"    f{fhr:03d}: --", end=' ', flush=True)

        # Rate limiting
        time.sleep(0.3)

    print()

    if successful > 0:
        with open(output_file, 'wb') as f:
            f.write(all_data)
        print(f"    Saved: {output_file} ({successful}/{len(FORECAST_HOURS)} hours)")
        return output_file

    return None


def fetch_date_range(start_date, end_date, cycle='00'):
    """Fetch GFS data for a date range."""
    print(f"Fetching GFS data from {start_date} to {end_date}")
    print(f"Cycle: {cycle}z")
    print()

    current = start_date
    results = []

    while current <= end_date:
        result = fetch_historical_day(current, cycle)
        if result:
            results.append(result)
        current += timedelta(days=1)

    print(f"\nDownloaded {len(results)} days of GFS data")
    return results


def main():
    """Main entry point."""
    print("=" * 60)
    print("GFS Historical Data Fetcher (AWS Open Data)")
    print("=" * 60)
    print()

    # Default: fetch last 30 days
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Check if it looks like a date (8 digits)
        if len(sys.argv[1]) == 8 and sys.argv[1].isdigit():
            # Date range: YYYYMMDD YYYYMMDD
            start_date = datetime.strptime(sys.argv[1], "%Y%m%d").date()
            if len(sys.argv) >= 3:
                end_date = datetime.strptime(sys.argv[2], "%Y%m%d").date()
        else:
            # Number of days back
            try:
                days_back = int(sys.argv[1])
                start_date = end_date - timedelta(days=days_back)
            except ValueError:
                print(f"Invalid argument: {sys.argv[1]}")
                print("Usage: fetch_gfs_historical.py [DAYS_BACK | START_DATE END_DATE]")
                sys.exit(1)

    print(f"Date range: {start_date} to {end_date}")
    print(f"Days: {(end_date - start_date).days + 1}")
    print()

    results = fetch_date_range(start_date, end_date)

    if results:
        print("\nDone! Run gfs_hdd_history.py to analyze.")
    else:
        print("\nNo data downloaded.")
        sys.exit(1)


if __name__ == "__main__":
    main()
