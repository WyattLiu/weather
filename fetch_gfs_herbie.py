#!/usr/bin/env python3
"""
Fetch GFS historical data using Herbie.
Herbie searches multiple sources: AWS, Google, Azure, NOMADS, NCAR RDA.
"""

import os
import sys
from datetime import datetime, timedelta
from herbie import Herbie
import xarray as xr

OUTPUT_DIR = "/home/wyatt/weather"

# US bounding box
US_LAT_MIN, US_LAT_MAX = 25, 50
US_LON_MIN, US_LON_MAX = 235, 295  # 0-360 format

# Forecast hours for daily temps (every 24h out to 16 days)
FORECAST_HOURS = list(range(0, 385, 24))  # 0, 24, 48, ... 384


def fetch_gfs_day(date, cycle='00'):
    """Fetch GFS forecast for a single day using Herbie."""
    date_str = date.strftime("%Y%m%d")
    output_file = os.path.join(OUTPUT_DIR, f"gfs_{date_str}_{cycle}z.grib2")

    # Skip if already exists
    if os.path.exists(output_file):
        print(f"  Already exists: {output_file}")
        return output_file

    print(f"  Fetching {date_str} {cycle}z...")

    try:
        # Get 2m temperature for each forecast hour
        all_data = []
        for fxx in FORECAST_HOURS:
            try:
                H = Herbie(
                    date.strftime("%Y-%m-%d"),
                    model='gfs',
                    fxx=fxx,
                    product='pgrb2.0p25'
                )

                # Download just TMP:2 m (2-meter temperature)
                ds = H.xarray(":TMP:2 m above ground:", remove_grib=True)

                if ds is not None:
                    all_data.append(ds)
                    print(f"    f{fxx:03d}: OK", end=' ', flush=True)
                else:
                    print(f"    f{fxx:03d}: --", end=' ', flush=True)

            except Exception as e:
                print(f"    f{fxx:03d}: ERR", end=' ', flush=True)
                continue

        print()

        if all_data:
            # Combine all forecast hours
            combined = xr.concat(all_data, dim='step')

            # Filter to US region
            lon_name = 'longitude' if 'longitude' in combined.coords else 'lon'
            lat_name = 'latitude' if 'latitude' in combined.coords else 'lat'

            combined_us = combined.where(
                (combined[lat_name] >= US_LAT_MIN) & (combined[lat_name] <= US_LAT_MAX) &
                (combined[lon_name] >= US_LON_MIN) & (combined[lon_name] <= US_LON_MAX),
                drop=True
            )

            # Save as GRIB2
            combined_us.to_netcdf(output_file.replace('.grib2', '.nc'))
            print(f"    Saved: {output_file.replace('.grib2', '.nc')} ({len(all_data)} hours)")
            return output_file.replace('.grib2', '.nc')

    except Exception as e:
        print(f"    Error: {e}")
        return None

    return None


def fetch_date_range(start_date, end_date, cycles=['00', '12']):
    """Fetch GFS for date range."""
    print(f"Fetching GFS data from {start_date} to {end_date}")
    print(f"Cycles: {cycles}")
    print(f"Sources: AWS, Google, Azure, NOMADS, NCAR RDA")
    print()

    current = start_date
    results = []

    while current <= end_date:
        for cycle in cycles:
            result = fetch_gfs_day(current, cycle)
            if result:
                results.append(result)
        current += timedelta(days=1)

    print(f"\nDownloaded {len(results)} GFS forecasts")
    return results


def main():
    print("=" * 60)
    print("GFS Historical Data Fetcher (Herbie - Multi-Source)")
    print("=" * 60)
    print()

    # Default: fetch last 30 days
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=30)

    # Parse command line
    if len(sys.argv) > 1:
        if len(sys.argv[1]) == 8 and sys.argv[1].isdigit():
            start_date = datetime.strptime(sys.argv[1], "%Y%m%d").date()
            if len(sys.argv) >= 3:
                end_date = datetime.strptime(sys.argv[2], "%Y%m%d").date()
        else:
            try:
                days_back = int(sys.argv[1])
                start_date = end_date - timedelta(days=days_back)
            except ValueError:
                print(f"Usage: {sys.argv[0]} [DAYS_BACK | START_DATE END_DATE]")
                sys.exit(1)

    print(f"Date range: {start_date} to {end_date}")
    print(f"Days: {(end_date - start_date).days + 1}")
    print()

    results = fetch_date_range(start_date, end_date)

    if results:
        print("\nDone!")
    else:
        print("\nNo data downloaded.")


if __name__ == "__main__":
    main()
