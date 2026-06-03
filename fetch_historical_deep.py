#!/usr/bin/env python3
"""
Fetch deep historical data for long-term fair value analysis.
- GFS: Available from 2021 on AWS
- ECMWF IFS: Available from early 2024 on Google/AWS

We'll fetch weekly samples (every 7 days) to get broad coverage efficiently.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import numpy as np
import pandas as pd
import time

OUTPUT_DIR = "/home/wyatt/weather/historical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# US bounding box
US_LAT_MIN, US_LAT_MAX = 25, 50
BASE_TEMP = 18.3


def fetch_gfs_historical(start_date, end_date, interval_days=7):
    """Fetch GFS data at regular intervals."""
    print(f"\n{'='*60}")
    print(f"Fetching GFS Historical Data")
    print(f"Range: {start_date} to {end_date}")
    print(f"Interval: Every {interval_days} days")
    print(f"{'='*60}\n")

    results = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        output_file = os.path.join(OUTPUT_DIR, f"gfs_hist_{date_str}.csv")

        if os.path.exists(output_file):
            print(f"  {date_str}: Already exists")
            results.append(output_file)
            current += timedelta(days=interval_days)
            continue

        print(f"  {date_str}: Fetching...", end=' ', flush=True)

        try:
            temps_by_day = {}

            # Fetch daily temps for 16 days
            for fxx in range(0, 385, 24):  # 0, 24, 48, ... 384 hours
                try:
                    H = Herbie(
                        current.strftime("%Y-%m-%d"),
                        model='gfs',
                        fxx=fxx,
                        product='pgrb2.0p25'
                    )

                    ds = H.xarray(":TMP:2 m above ground:", remove_grib=True)

                    if ds is not None:
                        # Get temperature variable
                        temp_var = None
                        for var in ds.data_vars:
                            if 't2m' in var.lower() or 't' == var.lower():
                                temp_var = var
                                break

                        if temp_var:
                            temp_data = ds[temp_var]
                            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
                            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

                            lons = ds[lon_name].values
                            if lons.min() < 0:
                                lon_min, lon_max = -125, -65
                            else:
                                lon_min, lon_max = 235, 295

                            temp_us = temp_data.where(
                                (ds[lat_name] >= US_LAT_MIN) & (ds[lat_name] <= US_LAT_MAX) &
                                (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max),
                                drop=True
                            )

                            us_mean_k = float(temp_us.mean().values)
                            us_mean_c = us_mean_k - 273.15
                            day = fxx // 24
                            temps_by_day[day] = us_mean_c

                    time.sleep(0.1)

                except Exception as e:
                    continue

            if temps_by_day:
                # Calculate HDD for each day
                hdds = {day: max(0, BASE_TEMP - temp) for day, temp in temps_by_day.items()}

                # Calculate period HDDs
                days_1_5 = [hdds.get(d, 0) for d in range(1, 6)]
                days_6_10 = [hdds.get(d, 0) for d in range(6, 11)]
                days_8_14 = [hdds.get(d, 0) for d in range(8, 15)]

                with open(output_file, 'w') as f:
                    f.write("date,hdd_1_5d,hdd_6_10d,hdd_8_14d\n")
                    f.write(f"{date_str},{np.mean(days_1_5):.2f},{np.mean(days_6_10):.2f},{np.mean(days_8_14):.2f}\n")

                print(f"OK ({len(temps_by_day)} days)")
                results.append(output_file)
            else:
                print("No data")

        except Exception as e:
            print(f"Error: {e}")

        current += timedelta(days=interval_days)
        time.sleep(0.5)

    return results


def fetch_ifs_historical(start_date, end_date, interval_days=7):
    """Fetch ECMWF IFS data at regular intervals."""
    print(f"\n{'='*60}")
    print(f"Fetching ECMWF IFS Historical Data")
    print(f"Range: {start_date} to {end_date}")
    print(f"Interval: Every {interval_days} days")
    print(f"{'='*60}\n")

    results = []
    current = start_date

    while current <= end_date:
        date_str = current.strftime("%Y%m%d")
        output_file = os.path.join(OUTPUT_DIR, f"ifs_hist_{date_str}.csv")

        if os.path.exists(output_file):
            print(f"  {date_str}: Already exists")
            results.append(output_file)
            current += timedelta(days=interval_days)
            continue

        print(f"  {date_str}: Fetching...", end=' ', flush=True)

        try:
            temps_by_day = {}

            # Fetch daily temps for 10 days
            for fxx in range(0, 241, 24):  # 0, 24, 48, ... 240 hours
                try:
                    H = Herbie(
                        current.strftime("%Y-%m-%d"),
                        model='ifs',
                        fxx=fxx,
                        product='oper'
                    )

                    ds = H.xarray(":2t:", remove_grib=True)

                    if ds is not None:
                        temp_var = None
                        for var in ds.data_vars:
                            if 't2m' in var.lower() or '2t' in var.lower():
                                temp_var = var
                                break

                        if temp_var:
                            temp_data = ds[temp_var]
                            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
                            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'

                            lons = ds[lon_name].values
                            if lons.min() < 0:
                                lon_min, lon_max = -125, -65
                            else:
                                lon_min, lon_max = 235, 295

                            temp_us = temp_data.where(
                                (ds[lat_name] >= US_LAT_MIN) & (ds[lat_name] <= US_LAT_MAX) &
                                (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max),
                                drop=True
                            )

                            us_mean_k = float(temp_us.mean().values)
                            us_mean_c = us_mean_k - 273.15
                            day = fxx // 24
                            temps_by_day[day] = us_mean_c

                    time.sleep(0.1)

                except Exception as e:
                    continue

            if temps_by_day:
                hdds = {day: max(0, BASE_TEMP - temp) for day, temp in temps_by_day.items()}

                days_1_5 = [hdds.get(d, 0) for d in range(1, 6)]
                days_6_10 = [hdds.get(d, 0) for d in range(6, 11)]

                with open(output_file, 'w') as f:
                    f.write("date,hdd_1_5d,hdd_6_10d\n")
                    f.write(f"{date_str},{np.mean(days_1_5):.2f},{np.mean(days_6_10):.2f}\n")

                print(f"OK ({len(temps_by_day)} days)")
                results.append(output_file)
            else:
                print("No data")

        except Exception as e:
            print(f"Error: {e}")

        current += timedelta(days=interval_days)
        time.sleep(0.5)

    return results


def main():
    print("=" * 70)
    print("DEEP HISTORICAL DATA FETCHER")
    print("For Long-Term Fair Value Analysis")
    print("=" * 70)

    # GFS: 2021 to present (weekly samples)
    gfs_start = datetime(2021, 1, 1)
    gfs_end = datetime.now(timezone.utc).date() - timedelta(days=1)
    gfs_end = datetime.combine(gfs_end, datetime.min.time())

    # IFS: Jan 2024 to present (weekly samples)
    ifs_start = datetime(2024, 1, 1)
    ifs_end = gfs_end

    # Parse command line for custom range
    interval = 7  # Weekly by default
    if len(sys.argv) > 1:
        if sys.argv[1] == 'gfs':
            gfs_results = fetch_gfs_historical(gfs_start, gfs_end, interval)
            print(f"\nGFS: {len(gfs_results)} files")
            return
        elif sys.argv[1] == 'ifs':
            ifs_results = fetch_ifs_historical(ifs_start, ifs_end, interval)
            print(f"\nIFS: {len(ifs_results)} files")
            return
        elif sys.argv[1].isdigit():
            interval = int(sys.argv[1])

    # Fetch both
    print(f"\nFetching with {interval}-day intervals...")
    print(f"GFS: {gfs_start.date()} to {gfs_end.date()} (~{(gfs_end - gfs_start).days // interval} samples)")
    print(f"IFS: {ifs_start.date()} to {ifs_end.date()} (~{(ifs_end - ifs_start).days // interval} samples)")

    gfs_results = fetch_gfs_historical(gfs_start, gfs_end, interval)
    ifs_results = fetch_ifs_historical(ifs_start, ifs_end, interval)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"GFS Historical: {len(gfs_results)} files")
    print(f"IFS Historical: {len(ifs_results)} files")
    print(f"\nData saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
