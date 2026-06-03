#!/usr/bin/env python3
"""
Fetch GFS, ECMWF IFS, and AIFS data using Herbie.
Herbie searches multiple sources: AWS, Google, Azure, NOMADS, ECMWF.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from herbie import Herbie
import numpy as np
import time

OUTPUT_DIR = "/home/wyatt/weather"

# US bounding box (for subsetting later)
US_LAT_MIN, US_LAT_MAX = 25, 50
US_LON_MIN, US_LON_MAX = 235, 295  # 0-360 format


def fetch_model_day(date, model, cycle='00', max_days=16):
    """Fetch forecast data for a single day using Herbie."""
    date_str = date.strftime("%Y%m%d")

    # Model-specific settings
    if model == 'gfs':
        product = 'pgrb2.0p25'
        search_pattern = ":TMP:2 m above ground:"
        forecast_hours = list(range(0, min(385, max_days * 24 + 1), 24))
    elif model == 'ifs':
        product = 'oper'
        search_pattern = ":2t:"  # 2m temperature in ECMWF
        forecast_hours = list(range(0, min(241, max_days * 24 + 1), 24))  # IFS goes to 10 days
    elif model == 'aifs':
        product = 'oper'
        search_pattern = ":2t:"
        forecast_hours = list(range(0, min(241, max_days * 24 + 1), 24))
    else:
        print(f"Unknown model: {model}")
        return None

    output_file = os.path.join(OUTPUT_DIR, f"{model}_{date_str}_{cycle}z.nc")

    # Skip if already exists
    if os.path.exists(output_file):
        print(f"  [{model}] Already exists: {output_file}")
        return output_file

    print(f"  [{model}] Fetching {date_str} {cycle}z...")

    temps_by_step = {}

    for fxx in forecast_hours:
        try:
            H = Herbie(
                date.strftime("%Y-%m-%d"),
                model=model,
                fxx=fxx,
                product=product
            )

            # Download temperature data
            ds = H.xarray(search_pattern, remove_grib=True)

            if ds is not None:
                # Get temperature variable
                temp_var = None
                for var in ds.data_vars:
                    if 't2m' in var.lower() or 't' == var.lower() or '2t' in var.lower():
                        temp_var = var
                        break

                if temp_var:
                    # Get US mean temperature
                    temp_data = ds[temp_var]

                    # Handle coordinates
                    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon' if 'lon' in ds.coords else 'x'
                    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat' if 'lat' in ds.coords else 'y'

                    if lon_name in ds.coords and lat_name in ds.coords:
                        lons = ds[lon_name].values
                        lats = ds[lat_name].values

                        # Convert longitude if needed
                        if lons.min() < 0:
                            lon_min, lon_max = -125, -65
                        else:
                            lon_min, lon_max = 235, 295

                        # Subset to US
                        temp_us = temp_data.where(
                            (ds[lat_name] >= US_LAT_MIN) & (ds[lat_name] <= US_LAT_MAX) &
                            (ds[lon_name] >= lon_min) & (ds[lon_name] <= lon_max),
                            drop=True
                        )

                        us_mean = float(temp_us.mean().values)
                        temps_by_step[fxx] = us_mean
                        print(f"f{fxx:03d}", end=' ', flush=True)

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            continue

    print()

    if temps_by_step:
        # Save as simple CSV for now
        csv_file = output_file.replace('.nc', '.csv')
        with open(csv_file, 'w') as f:
            f.write("forecast_hour,temp_k,temp_c\n")
            for fxx in sorted(temps_by_step.keys()):
                temp_k = temps_by_step[fxx]
                temp_c = temp_k - 273.15
                f.write(f"{fxx},{temp_k:.2f},{temp_c:.2f}\n")

        print(f"    Saved: {csv_file} ({len(temps_by_step)} steps)")
        return csv_file

    return None


def fetch_all_models(start_date, end_date, models=['gfs', 'ifs', 'aifs'], cycles=['00', '12']):
    """Fetch all models for date range."""
    print(f"Fetching data from {start_date} to {end_date}")
    print(f"Models: {models}")
    print(f"Cycles: {cycles}")
    print()

    results = {m: [] for m in models}

    current = start_date
    while current <= end_date:
        print(f"\n=== {current} ===")
        for cycle in cycles:
            for model in models:
                try:
                    result = fetch_model_day(current, model, cycle)
                    if result:
                        results[model].append(result)
                except Exception as e:
                    print(f"  [{model}] Error: {e}")
        current += timedelta(days=1)

    print(f"\n" + "=" * 50)
    print("Summary:")
    for model in models:
        print(f"  {model}: {len(results[model])} forecasts")

    return results


def main():
    print("=" * 60)
    print("Multi-Model Weather Data Fetcher (GFS, IFS, AIFS)")
    print("=" * 60)
    print()

    # Default: fetch last 30 days
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=30)
    models = ['gfs', 'ifs', 'aifs']

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

    results = fetch_all_models(start_date, end_date, models)

    print("\nDone!")


if __name__ == "__main__":
    main()
