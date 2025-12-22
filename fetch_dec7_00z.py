#!/usr/bin/env python3
from ecmwf.opendata import Client

client = Client()

print("Fetching Dec 7 00z forecast...")

try:
    client.retrieve(
        time='00',
        date='2025-12-07',
        type="fc",
        step=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
        param=["2t"],
        target="forecast_historical_20251207_00z.grib2"
    )
    print("✓ Successfully downloaded forecast_historical_20251207_00z.grib2")
except Exception as e:
    print(f"✗ Failed: {e}")
