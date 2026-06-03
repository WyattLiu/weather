#!/usr/bin/env python3
"""Fetch GFS data for Dec 4-11."""
import subprocess
from datetime import datetime, timedelta

dates = ['20251204', '20251205', '20251206', '20251207', '20251208', '20251209', '20251210', '20251211']
cycles = ['00', '12']

for date in dates:
    for cycle in cycles:
        arg = f"{date}_{cycle}"
        print(f"=== {date} {cycle}z ===")
        result = subprocess.run(['python3', 'fetch_gfs.py', arg], capture_output=False)

print("\nDone!")
