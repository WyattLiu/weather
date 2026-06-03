#!/usr/bin/env python3
"""Fetch multiple days of GFS data from NOMADS."""
import subprocess
import sys
from datetime import datetime, timedelta

# Fetch last N days
days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 10

end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=days_back)

print(f"Fetching GFS data from {start_date} to {end_date}")
print(f"Cycles: 00z, 12z")
print()

current = start_date
while current <= end_date:
    for cycle in ['00', '12']:
        date_str = current.strftime("%Y%m%d")
        arg = f"{date_str}_{cycle}"
        print(f"=== {date_str} {cycle}z ===")
        result = subprocess.run(
            ['python3', 'fetch_gfs.py', arg],
            capture_output=False
        )
    current += timedelta(days=1)

print("\nDone fetching GFS data!")
