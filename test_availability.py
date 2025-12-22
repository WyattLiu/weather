#!/usr/bin/env python3
from ecmwf.opendata import Client
from datetime import datetime, timedelta
import requests

# Create client
client = Client()

print("Testing how far back ECMWF open data is available...\n")

# Test going back day by day
available_dates = []
unavailable_dates = []

for days_back in range(1, 31):  # Test up to 30 days back
    date = datetime.now() - timedelta(days=days_back)
    date_str = date.strftime('%Y-%m-%d')
    
    # Test with a simple HEAD request to check availability
    for time_run in ['00', '12']:
        test_url = f"https://data.ecmwf.int/forecasts/{date.strftime('%Y%m%d')}/{time_run}z/ifs/0p25/oper/"
        
        try:
            response = requests.head(test_url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {date_str} {time_run}z - AVAILABLE")
                available_dates.append(f"{date_str} {time_run}z")
            else:
                print(f"✗ {date_str} {time_run}z - Not available (status: {response.status_code})")
                unavailable_dates.append(f"{date_str} {time_run}z")
        except Exception as e:
            print(f"✗ {date_str} {time_run}z - Not available ({str(e)[:40]})")
            unavailable_dates.append(f"{date_str} {time_run}z")

print("\n" + "="*60)
print(f"RESULTS:")
print(f"Available: {len(available_dates)} forecast runs")
print(f"Unavailable: {len(unavailable_dates)} forecast runs")

if available_dates:
    print(f"\nOldest available: {available_dates[-1]}")
    print(f"Newest available: {available_dates[0]}")
    print(f"\nECMWF keeps approximately {len(available_dates)//2} days of data")
print("="*60)
