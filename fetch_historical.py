#!/usr/bin/env python3
from ecmwf.opendata import Client
from datetime import datetime, timedelta
import os
import time as time_module

# Create client
client = Client()

print("Fetching historical forecast data for the past 3 weeks...")
print("This will download both 00z and 12z forecasts...\n")

# Get forecasts from the past 21 days, both 00z and 12z
dates_to_fetch = []
for i in range(1, 22):  # Last 21 days (3 weeks)
    date = datetime.now() - timedelta(days=i)
    dates_to_fetch.append(date)

successful_downloads = 0
failed_count = 0

for date in dates_to_fetch:
    date_str = date.strftime('%Y%m%d')
    
    # Try both 00z and 12z
    for time_run in ['00', '12']:
        filename = f"forecast_historical_{date_str}_{time_run}z.grib2"
        
        # Skip if already exists
        if os.path.exists(filename):
            print(f"✓ Already have {filename}")
            successful_downloads += 1
            continue
        
        try:
            print(f"Fetching {date_str} {time_run}:00 UTC...", end=' ')
            client.retrieve(
                time=time_run,
                date=date.strftime('%Y-%m-%d'),
                type="fc",
                step=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
                param=["2t"],
                target=filename
            )
            print(f"✓")
            successful_downloads += 1
            
            # Small delay to avoid hitting rate limits
            time_module.sleep(2)
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                print(f"✗ (not available)")
                failed_count += 1
            elif "429" in error_msg:
                print(f"✗ (rate limited - waiting 120s)")
                time_module.sleep(120)
                # Retry once after rate limit
                try:
                    client.retrieve(
                        time=time_run,
                        date=date.strftime('%Y-%m-%d'),
                        type="fc",
                        step=[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
                        param=["2t"],
                        target=filename
                    )
                    print(f"  Retry succeeded ✓")
                    successful_downloads += 1
                except:
                    print(f"  Retry failed ✗")
                    failed_count += 1
            else:
                print(f"✗ ({error_msg[:50]})")
                failed_count += 1

print(f"\n{'='*60}")
print(f"Completed. Downloaded {successful_downloads} forecast files.")
print(f"Failed: {failed_count}")
print(f"{'='*60}")
