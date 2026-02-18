#!/bin/bash
# Automated ECMWF forecast fetching script
# Checks for missing data and fetches all available forecasts

cd /home/wyatt/weather

# Activate virtual environment
source venv/bin/activate

# Log file
LOGFILE="/home/wyatt/weather/fetch.log"
SKIP_FILE="/home/wyatt/weather/.skip_dates"

echo "$(date): Starting forecast fetch check..." >> $LOGFILE

# Run Python script to check and fetch missing data
python3 << 'PYEOF'
from ecmwf.opendata import Client
from datetime import datetime, timedelta
import os
import time

client = Client()
logfile = "/home/wyatt/weather/fetch.log"
skip_file = "/home/wyatt/weather/.skip_dates"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"{timestamp}: {msg}\n")
    print(msg)

# Load skip list (dates we know are not available)
skip_dates = set()
if os.path.exists(skip_file):
    with open(skip_file, "r") as f:
        for line in f:
            skip_dates.add(line.strip())

# Check what we have
existing_files = set()
for f in os.listdir('.'):
    if f.startswith('forecast_historical_') and f.endswith('.grib2'):
        existing_files.add(f)

log(f"Found {len(existing_files)} existing forecast files")
if skip_dates:
    log(f"Skipping {len(skip_dates)} dates known to be unavailable")

# Check last 5 days only (ECMWF only keeps ~4 days)
today = datetime.now()
dates_to_check = []

for days_back in range(0, 5):  # Check today and past 4 days
    date = today - timedelta(days=days_back)
    for run_time in ['00', '12']:
        date_key = f"{date.strftime('%Y%m%d')}_{run_time}z"
        filename = f"forecast_historical_{date_key}.grib2"
        
        # Skip if we already have it or know it's unavailable
        if filename not in existing_files and date_key not in skip_dates:
            dates_to_check.append({
                'date': date,
                'run': run_time,
                'filename': filename,
                'date_key': date_key
            })

if not dates_to_check:
    log("All recent forecasts already downloaded or known unavailable")
else:
    log(f"Need to check {len(dates_to_check)} potentially missing forecasts")
    
    fetched = 0
    failed = 0
    new_skips = []
    
    for item in dates_to_check:
        date_str = item['date'].strftime('%Y-%m-%d')
        log(f"Fetching {date_str} {item['run']}z...")
        
        try:
            # Full 10-day forecast: 0-144h at 6-hourly, 150-240h at 6-hourly
            steps = list(range(0, 145, 6)) + list(range(150, 241, 6))
            client.retrieve(
                time=item['run'],
                date=date_str,
                type="fc",
                step=steps,
                param=["2t"],
                target=item['filename']
            )
            log(f"  ✓ Successfully downloaded {item['filename']}")
            fetched += 1
            time.sleep(2)  # Rate limiting
            
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                # Check if this is an old date (more than 5 days ago) vs future date
                days_old = (today - item['date']).days
                if days_old > 5:
                    log(f"  ✗ Not available (404) - too old, adding to skip list")
                    new_skips.append(item['date_key'])
                else:
                    log(f"  ✗ Not available (404) - too recent, will retry next time")
            elif "429" in error_msg:
                log(f"  ✗ Rate limited - will retry next time")
            else:
                log(f"  ✗ Error: {error_msg[:100]}")
            failed += 1
    
    # Save new skip entries
    if new_skips:
        with open(skip_file, "a") as f:
            for date_key in new_skips:
                f.write(f"{date_key}\n")
        log(f"Added {len(new_skips)} unavailable dates to skip list")
    
    log(f"Fetch complete: {fetched} successful, {failed} failed")
PYEOF

# Update the comparison chart if we got any new data
if grep -q "Successfully downloaded" $LOGFILE; then
    echo "$(date): Updating comparison charts..." >> $LOGFILE

    # Update HDD comparison chart
    python3 hdd_comparison.py >> $LOGFILE 2>&1
    echo "$(date): HDD chart updated" >> $LOGFILE

    # Fetch NG prices
    python3 fetch_ng_prices.py >> $LOGFILE 2>&1
fi

# Always update HDD vs NG/TTF chart (refreshes live prices from yfinance)
echo "$(date): Updating HDD vs NG/TTF chart..." >> $LOGFILE
python3 hdd_ng_comparison.py >> $LOGFILE 2>&1
echo "$(date): HDD vs NG/TTF chart updated" >> $LOGFILE
