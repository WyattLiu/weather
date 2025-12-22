#!/usr/bin/env python3
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

def get_ng_price_at_release(release_datetime_utc):
    """
    Get NG price for the hour bar containing the ECMWF release time.
    
    ECMWF releases:
    - 00z: Available ~03:00-04:00 UTC (10pm-11pm EST previous day)
    - 12z: Available ~15:00-16:00 UTC (10am-11am EST)
    
    We want the price bar that includes the release time.
    """
    ng = yf.Ticker("NG=F")
    
    # Convert UTC to EST for NG trading
    est = pytz.timezone('US/Eastern')
    release_est = release_datetime_utc.astimezone(est)
    
    # Fetch data around this time (48 hours window)
    start_date = release_est - timedelta(days=1)
    end_date = release_est + timedelta(days=1)
    
    try:
        df = ng.history(start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            return None
        
        # Find the closest bar to release time
        # Handle timezone issues
        if df.index.tz is not None:
            release_compare = release_est
        else:
            release_compare = release_est.replace(tzinfo=None)
        
        time_diffs = abs(df.index - release_compare)
        closest_idx = time_diffs.argmin()
        closest_time = df.index[closest_idx]
        
        price_data = {
            'datetime': closest_time,
            'open': df.loc[closest_time, 'Open'],
            'high': df.loc[closest_time, 'High'],
            'low': df.loc[closest_time, 'Low'],
            'close': df.loc[closest_time, 'Close'],
            'volume': df.loc[closest_time, 'Volume'],
            'time_diff_minutes': time_diffs[closest_idx].total_seconds() / 60
        }
        
        return price_data
    except Exception as e:
        print(f"Error fetching NG price for {release_est}: {e}")
        return None

if __name__ == "__main__":
    # Test with recent dates
    import glob
    import os
    
    print("Fetching NG prices for recent ECMWF releases...\n")
    
    files = sorted(glob.glob('forecast_historical_*z.grib2'))
    
    ng_prices = []
    
    for f in files:  # All files, not just last 10
        # Extract date and hour from filename
        # Format: forecast_historical_20251212_00z.grib2
        basename = os.path.basename(f)
        date_str = basename.split('_')[2]  # 20251212
        hour_str = basename.split('_')[3].replace('z.grib2', '')  # 00 or 12
        
        # Create UTC datetime for the model run
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(hour_str)
        
        release_time = datetime(year, month, day, hour, tzinfo=pytz.UTC)
        
        # Add 4 hours for when data is available (rough estimate)
        available_time = release_time + timedelta(hours=4)
        
        price_data = get_ng_price_at_release(available_time)
        
        if price_data:
            ng_prices.append({
                'forecast_date': f"{date_str} {hour_str}z",
                'release_time': available_time,
                'ng_datetime': price_data['datetime'],
                'ng_close': price_data['close'],
                'time_offset_min': price_data['time_diff_minutes']
            })
            print(f"{date_str} {hour_str}z: ${price_data['close']:.3f} (offset: {price_data['time_diff_minutes']:.0f} min)")
    
    # Save to CSV
    if ng_prices:
        df = pd.DataFrame(ng_prices)
        df.to_csv('ng_prices_at_releases.csv', index=False)
        print(f"\n✓ Saved {len(ng_prices)} NG prices to ng_prices_at_releases.csv")
