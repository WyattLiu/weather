#!/usr/bin/env python3
"""
Fetch CPC (Climate Prediction Center) Temperature Outlooks.
- 6-10 day outlook
- 8-14 day outlook

Source: https://www.cpc.ncep.noaa.gov/
Data: https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/
"""

import os
import sys
import requests
from datetime import datetime, timedelta
import zipfile
import io

# Configuration
OUTPUT_DIR = "/home/wyatt/weather"
CPC_BASE_URL = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst"

# Outlook types
OUTLOOK_TYPES = {
    '6-10': {
        'temp_file': 'temp610',
        'description': '6-10 Day Temperature Outlook'
    },
    '8-14': {
        'temp_file': 'temp814',
        'description': '8-14 Day Temperature Outlook'
    }
}


def download_cpc_outlook(outlook_type='6-10'):
    """
    Download CPC temperature outlook shapefiles.
    Returns path to downloaded file or None on failure.
    """
    if outlook_type not in OUTLOOK_TYPES:
        print(f"Unknown outlook type: {outlook_type}")
        return None

    config = OUTLOOK_TYPES[outlook_type]
    file_base = config['temp_file']

    # CPC updates daily, file names are like: temp610_YYYYMMDD.zip
    # Try today and yesterday
    today = datetime.utcnow()

    for days_back in range(3):
        check_date = today - timedelta(days=days_back)
        date_str = check_date.strftime("%Y%m%d")

        # Try different URL patterns
        urls_to_try = [
            f"{CPC_BASE_URL}/{file_base}.shp.zip",  # Current file (no date)
            f"{CPC_BASE_URL}/{file_base}_{date_str}.zip",  # Dated file
        ]

        for url in urls_to_try:
            try:
                print(f"  Trying: {url}")
                response = requests.get(url, timeout=30)

                if response.status_code == 200 and len(response.content) > 1000:
                    # Save the zip file
                    output_file = os.path.join(
                        OUTPUT_DIR,
                        f"cpc_{outlook_type.replace('-', '')}_{today.strftime('%Y%m%d')}.zip"
                    )

                    with open(output_file, 'wb') as f:
                        f.write(response.content)

                    print(f"  Downloaded: {output_file} ({len(response.content)} bytes)")

                    # Extract shapefile
                    extract_dir = os.path.join(OUTPUT_DIR, f"cpc_{outlook_type.replace('-', '')}")
                    os.makedirs(extract_dir, exist_ok=True)

                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        zf.extractall(extract_dir)
                        print(f"  Extracted to: {extract_dir}")

                    return output_file

            except Exception as e:
                print(f"  Error: {e}")
                continue

    return None


def download_cpc_grib():
    """
    Download CPC temperature outlook in GRIB format (if available).
    This provides gridded data similar to GFS/ECMWF.
    """
    # CPC also provides probability grids
    base_url = "https://ftp.cpc.ncep.noaa.gov/GIS/us_tempprcpfcst/grib"

    today = datetime.utcnow()
    date_str = today.strftime("%Y%m%d")

    files_to_try = [
        ('6-10', f"temp610_{date_str}.grb"),
        ('8-14', f"temp814_{date_str}.grb"),
    ]

    downloaded = []
    for outlook, filename in files_to_try:
        url = f"{base_url}/{filename}"
        output_file = os.path.join(OUTPUT_DIR, f"cpc_{outlook.replace('-', '')}_{date_str}.grb")

        try:
            print(f"  Trying GRIB: {url}")
            response = requests.get(url, timeout=30)

            if response.status_code == 200 and len(response.content) > 500:
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"  Downloaded: {output_file}")
                downloaded.append(output_file)
        except Exception as e:
            print(f"  GRIB not available: {e}")

    return downloaded


def fetch_cpc_text_forecast():
    """
    Fetch CPC text discussion which contains HDD/CDD forecasts.
    """
    url = "https://www.cpc.ncep.noaa.gov/products/predictions/610day/fxus06.html"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            output_file = os.path.join(OUTPUT_DIR, f"cpc_discussion_{datetime.utcnow().strftime('%Y%m%d')}.html")
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"Saved CPC discussion: {output_file}")
            return output_file
    except Exception as e:
        print(f"Error fetching discussion: {e}")

    return None


def main():
    """Main entry point."""
    print("=" * 60)
    print("CPC Temperature Outlook Fetcher")
    print("6-10 Day and 8-14 Day Forecasts")
    print("=" * 60)
    print()

    results = {}

    # Download shapefiles
    print("Downloading 6-10 day outlook...")
    results['6-10'] = download_cpc_outlook('6-10')

    print("\nDownloading 8-14 day outlook...")
    results['8-14'] = download_cpc_outlook('8-14')

    # Try GRIB format
    print("\nTrying GRIB format (gridded data)...")
    grib_files = download_cpc_grib()
    results['grib'] = grib_files

    # Get text discussion
    print("\nFetching CPC discussion...")
    results['discussion'] = fetch_cpc_text_forecast()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for key, value in results.items():
        status = "OK" if value else "FAILED"
        print(f"  {key}: {status}")

    success = any(results.values())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
