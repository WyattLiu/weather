"""Fetch NOAA CPC population-weighted degree days (HDD + CDD) → national daily.

CPC publishes daily population-weighted degree days by the 9 US Census Divisions:
  https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data/{YEAR}/
    Population.Heating.txt   (HDD)
    Population.Cooling.txt   (CDD)

We aggregate the 9 divisions to a NATIONAL series using 2020-census population
shares (gas demand is roughly population-weighted). Output: cache/degree_days_daily.csv
with columns date, hdd, cdd. Uses curl (NOAA, like the ONI fetcher).
"""
import os
import subprocess
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS, 'cache')
BASE = 'https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/daily_data'

# 2020-census population share by Census Division (1..9), sums ~1.0
DIV_W = {1: 0.045, 2: 0.126, 3: 0.140, 4: 0.064, 5: 0.205,
         6: 0.058, 7: 0.126, 8: 0.077, 9: 0.159}


def _fetch(url):
    r = subprocess.run(['curl', '-s', '-L', '--max-time', '40', url],
                       capture_output=True, text=True)
    return r.stdout


def _parse(text):
    """CPC file: header 'Region|YYYYMMDD|...' then 'div|v|v|...' rows.
    Returns a national daily Series (population-weighted across divisions)."""
    lines = [ln for ln in text.splitlines() if '|' in ln]
    hdr = None
    div_series = {}
    for ln in lines:
        parts = ln.split('|')
        if parts[0] == 'Region':
            hdr = pd.to_datetime(parts[1:], format='%Y%m%d', errors='coerce')
            continue
        if hdr is None:
            continue
        try:
            div = int(parts[0])
        except ValueError:
            continue
        if div not in DIV_W:
            continue
        vals = pd.to_numeric(pd.Series(parts[1:]), errors='coerce').values
        if len(vals) == len(hdr):
            div_series[div] = pd.Series(vals, index=hdr)
    if not div_series:
        return None
    df = pd.DataFrame(div_series)
    w = pd.Series(DIV_W)
    nat = (df * w).sum(axis=1) / w.sum()   # population-weighted national
    return nat.dropna()


def main(start_year=2021, end_year=2026):
    hdd_all, cdd_all = [], []
    for yr in range(start_year, end_year + 1):
        for kind, bucket in (('Heating', hdd_all), ('Cooling', cdd_all)):
            url = f'{BASE}/{yr}/Population.{kind}.txt'
            txt = _fetch(url)
            s = _parse(txt) if txt else None
            if s is not None and len(s):
                bucket.append(s)
                print(f'  {yr} {kind}: {len(s)} days '
                      f'({s.index.min().date()}→{s.index.max().date()})')
            else:
                print(f'  {yr} {kind}: NO DATA')
    hdd = pd.concat(hdd_all).sort_index() if hdd_all else pd.Series(dtype=float)
    cdd = pd.concat(cdd_all).sort_index() if cdd_all else pd.Series(dtype=float)
    out = pd.DataFrame({'hdd': hdd, 'cdd': cdd})
    out = out[~out.index.duplicated(keep='last')].sort_index()
    out.index.name = 'date'
    path = os.path.join(CACHE, 'degree_days_daily.csv')
    out.to_csv(path)
    print(f'\nSaved {len(out)} days → {path}')
    print(out.describe().round(2).to_string())
    return out


if __name__ == '__main__':
    main()
