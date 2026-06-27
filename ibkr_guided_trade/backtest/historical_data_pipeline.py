"""High-fidelity historical data pipeline for UNG wheel backtesting.

Pulls EVERYTHING we can use for accurate historical replay:
- UNG/KOLD/BOIL daily prices (yfinance)
- NG futures (NG=F) for technicals
- EIA weekly storage (already cached locally via ng_daily_forecast)
- Baker Hughes rig count (already cached)
- CFTC COT positioning
- FRED industrial production, DXY
- CPC weather (HDD/CDD)
- Realized vol surface for IV proxy

Output: a unified time series dataframe with all factors aligned by date,
ready for historical z-score computation and engine replay.

Run: python backtest/historical_data_pipeline.py [--years 5]
"""
import os
import sys
import argparse
import math
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_ung_kold_boil(years=5):
    """Daily prices for the relevant ETFs and NG futures."""
    import yfinance as yf
    print(f"[data] Fetching ETF + futures prices ({years}yr)...")
    period = f'{years}y'
    out = {}
    for sym in ['UNG', 'KOLD', 'BOIL', 'BOXX', 'NG=F', 'CL=F', 'DX-Y.NYB', '^VIX']:
        try:
            t = yf.Ticker(sym)
            df = t.history(period=period)
            if not df.empty:
                out[sym.replace('=F', '').replace('-Y.NYB', '_DXY').replace('^', '')] = df['Close']
                print(f"  {sym:>10}: {len(df)} bars, ${df['Close'].iloc[-1]:.2f}")
        except Exception as e:
            print(f"  {sym}: failed ({e})")
    return out


def compute_realized_iv_surface(spot_series, windows=(30, 60, 90)):
    """For each date, compute trailing realized vol at multiple windows.
    Used as IV proxy in historical option pricing. UNG IV typically trades
    at ~10-15% premium to realized — apply offset.
    """
    out = {}
    for w in windows:
        rv = spot_series.pct_change().rolling(w).std() * math.sqrt(252)
        iv_proxy = rv * 1.12  # +12% IV premium typical for UNG
        out[f'iv_{w}d'] = iv_proxy
    return pd.DataFrame(out)


def fetch_eia_historical(years=5):
    """Pull EIA historical monthly + weekly data via curl (avoids requests block).

    Key series:
      - N9070US2m: dry gas production (monthly Bcf)
      - N9140US2m: total consumption (monthly Bcf)
      - N9133US2m: LNG exports (monthly Bcf)
      - N9132US2m: pipeline exports
      - NW2_EPG0_SWO_R48_BCF: weekly storage Lower 48 (Bcf)
    """
    import subprocess
    EIA_BASE = 'https://www.eia.gov/dnav/ng/hist_xls/'
    files = {
        'production': 'N9070US2m.xls',
        'consumption': 'N9140US2m.xls',
        'lng_exports': 'N9133US2m.xls',
        'pipe_exports': 'N9132US2m.xls',
        'storage_weekly': 'NW2_EPG0_SWO_R48_BCFW.xls',
        'hh_spot_daily': 'RNGWHHDD.xls',
    }
    out = {}
    for name, fname in files.items():
        cache_path = os.path.join(CACHE_DIR, f'eia_{name}.xls')
        if not os.path.exists(cache_path) or _file_age_hours(cache_path) > 24:
            url = EIA_BASE + fname
            print(f"  EIA {name}: downloading...")
            try:
                subprocess.run(['curl', '-s', '-L', '-A', 'Mozilla/5.0', '-o', cache_path, url],
                              timeout=60, check=False)
            except Exception as e:
                print(f"    FAILED: {e}")
                continue
        else:
            print(f"  EIA {name}: cached ({_file_age_hours(cache_path):.0f}h old)")

        # Parse — keep all rows; storage is WEEKLY (~52/yr), monthly is ~12/yr
        # Old bug: tail(years*12+24) truncated weekly storage to only 1.6 years
        try:
            sheet = 'Data 1'
            df = pd.read_excel(cache_path, sheet_name=sheet, skiprows=2)
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            # Keep last N years worth of data (works for both weekly and monthly)
            cutoff = df.index.max() - pd.DateOffset(years=years + 1)
            df = df[df.index >= cutoff]
            out[name] = df['value']
            print(f"    parsed: {len(df)} rows {df.index.min().date()} → {df.index.max().date()}")
        except Exception as e:
            print(f"    parse failed: {e}")
    return out


def fetch_cot_history(years=5):
    """CFTC Commitment of Traders for NG (NYMEX)."""
    import urllib.request
    out_path = os.path.join(CACHE_DIR, 'cot_ng.csv')
    if os.path.exists(out_path) and _file_age_hours(out_path) < 24 * 7:
        print(f"  COT: cached ({_file_age_hours(out_path):.0f}h)")
    else:
        url = 'https://www.cftc.gov/dea/newcot/deafut.txt'
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                content = resp.read().decode('utf-8', errors='ignore')
            # NG NYMEX market code: 023391
            ng_rows = [l for l in content.split('\n') if '023391' in l]
            print(f"  COT: {len(ng_rows)} latest NG rows (need historical too)")
            # For history, would need to pull deahistfo_archive
        except Exception as e:
            print(f"  COT failed: {e}")
    return None


def build_master_dataset(years=5):
    """Combine everything into one daily dataframe."""
    print(f"\n=== Building master dataset ({years} years) ===\n")

    # Prices (daily)
    prices = fetch_ung_kold_boil(years)
    df = pd.DataFrame({k: v for k, v in prices.items()})
    df.index = df.index.tz_localize(None) if df.index.tz else df.index

    # IV surface from UNG
    iv = compute_realized_iv_surface(df['UNG'])
    df = df.join(iv)

    # EIA monthly — forward fill to daily
    print("\n[data] EIA monthly data:")
    eia = fetch_eia_historical(years)
    for name, series in eia.items():
        if series is None or series.empty:
            continue
        # Reindex to daily, forward fill
        daily = series.reindex(df.index, method='ffill')
        df[f'eia_{name}'] = daily

    # Days of supply (storage / consumption)
    # EIA consumption reported in MMcf/month, storage in Bcf
    # Convert: cons_bcfd = MMcf/month / 30 days / 1000 (MMcf→Bcf)
    # Typical: 2.3M MMcf/mo / 30 / 1000 = 77 Bcf/d → days = 2290 Bcf / 77 = ~30 days ✓
    if 'eia_storage_weekly' in df.columns and 'eia_consumption' in df.columns:
        cons_bcfd = df['eia_consumption'] / 30.0 / 1000.0  # MMcf/mo → Bcf/d
        df['days_supply'] = df['eia_storage_weekly'] / cons_bcfd.replace(0, np.nan)

    # NG term structure (NG=F + simple curve approximation)
    if 'NG' in df.columns:
        df['ng_ma200'] = df['NG'].rolling(200).mean()
        df['ng_trend'] = df['NG'] / df['ng_ma200'] - 1

    print(f"\n=== Master dataset built: {df.shape[0]} days × {df.shape[1]} columns ===")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # Save as CSV — no parquet dependency required
    out_path = os.path.join(CACHE_DIR, 'master_dataset.csv')
    df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")
    return df


def _file_age_hours(path):
    if not os.path.exists(path):
        return float('inf')
    return (datetime.now().timestamp() - os.path.getmtime(path)) / 3600


_SLOW_COLS = ('eia_production', 'eia_consumption', 'eia_lng_exports', 'eia_pipe_exports',
              'eia_storage_weekly', 'eia_hh_spot_daily')  # weekly/monthly — carry forward

# THETADATA is the authoritative source (same feed as the option chain → no cross-source
# skew). It serves the ETFs (stock EOD) + VIX (index EOD). NG/CL futures aren't in this
# subscription (/v3/futures = 404) → those columns are CARRIED FORWARD, never yahoo.
_THETA_BASE = 'http://127.0.0.1:25503'
_THETA_STOCK = ('UNG', 'KOLD', 'BOIL', 'BOXX')        # /v3/stock/history/eod
_THETA_INDEX = {'VIX': 'VIX'}                          # col -> ThetaData index symbol
_CARRY_PRICE = ('NG', 'CL', 'DX_DXY')                  # not on ThetaData → carry forward


def _thetadata_eod_closes(start, end):
    """Daily closes from ThetaData (authoritative, same feed as the options) for the symbols
    it serves: ETFs via stock EOD + VIX via index EOD. Returns DataFrame[date × colname] of
    closes. NG/CL/DXY are intentionally absent (carried forward by the caller)."""
    import requests
    s, e = pd.Timestamp(start).strftime('%Y%m%d'), pd.Timestamp(end).strftime('%Y%m%d')
    out = {}
    def _fetch(kind, sym, col):
        try:
            r = requests.get(f'{_THETA_BASE}/v3/{kind}/history/eod',
                             params={'symbol': sym, 'start_date': s, 'end_date': e}, timeout=15)
            if r.status_code != 200 or not r.text.strip():
                return
            df = pd.read_csv(StringIO(r.text))
            if 'created' not in df.columns or 'close' not in df.columns or df.empty:
                return
            df['d'] = pd.to_datetime(df['created']).dt.normalize()
            out[col] = df.groupby('d')['close'].last()
        except Exception:
            return
    for sym in _THETA_STOCK:
        _fetch('stock', sym, sym)
    for col, sym in _THETA_INDEX.items():
        _fetch('index', sym, col)
    if not out:
        return None
    return pd.DataFrame(out)


def refresh_to_today(df=None, live_spot=None, max_stale_min=20, persist=True):
    """SAME-DAY refresh: bring the master dataset up to the real today with FRESH prices,
    so the live decision's signals (z / regime / IV-rank / greeks) reflect today — not
    yesterday's close. Source = THETADATA (authoritative, same feed as the option chain —
    NOT yahoo): ETFs + VIX from ThetaData EOD, NG/CL/DXY carried forward (not in this
    subscription), today's row stamped from the live WS spot. Slow weekly/monthly EIA
    columns carried forward; price-derived factors recomputed.

    Guarded: skips the network if the dataset is already current OR was refreshed within
    `max_stale_min`. Defensive: any fetch failure returns the dataset unchanged (the caller
    still has the staleness flag). Returns the refreshed (or unchanged) DataFrame."""
    path = os.path.join(CACHE_DIR, 'master_dataset.csv')
    if df is None:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    # normalise the index to clean calendar dates (the raw file has tz-shifted 04:00/05:00
    # rows + NaN junk rows) — one row per date, keep the last non-null values.
    df.index = pd.to_datetime(df.index).normalize()
    df = df.groupby(level=0).last().sort_index()
    df = df.dropna(subset=['UNG'])
    today = pd.Timestamp.today().normalize()
    last = df.index[-1]
    marker = os.path.join(CACHE_DIR, '.last_intraday_refresh')
    if last >= today:
        # already have today's row — just track the latest live spot (no network) so
        # z / regime / greeks follow intraday UNG moves; recompute UNG-derived factors.
        if live_spot and live_spot == live_spot and float(live_spot) != float(df.at[last, 'UNG']):
            df.at[last, 'UNG'] = float(live_spot)
            iv = compute_realized_iv_surface(df['UNG'])
            for c in iv.columns:
                df[c] = iv[c]
            if persist:
                df.to_csv(path)
        return df                                   # already current
    if os.path.exists(marker) and _file_age_hours(marker) * 60 < max_stale_min and last < today:
        # refreshed very recently but still behind (e.g. weekend/holiday) — accept as-is
        return df
    try:
        # AUTHORITATIVE source = ThetaData (same feed as the option chain → no cross-source
        # skew vs the IV surface/quotes). NOT yahoo. ETFs + VIX from ThetaData; NG/CL/DXY
        # carried forward (not in this subscription). EOD lands after each session close, so
        # this brings the daily history to the prior close; the live WS spot tracks intraday.
        raw = _thetadata_eod_closes(last, today)
        if raw is None or raw.empty:
            return df
        raw.index = pd.to_datetime(raw.index).normalize()
        new_rows = raw.index[raw.index > last]
        if len(new_rows) == 0:
            return df
        for dt in new_rows:
            row = {}
            for col in raw.columns:
                v = raw.at[dt, col] if col in raw.columns else None
                if v == v and v is not None:        # not NaN
                    row[col] = float(v)
            # NaN-GUARD: any column yfinance didn't fill today (lagged ETF bar, slow EIA
            # weekly/monthly) is carried forward from the last known value — a same-day
            # refresh must never INTRODUCE a NaN that a kernel could then decide on.
            prev = df.iloc[-1]
            for col in df.columns:
                if col not in row or row.get(col) != row.get(col):   # missing or NaN
                    pv = prev.get(col)
                    if pv == pv:                     # prev not NaN
                        row[col] = pv
            df.loc[dt] = pd.Series(row)
        df = df.sort_index()
        # ThetaData EOD lags to the PRIOR close, so the latest row is yesterday. If we have a
        # live intraday spot (WS/broker — authoritative for execution), STAMP today's row from
        # it (carry forward everything else) so signals/regime/greeks reflect the current
        # session. Never overwrite a real prior close. (If today's row already exists, update it.)
        if live_spot and live_spot == live_spot:
            if df.index[-1] < today:
                prev = df.iloc[-1]
                trow = {c: (prev[c] if prev[c] == prev[c] else None) for c in df.columns}
                trow['UNG'] = float(live_spot)
                df.loc[today] = pd.Series(trow)
                df = df.sort_index()
            else:
                df.at[df.index[-1], 'UNG'] = float(live_spot)
        # recompute the price-DERIVED factors over the full (now-extended) series
        iv = compute_realized_iv_surface(df['UNG'])
        for c in iv.columns:
            df[c] = iv[c]
        if 'NG' in df.columns:
            df['ng_ma200'] = df['NG'].rolling(200).mean()
            df['ng_trend'] = df['NG'] / df['ng_ma200'] - 1
        if 'eia_storage_weekly' in df.columns and 'eia_consumption' in df.columns:
            cons_bcfd = df['eia_consumption'] / 30.0 / 1000.0
            df['days_supply'] = df['eia_storage_weekly'] / cons_bcfd.replace(0, np.nan)
        if persist:
            df.to_csv(path)
            with open(marker, 'w') as f:
                f.write(str(today.date()))
        return df
    except Exception as e:
        print(f"[refresh_to_today] skipped ({e!r}) — using existing data")
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=5)
    args = parser.parse_args()
    build_master_dataset(years=args.years)
