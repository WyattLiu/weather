"""Backfill UNG + DBA option chains (OI + EOD quotes) from ThetaData.

Uses the local ThetaTerminal v3 proxy on :25503 (same setup as
spx_strategies/scripts/data/backfill/spx_oi_thetadata.py).

Per expiration, two requests covering [expiry - WINDOW_DAYS, expiry]:
  /v3/option/history/open_interest  → daily OI per strike/right
  /v3/option/history/eod            → daily close/bid/ask per strike/right

Output: one parquet-ish CSV per (symbol, expiry) under
  research/gex/history/thetadata/{symbol}/{expiry}_oi.csv
  research/gex/history/thetadata/{symbol}/{expiry}_eod.csv
Resumable: skips expiries whose files already exist.

Usage:
    python research/gex/thetadata_backfill.py --start-year 2018
    python research/gex/thetadata_backfill.py --symbol UNG --start-year 2018
"""
import os
import sys
import csv
import json
import time
import argparse
from datetime import date, datetime, timedelta

import requests

THETA_BASE = 'http://127.0.0.1:25503'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(THIS_DIR, 'history', 'thetadata')
WINDOW_DAYS = 75
SESSION = requests.Session()


def _get(url, params, retries=5):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(min(60, 2 ** attempt + 5))
                continue
            if r.status_code in (404, 472):  # no data
                return []
            r.raise_for_status()
            return r.json().get('response', []) or []
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(min(30, 2 ** attempt))
            else:
                raise
    return []


def list_expirations(symbol):
    rows = _get(f'{THETA_BASE}/v3/option/list/expirations',
                {'symbol': symbol, 'format': 'json'})
    return sorted({r['expiration'] for r in rows})


def fetch_expiry(symbol, expiry):
    """Pull OI + EOD for one expiration. Returns (oi_rows, eod_rows)."""
    exp_d = datetime.strptime(expiry, '%Y-%m-%d').date()
    start = (exp_d - timedelta(days=WINDOW_DAYS)).strftime('%Y%m%d')
    end = exp_d.strftime('%Y%m%d')
    base_params = {'symbol': symbol, 'expiration': exp_d.strftime('%Y%m%d'),
                   'start_date': start, 'end_date': end,
                   'strike_range': '100', 'format': 'json'}

    oi_rows = []
    for c in _get(f'{THETA_BASE}/v3/option/history/open_interest', base_params):
        K = c['contract']['strike']
        right = c['contract']['right'][0]  # CALL→C, PUT→P
        for d in c.get('data', []):
            # timestamp is the next-morning OCC posting; OI belongs to the
            # PRIOR trading day's close — shift back one day
            ts = datetime.fromisoformat(d['timestamp']).date()
            oi_rows.append((str(ts), expiry, right, K, d['open_interest']))

    eod_rows = []
    for c in _get(f'{THETA_BASE}/v3/option/history/eod', base_params):
        K = c['contract']['strike']
        right = c['contract']['right'][0]
        for d in c.get('data', []):
            ts = (d.get('created') or d.get('last_trade') or '')[:10]
            if not ts:
                continue
            eod_rows.append((ts, expiry, right, K,
                             d.get('bid', 0), d.get('ask', 0),
                             d.get('close', 0), d.get('volume', 0)))
    return oi_rows, eod_rows


def is_monthly(expiry):
    """3rd Friday of month, or Thursday before when Friday is a holiday."""
    d = datetime.strptime(expiry, '%Y-%m-%d').date()
    if d.weekday() == 4:  # Friday
        return 15 <= d.day <= 21
    if d.weekday() == 3:  # Thursday (holiday-shifted)
        return 16 <= d.day <= 22
    return False


def _process_expiry(symbol, expiry, out_dir):
    oi_path = os.path.join(out_dir, f'{expiry}_oi.csv')
    eod_path = os.path.join(out_dir, f'{expiry}_eod.csv')
    if os.path.exists(oi_path) and os.path.exists(eod_path):
        return 'cached'
    try:
        oi_rows, eod_rows = fetch_expiry(symbol, expiry)
    except Exception as e:
        return f'FAILED: {e}'
    with open(oi_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['oi_date', 'expiry', 'right', 'strike', 'open_interest'])
        w.writerows(oi_rows)
    with open(eod_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['quote_date', 'expiry', 'right', 'strike',
                    'bid', 'ask', 'close', 'volume'])
        w.writerows(eod_rows)
    return f'ok ({len(oi_rows)} oi, {len(eod_rows)} eod)'


def backfill(symbol, start_year, monthlies_only=True, workers=4):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out_dir = os.path.join(OUT_ROOT, symbol.lower())
    os.makedirs(out_dir, exist_ok=True)
    expiries = [e for e in list_expirations(symbol)
                if int(e[:4]) >= start_year and e <= date.today().isoformat()]
    if monthlies_only:
        expiries = [e for e in expiries if is_monthly(e)]
    print(f'[{symbol}] {len(expiries)} expirations '
          f'({"monthlies" if monthlies_only else "all"}) from {start_year}', flush=True)
    done = cached = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_expiry, symbol, e, out_dir): e for e in expiries}
        for fut in as_completed(futs):
            expiry = futs[fut]
            status = fut.result()
            if status == 'cached':
                cached += 1
            elif status.startswith('FAILED'):
                failed += 1
                print(f'  {expiry}: {status}', flush=True)
            else:
                done += 1
                print(f'  {expiry}: {status}  [{done+cached}/{len(expiries)}]', flush=True)
    print(f'[{symbol}] complete: {done} fetched, {cached} cached, {failed} failed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default=None, help='UNG or DBA; default both')
    p.add_argument('--start-year', type=int, default=2018)
    p.add_argument('--all-expiries', action='store_true',
                   help='include weeklies (default: monthlies only)')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    symbols = [args.symbol] if args.symbol else ['UNG', 'DBA']
    for sym in symbols:
        backfill(sym, args.start_year,
                 monthlies_only=not args.all_expiries, workers=args.workers)
