"""UNG INTRADAY option backfill → Postgres (market_scanner.ung_options_history).

Mirrors the single-name template (tlt_options_history): bid/ask stored as SEPARATE
rows (data_type='BID'/'ASK'), OHLC + underlying_price, unique on
(trade_date, bar_time, expiration, strike, option_right, data_type).

BOUNDED CHAIN (user spec): only expiries ≤ MAX_DTE (60) and strikes within ±BAND_PCT
of spot — the contracts that actually act (ITM/OTM puts, CCs, roll/recall targets).
Source: ThetaData v3 (/v3/option/history/quote, hourly default). Idempotent + parallel.

Usage:
  venv/bin/python backtest/backfill_ung_intraday.py --start 2026-03-15 --end 2026-06-12 --interval 1h
  (then extend to multi-year like ung_iv_surface)
"""
import os
import sys
import time
import argparse
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import THETA_BASE, split_factor_on   # reuse v3 base + split map

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
MAX_DTE = 60
BAND_PCT = 0.25      # ± of spot — strikes that can act


def create_table():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ung_options_history (
            id SERIAL PRIMARY KEY,
            trade_date DATE NOT NULL,
            bar_time TIMESTAMP NOT NULL,
            expiration DATE NOT NULL,
            strike NUMERIC NOT NULL,
            option_right CHAR(1) NOT NULL,
            open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC,
            volume INTEGER, bar_count INTEGER,
            underlying_price NUMERIC,
            data_type VARCHAR(8) NOT NULL,
            collected_at TIMESTAMP DEFAULT now(),
            UNIQUE (trade_date, bar_time, expiration, strike, option_right, data_type)
        );
        CREATE INDEX IF NOT EXISTS idx_ung_oh_date ON ung_options_history(trade_date);
        CREATE INDEX IF NOT EXISTS idx_ung_oh_exp  ON ung_options_history(expiration);
        CREATE INDEX IF NOT EXISTS idx_ung_oh_strike ON ung_options_history(strike);
        CREATE INDEX IF NOT EXISTS idx_ung_oh_time ON ung_options_history(bar_time);
    """)
    conn.commit(); conn.close()


def _get(url, params):
    r = requests.get(url, params={**params, 'format': 'json'}, timeout=20)
    if r.status_code != 200:
        return None
    try:
        return r.json().get('response')
    except Exception:
        return None


def expirations():
    resp = _get(f'{THETA_BASE}/v3/option/list/expirations', {'symbol': 'UNG'})
    return sorted(e['expiration'] for e in (resp or []))


def strikes(expiration):
    resp = _get(f'{THETA_BASE}/v3/option/list/strikes',
                {'symbol': 'UNG', 'expiration': expiration.replace('-', '')})
    return sorted(float(s['strike']) for s in (resp or []))


def quote_bars(expiration, strike, right, date_str, interval):
    resp = _get(f'{THETA_BASE}/v3/option/history/quote', {
        'symbol': 'UNG', 'expiration': expiration.replace('-', ''),
        'right': right, 'strike': strike,
        'start_date': date_str.replace('-', ''), 'end_date': date_str.replace('-', ''),
        'interval': interval})
    if not resp or not resp[0].get('data'):
        return []
    return resp[0]['data']


def process_day(args):
    d_str, adj_spot, exps, interval = args
    sf = split_factor_on(d_str)
    raw_spot = adj_spot / sf
    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
    rows = []
    for exp in exps:
        try:
            exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = (exp_d - d_obj).days
        if dte < 0 or dte > MAX_DTE:
            continue
        try:
            ks = strikes(exp)
        except Exception:
            continue
        near = [k for k in ks if abs(k - raw_spot) / max(raw_spot, 1e-6) <= BAND_PCT]
        for K_raw in near:
            for right in ('P', 'C'):
                for bar in quote_bars(exp, K_raw, right, d_str, interval):
                    ts = bar.get('timestamp')
                    if not ts:
                        continue
                    bid, ask = bar.get('bid'), bar.get('ask')
                    # store raw strike + raw spot (split-consistent); adjust at query time
                    for dtype, val in (('BID', bid), ('ASK', ask)):
                        if val is None:
                            continue
                        rows.append((d_str, ts, exp, float(K_raw), right,
                                     float(val), float(val), float(val), float(val),
                                     0, 0, round(raw_spot, 4), dtype))
    if rows:
        conn = psycopg2.connect(**DB); cur = conn.cursor()
        execute_values(cur, """INSERT INTO ung_options_history
            (trade_date, bar_time, expiration, strike, option_right,
             open, high, low, close, volume, bar_count, underlying_price, data_type)
            VALUES %s ON CONFLICT (trade_date, bar_time, expiration, strike, option_right, data_type)
            DO NOTHING""", rows)
        conn.commit(); n = cur.rowcount; conn.close()
        return n
    return 0


def main(start, end, interval, workers):
    create_table()
    spot_df = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'),
                          index_col=0, parse_dates=True).loc[start:end, ['UNG']].dropna()
    exps = expirations()
    print(f"{len(spot_df)} trading days {spot_df.index[0].date()}→{spot_df.index[-1].date()}; "
          f"{len(exps)} expiries; ≤{MAX_DTE}DTE, ±{BAND_PCT:.0%} strikes, {interval} bars")
    tasks = [(d.strftime('%Y-%m-%d'), float(r['UNG']), exps, interval)
             for d, r in spot_df.iterrows()]
    t0 = time.time(); total = 0; done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_day, t): t[0] for t in tasks}
        for f in as_completed(futs):
            try:
                total += f.result()
            except Exception as e:
                print(f"  {futs[f]} FAILED: {repr(e)[:90]}")
            done += 1
            if done % 5 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {el:.0f}s, ETA {el/done*(len(tasks)-done)/60:.1f}min, {total:,} rows")
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT trade_date) FROM ung_options_history")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. ung_options_history: {cur.fetchone()}")
    conn.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2026-03-15')
    p.add_argument('--end', default='2026-06-12')
    p.add_argument('--interval', default='1h')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    main(a.start, a.end, a.interval, a.workers)
