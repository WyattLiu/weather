"""SPY daily OPEN INTEREST → Postgres (spy_options_oi). Mirrors ung_options_oi.

Pulls EOD open-interest for exactly the (expiration, strike, right) universe present in
spy_options_history (so OI lines up 1:1 with the minute quotes for the fill-quality model).
Source: ThetaData v3 /v3/option/history/open_interest. Idempotent + parallel.

  venv/bin/python backtest/backfill_spy_oi.py            # all contracts in the minute table
"""
import os
import sys
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import THETA_BASE

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}


def create_table():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spy_options_oi (
            trade_date DATE NOT NULL, expiration DATE NOT NULL,
            strike NUMERIC NOT NULL, option_right CHAR(1) NOT NULL,
            open_interest INTEGER, collected_at TIMESTAMP DEFAULT now(),
            UNIQUE (trade_date, expiration, strike, option_right));
        CREATE INDEX IF NOT EXISTS idx_spy_oi_date ON spy_options_oi(trade_date);
        CREATE INDEX IF NOT EXISTS idx_spy_oi_exp  ON spy_options_oi(expiration);
    """)
    conn.commit(); conn.close()


def contracts():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""SELECT expiration, strike, option_right, min(trade_date), max(trade_date)
                   FROM spy_options_history GROUP BY 1,2,3""")
    out = cur.fetchall(); conn.close()
    return out


def fetch_oi(exp, strike, right, d0, d1):
    try:
        r = requests.get(f'{THETA_BASE}/v3/option/history/open_interest', params={
            'symbol': 'SPY', 'expiration': exp.strftime('%Y%m%d'), 'right': right,
            'strike': float(strike), 'start_date': d0.strftime('%Y%m%d'),
            'end_date': d1.strftime('%Y%m%d'), 'format': 'json'}, timeout=25)
        if r.status_code != 200:
            return []
        resp = r.json().get('response') or []
    except Exception:
        return []
    if not resp or not resp[0].get('data'):
        return []
    return [(b['timestamp'][:10], b['open_interest']) for b in resp[0]['data']
            if b.get('timestamp') and b.get('open_interest') is not None]


def process(args):
    exp, strike, right, d0, d1 = args
    rows = [(dt, exp, float(strike), right, int(oi)) for dt, oi in fetch_oi(exp, strike, right, d0, d1)]
    if not rows:
        return 0
    # try/finally so the connection is ALWAYS released — previously any exception in execute_values/commit
    # skipped conn.close(), and over thousands of parallel contracts the leaked connections exhausted PG
    # ("sorry, too many clients already"), silently forcing real_chain fills to the model (2026-07 incident).
    conn = psycopg2.connect(**DB)
    try:
        cur = conn.cursor()
        execute_values(cur, """INSERT INTO spy_options_oi
            (trade_date, expiration, strike, option_right, open_interest)
            VALUES %s ON CONFLICT (trade_date, expiration, strike, option_right)
            DO UPDATE SET open_interest=EXCLUDED.open_interest""", rows)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def main(workers=2):
    create_table()
    cs = contracts()
    print(f"{len(cs):,} SPY contracts; fetching daily OI (parallel x{workers})", flush=True)
    t0 = time.time(); total = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, c): c for c in cs}
        for f in as_completed(futs):
            try:
                total += f.result()
            except Exception as e:
                print(f"  FAIL {futs[f][:3]}: {repr(e)[:70]}", flush=True)
            done += 1
            if done % 500 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(cs)}] {el:.0f}s ETA {el/done*(len(cs)-done)/60:.1f}min {total:,} rows", flush=True)
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT count(*), count(distinct trade_date), min(trade_date), max(trade_date) FROM spy_options_oi")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. spy_options_oi: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    main()
