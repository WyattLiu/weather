"""UNG daily OPEN INTEREST backfill → Postgres (market_scanner.ung_options_oi).

OI is an end-of-day settlement figure (OCC publishes once daily) — daily is the ONLY
granularity that exists; there is no intraday OI. We fetch the daily OI series per
contract from ThetaData v3 (/v3/option/history/open_interest) for exactly the contract
universe already in ung_options_history (raw strikes), so OI joins 1:1 with the minute
quotes. Used to (a) calibrate mid-fill likelihood (deep books fill at mid) and (b)
compute historical dealer GEX for regime/decision signals.

One API call per contract returns its whole life's OI — so ~17k calls, parallel, minutes.
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
        CREATE TABLE IF NOT EXISTS ung_options_oi (
            trade_date DATE NOT NULL,
            expiration DATE NOT NULL,
            strike NUMERIC NOT NULL,            -- RAW (ThetaData) scale, matches minute table
            option_right CHAR(1) NOT NULL,
            open_interest INTEGER,
            collected_at TIMESTAMP DEFAULT now(),
            UNIQUE (trade_date, expiration, strike, option_right));
        CREATE INDEX IF NOT EXISTS idx_ung_oi_date ON ung_options_oi(trade_date);
        CREATE INDEX IF NOT EXISTS idx_ung_oi_exp  ON ung_options_oi(expiration);
    """)
    conn.commit(); conn.close()


def contracts():
    """Exactly the (exp, strike, right) universe in the minute table + its date span."""
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""SELECT expiration, strike, option_right,
                          min(trade_date), max(trade_date)
                   FROM ung_options_history GROUP BY 1,2,3""")
    out = cur.fetchall(); conn.close()
    return out


def fetch_oi(exp, strike, right, d0, d1):
    r = requests.get(f'{THETA_BASE}/v3/option/history/open_interest', params={
        'symbol': 'UNG', 'expiration': exp.strftime('%Y%m%d'), 'right': right,
        'strike': float(strike), 'start_date': d0.strftime('%Y%m%d'),
        'end_date': d1.strftime('%Y%m%d'), 'format': 'json'}, timeout=20)
    if r.status_code != 200:
        return []
    try:
        resp = r.json().get('response') or []
    except Exception:
        return []
    if not resp or not resp[0].get('data'):
        return []
    out = []
    for bar in resp[0]['data']:
        ts, oi = bar.get('timestamp'), bar.get('open_interest')
        if ts and oi is not None:
            out.append((ts[:10], oi))
    return out


def process(args):
    exp, strike, right, d0, d1 = args
    rows = [(dt, exp, float(strike), right, int(oi))
            for dt, oi in fetch_oi(exp, strike, right, d0, d1)]
    if not rows:
        return 0
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    execute_values(cur, """INSERT INTO ung_options_oi
        (trade_date, expiration, strike, option_right, open_interest)
        VALUES %s ON CONFLICT (trade_date, expiration, strike, option_right)
        DO UPDATE SET open_interest=EXCLUDED.open_interest""", rows)
    conn.commit(); n = cur.rowcount; conn.close()
    return n


def main(workers=6):
    create_table()
    cs = contracts()
    print(f"{len(cs):,} contracts; fetching daily OI (parallel x{workers})", flush=True)
    t0 = time.time(); total = 0; done = 0
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
    cur.execute("SELECT count(*), count(DISTINCT trade_date), min(trade_date), max(trade_date) FROM ung_options_oi")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. ung_options_oi: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    main()
