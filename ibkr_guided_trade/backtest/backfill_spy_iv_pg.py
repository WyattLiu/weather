"""SPY IV surface → Postgres (spy_iv_surface), built FROM the minute table.

Mirrors ung_iv_surface (same columns → works with replay_engine.iv_from_surface). Instead of
re-fetching from ThetaData, it derives the EOD smile from spy_options_history we already have:
the LAST RTH bar per (date, expiration, strike, right) → mid=(bid+ask)/2 → BS implied vol.
SPY does not split → strike_adj == strike_real, split_factor = 1. Idempotent + auto-resume.

  venv/bin/python backtest/backfill_spy_iv_pg.py                 # resume → latest quote date
  venv/bin/python backtest/backfill_spy_iv_pg.py --start 2021-01-01 --end 2026-06-18
"""
import os
import sys
import time
import argparse
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import bs_implied_vol

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
R = 0.045


def create_table():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spy_iv_surface (
            date DATE NOT NULL, expiration DATE NOT NULL, dte INTEGER,
            strike_adj NUMERIC NOT NULL, strike_real NUMERIC, option_right CHAR(1) NOT NULL,
            spot_adj NUMERIC, spot_real NUMERIC, mid NUMERIC, iv NUMERIC,
            split_factor NUMERIC DEFAULT 1, created_at TIMESTAMPTZ DEFAULT now(),
            UNIQUE (date, expiration, strike_adj, option_right)
        );
        CREATE INDEX IF NOT EXISTS idx_spy_ivs_date ON spy_iv_surface(date);
    """)
    conn.commit(); conn.close()


def _dates(start, end):
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT DISTINCT trade_date FROM spy_options_history "
                "WHERE trade_date BETWEEN %s AND %s ORDER BY trade_date", (start, end))
    out = [r[0] for r in cur.fetchall()]; conn.close()
    return out


def process_date(d):
    """EOD smile for one date: last RTH bar per contract → mid → IV → spy_iv_surface."""
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    # last bid and last ask per (expiration, strike, right) on this date
    cur.execute("""
        SELECT DISTINCT ON (expiration, strike, option_right, data_type)
               expiration, strike, option_right, data_type, close, underlying_price
        FROM spy_options_history
        WHERE trade_date = %s
        ORDER BY expiration, strike, option_right, data_type, bar_time DESC
    """, (d,))
    legs = {}   # (exp,K,right) -> {'BID':x,'ASK':y,'spot':s}
    for exp, K, right, dt, close, spot in cur.fetchall():
        k = (exp, float(K), right)
        legs.setdefault(k, {})[dt] = float(close)
        legs[k]['spot'] = float(spot) if spot is not None else None
    rows = []
    for (exp, K, right), v in legs.items():
        bid, ask, spot = v.get('BID'), v.get('ASK'), v.get('spot')
        if not bid or not ask or not spot or ask <= bid:
            continue
        mid = (bid + ask) / 2
        dte = (exp - d).days
        if dte <= 0:
            continue
        iv = bs_implied_vol(mid, spot, K, dte / 365, R, right)
        if iv is None or not (0.03 < iv < 3.0):
            continue
        rows.append((d, exp, int(dte), float(K), float(K), right,
                     float(round(spot, 4)), float(round(spot, 4)),
                     float(round(mid, 4)), float(round(iv, 5)), 1.0))
    if rows:
        execute_values(cur, """INSERT INTO spy_iv_surface
            (date, expiration, dte, strike_adj, strike_real, option_right,
             spot_adj, spot_real, mid, iv, split_factor)
            VALUES %s ON CONFLICT (date, expiration, strike_adj, option_right) DO NOTHING""", rows)
        conn.commit(); n = cur.rowcount; conn.close(); return n
    conn.close(); return 0


def _resume_start(default='2021-01-01'):
    try:
        conn = psycopg2.connect(**DB); cur = conn.cursor()
        cur.execute("SELECT max(date) FROM spy_iv_surface"); m = cur.fetchone()[0]; conn.close()
        if m:
            import pandas as pd
            return (pd.Timestamp(m) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    except Exception:
        pass
    return default


def main(start, end):
    create_table()
    ds = _dates(start, end)
    if not ds:
        print(f"No quote dates in {start}→{end} (surface already current)."); return
    print(f"Building SPY IV surface for {len(ds)} dates {ds[0]}→{ds[-1]} (from minute table)", flush=True)
    t0 = time.time(); total = 0
    for i, d in enumerate(ds, 1):
        total += process_date(d)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"  [{i}/{len(ds)}] {el:.0f}s ETA {el/i*(len(ds)-i)/60:.1f}min {total:,} rows", flush=True)
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT count(*), min(date), max(date), count(distinct date) FROM spy_iv_surface")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. spy_iv_surface: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    a = p.parse_args()
    import pandas as pd
    start = a.start or _resume_start()
    end = a.end or pd.Timestamp.today().strftime('%Y-%m-%d')
    main(start, end)
