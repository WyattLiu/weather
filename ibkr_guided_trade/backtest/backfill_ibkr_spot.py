"""Minute SPOT (underlying) bars from IBKR → Postgres (etf_spot_minute).

ThetaData stock-minute is subscription-gated, so we source the intraday underlying from IBKR
(ib_insync) for SPY/QQQ/IWM, multi-year, 1-min RTH TRADES. Pages back in 20-day chunks with
IBKR pacing. Idempotent (ON CONFLICT) + skips chunks already fully present ("we might have some").
Pairs with the minute OPTION tables → full intraday execution data (spot + option bid/ask).

  venv/bin/python backtest/backfill_ibkr_spot.py 2018-01-01     # back to 2018
"""
import os
import sys
import time
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.common import connect          # IBKR connect helper (192.168.1.127:20009)
from ib_insync import Stock

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
SYMBOLS = ('SPY', 'QQQ', 'IWM')
CHUNK_DAYS = 20
PACE_SEC = 11        # ≤6 historical reqs/min → avoid IBKR pacing violations


def create_table():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS etf_spot_minute (
            symbol TEXT NOT NULL, bar_time TIMESTAMPTZ NOT NULL,
            open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume BIGINT,
            UNIQUE (symbol, bar_time));
        CREATE INDEX IF NOT EXISTS idx_esm_sym_time ON etf_spot_minute(symbol, bar_time);
    """)
    conn.commit(); conn.close()


def chunk_present(cur, sym, d0, d1):
    cur.execute("SELECT count(*) FROM etf_spot_minute WHERE symbol=%s AND bar_time>=%s AND bar_time<%s",
                (sym, d0, d1))
    return cur.fetchone()[0] >= 300          # ~1 RTH session of minutes ⇒ treat as present


def insert(cur, sym, bars):
    rows = [(sym, b.date, float(b.open), float(b.high), float(b.low), float(b.close), int(b.volume))
            for b in bars]
    if rows:
        execute_values(cur, """INSERT INTO etf_spot_minute (symbol,bar_time,open,high,low,close,volume)
            VALUES %s ON CONFLICT (symbol,bar_time) DO NOTHING""", rows)
    return len(rows)


def main(start):
    create_table()
    ib = connect(client_id=78)
    conn = psycopg2.connect(**DB); conn.autocommit = True; cur = conn.cursor()
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    t0 = time.time(); total = 0
    for sym in SYMBOLS:
        c = Stock(sym, 'SMART', 'USD')
        ib.qualifyContracts(c)
        end = datetime.now()
        got = 0; reqs = 0
        while end > start_dt:
            d0 = end - timedelta(days=CHUNK_DAYS)
            if not chunk_present(cur, sym, d0, end):
                try:
                    bars = ib.reqHistoricalData(c, endDateTime=end, durationStr=f'{CHUNK_DAYS} D',
                                                barSizeSetting='1 min', whatToShow='TRADES',
                                                useRTH=True, formatDate=1)
                except Exception as e:
                    print(f"  {sym} {end.date()} req fail: {repr(e)[:60]}", flush=True); bars = []
                reqs += 1
                if bars:
                    n = insert(cur, sym, bars); got += n; total += n
                    end = bars[0].date.replace(tzinfo=None)   # page back from earliest bar
                else:
                    end = d0                                  # nothing → step back a chunk
                time.sleep(PACE_SEC)
            else:
                end = d0                                       # already have it → skip, step back
            if reqs and reqs % 10 == 0:
                print(f"  {sym}: {got:,} rows, back to {end.date()}, {reqs} reqs, {(time.time()-t0)/60:.0f}min", flush=True)
        print(f"{sym} done: {got:,} rows back to {start}", flush=True)
    ib.disconnect()
    cur.execute("SELECT symbol, count(*), min(bar_time), max(bar_time) FROM etf_spot_minute GROUP BY symbol")
    for r in cur.fetchall():
        print(f"  {r[0]}: {r[1]:,} bars {r[2]}→{r[3]}", flush=True)
    conn.close()
    print(f"\nDONE {(time.time()-t0)/60:.1f}min, {total:,} new rows", flush=True)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '2018-01-01')
