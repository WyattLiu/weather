"""Generalized ETF MINUTE option backfill → {sym}_options_history (SPY-grade for any symbol).

Same as backfill_spy_intraday but parameterized by symbol, reading spot from etf_vix_daily.csv.
GAP-FILLING: skips trade_dates already present in the table, so re-running only fetches missing
sessions (cheap to top up QQQ/IWM's patchy coverage or extend SPY 2018-2020). 1m, RTH-only,
two-sided-only, monthly expiries ≤95 DTE, ±7% strikes. Idempotent.

  venv/bin/python backtest/backfill_etf_intraday.py QQQ 2018-01-01 2026-06-18
  venv/bin/python backtest/backfill_etf_intraday.py SPY 2018-01-01 2021-01-01   # extend
"""
import os
import sys
import time
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import THETA_BASE

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
DAILY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'research', 'spy_vol', 'cache', 'etf_vix_daily.csv')
MAX_DTE, BAND, RTH_OPEN, RTH_CLOSE = 95, 0.07, '09:30', '16:00'


def create_table(sym):
    t = f'{sym.lower()}_options_history'
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute(f"""CREATE TABLE IF NOT EXISTS {t} (
        id BIGSERIAL PRIMARY KEY, trade_date DATE NOT NULL, bar_time TIMESTAMP NOT NULL,
        expiration DATE NOT NULL, strike NUMERIC NOT NULL, option_right CHAR(1) NOT NULL,
        open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume INTEGER, bar_count INTEGER,
        underlying_price NUMERIC, data_type VARCHAR(8) NOT NULL, collected_at TIMESTAMP DEFAULT now(),
        UNIQUE (trade_date, bar_time, expiration, strike, option_right, data_type));
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_date ON {t}(trade_date);
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_exp ON {t}(expiration);
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_strike ON {t}(strike);""")
    conn.commit(); conn.close()
    return t


def existing_dates(table):
    """Dates with REAL minute coverage (>10 distinct bar_times) — EOD-only overlay days
    (1 bar at 16:00 from the earlier eod backfill) are NOT counted, so they get filled."""
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute(f"SELECT trade_date FROM {table} GROUP BY trade_date "
                f"HAVING count(DISTINCT bar_time) > 10")
    out = {r[0].isoformat() for r in cur.fetchall()}; conn.close()
    return out


def _get(url, params):
    try:
        r = requests.get(url, params={**params, 'format': 'json'}, timeout=25)
        return r.json().get('response') if r.status_code == 200 else None
    except Exception:
        return None


def expirations(sym):
    return sorted(e['expiration'] for e in (_get(f'{THETA_BASE}/v3/option/list/expirations', {'symbol': sym}) or []))


def strikes(sym, exp):
    return sorted(float(s['strike']) for s in (_get(f'{THETA_BASE}/v3/option/list/strikes',
                 {'symbol': sym, 'expiration': exp.replace('-', '')}) or []))


def is_3rd_fri(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def quote_bars(sym, exp, K, right, d):
    resp = _get(f'{THETA_BASE}/v3/option/history/quote', {
        'symbol': sym, 'expiration': exp.replace('-', ''), 'right': right, 'strike': K,
        'start_date': d.replace('-', ''), 'end_date': d.replace('-', ''), 'interval': '1m'})
    return resp[0]['data'] if (resp and resp[0].get('data')) else []


def process_day(args):
    sym, table, d_str, spot, exps = args
    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
    rows = []
    for exp in exps:
        try:
            exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = (exp_d - d_obj).days
        if dte < 0 or dte > MAX_DTE or not is_3rd_fri(exp_d):
            continue
        try:
            ks = strikes(sym, exp)
        except Exception:
            continue
        for K in [k for k in ks if abs(k - spot) / max(spot, 1e-6) <= BAND]:
            for right in ('P', 'C'):
                for bar in quote_bars(sym, exp, K, right, d_str):
                    ts = bar.get('timestamp')
                    if not ts:
                        continue
                    hhmm = ts[11:16] if len(ts) >= 16 else ''
                    if hhmm and not (RTH_OPEN <= hhmm <= RTH_CLOSE):
                        continue
                    bid, ask = bar.get('bid'), bar.get('ask')
                    if bid is None or ask is None or bid <= 0 or ask <= bid:
                        continue
                    for dtype, val in (('BID', bid), ('ASK', ask)):
                        rows.append((d_str, ts, exp, float(K), right, float(val), float(val),
                                     float(val), float(val), 0, 0, round(spot, 4), dtype))
    if rows:
        conn = psycopg2.connect(**DB); cur = conn.cursor()
        execute_values(cur, f"""INSERT INTO {table} (trade_date,bar_time,expiration,strike,option_right,
            open,high,low,close,volume,bar_count,underlying_price,data_type) VALUES %s
            ON CONFLICT (trade_date,bar_time,expiration,strike,option_right,data_type) DO NOTHING""", rows)
        conn.commit(); n = cur.rowcount; conn.close()
        return n
    return 0


def main(sym, start, end, workers=6):
    table = create_table(sym)
    have = existing_dates(table)
    spy = pd.read_csv(DAILY, index_col=0, parse_dates=True)[sym].dropna().loc[start:end]
    exps = expirations(sym)
    tasks = [(sym, table, d.strftime('%Y-%m-%d'), float(v), exps)
             for d, v in spy.items() if d.strftime('%Y-%m-%d') not in have]
    print(f"{sym}: {len(spy)} sessions in range, {len(spy)-len(tasks)} already have data, "
          f"{len(tasks)} to fetch", flush=True)
    if not tasks:
        print(f"{sym}: nothing to do."); return
    t0 = time.time(); total = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_day, t): t[2] for t in tasks}
        for f in as_completed(futs):
            try:
                total += f.result()
            except Exception as e:
                print(f"  {futs[f]} FAIL {repr(e)[:70]}", flush=True)
            done += 1
            if done % 20 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {el:.0f}s ETA {el/done*(len(tasks)-done)/60:.0f}min {total:,} rows", flush=True)
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute(f"SELECT count(*), count(distinct trade_date), min(trade_date), max(trade_date) FROM {table}")
    print(f"\n{sym} done {(time.time()-t0)/60:.1f}min. {table}: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
