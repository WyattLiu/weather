"""Multi-ETF EOD option backfill → Postgres ({sym}_options_history). FAST path for the
cross-asset / long-history vega study (EOD only — one row/contract/day, ~400x less than minute).

One ThetaData /v3/option/history/eod call returns a contract's WHOLE history, so we enumerate
contracts (monthly expiry ≤95 DTE, strikes within ±BAND of the spot range over the expiry's
active window) and fetch each once. Stored as BID/ASK rows at bar_time = date 16:00 (same schema
as the minute tables, so the study's eod_mid works unchanged). Idempotent.

  venv/bin/python backtest/backfill_etf_eod.py QQQ 2018-01-01 2026-06-18
  venv/bin/python backtest/backfill_etf_eod.py SPY 2018-01-01 2021-01-01   # 2018-2020 extension
"""
import os
import sys
import time
import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import THETA_BASE

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
DAILY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'research', 'spy_vol', 'cache', 'etf_vix_daily.csv')
MAX_DTE, BAND = 95, 0.08


def create_table(sym):
    t = f'{sym.lower()}_options_history'
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {t} (
            id BIGSERIAL PRIMARY KEY, trade_date DATE NOT NULL, bar_time TIMESTAMP NOT NULL,
            expiration DATE NOT NULL, strike NUMERIC NOT NULL, option_right CHAR(1) NOT NULL,
            open NUMERIC, high NUMERIC, low NUMERIC, close NUMERIC, volume INTEGER, bar_count INTEGER,
            underlying_price NUMERIC, data_type VARCHAR(8) NOT NULL, collected_at TIMESTAMP DEFAULT now(),
            UNIQUE (trade_date, bar_time, expiration, strike, option_right, data_type));
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_date ON {t}(trade_date);
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_exp ON {t}(expiration);
        CREATE INDEX IF NOT EXISTS idx_{sym.lower()}_oh_strike ON {t}(strike);
    """)
    conn.commit(); conn.close()
    return t


def _get(url, params):
    try:
        r = requests.get(url, params={**params, 'format': 'json'}, timeout=30)
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


def fetch_contract(sym, exp, K, right, d0, d1, table):
    resp = _get(f'{THETA_BASE}/v3/option/history/eod', {
        'symbol': sym, 'expiration': exp.replace('-', ''), 'right': right, 'strike': K,
        'start_date': d0.replace('-', ''), 'end_date': d1.replace('-', '')})
    if not resp or not resp[0].get('data'):
        return 0
    rows = []
    for b in resp[0]['data']:
        ts = b.get('created'); bid, ask = b.get('bid'), b.get('ask')
        if not ts or bid is None or ask is None or bid <= 0 or ask <= bid:
            continue
        d = ts[:10]; bt = f'{d} 16:00:00'
        for dtype, val in (('BID', bid), ('ASK', ask)):
            rows.append((d, bt, exp, float(K), right, float(val), float(val), float(val), float(val),
                         0, 0, None, dtype))
    if not rows:
        return 0
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    execute_values(cur, f"""INSERT INTO {table} (trade_date,bar_time,expiration,strike,option_right,
        open,high,low,close,volume,bar_count,underlying_price,data_type) VALUES %s
        ON CONFLICT (trade_date,bar_time,expiration,strike,option_right,data_type) DO NOTHING""", rows)
    conn.commit(); n = cur.rowcount; conn.close()
    return n


def main(sym, start, end, workers=6):
    table = create_table(sym)
    daily = pd.read_csv(DAILY, index_col=0, parse_dates=True)[sym].dropna()
    exps = [e for e in expirations(sym)
            if start <= e <= (pd.Timestamp(end) + pd.Timedelta(days=MAX_DTE)).strftime('%Y-%m-%d')
            and is_3rd_fri(pd.Timestamp(e).date())]
    tasks = []
    for exp in exps:
        ed = pd.Timestamp(exp)
        a0 = max(pd.Timestamp(start), ed - pd.Timedelta(days=MAX_DTE)); a1 = min(pd.Timestamp(end), ed)
        if a0 >= a1:
            continue
        win = daily.loc[a0:a1]
        if win.empty:
            continue
        lo, hi = win.min() * (1 - BAND), win.max() * (1 + BAND)
        ks = [k for k in strikes(sym, exp) if lo <= k <= hi]
        for K in ks:
            for r in ('C', 'P'):
                tasks.append((exp, K, r, a0.strftime('%Y-%m-%d'), a1.strftime('%Y-%m-%d')))
    print(f"{sym}: {len(exps)} monthly expiries, {len(tasks)} contracts {start}→{end}", flush=True)
    t0 = time.time(); total = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_contract, sym, e, k, r, d0, d1, table): 1 for e, k, r, d0, d1 in tasks}
        for f in as_completed(futs):
            try:
                total += f.result()
            except Exception:
                pass
            done += 1
            if done % 1000 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {el:.0f}s ETA {el/done*(len(tasks)-done)/60:.1f}min {total:,} rows", flush=True)
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute(f"SELECT count(*), min(trade_date), max(trade_date) FROM {table}")
    print(f"\n{sym} done {(time.time()-t0)/60:.1f}min. {table}: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    sym, start, end = sys.argv[1], sys.argv[2], sys.argv[3]
    main(sym, start, end)
