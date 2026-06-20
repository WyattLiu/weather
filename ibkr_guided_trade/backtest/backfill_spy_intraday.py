"""SPY INTRADAY option backfill → Postgres (market_scanner.spy_options_history).

Mirrors the UNG pipeline (ung_options_history) so SPY reaches the same maturity: 1m bid/ask
bars, RTH-only, two-sided-only, unique on (trade_date,bar_time,expiration,strike,right,data_type).
Source: ThetaData v3 (/v3/option/history/quote, interval=1m). Idempotent + parallel + auto-resume.

SCOPED for the low-VIX VEGA-SCRAPING straddle/strangle study (full SPY chain would be ~600M rows):
  • MONTHLY expiries only (3rd-Friday standard contracts — the liquid ones for 30-90 DTE vega).
  • DTE 0..MAX_DTE (track a 30-90 DTE entry all the way down to expiry).
  • strikes within ±BAND_PCT of spot (ATM straddle + strangle wings + the hold's moves).
SPY does not split in-period → no split adjustment. Underlying spot read from spy_vix_daily.csv.

  venv/bin/python backtest/backfill_spy_intraday.py --start 2024-01-01 --end 2026-06-18
  venv/bin/python backtest/backfill_spy_intraday.py            # auto-resume → today
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
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import THETA_BASE

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
SPY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'research', 'spy_vol', 'cache', 'spy_vix_daily.csv')
MAX_DTE = 95
BAND_PCT = 0.07
RTH_OPEN, RTH_CLOSE = '09:30', '16:00'


def create_table():
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spy_options_history (
            id BIGSERIAL PRIMARY KEY,
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
        CREATE INDEX IF NOT EXISTS idx_spy_oh_date ON spy_options_history(trade_date);
        CREATE INDEX IF NOT EXISTS idx_spy_oh_exp  ON spy_options_history(expiration);
        CREATE INDEX IF NOT EXISTS idx_spy_oh_strike ON spy_options_history(strike);
        CREATE INDEX IF NOT EXISTS idx_spy_oh_time ON spy_options_history(bar_time);
    """)
    conn.commit(); conn.close()


def _get(url, params):
    r = requests.get(url, params={**params, 'format': 'json'}, timeout=25)
    if r.status_code != 200:
        return None
    try:
        return r.json().get('response')
    except Exception:
        return None


def expirations():
    resp = _get(f'{THETA_BASE}/v3/option/list/expirations', {'symbol': 'SPY'})
    return sorted(e['expiration'] for e in (resp or []))


def _is_third_friday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21    # standard monthly expiry


def strikes(expiration):
    resp = _get(f'{THETA_BASE}/v3/option/list/strikes',
                {'symbol': 'SPY', 'expiration': expiration.replace('-', '')})
    return sorted(float(s['strike']) for s in (resp or []))


def quote_bars(expiration, strike, right, date_str, interval):
    resp = _get(f'{THETA_BASE}/v3/option/history/quote', {
        'symbol': 'SPY', 'expiration': expiration.replace('-', ''),
        'right': right, 'strike': strike,
        'start_date': date_str.replace('-', ''), 'end_date': date_str.replace('-', ''),
        'interval': interval})
    if not resp or not resp[0].get('data'):
        return []
    return resp[0]['data']


def process_day(args):
    d_str, spot, exps, interval = args
    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()
    rows = []
    for exp in exps:
        try:
            exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = (exp_d - d_obj).days
        if dte < 0 or dte > MAX_DTE or not _is_third_friday(exp_d):
            continue
        try:
            ks = strikes(exp)
        except Exception:
            continue
        near = [k for k in ks if abs(k - spot) / max(spot, 1e-6) <= BAND_PCT]
        for K in near:
            for right in ('P', 'C'):
                for bar in quote_bars(exp, K, right, d_str, interval):
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
                        rows.append((d_str, ts, exp, float(K), right,
                                     float(val), float(val), float(val), float(val),
                                     0, 0, round(spot, 4), dtype))
    if rows:
        conn = psycopg2.connect(**DB); cur = conn.cursor()
        execute_values(cur, """INSERT INTO spy_options_history
            (trade_date, bar_time, expiration, strike, option_right,
             open, high, low, close, volume, bar_count, underlying_price, data_type)
            VALUES %s ON CONFLICT (trade_date, bar_time, expiration, strike, option_right, data_type)
            DO NOTHING""", rows)
        conn.commit(); n = cur.rowcount; conn.close()
        return n
    return 0


def _resume_start(default='2024-01-01'):
    try:
        conn = psycopg2.connect(**DB); cur = conn.cursor()
        cur.execute("SELECT max(trade_date) FROM spy_options_history")
        m = cur.fetchone()[0]; conn.close()
        if m:
            return (pd.Timestamp(m) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    except Exception:
        pass
    return default


def main(start, end, interval, workers):
    create_table()
    spy = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)['SPY'].dropna().loc[start:end]
    if spy.empty:
        print(f"No SPY sessions in {start}→{end} (already current).")
        return
    exps = expirations()
    print(f"{len(spy)} sessions {spy.index[0].date()}→{spy.index[-1].date()}; "
          f"{len(exps)} expiries; MONTHLY only, ≤{MAX_DTE}DTE, ±{BAND_PCT:.0%} strikes, {interval} bars",
          flush=True)
    tasks = [(d.strftime('%Y-%m-%d'), float(v), exps, interval) for d, v in spy.items()]
    t0 = time.time(); total = done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_day, t): t[0] for t in tasks}
        for f in as_completed(futs):
            try:
                total += f.result()
            except Exception as e:
                print(f"  {futs[f]} FAILED: {repr(e)[:90]}", flush=True)
            done += 1
            if done % 10 == 0:
                el = time.time() - t0
                print(f"  [{done}/{len(tasks)}] {el:.0f}s ETA {el/done*(len(tasks)-done)/60:.1f}min "
                      f"{total:,} rows", flush=True)
    conn = psycopg2.connect(**DB); cur = conn.cursor()
    cur.execute("SELECT count(*), min(trade_date), max(trade_date), count(DISTINCT trade_date) "
                "FROM spy_options_history")
    print(f"\nDone in {(time.time()-t0)/60:.1f}min. spy_options_history: {cur.fetchone()}", flush=True)
    conn.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default=None, help='default: resume from last ingested date')
    p.add_argument('--end', default=None, help='default: today')
    p.add_argument('--interval', default='1m')
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    start = a.start or _resume_start()
    end = a.end or pd.Timestamp.today().strftime('%Y-%m-%d')
    main(start, end, a.interval, a.workers)
