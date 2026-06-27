"""Full UNG IV surface backfill from ThetaData → Postgres.

Parallel-fetches historical NBBO quotes for UNG options at ATM ± N strikes
near front-month expiry, inverts to IV with split-adjusted spot/strikes,
writes to `ung_iv_surface` table on market_scanner@192.168.1.172.

Idempotent: skips (date, expiration, strike_adj, right) tuples already in PG.
Concurrent: 4 worker threads (matches ThetaData terminal cap).
Resume-safe: kill and rerun anytime.

Usage:
  cd /home/wyatt/weather/ibkr_guided_trade
  ../venv/bin/python backtest/backfill_ung_iv_pg.py --start 2021-06-01 --end 2026-06-03

Expected runtime: ~30-60 min for 5yr coverage (parallel @ 4 workers).
"""
import os
import sys
import time
import argparse
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import (
    split_factor_on,
    get_expirations, get_strikes, get_quote_eod, bs_implied_vol,
)

DB_PARAMS = {
    'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
    'user': 'postgres', 'password': 'shinobi2025',
}


def get_existing_keys(date_str):
    """Return set of (expiration, strike_adj, right) already in PG for this date."""
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute(
        "SELECT expiration, strike_adj, option_right FROM ung_iv_surface WHERE date = %s",
        (date_str,)
    )
    rows = set((r[0].isoformat(), float(r[1]), r[2]) for r in cur.fetchall())
    conn.close()
    return rows


def insert_rows(rows):
    if not rows:
        return 0
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    execute_values(cur, '''
        INSERT INTO ung_iv_surface
        (date, expiration, dte, strike_adj, strike_real, option_right,
         spot_adj, spot_real, mid, iv, split_factor)
        VALUES %s
        ON CONFLICT (date, expiration, strike_adj, option_right) DO NOTHING
    ''', rows)
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n


def process_one_date(args):
    """Process all strikes for one date. Returns rows-inserted count.

    Tries multiple expiries in fallback order: dte_target, +15, +30, +60, then
    weekly (~7-14 DTE). ThetaData NBBO populates ~2-3 wks before expiry, so
    on older dates the first ≥30-DTE contract may have no quotes yet.
    """
    d_str, adj_spot, expirations, dte_target, n_strikes = args
    sf = split_factor_on(d_str)
    real_spot = adj_spot / sf
    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()

    # Build candidate expiry list, ordered by preference (closer to target first,
    # then longer-dated fallbacks, then weekly fallback as last resort)
    candidates = []
    seen = set()
    for dte_pref in (dte_target, dte_target + 15, dte_target + 30, dte_target + 60, 14, 7):
        for exp in expirations:
            try:
                exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
            except Exception:
                continue
            actual = (exp_d - d_obj).days
            if actual >= dte_pref and exp not in seen:
                candidates.append((exp, actual))
                seen.add(exp)
                break

    if not candidates:
        return 0

    existing = get_existing_keys(d_str)

    # Try each candidate; first one that yields >=3 valid quotes wins
    for target_exp, actual_dte in candidates:
        try:
            strikes = get_strikes('UNG', target_exp)
        except Exception:
            continue
        if not strikes:
            continue
        strikes_near = sorted(strikes, key=lambda k: abs(k - real_spot))[:n_strikes*2 + 1]
        pending = []
        for K_real in strikes_near:
            K_adj = K_real * sf
            for right in ['C', 'P']:
                if (target_exp, K_adj, right) in existing:
                    continue
                pending.append((K_real, K_adj, right))
        if not pending:
            # Already complete for this expiry — count as success, no fallback needed
            return 0

        rows_to_insert = []
        for K_real, K_adj, right in pending:
            mid = get_quote_eod('UNG', target_exp, K_real, right, d_str)
            if mid is None or mid <= 0:
                continue
            T = actual_dte / 365
            iv = bs_implied_vol(mid, real_spot, K_real, T, 0.045, right)
            if iv is None or not (0.05 < iv < 3.0):
                continue
            rows_to_insert.append((
                d_str, target_exp, int(actual_dte),
                float(round(K_adj, 4)), float(round(K_real, 4)), right,
                float(round(adj_spot, 4)), float(round(real_spot, 4)),
                float(round(mid, 4)), float(round(iv, 5)), float(round(sf, 4))
            ))

        if len(rows_to_insert) >= 3:
            # Good enough — this expiry has real data
            return insert_rows(rows_to_insert)
        # Otherwise fall through to next candidate expiry

    return 0


def main(start, end, dte_target, n_strikes, max_workers):
    print("Loading dataset...")
    spot_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'master_dataset.csv')
    spot_df = pd.read_csv(spot_csv, index_col=0, parse_dates=True)
    spot_df = spot_df.loc[start:end, ['UNG']].dropna()
    if spot_df.empty:
        print(f"  No new sessions to build in {start}→{end} (surface already current).")
        return
    print(f"  {len(spot_df)} business days from {spot_df.index[0].date()} to {spot_df.index[-1].date()}")

    print("Fetching ThetaData expirations list...")
    expirations = get_expirations('UNG')
    print(f"  {len(expirations)} historical expirations")

    tasks = [(d.strftime('%Y-%m-%d'), float(row['UNG']), expirations, dte_target, n_strikes)
             for d, row in spot_df.iterrows()]

    print(f"Starting parallel backfill ({max_workers} workers, {len(tasks)} dates)...")
    t0 = time.time()
    total_inserted = 0
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_date, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            d_str = futures[fut]
            try:
                inserted = fut.result()
                total_inserted += inserted
            except Exception as e:
                print(f"  {d_str} FAILED: {e}")
            n_done += 1
            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(tasks) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(tasks)}] elapsed {elapsed:.0f}s, ETA {eta/60:.1f}min, total inserted {total_inserted}")

    elapsed = time.time() - t0
    print(f"\nDone. {total_inserted} rows inserted, {elapsed/60:.1f}min total.")

    # Final summary
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT date), MIN(date), MAX(date) FROM ung_iv_surface")
    print(f"PG table now: {cur.fetchone()}")
    conn.close()


def _resume_start(default='2021-06-01'):
    """Day AFTER the last surface date → a daily run only rebuilds new sessions instead of
    rescanning all of history. Falls back to `default` (full history) if the table is empty."""
    try:
        conn = psycopg2.connect(**DB_PARAMS); cur = conn.cursor()
        cur.execute("SELECT max(date) FROM ung_iv_surface")
        m = cur.fetchone()[0]; conn.close()
        if m:
            return (pd.Timestamp(m) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    except Exception:
        pass
    return default


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default=None, help='default: resume from last surface date (full history if empty)')
    p.add_argument('--end', default=None, help='default: today')
    p.add_argument('--dte', type=int, default=30)
    p.add_argument('--n-strikes', type=int, default=5)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    start = args.start or _resume_start()
    end = args.end or date.today().isoformat()
    main(start, end, args.dte, args.n_strikes, args.workers)
