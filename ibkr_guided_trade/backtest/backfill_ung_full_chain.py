"""Full UNG option chain backfill — ALL expirations per date.

Extends backfill_ung_iv_pg.py to track every listed expiration's strikes
(not just front-month). Gives the backtest exact per-(date, expiration)
strike availability so it matches live constraints.

For each business day:
  1. Pull all UNG expirations active that day from ThetaData
  2. Filter to a reasonable window: 0-120 DTE + LEAPS  (cap ~10 per day)
  3. For each expiration: fetch ATM±5 strike grid, EOD quote, invert to IV
  4. Insert all (date, exp, strike, right) rows to PG

Idempotent: skips existing (date, exp, strike, right) tuples.
Parallelism: 4 workers (ThetaData terminal cap).

Expected runtime: 1.5-3 hrs for 5yr coverage at 4 workers.
"""
from __future__ import annotations
import os
import sys
import time
import argparse
import psycopg2
from psycopg2.extras import execute_values
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_thetadata_iv import (
    THETA_BASE, split_factor_on, get_expirations, get_strikes,
    get_quote_eod, bs_implied_vol,
)
from backfill_ung_iv_pg import DB_PARAMS, insert_rows, get_existing_keys


# How many expirations per day to track. Strategy: all weeklies in first 60d
# + monthlies (3rd Friday) out to 200 DTE + LEAPS (Jan/Jun of future years).
# Cap at MAX_EXP_PER_DAY to control cost.
MAX_EXP_PER_DAY = 12
MAX_DTE_TRACKED = 250  # don't track expiries beyond ~8 months out
N_STRIKES_EACH_SIDE = 5


def _is_third_friday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21


def _select_expirations_for_date(d_obj, all_expirations):
    """Pick which expirations to backfill for a given date. Returns sorted list.
    Strategy: all weeklies in first 30 DTE + monthlies through 200 DTE + nearest LEAP.
    """
    selected = []
    for exp in all_expirations:
        try:
            exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
        except Exception:
            continue
        dte = (exp_d - d_obj).days
        if dte < 0:
            continue
        if dte > MAX_DTE_TRACKED:
            # Allow LEAPS (1 per year of future)
            if exp_d.month in (1, 6) and exp_d.year > d_obj.year:
                selected.append((exp, dte))
                # only track first LEAP per cycle
            continue
        # In-window: include all weeklies for short DTE, monthlies for longer
        if dte <= 30:
            selected.append((exp, dte))  # all weeklies+monthlies in 0-30d
        elif _is_third_friday(exp_d):
            selected.append((exp, dte))  # monthlies only beyond 30d
    selected.sort(key=lambda x: x[1])
    return [s[0] for s in selected[:MAX_EXP_PER_DAY]]


def process_one_date_full(args):
    """Process ALL selected expirations for one date. Returns rows inserted."""
    d_str, adj_spot, all_expirations = args
    sf = split_factor_on(d_str)
    real_spot = adj_spot / sf
    d_obj = datetime.strptime(d_str, '%Y-%m-%d').date()

    target_exps = _select_expirations_for_date(d_obj, all_expirations)
    if not target_exps:
        return 0

    existing = get_existing_keys(d_str)
    total_inserted = 0

    for target_exp in target_exps:
        exp_d = datetime.strptime(target_exp, '%Y-%m-%d').date()
        actual_dte = (exp_d - d_obj).days

        try:
            strikes = get_strikes('UNG', target_exp)
        except Exception:
            continue
        if not strikes:
            continue
        strikes_near = sorted(strikes, key=lambda k: abs(k - real_spot))[:N_STRIKES_EACH_SIDE*2 + 1]

        pending = []
        for K_real in strikes_near:
            K_adj = K_real * sf
            for right in ['C', 'P']:
                if (target_exp, K_adj, right) in existing:
                    continue
                pending.append((K_real, K_adj, right))
        if not pending:
            continue

        rows_to_insert = []
        for K_real, K_adj, right in pending:
            mid = get_quote_eod('UNG', target_exp, K_real, right, d_str)
            if mid is None or mid <= 0:
                continue
            T = max(1, actual_dte) / 365
            iv = bs_implied_vol(mid, real_spot, K_real, T, 0.045, right)
            if iv is None or not (0.05 < iv < 3.0):
                continue
            rows_to_insert.append((
                d_str, target_exp, int(actual_dte),
                float(round(K_adj, 4)), float(round(K_real, 4)), right,
                float(round(adj_spot, 4)), float(round(real_spot, 4)),
                float(round(mid, 4)), float(round(iv, 5)), float(round(sf, 4))
            ))

        if rows_to_insert:
            n_ins = insert_rows(rows_to_insert)
            total_inserted += n_ins
            # Update local existing set so we don't repeat strikes within this date
            for K_real, K_adj, right in pending:
                existing.add((target_exp, K_adj, right))

    return total_inserted


def main(start, end, max_workers):
    import pandas as pd
    print(f"Loading dataset...")
    spot_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cache', 'master_dataset.csv')
    spot_df = pd.read_csv(spot_csv, index_col=0, parse_dates=True)
    spot_df = spot_df.loc[start:end, ['UNG']].dropna()
    print(f"  {len(spot_df)} business days from {spot_df.index[0].date()} to {spot_df.index[-1].date()}")

    print(f"Fetching ThetaData expirations list...")
    all_exps = get_expirations('UNG')
    print(f"  {len(all_exps)} historical expirations")

    tasks = [(d.strftime('%Y-%m-%d'), float(row['UNG']), all_exps)
             for d, row in spot_df.iterrows()]

    print(f"Starting parallel FULL-CHAIN backfill ({max_workers} workers, {len(tasks)} dates)...")
    t0 = time.time()
    total_inserted = 0
    n_done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one_date_full, t): t[0] for t in tasks}
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
                print(f"  [{n_done}/{len(tasks)}] elapsed {elapsed:.0f}s, "
                      f"ETA {eta/60:.1f}min, total inserted {total_inserted}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {total_inserted} rows inserted, {elapsed/60:.1f}min total.")

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("""SELECT COUNT(*), COUNT(DISTINCT date),
                          COUNT(DISTINCT (date, expiration)) as date_exp_pairs
                   FROM ung_iv_surface""")
    rows, days, pairs = cur.fetchone()
    print(f"PG table now: {rows} rows, {days} days, {pairs} (date,exp) pairs")
    conn.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2021-06-01')
    p.add_argument('--end', default=date.today().isoformat())
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    main(args.start, args.end, args.workers)
