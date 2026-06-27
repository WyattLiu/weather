"""IV data quality checks + cross-source validation for ung_iv_surface.

Three layers of validation:

1. INTRA-SOURCE sanity (PG data only):
   - IV range bounds (0.05 ≤ IV ≤ 3.0 — bs_implied_vol already enforces)
   - Put-call parity: C - P ≈ S - K·e^(-rT) per (date, exp, strike)
   - Strike monotonicity: |ΔIV/ΔK| not absurdly large (no jumps)
   - Term-structure monotonicity: same K across expiries shouldn't flip wildly
   - Mid > 0, intrinsic-respecting (call mid ≥ max(0, S-K))

2. CROSS-SOURCE vs yfinance (recent days only):
   - Compare today's PG IV vs yfinance impliedVolatility column
   - Flag legs where PG IV deviates >50% from yfinance

3. CROSS-SOURCE vs realized vol baseline:
   - PG IV should be > realized 30d vol most days (vol risk premium)
   - Flag days where median IV is far below RV30

Usage:
    venv/bin/python backtest/iv_quality_checks.py [--recent N]
"""
from __future__ import annotations
import os
import math
import argparse
import psycopg2
from datetime import date as _date
from collections import defaultdict

DB_PARAMS = {
    'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
    'user': 'postgres', 'password': 'shinobi2025',
}


def _connect():
    return psycopg2.connect(**DB_PARAMS, connect_timeout=8)


def _pull_recent(n_days=60):
    """Pull PG IV data for last N days; returns list of dicts."""
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, expiration, dte, strike_real, option_right,
               spot_real, mid, iv
        FROM ung_iv_surface
        WHERE date >= (CURRENT_DATE - INTERVAL '%s days')
        ORDER BY date, expiration, strike_real, option_right
    """, (n_days,))
    rows = cur.fetchall()
    conn.close()
    keys = ['date', 'exp', 'dte', 'K', 'right', 'spot', 'mid', 'iv']
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        # Decimal → float for arithmetic
        for k in ('K', 'spot', 'mid', 'iv'):
            d[k] = float(d[k]) if d[k] is not None else 0.0
        d['dte'] = int(d['dte']) if d['dte'] is not None else 0
        out.append(d)
    return out


def check_intrinsic_violation(data):
    """Mid should be ≥ intrinsic (max(0, S-K) for call, max(0, K-S) for put)."""
    violations = []
    for r in data:
        intr = max(0, r['spot'] - r['K']) if r['right'] == 'C' else max(0, r['K'] - r['spot'])
        # Allow 1% tolerance for round/spread
        if r['mid'] + 0.01 < intr:
            violations.append({
                'date': r['date'].isoformat(), 'exp': r['exp'].isoformat(),
                'K': r['K'], 'right': r['right'],
                'mid': r['mid'], 'intrinsic': intr,
                'gap': intr - r['mid'],
            })
    return violations


def check_put_call_parity(data, r=0.045, tolerance=0.20):
    """C - P ≈ S - K·e^(-rT). Returns violations with |residual| > tolerance × spot."""
    violations = []
    # Group by (date, exp, K)
    grouped = defaultdict(dict)
    for r_ in data:
        key = (r_['date'], r_['exp'], r_['K'])
        grouped[key][r_['right']] = r_
    for (d, exp, K), legs in grouped.items():
        if 'C' not in legs or 'P' not in legs:
            continue
        S = legs['C']['spot']
        dte = legs['C']['dte']
        T = max(1, dte) / 365.0
        expected_lhs = legs['C']['mid'] - legs['P']['mid']
        expected_rhs = S - K * math.exp(-r * T)
        residual = expected_lhs - expected_rhs
        # Tolerance scaled to spot (parity violations of >20% of spot = bad data)
        if abs(residual) > tolerance * S:
            violations.append({
                'date': d.isoformat(), 'exp': exp.isoformat(), 'K': K,
                'C': legs['C']['mid'], 'P': legs['P']['mid'],
                'S': S, 'lhs (C-P)': round(expected_lhs, 3),
                'rhs (S-Ke^-rT)': round(expected_rhs, 3),
                'residual': round(residual, 3),
                'residual_pct_of_spot': round(residual / S * 100, 1),
            })
    return violations


def check_iv_smile_jumps(data, max_jump_pct=100):
    """Adjacent strikes (same date, exp, right) shouldn't have >100% IV change."""
    violations = []
    # Group by (date, exp, right) — sorted by K
    grouped = defaultdict(list)
    for r in data:
        grouped[(r['date'], r['exp'], r['right'])].append(r)
    for key, legs in grouped.items():
        legs.sort(key=lambda x: x['K'])
        for i in range(1, len(legs)):
            iv_prev, iv_cur = legs[i-1]['iv'], legs[i]['iv']
            if iv_prev <= 0:
                continue
            change = abs(iv_cur - iv_prev) / iv_prev * 100
            if change > max_jump_pct:
                violations.append({
                    'date': key[0].isoformat(), 'exp': key[1].isoformat(), 'right': key[2],
                    'K_prev': legs[i-1]['K'], 'K_cur': legs[i]['K'],
                    'iv_prev': round(iv_prev, 3), 'iv_cur': round(iv_cur, 3),
                    'change_pct': round(change, 1),
                })
    return violations


def cross_check_yfinance(latest_n_days=3):
    """Compare PG IV against yfinance live chain for recent days.
    yfinance only has TODAY's chain, so we can only validate today's PG data.
    """
    out = {'compared': 0, 'matched': 0, 'deviations': []}
    try:
        import yfinance as yf
    except Exception:
        out['error'] = 'yfinance not available'
        return out
    _date.today()
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, expiration, strike_real, option_right, iv, mid
        FROM ung_iv_surface
        WHERE date = (SELECT MAX(date) FROM ung_iv_surface)
        ORDER BY expiration, strike_real, option_right
    """)
    pg_rows = cur.fetchall()
    conn.close()
    if not pg_rows:
        out['error'] = 'no recent PG data'
        return out
    pg_rows[0][0]

    # Group PG by expiration
    pg_by_exp = defaultdict(dict)
    for d, exp, K, right, iv, mid in pg_rows:
        pg_by_exp[exp][(float(K), right)] = (float(iv), float(mid))

    ung = yf.Ticker('UNG')
    for exp_d, contracts in pg_by_exp.items():
        exp_str = exp_d.isoformat()
        if exp_str not in ung.options:
            continue
        try:
            chain = ung.option_chain(exp_str)
        except Exception:
            continue
        for side_name, side_df in [('P', chain.puts), ('C', chain.calls)]:
            for _, row in side_df.iterrows():
                K = float(row['strike'])
                key = (K, side_name)
                if key not in contracts:
                    continue
                pg_iv, pg_mid = contracts[key]
                yf_iv = float(row.get('impliedVolatility') or 0)
                if yf_iv <= 0:
                    continue
                out['compared'] += 1
                dev_pct = abs(pg_iv - yf_iv) / yf_iv * 100
                if dev_pct < 25:
                    out['matched'] += 1
                else:
                    out['deviations'].append({
                        'exp': exp_str, 'K': K, 'right': side_name,
                        'pg_iv': round(pg_iv, 3), 'yf_iv': round(yf_iv, 3),
                        'dev_pct': round(dev_pct, 1),
                    })
    return out


def check_iv_vs_realized_vol():
    """IV should exceed realized 30d vol most days (volatility risk premium)."""
    import pandas as pd
    import numpy as np
    csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'cache', 'master_dataset.csv')
    if not os.path.exists(csv):
        return {'error': 'master_dataset missing'}
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    # master_dataset has duplicate rows per date (one with UNG value, one NaN) —
    # drop NaNs first so pct_change connects consecutive real values
    ung = df['UNG'].dropna()
    rets = ung.pct_change().dropna()
    rv30 = rets.rolling(30).std() * np.sqrt(252)
    conn = _connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT date, AVG(iv) AS med_iv
        FROM ung_iv_surface
        WHERE ABS(strike_real - spot_real) <= 1.0
        GROUP BY date ORDER BY date
    """)
    rows = cur.fetchall()
    conn.close()
    iv_below_rv_days = 0
    total = 0
    big_gaps = []
    # rv30 is indexed by timestamps; normalize to a date-keyed dict for lookups
    rv30_by_date = {ts.date(): float(v) for ts, v in rv30.items()
                    if not pd.isna(v) and v > 0}
    for d, med_iv in rows:
        rv = rv30_by_date.get(d)
        if rv is None or rv <= 0:
            continue
        med_iv_f = float(med_iv)
        total += 1
        if med_iv_f < rv * 0.85:
            iv_below_rv_days += 1
            if med_iv_f < rv * 0.5:
                big_gaps.append({'date': d.isoformat(), 'med_iv': round(med_iv_f, 3),
                                 'rv30': round(rv, 3),
                                 'ratio': round(med_iv_f/rv, 2)})
    return {
        'days_compared': total,
        'days_iv_below_rv': iv_below_rv_days,
        'pct_anomaly': round(iv_below_rv_days/total*100, 1) if total > 0 else 0,
        'big_gaps_count': len(big_gaps),
        'big_gaps_sample': big_gaps[:5],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--recent', type=int, default=60, help='days to scan (intra checks)')
    args = p.parse_args()

    print(f'=== IV QUALITY CHECKS (last {args.recent} days) ===\n')
    data = _pull_recent(args.recent)
    print(f'Loaded {len(data)} rows for intra-source checks.\n')

    print('--- 1. INTRINSIC VIOLATIONS (mid < intrinsic) ---')
    iv = check_intrinsic_violation(data)
    print(f'  {len(iv)} violations found')
    for v in iv[:5]:
        print(f'    {v}')
    print()

    print('--- 2. PUT-CALL PARITY (|C - P - (S - Ke^-rT)| > 20% spot) ---')
    pcv = check_put_call_parity(data)
    print(f'  {len(pcv)} violations found')
    for v in pcv[:5]:
        print(f'    {v}')
    print()

    print('--- 3. IV SMILE JUMPS (|ΔIV/IV| > 100% between adjacent K) ---')
    sj = check_iv_smile_jumps(data)
    print(f'  {len(sj)} violations found')
    for v in sj[:5]:
        print(f'    {v}')
    print()

    print('--- 4. CROSS-SOURCE: PG vs yfinance (today only) ---')
    cs = cross_check_yfinance()
    print(f'  compared: {cs.get("compared", 0)}, matched: {cs.get("matched", 0)}')
    print(f'  deviations >25%: {len(cs.get("deviations", []))}')
    for v in cs.get('deviations', [])[:5]:
        print(f'    {v}')
    if cs.get('error'):
        print(f'  ERROR: {cs["error"]}')
    print()

    print('--- 5. CROSS-CHECK: IV vs realized 30d vol (VRP sanity) ---')
    vrp = check_iv_vs_realized_vol()
    if 'error' in vrp:
        print(f'  ERROR: {vrp["error"]}')
    else:
        print(f'  days compared: {vrp["days_compared"]}')
        print(f'  days IV < 85% of RV: {vrp["days_iv_below_rv"]} ({vrp["pct_anomaly"]}%)')
        print(f'  big gaps (IV < 50% RV): {vrp["big_gaps_count"]}')
        for v in vrp.get('big_gaps_sample', []):
            print(f'    {v}')
    print()

    # Summary
    issues = len(iv) + len(pcv) + len(sj) + len(cs.get('deviations', []))
    print(f'=== TOTAL ISSUES: {issues} ===')
    if issues == 0:
        print('All quality checks PASSED.')


if __name__ == '__main__':
    main()
