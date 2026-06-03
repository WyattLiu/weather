"""Fetch UNG historical option chain + greeks from ThetaData.

CRITICAL CAVEAT (discovered 2026-06-03):
ThetaData returns UNADJUSTED prices and UNADJUSTED strikes. UNG has had
multiple reverse splits (Feb 2017, Apr 2022, Apr 2024 — all 1-for-4).
So adjusted spot $21.76 on 2024-01-03 corresponds to UNADJUSTED ~$5.44.
A strike "$22" query returns DEEP ITM contracts (put bid $16.50 because
true market strike was 22 vs $5.44 spot).

To use ThetaData for backtest pricing we MUST first build a split-
adjustment map: yfinance_adjusted_strike → ThetaData_unadjusted_strike
per historical date. Until then, use IBKR IV30 (single time-series,
no strike mapping needed).


Per ~/spx_strategies/docs/data_sources/api_reference.md, ThetaData Options
STANDARD covers full OPRA NBBO + greeks for any underlying, including UNG.
Terminal at http://127.0.0.1:25503.

This gives us per-strike historical IV with FULL SKEW — much better than
the single IV30 series we get from IBKR. Backtest option pricing for tail
strikes (deep OTM wings, momentum calls) will be accurate.

Strategy:
- For each historical date, pick the front-month expiry (closest >=21d out)
- Fetch quotes for ATM ± N strikes
- Back out IV from BSM using mid quote
- Build (date, dte, K, right) → IV table

Output: backtest/cache/ung_iv_surface.csv
"""
import os
import sys
import time
import math
import argparse
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from scipy.stats import norm

THETA_BASE = 'http://127.0.0.1:25503'
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# UNG split history (from yfinance .splits): (date, ratio) where ratio < 1
# means reverse split. Cumulative factor going FORWARD from each date to
# today, used to convert yfinance-adjusted prices to real market prices.
UNG_SPLITS = [
    ('2011-03-09', 0.50),
    ('2012-02-22', 0.25),
    ('2018-01-05', 0.25),
    ('2024-01-24', 0.25),
]


def split_factor_on(date_str):
    """Cumulative split factor as of a given date.

    adjusted_price_yfinance = real_market_price * factor
    real_market_price = adjusted_price_yfinance / factor
    real_strike = adjusted_strike / factor
    """
    factor = 1.0
    for split_date, ratio in UNG_SPLITS:
        if date_str < split_date:
            factor /= ratio  # going backward through a 0.25 split = 4x
    return factor


def bs_implied_vol(target_price, S, K, T, r, right):
    """Newton-Raphson IV from BSM mid price. Returns None on failure."""
    if T <= 0 or target_price <= 0:
        return None
    sigma = 0.5  # initial guess
    for _ in range(50):
        d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        if right == 'C':
            price = S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
        else:
            price = K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        vega = S*math.sqrt(T)*norm.pdf(d1)
        if vega < 1e-6:
            return None
        diff = price - target_price
        if abs(diff) < 1e-4:
            return max(0.01, min(5.0, sigma))
        sigma -= diff / vega
        if sigma < 0.001 or sigma > 10:
            return None
    return None


def get_expirations(symbol='UNG'):
    r = requests.get(f'{THETA_BASE}/v3/option/list/expirations',
                     params={'symbol': symbol, 'format': 'json'}, timeout=15)
    r.raise_for_status()
    return sorted([e['expiration'] for e in r.json()['response']])


def get_strikes(symbol, expiration):
    r = requests.get(f'{THETA_BASE}/v3/option/list/strikes',
                     params={'symbol': symbol, 'expiration': expiration.replace('-',''),
                             'format': 'json'}, timeout=15)
    r.raise_for_status()
    return sorted([s['strike'] for s in r.json()['response']])


def get_quote_eod(symbol, expiration, strike, right, date_str):
    """EOD-ish quote for one option contract on one date.

    Uses /v3/option/history/quote with 1h interval, takes any post-noon bar
    with non-zero quote. Returns mid price or None.
    """
    r = requests.get(f'{THETA_BASE}/v3/option/history/quote', params={
        'symbol': symbol, 'expiration': expiration.replace('-',''),
        'right': right, 'strike': strike,
        'start_date': date_str.replace('-',''),
        'end_date': date_str.replace('-',''),
        'interval': '1h', 'format': 'json',
    }, timeout=15)
    if r.status_code != 200:
        return None
    try:
        data = r.json().get('response', [])
    except Exception:
        return None
    if not data or not data[0].get('data'):
        return None
    bars = data[0]['data']
    # Pick any bar from afternoon (12:00+) with valid 2-sided quote
    best_mid = None
    for bar in bars:
        ts = bar.get('timestamp', '')
        # ts like '2024-01-08T13:30:00.000'
        hh = ts.split('T')[1][:2] if 'T' in ts else '00'
        try:
            hour = int(hh)
        except Exception:
            continue
        if hour < 12:
            continue
        bid, ask = bar.get('bid', 0), bar.get('ask', 0)
        if bid > 0 and ask > 0 and ask > bid:
            best_mid = (bid + ask) / 2  # take last valid
    return best_mid


def fetch_iv_surface(symbol='UNG', start_date='2021-06-01', end_date=None,
                     dte_target=30, n_strikes_each_side=5):
    """Build IV surface table: for each business day, pick front-month
    expiry near dte_target, fetch quotes for ATM ± N strikes, back out IV.

    SLOW (~4s per (date, expiry) request, ~10 strikes × 2 rights per date).
    Run in background.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    spot_csv = os.path.join(CACHE_DIR, 'master_dataset.csv')
    spot_df = pd.read_csv(spot_csv, index_col=0, parse_dates=True)
    spot_df = spot_df.loc[start_date:end_date, ['UNG']].dropna()

    expirations = get_expirations(symbol)
    print(f"Found {len(expirations)} historical expirations")

    rows = []
    n_total = len(spot_df)
    t0 = time.time()
    for i, (d, row) in enumerate(spot_df.iterrows()):
        adj_spot = float(row['UNG'])
        d_str = d.strftime('%Y-%m-%d')
        # Convert to real market spot for strike matching
        sf = split_factor_on(d_str)
        spot = adj_spot / sf
        # Find front-month expiry ≥ dte_target days out
        target_exp = None
        for exp in expirations:
            try:
                exp_d = datetime.strptime(exp, '%Y-%m-%d').date()
            except Exception:
                continue
            days = (exp_d - d.date()).days
            if days >= dte_target:
                target_exp = exp
                break
        if not target_exp:
            continue
        actual_dte = (datetime.strptime(target_exp, '%Y-%m-%d').date() - d.date()).days

        try:
            strikes = get_strikes(symbol, target_exp)
        except Exception as e:
            continue
        if not strikes:
            continue
        # Pick strikes closest to spot
        strikes_near = sorted(strikes, key=lambda k: abs(k - spot))[:n_strikes_each_side*2 + 1]

        for K in strikes_near:
            for right in ['C', 'P']:
                mid = get_quote_eod(symbol, target_exp, K, right, d_str)
                if mid is None or mid <= 0:
                    continue
                T = actual_dte / 365
                iv = bs_implied_vol(mid, spot, K, T, 0.045, right)
                if iv is not None and 0.05 < iv < 3.0:
                    # Store ADJUSTED strike so backtest can lookup directly
                    rows.append({'date': d_str, 'expiration': target_exp,
                                 'dte': actual_dte,
                                 'strike_adj': K * sf,  # back to adjusted space
                                 'strike_real': K,
                                 'right': right,
                                 'spot_adj': adj_spot, 'spot_real': spot,
                                 'mid': mid, 'iv': iv,
                                 'split_factor': sf})

        if (i+1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_total} dates done ({elapsed:.0f}s, est total {elapsed/(i+1)*n_total/60:.1f}min, {len(rows)} rows)")

    out = pd.DataFrame(rows)
    out_path = os.path.join(CACHE_DIR, 'ung_iv_surface.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved {len(out)} IV observations to {out_path}")
    if len(out) > 0:
        print(f"Coverage: {out['date'].nunique()} unique dates")
        print(f"IV median: {out['iv'].median():.3f}, range {out['iv'].min():.3f}-{out['iv'].max():.3f}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='UNG')
    parser.add_argument('--start', default='2021-06-01')
    parser.add_argument('--end', default=None)
    parser.add_argument('--dte', type=int, default=30)
    parser.add_argument('--n-strikes', type=int, default=5)
    args = parser.parse_args()
    fetch_iv_surface(args.symbol, args.start, args.end, args.dte, args.n_strikes)
