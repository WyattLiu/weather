"""Reconstruct daily GEX profiles from ThetaData backfill + backtest walls.

Inputs (from thetadata_backfill.py):
  history/thetadata/{symbol}/{expiry}_oi.csv   — daily OI per strike/right
  history/thetadata/{symbol}/{expiry}_eod.csv  — daily close/bid/ask

Spot: research/dba/cache/master_panel.csv (UNG + DBA daily closes).

For each trade date:
  GEX(K) = Σ_expiries  OI(K) × BSM_gamma(S, K, T, IV) × 100 × S² × 0.01
           (calls +, puts −, dealer convention)
  IV backed out per strike from EOD mid; fallback 30d realized × 1.12.

Backtests:
  1. PIN: |expiry_close − argmax_K GEX| vs |expiry_close − random strike|
  2. WALL RESPECT: P(high of final week ≤ call wall) and
                   P(low of final week ≥ put wall)
  3. GEX-FLIP: forward 5d realized vol when net GEX < 0 vs > 0

Run: venv/bin/python research/gex/gex_backtest.py --symbol UNG
"""
import os
import math
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TD_DIR = os.path.join(THIS_DIR, 'history', 'thetadata')
PANEL = os.path.join(os.path.dirname(THIS_DIR), 'dba', 'cache', 'master_panel.csv')
R = 0.045


def bsm_price(S, K, T, sig, right):
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_gamma(S, K, T, sig):
    d1 = (math.log(S / K) + (R + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return norm.pdf(d1) / (S * sig * math.sqrt(T))


def implied_vol(S, K, T, mid, right):
    if mid <= 0.01 or T <= 0:
        return None
    try:
        return brentq(lambda s: bsm_price(S, K, T, s, right) - mid,
                      0.03, 4.0, xtol=1e-4)
    except Exception:
        return None


def load_symbol(symbol):
    """Merge all per-expiry files into (oi_df, eod_df)."""
    sdir = os.path.join(TD_DIR, symbol.lower())
    oi_parts, eod_parts = [], []
    for p in sorted(glob.glob(os.path.join(sdir, '*_oi.csv'))):
        try:
            df = pd.read_csv(p)
            if len(df):
                oi_parts.append(df)
        except Exception:
            continue
    for p in sorted(glob.glob(os.path.join(sdir, '*_eod.csv'))):
        try:
            df = pd.read_csv(p)
            if len(df):
                eod_parts.append(df)
        except Exception:
            continue
    if not oi_parts:
        raise SystemExit(f'no backfill data yet for {symbol} in {sdir}')
    oi = pd.concat(oi_parts, ignore_index=True)
    eod = pd.concat(eod_parts, ignore_index=True) if eod_parts else pd.DataFrame()
    # OI timestamp is next-morning posting → belongs to prior trade date.
    # Shift back 1 day so OI aligns with the trade date it describes.
    oi['oi_date'] = pd.to_datetime(oi['oi_date']) - pd.Timedelta(days=1)
    if len(eod):
        eod['quote_date'] = pd.to_datetime(eod['quote_date'])
    return oi, eod


# Reverse splits: yfinance closes are split-ADJUSTED but ThetaData strikes
# are the raw values traded on the day. Un-adjust spot to raw by dividing
# by every split factor that occurred AFTER the date.
SPLITS = {
    'UNG': [('2018-01-05', 4.0), ('2024-01-24', 4.0)],  # two 1-for-4 reverse
    'DBA': [],
}


def unadjust(spot, symbol):
    raw = spot.copy()
    for split_date, factor in SPLITS.get(symbol, []):
        raw.loc[raw.index < pd.Timestamp(split_date)] /= factor
    return raw


def daily_gex(symbol):
    """Build per-(date, strike) net GEX panel. Returns DataFrame."""
    oi, eod = load_symbol(symbol)
    spot_adj = pd.read_csv(PANEL, index_col=0, parse_dates=True)[symbol].dropna()
    spot = unadjust(spot_adj, symbol)  # raw prices matching historical strikes
    # realized vol from the ADJUSTED series — the raw series has fake 4x
    # jumps at split dates that would poison the rolling window
    rv = (spot_adj.pct_change().rolling(30).std() * math.sqrt(252) * 1.12).bfill()

    # Pre-index EOD mids: (date, expiry, right, strike) → mid
    mid_lookup = {}
    if len(eod):
        eod['mid'] = np.where((eod['bid'] > 0) & (eod['ask'] > 0),
                              (eod['bid'] + eod['ask']) / 2, eod['close'])
        for t in eod.itertuples():
            mid_lookup[(t.quote_date, t.expiry, t.right, t.strike)] = t.mid

    rows = []
    for (d, expiry), grp in oi.groupby(['oi_date', 'expiry']):
        if d not in spot.index:
            continue
        S = float(spot.loc[d])
        exp_d = pd.Timestamp(expiry)
        T = max(0.5, (exp_d - d).days) / 365.0
        sigma_fb = float(rv.loc[d]) if d in rv.index else 0.45
        for t in grp.itertuples():
            if t.open_interest <= 0:
                continue
            mid = mid_lookup.get((d, expiry, t.right, t.strike))
            iv = implied_vol(S, t.strike, T, mid, t.right) if mid else None
            sig = iv or sigma_fb
            g = bsm_gamma(S, t.strike, T, sig)
            dollar = g * t.open_interest * 100 * S * S * 0.01
            rows.append((d, expiry, t.strike,
                         dollar if t.right == 'C' else -dollar,
                         t.open_interest, t.right))
    df = pd.DataFrame(rows, columns=['date', 'expiry', 'strike',
                                     'gex', 'oi', 'right'])
    return df, spot


def backtest(symbol):
    print(f'[gex] building daily GEX panel for {symbol}...')
    panel, spot = daily_gex(symbol)
    out_path = os.path.join(TD_DIR, f'{symbol.lower()}_gex_panel.csv')
    panel.to_csv(out_path, index=False)
    print(f'  {len(panel)} rows → {out_path}')

    # Aggregate: per (date) → net GEX by strike across expiries
    by_strike = panel.groupby(['date', 'strike'])['gex'].sum().reset_index()

    # ── 1. PIN TEST ────────────────────────────────────────────────────
    # For each monthly expiry: GEX wall measured 5 trading days before
    # expiry; compare |close_at_expiry − wall| to |close − each other
    # candidate strike| percentile.
    pin_results = []
    for expiry in sorted(panel['expiry'].unique()):
        exp_d = pd.Timestamp(expiry)
        sub = panel[panel['expiry'] == expiry]
        days = sorted(sub['date'].unique())
        if len(days) < 10:
            continue
        # measurement day ≈ 5 trading days pre-expiry
        meas_day = days[-6] if len(days) >= 6 else days[0]
        prof = (sub[sub['date'] == meas_day]
                .groupby('strike')['gex'].sum())
        prof = prof[prof.abs() > 0]
        if len(prof) < 5:
            continue
        wall = prof.idxmax()
        # expiry close
        e_dates = spot.index[spot.index <= exp_d]
        if not len(e_dates):
            continue
        close = float(spot.loc[e_dates[-1]])
        S_meas = float(spot.loc[meas_day]) if meas_day in spot.index else None
        if S_meas is None:
            continue
        dist_wall = abs(close - wall) / S_meas
        dist_all = [abs(close - k) / S_meas for k in prof.index]
        pct_rank = sum(1 for x in dist_all if x < dist_wall) / len(dist_all)
        pin_results.append({
            'expiry': expiry, 'wall': wall, 'close': close,
            'spot_at_meas': S_meas, 'dist_pct': dist_wall,
            'rank_among_strikes': pct_rank,
            'moved_toward_wall': abs(close - wall) < abs(S_meas - wall),
        })
    pins = pd.DataFrame(pin_results)

    print(f'\n=== PIN TEST ({len(pins)} monthly expiries) ===')
    if len(pins):
        print(f'  avg |close − wall| / spot: {pins["dist_pct"].mean():.2%}')
        print(f'  spot moved TOWARD wall in final week: '
              f'{pins["moved_toward_wall"].mean():.1%} of expiries')
        print(f'  wall closer than median strike: '
              f'{(pins["rank_among_strikes"] < 0.5).mean():.1%}')

    # ── 2. WALL RESPECT ───────────────────────────────────────────────
    respect = []
    for r in pin_results:
        exp_d = pd.Timestamp(r['expiry'])
        wk = spot[(spot.index > exp_d - pd.Timedelta(days=7)) & (spot.index <= exp_d)]
        if not len(wk):
            continue
        respect.append({'expiry': r['expiry'],
                        'call_wall_held': wk.max() <= r['wall'] * 1.02})
    if respect:
        rdf = pd.DataFrame(respect)
        print('\n=== CALL WALL RESPECT (final week, +2% tolerance) ===')
        print(f'  held: {rdf["call_wall_held"].mean():.1%} of {len(rdf)} expiries')

    # ── 3. GEX-FLIP vs FORWARD VOL ────────────────────────────────────
    net_daily = by_strike.groupby('date')['gex'].sum()
    fwd_vol = (spot.pct_change().shift(-5).rolling(5).std() * math.sqrt(252))
    j = pd.DataFrame({'net_gex': net_daily}).join(fwd_vol.rename('fwd_vol5d')).dropna()
    if len(j) > 50:
        neg = j[j['net_gex'] < 0]['fwd_vol5d']
        pos = j[j['net_gex'] > 0]['fwd_vol5d']
        print('\n=== GEX-FLIP vs FORWARD 5d REALIZED VOL ===')
        print(f'  net GEX < 0 ({len(neg)} days): fwd vol {neg.mean():.1%}')
        print(f'  net GEX > 0 ({len(pos)} days): fwd vol {pos.mean():.1%}')
        if len(neg) > 10:
            print(f'  ratio: {neg.mean()/pos.mean():.2f}× '
                  f'(>1 = negative-GEX days precede higher vol, as theory predicts)')

    pins.to_csv(os.path.join(TD_DIR, f'{symbol.lower()}_pin_test.csv'), index=False)
    return pins


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='UNG')
    args = p.parse_args()
    backtest(args.symbol)
