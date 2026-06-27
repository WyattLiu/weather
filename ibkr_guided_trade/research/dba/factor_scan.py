"""DBA alpha factor scan — forward returns conditional on each factor.

Factors (all from data already on disk):
  mom_1m / mom_3m / mom_6m / mom_12m  — time-series momentum (CTA classic)
  month         — seasonality (ag planting/harvest cycle)
  dxy_trend     — USD 3m trend (strong USD bearish ag)
  ng_trend      — NG 3m trend (fertilizer cost channel: NG→ammonia→grain)
  oni / dsci_z  — ENSO + drought (existing)
  iv_rank       — DBA ATM IV percentile from ThetaData EOD backfill
                  (premium richness — matters for the SHORT-PUT leg)

Method: quintile forward returns (21d / 63d) + Newey-West-ish t-stat on
top-minus-bottom. Monthly-sampled to avoid overlapping-window inflation.

Run: venv/bin/python research/dba/factor_scan.py
"""
import os
import glob
import math
import pandas as pd
from scipy import stats as sstats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
CACHE = os.path.join(THIS_DIR, 'cache')
TD_DBA = os.path.join(ROOT, 'research', 'gex', 'history', 'thetadata', 'dba')


def dba_iv_series():
    """ATM IV per day from ThetaData EOD quotes (nearest-monthly, ~ATM)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    spot = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                       index_col=0, parse_dates=True)['DBA'].dropna()

    def bsm_p(S, K, T, sig):
        d1 = (math.log(S/K) + (0.045 + sig*sig/2)*T) / (sig*math.sqrt(T))
        d2 = d1 - sig*math.sqrt(T)
        return K*math.exp(-0.045*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    ivs = {}
    for p in sorted(glob.glob(os.path.join(TD_DBA, '*_eod.csv'))):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        df = df[df['right'] == 'P']
        df['quote_date'] = pd.to_datetime(df['quote_date'])
        expiry = pd.Timestamp(df['expiry'].iloc[0])
        for d, grp in df.groupby('quote_date'):
            if d not in spot.index:
                continue
            S = float(spot.loc[d])
            T = max(5, (expiry - d).days) / 365.0
            if T > 0.20 or T < 0.02:   # keep ~1wk-10wk window
                continue
            atm = grp.iloc[(grp['strike'] - S).abs().argsort()[:1]]
            mid = float((atm['bid'].iloc[0] + atm['ask'].iloc[0]) / 2)
            if mid <= 0.02:
                continue
            K = float(atm['strike'].iloc[0])
            try:
                iv = brentq(lambda s: bsm_p(S, K, T, s) - mid, 0.03, 3.0, xtol=1e-4)
            except Exception:
                continue
            # keep the nearest-expiry estimate per day
            if d not in ivs or T < ivs[d][1]:
                ivs[d] = (iv, T)
    s = pd.Series({d: v[0] for d, v in ivs.items()}).sort_index()
    return s


def build_factors():
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    dba = panel['DBA'].dropna()
    f = pd.DataFrame(index=dba.index)
    f['dba'] = dba
    for m, n in [(1, 21), (3, 63), (6, 126), (12, 252)]:
        f[f'mom_{m}m'] = dba.pct_change(n)
    f['month'] = f.index.month
    if 'UNG' in panel.columns:
        pass
    # DXY + NG from the kernel master dataset (has DXY column)
    md = pd.read_csv(os.path.join(ROOT, 'backtest', 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    _idx = pd.to_datetime(md.index, utc=True).tz_localize(None)
    md.index = _idx.normalize()
    # dataset has duplicate per-day rows (04:00 + 05:00 DST artifacts) with
    # values split between them — groupby.first() takes first non-NaN per col
    md = md.groupby(md.index).first()
    for src, name in [('DX_DXY', 'dxy'), ('NG', 'ng')]:
        if src in md.columns:
            f[f'{name}_trend'] = md[src].pct_change(63).reindex(f.index).ffill()
    f['oni'] = panel['oni'].reindex(f.index).ffill()
    f['dsci_z'] = panel['dsci_z'].reindex(f.index).ffill()

    print('[scan] computing DBA IV series from ThetaData EOD backfill...')
    iv = dba_iv_series()
    print(f'  {len(iv)} IV observations {iv.index.min().date()} → {iv.index.max().date()}')
    f['iv'] = iv.reindex(f.index).ffill()
    f['iv_rank'] = f['iv'].rolling(252, min_periods=120).rank(pct=True)

    # forward returns
    f['fwd_21d'] = dba.pct_change(21).shift(-21)
    f['fwd_63d'] = dba.pct_change(63).shift(-63)
    return f


def quintile_table(f, factor, fwd='fwd_21d', monthly=True):
    sub = f[[factor, fwd]].dropna()
    if monthly:
        sub = sub.iloc[::21]  # non-overlapping-ish sampling
    if len(sub) < 40:
        return None
    sub = sub.copy()
    try:
        sub['q'] = pd.qcut(sub[factor], 5, labels=False, duplicates='drop')
    except ValueError:
        return None
    g = sub.groupby('q')[fwd].agg(['mean', 'count'])
    if len(g) < 4:
        return None
    top, bot = g['mean'].iloc[-1], g['mean'].iloc[0]
    spread = top - bot
    # t-stat of spread
    t_top = sub[sub['q'] == sub['q'].max()][fwd]
    t_bot = sub[sub['q'] == sub['q'].min()][fwd]
    t, p = sstats.ttest_ind(t_top, t_bot, equal_var=False)
    return {'factor': factor, 'fwd': fwd,
            'q1_mean': round(bot, 4), 'q5_mean': round(top, 4),
            'spread': round(spread, 4), 't': round(t, 2), 'p': round(p, 3),
            'n': len(sub)}


def seasonality_table(f):
    sub = f[['month', 'fwd_21d']].dropna().iloc[::21]
    g = sub.groupby('month')['fwd_21d'].agg(['mean', 'count'])
    return g


def main():
    f = build_factors()
    f.to_csv(os.path.join(CACHE, 'dba_factor_panel.csv'))

    rows = []
    factors = ['mom_1m', 'mom_3m', 'mom_6m', 'mom_12m',
               'dxy_trend', 'ng_trend', 'oni', 'dsci_z', 'iv_rank']
    for fac in factors:
        if fac not in f.columns or f[fac].dropna().empty:
            continue
        for fwd in ('fwd_21d', 'fwd_63d'):
            r = quintile_table(f, fac, fwd)
            if r:
                rows.append(r)
    res = pd.DataFrame(rows).sort_values('p')
    print('\n=== FACTOR SCAN — quintile top-minus-bottom forward returns ===')
    print(res.to_string(index=False))

    print('\n=== SEASONALITY (fwd 21d by calendar month) ===')
    print((seasonality_table(f) * 100).round(2).to_string())

    res.to_csv(os.path.join(CACHE, 'dba_factor_scan.csv'), index=False)
    print(f'\n→ {CACHE}/dba_factor_scan.csv')


if __name__ == '__main__':
    main()
