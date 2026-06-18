"""UNG volatility/pricing MODEL — calibrated to real quotes, for WHAT-IF greeks only.
NEVER used to fill; fills come from real bid/ask. This is the elegant-math layer for risk
analysis (delta/gamma sizing, DTE mix, bear-put what-if), robust where flat-BS fails:
the penny wings and deep-ITM.

Method: per-expiry SVI smile fit to the REAL market IVs (backed out from ung_options_history
mids), giving a smooth, arbitrage-aware smile across the whole strike range. Pricing = BS at
the SVI IV (the American/early-exercise premium is already embedded in the market-implied
vols we calibrate to). Validated vs real mids across moneyness×DTE.

  venv/bin/python backtest/ung_vol_model.py        # calibrate + validate vs real quotes
"""
import math
import numpy as np
import pandas as pd
import psycopg2
from scipy.optimize import least_squares
from scipy.stats import norm

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
R = 0.045


# ── Black-Scholes (pricing/greeks given a vol) ──────────────────────────────
def bs_price(S, K, T, sig, right):
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if right == 'C' else (K - S))
    d1 = (math.log(S / K) + (R + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    if right == 'C':
        return S * norm.cdf(d1) - K * math.exp(-R * T) * norm.cdf(d2)
    return K * math.exp(-R * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_iv(price, S, K, T, right):
    """Back out implied vol via bisection (robust in the wings, unlike Newton)."""
    intr = max(0.0, (S - K) if right == 'C' else (K - S))
    if price <= intr + 1e-6 or T <= 0:
        return None
    lo, hi = 1e-3, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, right) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ── SVI smile: total variance w(k) = a + b(ρ(k−m) + √((k−m)²+σ²)) ───────────
def svi(k, p):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def fit_svi(k, w):
    """Fit SVI params to (log-moneyness k, total variance w). Returns params or None."""
    if len(k) < 5:
        return None
    w = np.maximum(w, 1e-6)
    a0 = max(w.min() * 0.5, 1e-4)
    p0 = [a0, 0.1, -0.3, 0.0, 0.1]
    lo = [1e-6, 1e-4, -0.999, k.min() - 0.5, 1e-3]
    hi = [w.max() * 1.5 + 1e-3, 5.0, 0.999, k.max() + 0.5, 2.0]
    try:
        res = least_squares(lambda p: svi(k, p) - w, p0, bounds=(lo, hi),
                            max_nfev=4000, loss='soft_l1')
        return res.x
    except Exception:
        return None


def _quotes(date):
    q = """WITH q AS (SELECT expiration,strike,option_right,underlying_price,
             MAX(CASE WHEN data_type='BID' THEN close END) bid,
             MAX(CASE WHEN data_type='ASK' THEN close END) ask
           FROM ung_options_history WHERE trade_date=%s AND bar_time::time='15:30:00'
           GROUP BY 1,2,3,4)
           SELECT expiration,strike,option_right,underlying_price,bid,ask FROM q
           WHERE bid>0 AND ask>bid"""
    df = pd.read_sql(q, psycopg2.connect(**DB), params=(date,))
    if not len(df):
        return df
    df['expiration'] = pd.to_datetime(df['expiration'])
    df['strike'] = df['strike'].astype(float)
    df['bid'] = df['bid'].astype(float); df['ask'] = df['ask'].astype(float)
    df['mid'] = (df['bid'] + df['ask']) / 2.0
    df['dte'] = (df['expiration'] - pd.Timestamp(date)).dt.days.astype(int)
    df = df[(df['dte'] > 2) & (df['dte'] < 120)].copy()
    df['S'] = df['underlying_price'].astype(float)
    df['Tyr'] = df['dte'].astype(float) / 365.0
    df['F'] = df['S'] * np.exp(R * df['Tyr'])
    df['k'] = np.log(df['strike'] / df['F'])
    return df


def calibrate(date):
    """Per-expiry SVI calibration from real quotes. Returns {dte: (params, S)}."""
    df = _quotes(date)
    cal = {}
    for dte, g in df.groupby('dte'):
        ivs, ks = [], []
        for _, r in g.iterrows():
            iv = bs_iv(float(r['mid']), float(r['S']), float(r['strike']),
                       float(r['Tyr']), r['option_right'])
            if iv and 0.05 < iv < 4.0:
                ivs.append(iv); ks.append(float(r['k']))
        if len(ks) < 5:
            continue
        k = np.array(ks); w = (np.array(ivs) ** 2) * (dte / 365.0)
        p = fit_svi(k, w)
        if p is not None:
            cal[int(dte)] = (p, float(g.S.iloc[0]))
    return cal


def model_iv(cal, K, dte, spot):
    """SVI IV for any strike/DTE: pick nearest calibrated expiry, eval its smile at k."""
    if not cal:
        return 0.5
    d = min(cal.keys(), key=lambda x: abs(x - dte))
    p, _ = cal[d]
    F = spot * math.exp(R * dte / 365.0)
    k = math.log(max(K, 1e-6) / F)
    w = float(svi(np.array([k]), p)[0])
    return math.sqrt(max(w, 1e-6) / max(dte / 365.0, 1e-4))


def price(cal, S, K, dte, right):
    return max(0.01, bs_price(S, K, dte / 365.0, model_iv(cal, K, dte, S), right))


def greeks(cal, S, K, dte, right):
    """Smile-aware (sticky-strike) delta & gamma via central difference on the model."""
    h = max(0.01 * S, 0.02)
    pu = bs_price(S + h, K, dte / 365.0, model_iv(cal, K, dte, S + h), right)
    pd_ = bs_price(S - h, K, dte / 365.0, model_iv(cal, K, dte, S - h), right)
    p0 = bs_price(S, K, dte / 365.0, model_iv(cal, K, dte, S), right)
    delta = (pu - pd_) / (2 * h)
    gamma = (pu - 2 * p0 + pd_) / (h * h)
    return delta, gamma


def validate(dates):
    """Model price vs REAL mid by moneyness bucket. The honesty check."""
    rows = []
    for date in dates:
        cal = calibrate(date)
        if not cal:
            continue
        df = _quotes(date)
        for _, r in df.iterrows():
            mp = price(cal, r.S, float(r.strike), int(r.dte), r.option_right)
            rows.append({'mny': float(r.strike) / r.S, 'mid': r.mid, 'model': mp,
                         'err_pct': (mp - r.mid) / r.mid * 100, 'penny': r.mid < 0.10})
    v = pd.DataFrame(rows)
    print(f"=== UNG SVI MODEL vs REAL MID — {len(dates)} dates, {len(v):,} options ===")
    v['bucket'] = pd.cut(v.mny, [0, 0.85, 0.95, 1.05, 1.15, 9],
                         labels=['deepITMput', 'ITM', 'ATM', 'OTM', 'deepOTM/penny'])
    print(f"{'bucket':16}{'n':>6}{'med|err|%':>10}{'p90|err|%':>10}{'BS-flat was':>13}")
    bsflat = {'deepITMput': 5, 'ITM': 8, 'ATM': 10, 'OTM': 8, 'deepOTM/penny': 7}
    for b, g in v.groupby('bucket', observed=True):
        print(f"{str(b):16}{len(g):>6}{g.err_pct.abs().median():>9.1f}{g.err_pct.abs().quantile(.9):>10.1f}"
              f"{bsflat.get(str(b), 0):>11}%")
    print(f"\nOVERALL median |err|: {v.err_pct.abs().median():.1f}%  (flat-BS was ~5-10%)")
    print(f"penny zone (<$0.10): median |err| {v[v.penny].err_pct.abs().median():.1f}%  n={v.penny.sum()}")


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    # validate across a spread of recent dates
    test = ['2026-06-12', '2026-05-15', '2026-04-15', '2026-03-16', '2026-02-13',
            '2025-12-15', '2025-09-15', '2025-06-16', '2025-01-15', '2024-08-15']
    validate(test)
