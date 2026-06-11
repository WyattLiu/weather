"""Live GEX wall from current WS option chains.

Backtest evidence (research/gex/gex_backtest.py, 100 UNG monthlies
2018-2026): final-week high stayed under the call wall (+2%) 74% of the
time vs 69% for a naive same-OTM strike → +5pp. Pin/magnetism and
GEX-flip vol signals NOT confirmed. So the wall is used ONLY as a CC
strike floor, never for sizing.

Usage:
    from live_wall import current_gex_wall
    wall = current_gex_wall('UNG', spot)   # → dict or None
"""
import os
import sys
import math
import time
from datetime import date, timedelta

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'live'))

_CACHE = {}
_CACHE_TTL = 900  # 15 min — walls move slowly intraday


def _front_monthlies(n=2):
    """Next n monthly expiries (3rd Friday, with Thursday fallback tried
    at fetch time)."""
    out = []
    d = date.today().replace(day=1)
    while len(out) < n:
        # 3rd Friday of month d
        fridays = [d + timedelta(days=i) for i in range(31)
                   if (d + timedelta(days=i)).month == d.month
                   and (d + timedelta(days=i)).weekday() == 4]
        third = fridays[2]
        if third >= date.today():
            out.append(third)
        d = (d + timedelta(days=32)).replace(day=1)
    return out


def _bsm_gamma(S, K, T, sig):
    from scipy.stats import norm
    d1 = (math.log(S / K) + (0.045 + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return norm.pdf(d1) / (S * sig * math.sqrt(T))


def _iv(S, K, T, mid, right):
    from scipy.stats import norm
    from scipy.optimize import brentq

    def px(sig):
        d1 = (math.log(S / K) + (0.045 + sig * sig / 2) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        if right == 'C':
            return S * norm.cdf(d1) - K * math.exp(-0.045 * T) * norm.cdf(d2)
        return K * math.exp(-0.045 * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if mid <= 0.01 or T <= 0:
        return None
    try:
        return brentq(lambda s: px(s) - mid, 0.03, 4.0, xtol=1e-4)
    except Exception:
        return None


def current_gex_wall(symbol, spot, fallback_iv=0.45):
    """Compute net GEX by strike from the front 2 monthly chains.
    Returns {'wall': K, 'wall_gex': $, 'put_wall': K, 'net_gex': $,
             'expiries': [...]} or None on failure."""
    key = symbol.upper()
    now = time.time()
    if key in _CACHE and now - _CACHE[key][0] < _CACHE_TTL:
        return _CACHE[key][1]

    try:
        from ws_option_resolver import fetch_chain
    except ImportError:
        return None

    gex = {}
    expiries_used = []
    for exp_d in _front_monthlies(2):
        T = max(1, (exp_d - date.today()).days) / 365.0
        for try_d in (exp_d, exp_d - timedelta(days=1)):  # Thu fallback
            expiry = try_d.isoformat()
            got = False
            for right in ('C', 'P'):
                try:
                    chain = fetch_chain(symbol, expiry, right)
                except Exception:
                    chain = {}
                if not chain:
                    continue
                got = True
                for K, leg in chain.items():
                    oi = leg.get('oi', 0)
                    if oi <= 0:
                        continue
                    mid = ((leg['bid'] + leg['ask']) / 2
                           if leg['bid'] > 0 and leg['ask'] > 0 else leg['last'])
                    sig = _iv(spot, K, T, mid, right) or fallback_iv
                    g = _bsm_gamma(spot, K, T, sig)
                    dollar = g * oi * 100 * spot * spot * 0.01
                    gex[K] = gex.get(K, 0) + (dollar if right == 'C' else -dollar)
            if got:
                expiries_used.append(expiry)
                break

    if not gex:
        return None
    wall_k = max(gex, key=gex.get)
    put_k = min(gex, key=gex.get)
    out = {
        'wall': wall_k,
        'wall_gex': round(gex[wall_k], 0),
        'put_wall': put_k,
        'put_wall_gex': round(gex[put_k], 0),
        'net_gex': round(sum(gex.values()), 0),
        'expiries': expiries_used,
        'spot_used': spot,
    }
    _CACHE[key] = (now, out)
    return out


if __name__ == '__main__':
    import json
    import yfinance as yf
    spot = float(yf.Ticker('UNG').fast_info.last_price)
    print(json.dumps(current_gex_wall('UNG', spot), indent=2))
