"""Real-chain option pricing for the kernel backtest (tier-3 fidelity).

Loads actual historical UNG option quotes (ThetaData EOD, ~100 monthly
expiries 2018-2026) and prices any (date, strike, dte, right) at the
REAL bid/ask/mid that traded that day — not a Black-Scholes model.

Returns (bid, ask, mid, is_real). When no real quote exists (off-grid
date such as a weekly entry, or a strike/expiry never quoted), is_real
is False and the caller falls back to the BS x empirical-fill-grid
estimate. So: real chains where they exist, calibrated-real elsewhere.

Coverage caveat: UNG ThetaData backfill is MONTHLY expiries only. Kernel
30/45/60-DTE entries map near monthlies (good real coverage); exact-DTE
and any weekly legs fall back to the grid. Honest hybrid, never silent.
"""
import os
import glob
import bisect
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
UNG_DIR = os.path.join(os.path.dirname(THIS_DIR),
                       'research', 'gex', 'history', 'thetadata', 'ung')

# split-adjust: yfinance/master_dataset spot is adjusted; ThetaData strikes
# are raw. To match a kernel (adjusted) spot to raw strikes, divide spot.
_SPLITS = [('2018-01-05', 4.0), ('2024-01-24', 4.0)]

_LOOKUP = None      # {(date, expiry, right): {strike: (bid, ask, mid)}}
_EXPIRIES = None    # sorted list of expiry Timestamps
_DATES = None       # set of quote dates


def _load():
    global _LOOKUP, _EXPIRIES, _DATES
    if _LOOKUP is not None:
        return
    _LOOKUP = {}
    exps = set(); dates = set()
    for f in sorted(glob.glob(os.path.join(UNG_DIR, '*_eod.csv'))):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        df['quote_date'] = pd.to_datetime(df['quote_date']).dt.normalize()
        df['expiry'] = pd.to_datetime(df['expiry']).dt.normalize()
        for (d, e, r), g in df.groupby(['quote_date', 'expiry', 'right']):
            book = {}
            for t in g.itertuples():
                b, a = float(t.bid), float(t.ask)
                if b <= 0 and a <= 0:
                    continue
                mid = (b + a) / 2 if (b > 0 and a > 0) else (b or a or float(t.close))
                book[float(t.strike)] = (b, a, mid)
            if book:
                _LOOKUP[(d, e, r)] = book
                exps.add(e); dates.add(d)
    _EXPIRIES = sorted(exps); _DATES = dates


def _adjust_spot_to_raw(spot, date):
    """Convert an adjusted spot to the raw scale matching ThetaData strikes."""
    raw = spot
    for sd, factor in _SPLITS:
        if date < pd.Timestamp(sd):
            raw /= factor
    return raw


def price(date, strike_adj, dte_target, right, spot_adj=None):
    """Real bid/ask/mid for the option nearest (strike, dte) on `date`.
    strike_adj is in ADJUSTED scale (kernel's); converted to raw internally
    when spot_adj is given. right in {'P','C'}. Returns (bid,ask,mid,is_real)."""
    _load()
    d = pd.Timestamp(date).normalize()
    if d not in _DATES or not _EXPIRIES:
        return None, None, None, False
    # nearest real expiry to target DTE
    target_exp = d + pd.Timedelta(days=int(dte_target))
    i = bisect.bisect_left(_EXPIRIES, target_exp)
    cands = [e for e in _EXPIRIES[max(0, i-1):i+2] if e > d]
    if not cands:
        return None, None, None, False
    exp = min(cands, key=lambda e: abs((e - target_exp).days))
    book = _LOOKUP.get((d, exp, right))
    if not book:
        return None, None, None, False
    # strike in raw scale
    K_raw = _adjust_spot_to_raw(strike_adj, d) if spot_adj is not None else strike_adj
    K = min(book, key=lambda k: abs(k - K_raw))
    if abs(K - K_raw) / max(K_raw, 1) > 0.10:   # no strike within 10% → no real quote
        return None, None, None, False
    b, a, m = book[K]
    # scale prices back to adjusted dollars (option px scales with split too)
    if spot_adj is not None:
        for sd, factor in _SPLITS:
            if d < pd.Timestamp(sd):
                b, a, m = b * factor, a * factor, m * factor
    return b, a, m, True


def coverage_report():
    _load()
    return {'dates': len(_DATES), 'expiries': len(_EXPIRIES),
            'first': str(min(_DATES).date()) if _DATES else None,
            'last': str(max(_DATES).date()) if _DATES else None}


if __name__ == '__main__':
    print('UNG real-chain coverage:', coverage_report())
    # smoke: price a ~45d 5% OTM put on a sample date
    b, a, m, real = price(pd.Timestamp('2023-06-15'), 6.5, 45, 'P', spot_adj=6.8)
    print(f'2023-06-15 ~P6.5 45d: bid {b} ask {a} mid {m} real={real}')
