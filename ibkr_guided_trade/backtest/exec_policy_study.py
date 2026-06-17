"""Sophisticated intraday EXECUTION-POLICY study — what is the best way to work a roll
or an open, on the REAL minute path?

For every order the kernel places, we take the actual intraday bid/ask path of that exact
contract and measure IMPLEMENTATION SHORTFALL (fill vs the arrival mid, in cents) under
competing execution policies. We then condition on state (day-of-week, hour, DTE, OI,
arrival-spread, daily move regime) to find the STATE-DEPENDENT best policy — not one rule.

Policies (all RTH, Thursday pre-11:00 EIA window excluded):
  cross_now      cross the spread immediately at arrival      (cost = half arrival spread)
  tightest       wait, cross at the tightest-spread minute    (timing value, but price risk)
  patient_mid    rest at arrival mid; fill at mid if the path trades through, else cross tightest
  twap           slice across the day, average the touch       (dampens timing luck)
  opportunistic  cross the first minute spread tightens <= thr, else cross at deadline

Lower mean shortfall = better. Negative = beat arrival mid (price drifted your way).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import replay_engine as R
import intraday_fill as IF

DOW = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
OPP_THRESH_PCT = 8.0     # opportunistic: take when intraday spread tightens below this


def _path(date, K_adj, dte, right, expiry=None):
    """Real RTH two-sided minute path (t, bid, ask) for the order's contract, split-adj."""
    d = pd.Timestamp(date).normalize(); ds = d.date().isoformat()
    exps = IF._expiries_on(ds)
    if not exps:
        return None, None
    if expiry:
        want = pd.Timestamp(expiry).date()
        exp = min(exps, key=lambda e: abs((e - want).days))
        if abs((exp - want).days) > 7:
            expiry = None
    if not expiry:
        target = (d + pd.Timedelta(days=int(dte or 30))).date()
        exp = min(exps, key=lambda e: abs((e - target).days))
        if abs((exp - target).days) > 12:
            return None, None
    K_raw = float(K_adj)
    for sd, f in IF._SPLITS:
        if d < pd.Timestamp(sd):
            K_raw /= f
    sf = 1.0
    for sd, f in IF._SPLITS:
        if d < pd.Timestamp(sd):
            sf = f
    bars = [(t, b * sf, a * sf) for (t, b, a) in IF._bars(ds, exp.isoformat(), round(K_raw, 1), right)
            if b and a and a > b and b > 0]
    is_thu = d.dayofweek == 3
    win = [(t, b, a) for (t, b, a) in bars if not (is_thu and t.hour < 11)]
    return (win or None), sf


def _policies(win, side):
    """Return {policy: fill_price} on the real path. side 'sell'/'buy'."""
    arr_mid = (win[0][1] + win[0][2]) / 2.0
    touch = (lambda b, a: b) if side == 'sell' else (lambda b, a: a)
    out = {}
    out['cross_now'] = touch(win[0][1], win[0][2])
    tb = min(win, key=lambda x: (x[2] - x[1]) / ((x[1] + x[2]) / 2))
    out['tightest'] = touch(tb[1], tb[2])
    # patient_mid: rest at arrival mid; fill at mid if the path trades through, else cross tightest
    pm = None
    for (t, b, a) in win:
        if (side == 'sell' and b >= arr_mid) or (side == 'buy' and a <= arr_mid):
            pm = arr_mid; break
    out['patient_mid'] = pm if pm is not None else touch(tb[1], tb[2])
    out['twap'] = np.mean([touch(b, a) for (_, b, a) in win])
    # opportunistic: cross the first minute spread tightens below threshold, else deadline
    opp = None
    for (t, b, a) in win:
        if (a - b) / ((a + b) / 2) * 100 <= OPP_THRESH_PCT:
            opp = touch(b, a); break
    out['opportunistic'] = opp if opp is not None else touch(win[-1][1], win[-1][2])
    return out, arr_mid


def main(kernel, start, end, sample):
    df = pd.read_csv(os.path.join(R.CACHE_DIR, 'master_dataset.csv'), index_col=0, parse_dates=True)
    df = R.precompute_factor_z(df).dropna(subset=['UNG']).loc[start:end]
    ret = (df['UNG'] / df['UNG'].shift(1) - 1) * 100
    # order list (fast: model/EOD fills just to get the schedule)
    _, t = R.run_strategy_simple(df, R.STRATEGIES[kernel], 48000, 6200)
    legs = {'OPEN_PUT': ('sell', 'P'), 'OPEN_CC': ('sell', 'C'), 'OPEN_ITM_CC': ('sell', 'C'),
            'PUT_TP': ('buy', 'P'), 'CALL_TP': ('buy', 'C'), 'PUT_ROLL_DOWN': ('buy', 'P')}
    orders = t[t['type'].isin(legs)].copy()
    if sample and len(orders) > sample:
        orders = orders.iloc[:: max(1, len(orders) // sample)]
    print(f"{kernel}: {len(orders)} orders {start}→{end}; replaying minute paths...", flush=True)
    rows = []
    POL = ['cross_now', 'tightest', 'patient_mid', 'twap', 'opportunistic']
    for _, o in orders.iterrows():
        K, dte, ty = o.get('K'), o.get('dte'), o['type']
        if K != K or K is None:
            continue
        side, right = legs[ty]
        win, sf = _path(o['date'], float(K), int(dte) if dte == dte and dte else 30, right)
        if not win:
            continue
        fills, arr_mid = _policies(win, side)
        sgn = 1.0 if side == 'sell' else -1.0   # cost = sgn*(arr_mid - fill) → cents, lower better
        d = pd.Timestamp(o['date'])
        rmv = ret.get(d, np.nan)
        rec = {'date': d, 'type': ty, 'side': side, 'right': right,
               'dow': d.dayofweek, 'dte': int(dte) if dte == dte and dte else 30,
               'arr_spread_pct': round((win[0][2] - win[0][1]) / arr_mid * 100, 1),
               'day_ret': rmv}
        for p in POL:
            rec[p] = sgn * (arr_mid - fills[p]) * 100      # implementation shortfall, cents
        rows.append(rec)
    r = pd.DataFrame(rows)
    if not len(r):
        print("no orders with minute paths"); return
    r.to_csv(os.path.join(THIS, 'results', 'exec_policy_costs.csv'), index=False)
    print(f"\n=== IMPLEMENTATION SHORTFALL (cents vs arrival mid; LOWER=better, neg=beat mid) ===")
    print(f"n={len(r)} order-fills\n")
    summ = r[POL].mean().sort_values()
    for p, v in summ.items():
        print(f"  {p:14} mean {v:+6.2f}¢   median {r[p].median():+6.2f}¢   p90 {r[p].quantile(.9):+6.2f}¢")
    best = summ.index[0]
    print(f"\n  BEST overall: {best} ({summ.iloc[0]:+.2f}¢)  vs cross_now ({r['cross_now'].mean():+.2f}¢) "
          f"→ saves {r['cross_now'].mean()-summ.iloc[0]:.2f}¢/order")
    # state conditioning
    print(f"\n=== BEST POLICY BY STATE (mean shortfall ¢) ===")
    print("by DAY OF WEEK:")
    for dw, sub in r.groupby('dow'):
        m = sub[POL].mean(); b = m.idxmin()
        print(f"  {DOW.get(dw):3}  best={b:13} {m.min():+5.2f}¢   (cross_now {sub['cross_now'].mean():+5.2f}¢, n={len(sub)})")
    print("by ARRIVAL SPREAD:")
    r['sp_bkt'] = pd.cut(r['arr_spread_pct'], [0, 6, 12, 20, 999], labels=['<6%', '6-12%', '12-20%', '>20%'])
    for bk, sub in r.groupby('sp_bkt', observed=True):
        m = sub[POL].mean(); b = m.idxmin()
        print(f"  {str(bk):8} best={b:13} {m.min():+5.2f}¢   (cross_now {sub['cross_now'].mean():+5.2f}¢, n={len(sub)})")
    print("by DAILY MOVE regime (spike proxy):")
    r['mv_bkt'] = pd.cut(r['day_ret'], [-99, -3, 3, 99], labels=['down>3%', 'flat', 'up>3%'])
    for bk, sub in r.groupby('mv_bkt', observed=True):
        for sd in ('sell', 'buy'):
            ss = sub[sub['side'] == sd]
            if len(ss) < 10:
                continue
            m = ss[POL].mean(); b = m.idxmin()
            print(f"  {str(bk):8} {sd:4} best={b:13} {m.min():+5.2f}¢  (n={len(ss)})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kernel', default='champion_kold15_ivrank_kbh')
    ap.add_argument('--start', default='2024-01-02')
    ap.add_argument('--end', default='2026-06-12')
    ap.add_argument('--sample', type=int, default=0)
    a = ap.parse_args()
    main(a.kernel, a.start, a.end, a.sample)
