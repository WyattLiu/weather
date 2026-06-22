"""STRUCTURE comparison for the SPY vega-scrape: spot-ATM straddle vs FORWARD/DELTA-ZERO straddle
vs 5% strangle. Same green entries (VIX<=16 & IV>=RV), same exit (+30%/-40%/30d, no volpop),
enter ask / exit bid. Reports per-trade return / win / Sharpe / entry cost / entry net delta.

Why: spot-ATM carries a small NET DELTA (forward > spot from carry, + put skew) → directional noise.
FORWARD/DELTA-ZERO (strike ~ spot·e^{rT}, where straddle delta 2N(d1)-1 = 0) is the pure-vega bet.
Strangle = cheaper/lower-vega tail bet.

  venv/bin/python research/spy_vol/spy_structure_compare.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, eod_mid

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
R, PT, STOP, MAXHOLD = 0.045, 0.30, 0.40, 30


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def straddle_delta(S, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return 0.0
    F = S * math.exp(R * T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return 2 * N(d1) - 1                       # call_delta + put_delta


def sim(cpath, ppath, d):
    """enter ask, exit bid on first of +PT / -STOP / MAXHOLD / expiry. Return (ret, cost_at_entry)."""
    if d not in cpath or d not in ppath:
        return None
    entry = cpath[d][2] + ppath[d][2]
    if entry <= 0:
        return None
    fut = [t for t in sorted(set(cpath) & set(ppath)) if t > d]
    for i, t in enumerate(fut):
        mid = cpath[t][0] + ppath[t][0]; bid = cpath[t][1] + ppath[t][1]
        ret = mid / entry - 1; held = (t - d).days
        if ret >= PT or ret <= -STOP or held >= MAXHOLD or i == len(fut) - 1:
            return bid / entry - 1, entry
    return None


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    cur = _conn().cursor()
    pcache = {}

    def paths(exp, K, right, d):
        key = (exp, K, right)
        if key not in pcache:
            pcache[key] = eod_mid(cur, exp, K, right, d, exp)
        return pcache[key]

    R_atm, R_fwd, R_str = [], [], []
    cost = {'ATM': [], 'FWD': [], 'STR5': []}
    delt = {'ATM': [], 'FWD': []}
    for d_ts in spv.index:
        d = d_ts.date(); S = float(spy.loc[d_ts]); vx = float(vix.loc[d_ts])
        if vx > 16 or math.isnan(rv20.loc[d_ts]):
            continue
        pe = pick_entry(cur, d, S, 38, 52)
        if not pe:
            continue
        exp, K0, dte = pe
        T = dte / 365
        cur.execute("SELECT DISTINCT strike FROM spy_options_history WHERE trade_date=%s AND expiration=%s", (d, exp))
        ks = sorted(float(r[0]) for r in cur.fetchall())
        if not ks:
            continue
        Katm = min(ks, key=lambda k: abs(k - S))
        ca = paths(exp, Katm, 'C', d); pa = paths(exp, Katm, 'P', d)
        if d not in ca or d not in pa:
            continue
        emid = ca[d][0] + pa[d][0]
        iv = emid / (0.7979 * S * math.sqrt(T)) if T > 0 else 0
        if iv < float(rv20.loc[d_ts]):              # skip "cheap" (the trap)
            continue
        # ATM
        a = sim(ca, pa, d)
        if not a:
            continue
        R_atm.append(a[0]); cost['ATM'].append(a[1] / S); delt['ATM'].append(straddle_delta(S, Katm, T, iv))
        # FORWARD / delta-zero straddle: strike nearest spot*e^{rT}
        Kf = min(ks, key=lambda k: abs(k - S * math.exp(R * T)))
        cf = paths(exp, Kf, 'C', d); pf = paths(exp, Kf, 'P', d)
        f = sim(cf, pf, d)
        if f:
            R_fwd.append(f[0]); cost['FWD'].append(f[1] / S); delt['FWD'].append(straddle_delta(S, Kf, T, iv))
        # 5% strangle
        Kp = min(ks, key=lambda k: abs(k - S * 0.95)); Kc = min(ks, key=lambda k: abs(k - S * 1.05))
        cs = paths(exp, Kc, 'C', d); ps = paths(exp, Kp, 'P', d)
        s = sim(cs, ps, d)
        if s:
            R_str.append(s[0]); cost['STR5'].append(s[1] / S)

    def row(name, r, cst, dl=None):
        r = np.array(r)
        sh = r.mean() / r.std() * math.sqrt(len(r)) if (len(r) and r.std() > 0) else 0
        dtxt = f"{np.mean(dl):>+8.2f}" if dl else f"{'-':>8}"
        return (f"{name:<26}{len(r):>5}{r.mean():>+9.2%}{(r>0).mean()*100:>6.0f}%{sh:>8.2f}"
                f"{np.mean(cst)*100:>9.1f}%{dtxt}")

    print("=== SPY STRUCTURE COMPARISON (green entries, +30/-40/30d, enter ask/exit bid) ===\n")
    print(f"{'structure':<26}{'n':>5}{'avg ret':>9}{'win%':>7}{'t-stat':>8}{'cost/S':>10}{'net Δ':>8}")
    print('-' * 73)
    print(row('spot-ATM straddle', R_atm, cost['ATM'], delt['ATM']))
    print(row('FORWARD/Δ-zero straddle', R_fwd, cost['FWD'], delt['FWD']))
    print(row('5% strangle', R_str, cost['STR5']))
    print("\n  cost/S = entry debit as % of spot; net Δ = straddle delta at entry (≈0 = pure vega)")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
