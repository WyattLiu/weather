"""#3 CRASH-HEDGE PORTFOLIO: does a small long-vol sleeve cut the drawdown of a long book?
Long book proxy = buy&hold SPY. Hedge sleeve = BOXX-idle + a long-vol structure deployed on the
kernel's low-VIX signal (VIX<=16 & IV>=RV, conc=1, +30/-40/30d, ask-in/bid-out). Combine at constant
weight and report CAGR / Sharpe / MaxDD vs unhedged, for straddle / 1C2P put-tilt / long-put sleeves.

  venv/bin/python research/spy_vol/spy_crashhedge.py
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
RF_D = (1.045) ** (1 / 252) - 1
PT, STOP, MAXHOLD = 0.30, 0.40, 30


def sleeve_nav(cur, days, vix, spy, rv20, legmul, alloc, pc):
    """legmul = (n_call, n_put). Daily NAV of the hedge sleeve (idle->BOXX, conc=1)."""
    def pth(exp, K, r, d):
        k = (exp, K, r)
        if k not in pc:
            pc[k] = eod_mid(cur, exp, K, r, d, exp)
        return pc[k]
    nc, npu = legmul
    cash = 1.0; pos = None; hist = []
    for d_ts in days:
        d = d_ts.date(); cash *= (1 + RF_D); S = float(spy.loc[d_ts]); vx = float(vix.loc[d_ts])
        if pos is not None:
            c, p = pos['c'], pos['p']
            if d in c and d in p:
                mid = nc * c[d][0] + npu * p[d][0]; bid = nc * c[d][1] + npu * p[d][1]
                pos['mv'] = mid / pos['eask']; pos['bidmv'] = bid / pos['eask']
                ret = mid / pos['eask'] - 1; held = (d - pos['edate']).days
                if ret >= PT or ret <= -STOP or held >= MAXHOLD or d >= pos['exp']:
                    cash += pos['alloc'] * pos['bidmv']; pos = None
            elif d >= pos['exp']:
                cash += pos['alloc'] * pos['bidmv']; pos = None
        nav = cash + (pos['alloc'] * pos['mv'] if pos else 0)
        if pos is None and vx <= 16 and not math.isnan(rv20.loc[d_ts]):
            pe = pick_entry(cur, d, S, 38, 52)
            if pe:
                exp, K, dte = pe; T = dte / 365
                c = pth(exp, K, 'C', d); p = pth(exp, K, 'P', d)
                if d in c and d in p:
                    emid = c[d][0] + p[d][0]
                    iv = emid / (0.7979 * S * math.sqrt(T)) if T > 0 else 0
                    eask = nc * c[d][2] + npu * p[d][2]
                    if iv >= float(rv20.loc[d_ts]) and eask > 0:
                        a = alloc * nav
                        if a <= cash:
                            cash -= a
                            pos = {'eask': eask, 'alloc': a, 'edate': d, 'exp': exp, 'c': c, 'p': p,
                                   'mv': (nc * c[d][0] + npu * p[d][0]) / eask,
                                   'bidmv': (nc * c[d][1] + npu * p[d][1]) / eask}
        nav = cash + (pos['alloc'] * pos['mv'] if pos else 0)
        hist.append(nav)
    return pd.Series(hist, index=days)


def metr(nav, days):
    r = nav.pct_change().dropna(); yrs = (days[-1] - days[0]).days / 365.25
    return (nav.iloc[-1] ** (1 / yrs) - 1, r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0,
            (nav / nav.cummax() - 1).min())


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    days = list(spv.index)
    cur = _conn().cursor(); pc = {}
    spy_ret = spy.pct_change().fillna(0)

    sleeves = {'straddle 1C1P': (1, 1), '1C2P put-tilt': (1, 2), 'long put 0C1P': (0, 1)}
    navs = {n: sleeve_nav(cur, days, vix, spy, rv20, lm, 0.20, pc) for n, lm in sleeves.items()}

    base = spy / spy.iloc[0]
    bc = metr(base, days)
    print("=== #3 CRASH-HEDGE: long SPY + small long-vol sleeve (constant-weight) ===\n")
    print(f"  UNHEDGED buy&hold SPY: CAGR {bc[0]:+.1%}  Sharpe {bc[1]:.2f}  MaxDD {bc[2]:.1%}\n")
    for sname, snav in navs.items():
        sret = snav.pct_change().fillna(0)
        sm = metr(snav, days)
        print(f"  hedge sleeve [{sname}] standalone: CAGR {sm[0]:+.1%} Sharpe {sm[1]:.2f} MaxDD {sm[2]:.1%}")
        for wh in (0.05, 0.10, 0.15):
            cret = (1 - wh) * spy_ret + wh * sret
            cnav = (1 + cret).cumprod()
            cm = metr(cnav, days)
            print(f"     +{wh:.0%} hedge → CAGR {cm[0]:+.1%}  Sharpe {cm[1]:.2f}  MaxDD {cm[2]:.1%}  "
                  f"(ΔCAGR {cm[0]-bc[0]:+.1%}, ΔMaxDD {cm[2]-bc[2]:+.1%}, ΔSharpe {cm[1]-bc[1]:+.2f})")
        print()
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
