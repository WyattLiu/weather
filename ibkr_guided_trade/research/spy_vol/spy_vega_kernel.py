"""SPY VEGA-SCRAPE KERNEL v2 — deployable strategy, optimized for return / Sharpe / MaxDD.

Mirrors UNG regime_wheel_boxx: hold BOXX/T-bills (~rf) when idle, deploy a long ~45 DTE ATM straddle
only when the firmed-up signal is GREEN:
    LOW VIX (≤ thr, ABSOLUTE)  AND  NOT-CHEAP (ATM IV ≥ RV20)  [AND flat skew]
Entry crosses to ASK, daily MTM at mid (carried forward when a quote is missing), exit at BID on first
of: +PT / −STOP / MAXHOLD / expiry. Idle cash compounds at rf. Daily NAV → CAGR, Sharpe, MaxDD.

v2 fixes from the trade audit (spy_vega_audit.py):
  * DROP the vol-pop exit — it sold at the first +3 VIX tick, i.e. right as the expansion began,
    capping winners (+13.2% vs +6.9%/trade). Default volpop=0 (guarded so 0 never triggers).
  * MAXHOLD 15→30 days — 15 was cutting winners mid-expansion.
  * No leverage — an entry is skipped if its dollar size exceeds available cash (Σpos ≤ NAV).
  * NAV carry-forward — positions with no quote on a day are marked at last-known value (no NAV gaps).

  venv/bin/python research/spy_vol/spy_vega_kernel.py
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
RF = 0.045
RF_D = (1 + RF) ** (1 / 252) - 1


def wing_skew(cur, exp, d, S, dte):
    cur.execute("SELECT DISTINCT strike FROM spy_options_history WHERE trade_date=%s AND expiration=%s", (d, exp))
    ks = [float(r[0]) for r in cur.fetchall()]
    if not ks:
        return None
    Kp = min(ks, key=lambda k: abs(k - S * 0.95)); Kc = min(ks, key=lambda k: abs(k - S * 1.05))
    cp = eod_mid(cur, exp, Kp, 'P', d, d); cc = eod_mid(cur, exp, Kc, 'C', d, d)
    if d not in cp or d not in cc:
        return None
    T = dte / 365
    return cp[d][0] / (0.40 * S * math.sqrt(T)) - cc[d][0] / (0.40 * S * math.sqrt(T))


def backtest(cur, days, vix, spy, rv20, vix_thr_series, alloc, require_flat, max_conc,
             pt=0.30, stop=0.40, volpop=0.0, maxhold=30, cache=None):
    if cache is None:
        cache = {}
    cash = 1.0
    pos = []
    nav_hist = []
    trades = 0
    deployed_days = 0
    for d_ts in days:
        d = d_ts.date()
        cash *= (1 + RF_D)
        S = float(spy.loc[d_ts]); vx = float(vix.loc[d_ts])
        # ---- mark & exit ----
        keep = []
        for q in pos:
            c, p = q['c'], q['p']
            if d in c and d in p:
                mid = c[d][0] + p[d][0]; bid = c[d][1] + p[d][1]
                q['mv'] = mid / q['eask']; q['bidmv'] = bid / q['eask']
                ret = mid / q['eask'] - 1; held = (d - q['edate']).days
                if (ret >= pt or ret <= -stop or (volpop > 0 and vx >= q['vix0'] + volpop)
                        or held >= maxhold or d >= q['exp']):
                    cash += q['alloc'] * q['bidmv']            # sell at bid
                    continue
            elif d >= q['exp']:
                cash += q['alloc'] * q['bidmv']                # settle at last-known bid
                continue
            keep.append(q)
        pos = keep
        nav_now = cash + sum(q['alloc'] * q['mv'] for q in pos)
        # ---- entry ----
        green = vx <= vix_thr_series.loc[d_ts]
        if green and len(pos) < max_conc and not math.isnan(rv20.loc[d_ts]):
            pe = cache.get((d, 'pe'))
            if pe is None:
                pe = pick_entry(cur, d, S, 38, 52); cache[(d, 'pe')] = pe or False
            if pe and pe is not False:
                exp, K, dte = pe
                paths = cache.get((exp, K))
                if paths is None:
                    paths = (eod_mid(cur, exp, K, 'C', d, exp), eod_mid(cur, exp, K, 'P', d, exp))
                    cache[(exp, K)] = paths
                c, p = paths
                if d in c and d in p:
                    eask = c[d][2] + p[d][2]; emid = c[d][0] + p[d][0]
                    T = dte / 365
                    iv = emid / (0.7979 * S * math.sqrt(T)) if T > 0 else 0
                    not_cheap = iv >= float(rv20.loc[d_ts])
                    flat_ok = True
                    if require_flat:
                        sk = wing_skew(cur, exp, d, S, dte)
                        flat_ok = (sk is not None and sk < 0.04)
                    a = alloc * nav_now
                    if eask > 0 and not_cheap and flat_ok and 0 < a <= cash:   # no leverage
                        cash -= a
                        pos.append({'eask': eask, 'alloc': a, 'edate': d, 'exp': exp, 'vix0': vx,
                                    'c': c, 'p': p, 'mv': emid / eask, 'bidmv': (c[d][1] + p[d][1]) / eask})
                        trades += 1
        nav_now = cash + sum(q['alloc'] * q['mv'] for q in pos)
        if pos:
            deployed_days += 1
        nav_hist.append(nav_now)
    nav = pd.Series(nav_hist, index=days)
    rets = nav.pct_change().dropna()
    yrs = (days[-1] - days[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    return {'cagr': cagr, 'sharpe': sharpe, 'mdd': mdd, 'trades': trades,
            'deploy%': deployed_days / len(days), 'final': nav.iloc[-1], 'nav': nav}


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    vix16 = pd.Series(16.0, index=vix.index)
    days = list(spv.index)
    cur = _conn().cursor()

    def metr(nav):
        r = nav.pct_change().dropna(); yrs = (days[-1] - days[0]).days / 365.25
        return nav.iloc[-1] ** (1 / yrs) - 1, (r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0), (nav / nav.cummax() - 1).min()

    boxx = pd.Series((1 + RF_D) ** np.arange(len(days)), index=days)
    bh = spy / spy.iloc[0]
    print("=== SPY VEGA-SCRAPE KERNEL v2 (2018-2026, ask-in/bid-out, idle→BOXX, +30%/30d, NO volpop) ===\n")
    print(f"{'config':<40}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>8}{'trades':>7}{'deploy':>7}")
    print('-' * 75)
    bc = metr(boxx); print(f"{'BENCH all-BOXX':<40}{bc[0]:>6.1%}{'n/a':>6}{bc[2]:>8.1%}{'-':>7}{'-':>7}")
    bm = metr(bh); print(f"{'BENCH buy&hold SPY':<40}{bm[0]:>6.1%}{bm[1]:>6.2f}{bm[2]:>8.1%}{'-':>7}{'-':>7}")
    print('-' * 75)
    cache = {}
    configs = [
        ('alloc10 conc1', 0.10, 1, {}),
        ('alloc20 conc1', 0.20, 1, {}),
        ('alloc20 conc2', 0.20, 2, {}),
        ('alloc30 conc2', 0.30, 2, {}),
        ('alloc30 conc3', 0.30, 3, {}),
        ('alloc20 conc2 +flat', 0.20, 2, {'require_flat': True}),
        ('OLD-EXIT alloc20 conc2 (VIX+3/15d)', 0.20, 2, {'volpop': 3.0, 'maxhold': 15}),
    ]
    best = None
    for name, alloc, conc, kw in configs:
        rf = kw.pop('require_flat', False)
        r = backtest(cur, days, vix, spy, rv20, vix16, alloc, rf, conc, cache=cache, **kw)
        print(f"{name:<40}{r['cagr']:>6.1%}{r['sharpe']:>6.2f}{r['mdd']:>8.1%}{r['trades']:>7}{r['deploy%']:>6.0%}")
        if 'OLD' not in name and (best is None or r['sharpe'] > best[1]['sharpe']):
            best = (name, r)
    print('-' * 75)
    print(f"\nBEST Sharpe (v2): {best[0]} → CAGR {best[1]['cagr']:.1%}, Sharpe {best[1]['sharpe']:.2f}, "
          f"MaxDD {best[1]['mdd']:.1%}, {best[1]['trades']} trades")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
