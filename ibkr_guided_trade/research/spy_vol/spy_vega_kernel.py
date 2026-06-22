"""SPY VEGA-SCRAPE KERNEL — deployable strategy, optimized for return / Sharpe / MaxDD.

Mirrors the UNG regime_wheel_boxx idea: the long-vol edge is RARE (only ~low-VIX windows), so the
kernel holds BOXX/T-bills (~rf) when idle and only deploys into a long ~45 DTE ATM straddle when the
firmed-up signal is GREEN:
    LOW VIX (≤ thr)  AND  NOT-CHEAP (ATM IV ≥ RV20, i.e. not just-after a vol spike)  [AND flat skew]
Entry crosses to ASK, daily MTM at mid, exit at BID on first of: +PT / −STOP / VIX≥vix0+VOLPOP /
MAXHOLD / expiry. Idle cash compounds at rf. Daily NAV → CAGR, annualized Sharpe, MaxDD.

Sweeps allocation / VIX threshold / flat-skew / max-concurrent and prints a return-Sharpe-MaxDD Pareto.

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
    ivp = cp[d][0] / (0.40 * S * math.sqrt(T)); ivc = cc[d][0] / (0.40 * S * math.sqrt(T))
    return ivp - ivc


def backtest(cur, days, vix, spy, rv20, vix_thr_series, alloc, require_flat, max_conc,
             pt=0.30, stop=0.40, volpop=3.0, maxhold=15, cache={}):
    cash = 1.0
    pos = []
    nav_hist = []
    trades = 0
    deployed_days = 0
    for d_ts in days:
        d = d_ts.date()
        cash *= (1 + RF_D)
        S = float(spy.loc[d_ts]); vx = float(vix.loc[d_ts])
        # ---- mark & exit open positions ----
        keep = []
        for q in pos:
            c, p = q['c'], q['p']
            if d not in c or d not in p:
                if d >= q['exp']:                       # expired w/o quote → settle at last known
                    cash += q['alloc'] * (q['last'] / q['entry'])
                else:
                    keep.append(q)
                continue
            mid = c[d][0] + p[d][0]; bid = c[d][1] + p[d][1]
            q['last'] = bid
            ret = mid / q['entry'] - 1; held = (d - q['edate']).days
            if ret >= pt or ret <= -stop or vx >= q['vix0'] + volpop or held >= maxhold or d >= q['exp']:
                cash += q['alloc'] * (bid / q['entry'])     # sell at bid
            else:
                keep.append(q)
        pos = keep
        # ---- entry decision ----
        nav_now = cash + sum(q['alloc'] * ((q['c'][d][0] + q['p'][d][0]) / q['entry'])
                             for q in pos if d in q['c'] and d in q['p'])
        green = vx <= vix_thr_series.loc[d_ts]
        if green and len(pos) < max_conc and not math.isnan(rv20.loc[d_ts]):
            key = (d, 'pe')
            pe = cache.get(key)
            if pe is None:
                pe = pick_entry(cur, d, S, 38, 52); cache[key] = pe or False
            if pe and pe is not False:
                exp, K, dte = pe
                ck = (exp, K)
                paths = cache.get(ck)
                if paths is None:
                    c = eod_mid(cur, exp, K, 'C', d, exp); p = eod_mid(cur, exp, K, 'P', d, exp)
                    cache[ck] = (c, p); paths = (c, p)
                c, p = paths
                if d in c and d in p:
                    entry_ask = c[d][2] + p[d][2]; entry_mid = c[d][0] + p[d][0]
                    T = dte / 365
                    iv = entry_mid / (0.7979 * S * math.sqrt(T)) if T > 0 else 0
                    not_cheap = iv >= float(rv20.loc[d_ts])
                    flat_ok = True
                    if require_flat:
                        sk = wing_skew(cur, exp, d, S, dte)
                        flat_ok = (sk is not None and sk < 0.04)
                    if entry_ask > 0 and not_cheap and flat_ok:
                        a = alloc * nav_now
                        cash -= a
                        pos.append({'entry': entry_ask, 'alloc': a, 'edate': d, 'exp': exp,
                                    'vix0': vx, 'c': c, 'p': p, 'last': c[d][1] + p[d][1]})
                        trades += 1
        nav_now = cash + sum(q['alloc'] * ((q['c'][d][0] + q['p'][d][0]) / q['entry'])
                             for q in pos if d in q['c'] and d in q['p'])
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
    # adaptive "low VIX" = trailing 1yr 35th pctile (no lookahead), floored at 13
    vix_pct35 = vix.rolling(252, min_periods=60).quantile(0.35).clip(lower=13)
    vix_abs16 = pd.Series(16.0, index=vix.index)
    days = list(spv.index)
    cur = _conn().cursor()

    # benchmarks
    boxx = pd.Series((1 + RF_D) ** np.arange(len(days)), index=days)
    bh = spy / spy.iloc[0]
    def metr(nav):
        r = nav.pct_change().dropna(); yrs = (days[-1]-days[0]).days/365.25
        return nav.iloc[-1]**(1/yrs)-1, (r.mean()/r.std()*math.sqrt(252) if r.std()>0 else 0), (nav/nav.cummax()-1).min()

    print("=== SPY VEGA-SCRAPE KERNEL (2018-2026, realistic ask-in/bid-out, idle→BOXX) ===\n")
    print(f"{'config':<46}{'CAGR':>7}{'Shrp':>6}{'MaxDD':>7}{'trades':>7}{'deploy':>7}")
    print('-' * 80)
    bc = metr(boxx); print(f"{'BENCH: all-BOXX (rf 4.5%)':<46}{bc[0]:>6.1%}{bc[1]:>6.1f}{bc[2]:>7.1%}{'-':>7}{'-':>7}")
    bhm = metr(bh); print(f"{'BENCH: buy&hold SPY':<46}{bhm[0]:>6.1%}{bhm[1]:>6.1f}{bhm[2]:>7.1%}{'-':>7}{'-':>7}")
    print('-' * 80)
    configs = [
        ('alloc10 VIX16 conc1', 0.10, vix_abs16, False, 1),
        ('alloc20 VIX16 conc1', 0.20, vix_abs16, False, 1),
        ('alloc20 VIX16 conc2', 0.20, vix_abs16, False, 2),
        ('alloc30 VIX16 conc2', 0.30, vix_abs16, False, 2),
        ('alloc20 VIX16 conc2 +flat', 0.20, vix_abs16, True, 2),
        ('alloc20 adaptVIX35 conc2', 0.20, vix_pct35, False, 2),
        ('alloc30 adaptVIX35 conc2', 0.30, vix_pct35, False, 2),
        ('alloc30 adaptVIX35 conc3', 0.30, vix_pct35, False, 3),
        ('alloc50 adaptVIX35 conc3', 0.50, vix_pct35, False, 3),
    ]
    best = None
    for name, alloc, vthr, flat, conc in configs:
        r = backtest(cur, days, vix, spy, rv20, vthr, alloc, flat, conc)
        print(f"{name:<46}{r['cagr']:>6.1%}{r['sharpe']:>6.2f}{r['mdd']:>7.1%}{r['trades']:>7}{r['deploy%']:>6.0%}")
        if best is None or r['sharpe'] > best[1]['sharpe']:
            best = (name, r)
    print('-' * 80)
    print(f"\nBEST Sharpe: {best[0]} → CAGR {best[1]['cagr']:.1%}, Sharpe {best[1]['sharpe']:.2f}, "
          f"MaxDD {best[1]['mdd']:.1%}, {best[1]['trades']} trades, deployed {best[1]['deploy%']:.0%} of days")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
