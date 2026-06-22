"""WALK-FORWARD / regime-robustness of the SPY vega kernel v2.

Runs the kernel over 2018-2026 and decomposes the daily-NAV equity curve by CALENDAR YEAR, by HALF
(2018-21 vs 2022-26), and reports the WORST rolling-12-month return + per-year trade attribution.
Question: is the edge broad, or carried by 2018/2020/2024 vol events? Done for two configs.

  venv/bin/python research/spy_vol/spy_vega_walkforward.py
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_kernel import backtest, RF_D
from spy_vega_study import _conn

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')


def seg(nav):
    r = nav.pct_change().dropna()
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    return ret, sh, mdd


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    spv = spv[spv.index >= '2018-01-01']
    vix = spv['VIX']; spy = spv['SPY']
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    vix16 = pd.Series(16.0, index=vix.index)
    days = list(spv.index)
    cur = _conn().cursor()
    cache = {}

    for name, alloc, conc in [('alloc10 / 1-pos (best Sharpe)', 0.10, 1),
                              ('alloc20 / 2-pos (aggressive)', 0.20, 2)]:
        r = backtest(cur, days, vix, spy, rv20, vix16, alloc, False, conc, cache=cache)
        nav = r['nav']
        print(f"\n{'='*72}\n=== {name}  |  full: CAGR {r['cagr']:.1%}  Sharpe {r['sharpe']:.2f}  MaxDD {r['mdd']:.1%} ===")
        tl = pd.DataFrame(r['tradelog'])
        tl['yr'] = pd.to_datetime(tl['xdate']).dt.year

        print(f"\n{'year':<6}{'NAV ret':>9}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}{'trade avg':>11}{'win%':>7}")
        print('-' * 57)
        for y in range(2018, 2027):
            ny = nav[nav.index.year == y]
            if len(ny) < 2:
                continue
            ret, sh, mdd = seg(ny)
            ty = tl[tl['yr'] == y]
            tinfo = (f"{len(ty):>8}{ty['ret'].mean():>+11.1%}{(ty['ret']>0).mean()*100:>6.0f}%"
                     if len(ty) else f"{0:>8}{'-':>11}{'-':>7}")
            print(f"{y:<6}{ret:>+9.1%}{sh:>8.2f}{mdd:>8.1%}{tinfo}")

        # halves
        print(f"\n  {'half':<14}{'ann.ret':>9}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
        for lab, lo, hi in [('2018-2021', 2018, 2021), ('2022-2026', 2022, 2026)]:
            nh = nav[(nav.index.year >= lo) & (nav.index.year <= hi)]
            if len(nh) < 2:
                continue
            ret, sh, mdd = seg(nh)
            yrs = (nh.index[-1] - nh.index[0]).days / 365.25
            ann = (1 + ret) ** (1 / yrs) - 1
            th = tl[(tl['yr'] >= lo) & (tl['yr'] <= hi)]
            print(f"  {lab:<14}{ann:>+9.1%}{sh:>8.2f}{mdd:>8.1%}{len(th):>8}")

        # worst rolling 12m (252-trading-day) NAV return
        roll = nav.pct_change().rolling(252).apply(lambda w: np.prod(1 + w) - 1, raw=True)
        wr = roll.min()
        wr_end = roll.idxmin()
        print(f"\n  worst rolling-12mo NAV return: {wr:+.1%} (ending {wr_end.date() if pd.notna(wr_end) else 'n/a'})")
        # contribution: top-3 trades' share of total log-growth
        tl_sorted = tl.reindex(tl['ret'].abs().sort_values(ascending=False).index)
        print(f"  biggest single trades: " + ", ".join(f"{x:+.0%}" for x in tl_sorted['ret'].head(5)))
        pos_sum = tl[tl['ret'] > 0]['ret'].sum()
        top3 = tl.nlargest(3, 'ret')['ret'].sum()
        print(f"  top-3 winners = {top3/pos_sum*100:.0f}% of all positive trade-return (concentration check)")
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
