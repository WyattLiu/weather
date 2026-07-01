"""COMBINED multi-kernel backtest: UNG wheel (regime_wheel_boxx_greeks) + SPY vega overlay funded by
liquidating BOXX (shared dry-powder pool). The slow UNG champion runs ONCE (cached to disk); the SPY
overlay is then swept cheaply across allocation sizes to give the sizing frontier.

  venv/bin/python research/spy_vol/spy_ung_portfolio.py            # sweep default allocs
  venv/bin/python research/spy_vol/spy_ung_portfolio.py --fresh    # force-rerun the UNG champion
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
BT = os.path.join(THIS, '..', '..', 'backtest')
sys.path.insert(0, BT)
sys.path.insert(0, THIS)
from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z   # noqa: E402
from spy_vega_study import _conn, pick_entry, eod_mid                            # noqa: E402

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')
UNG_CACHE = os.path.join(THIS, 'cache', 'ung_champion_hist.csv')
PT, STOP, MAXHOLD = 0.30, 0.40, 30


def metr(nav, idx):
    nav = pd.Series(nav, index=idx); r = nav.pct_change().dropna()
    yrs = (idx[-1] - idx[0]).days / 365.25
    return ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0,
            (nav / nav.cummax() - 1).min())


def run_ung(fresh=False):
    if not fresh and os.path.exists(UNG_CACHE):
        h = pd.read_csv(UNG_CACHE, index_col=0, parse_dates=True)
        print(f"  loaded cached UNG champion ({len(h)} days)", flush=True)
        return h
    print("  running UNG champion (regime_wheel_boxx_greeks)… (~10min)", flush=True)
    df = pd.read_csv(os.path.join(BT, 'cache', 'master_dataset.csv'), index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG']).loc['2021-01-01':]
    hist, _ = run_strategy_simple(df, {**STRATEGIES['regime_wheel_boxx_greeks'],
                                       'use_real_chain_fills': True}, 100000, 0)  # HONEST fills (real PG chain)
    hist = hist.set_index('date')
    boxx_px = df['BOXX'].reindex(hist.index).ffill().fillna(117.0)
    out = pd.DataFrame({'nav': hist['nav'], 'boxx_usd': hist['boxx'] * boxx_px})
    out.to_csv(UNG_CACHE)
    print(f"  UNG champion done, cached ({len(out)} days)", flush=True)
    return out


def overlay(ung, vix, spy, rv20, cur, pc, alloc):
    """SPY vega sleeve funded from BOXX. Returns (combined_nav_series, funded, skipped, pnl, min_boxx)."""
    cum_pnl = 0.0; pos = None; funded = skipped = 0; min_boxx = 1e12; out = []
    for d_ts in ung.index:
        d = d_ts.date()
        ung_nav = float(ung.loc[d_ts, 'nav']); boxx_usd = float(ung.loc[d_ts, 'boxx_usd'])
        unreal = 0.0
        if pos is not None:
            c, p = pos['c'], pos['p']
            if d in c and d in p:
                mid = c[d][0] + p[d][0]; bid = c[d][1] + p[d][1]
                pos['mv'] = pos['cost'] * (mid / pos['eask'])
                ret = mid / pos['eask'] - 1; held = (d - pos['edate']).days
                if ret >= PT or ret <= -STOP or held >= MAXHOLD or d >= pos['exp']:
                    cum_pnl += pos['cost'] * (bid / pos['eask']) - pos['cost']; pos = None
            elif d >= pos['exp']:
                cum_pnl += pos['mv'] - pos['cost']; pos = None
        if pos is None and float(vix.loc[d_ts]) <= 16 and not math.isnan(rv20.loc[d_ts]):
            S = float(spy.loc[d_ts]); pe = pick_entry(cur, d, S, 38, 52)
            if pe:
                exp, K, dte = pe
                key = (exp, K)
                if key not in pc:
                    pc[key] = (eod_mid(cur, exp, K, 'C', d, exp), eod_mid(cur, exp, K, 'P', d, exp))
                c, p = pc[key]
                if d in c and d in p:
                    eask = c[d][2] + p[d][2]; emid = c[d][0] + p[d][0]
                    iv = emid / (0.7979 * S * math.sqrt(dte / 365)) if dte > 0 else 0
                    if iv >= float(rv20.loc[d_ts]) and eask > 0:
                        want = alloc * (ung_nav + cum_pnl)
                        deploy = min(want, max(0.0, boxx_usd))
                        if deploy >= 0.5 * want and deploy > 1000:
                            pos = {'eask': eask, 'cost': deploy, 'edate': d, 'exp': exp,
                                   'c': c, 'p': p, 'mv': deploy * (emid / eask)}
                            funded += 1; min_boxx = min(min_boxx, boxx_usd - deploy)
                        else:
                            skipped += 1
        if pos is not None:
            unreal = pos['mv'] - pos['cost']
        out.append(ung_nav + cum_pnl + unreal)
    return out, funded, skipped, cum_pnl, min_boxx


def main(fresh=False):
    ung = run_ung(fresh)
    idx = ung.index
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX'].reindex(idx).ffill(); spy = spv['SPY'].reindex(idx).ffill()
    rv20 = np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252)
    cur = _conn().cursor(); pc = {}

    cu, su, mu = metr(ung['nav'].values, idx)
    print(f"\n=== UNG + BOXX-funded SPY overlay — sizing sweep ({idx[0].date()}→{idx[-1].date()}) ===\n")
    print(f"  {'config':<22}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'funded':>8}{'skip':>6}{'minBOXX':>10}")
    print('  ' + '-' * 68)
    print(f"  {'UNG alone':<22}{cu:>7.1%}{su:>8.2f}{mu:>8.1%}{'-':>8}{'-':>6}{'-':>10}")
    for alloc in (0.10, 0.15, 0.20, 0.25, 0.30):
        nav, funded, skipped, pnl, minb = overlay(ung, vix, spy, rv20, cur, pc, alloc)
        cc, sc, mc = metr(nav, idx)
        print(f"  {'+SPY '+f'{alloc:.0%}/trade':<22}{cc:>7.1%}{sc:>8.2f}{mc:>8.1%}{funded:>8}{skipped:>6}"
              f"{('$'+format(int(minb),',')) if minb<1e11 else '-':>10}")
    print(f"\n  avg BOXX dry powder: ${ung['boxx_usd'].mean():,.0f}  ·  UNG-alone {cu:.1%}/{su:.2f}/{mu:.1%}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main('--fresh' in sys.argv)
