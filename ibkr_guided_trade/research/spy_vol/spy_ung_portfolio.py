"""COMBINED multi-kernel backtest: UNG wheel (regime_wheel_boxx_greeks) + SPY vega overlay funded by
liquidating BOXX. One shared dry-powder pool — the SPY straddle can only deploy what the wheel has
parked in BOXX, and proceeds flow back. Reports UNG-alone vs combined (CAGR/Sharpe/MaxDD), SPY setups
funded vs skipped (BOXX-constrained), and BOXX utilization.

  venv/bin/python research/spy_vol/spy_ung_portfolio.py
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
PT, STOP, MAXHOLD = 0.30, 0.40, 30


def metr(nav, idx):
    nav = pd.Series(nav, index=idx)
    r = nav.pct_change().dropna()
    yrs = (idx[-1] - idx[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    mdd = (nav / nav.cummax() - 1).min()
    return cagr, sh, mdd


def run_ung():
    df = pd.read_csv(os.path.join(BT, 'cache', 'master_dataset.csv'), index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    df = df.loc['2021-01-01':]                       # SPY clean minute starts 2021
    hist, _ = run_strategy_simple(df, STRATEGIES['regime_wheel_boxx_greeks'], 100000, 0)
    hist = hist.set_index('date')
    boxx_px = df['BOXX'].reindex(hist.index).ffill().fillna(117.0)
    hist['boxx_usd'] = hist['boxx'] * boxx_px        # BOXX dry powder ($)
    return hist


def straddle_path(cur, exp, K, d, pc):
    key = (exp, K)
    if key not in pc:
        pc[key] = (eod_mid(cur, exp, K, 'C', d, exp), eod_mid(cur, exp, K, 'P', d, exp))
    return pc[key]


def main(alloc=0.15):
    print("Running UNG champion (regime_wheel_boxx_greeks)…", flush=True)
    ung = run_ung()
    idx = ung.index
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX'].reindex(idx).ffill(); spy = spv['SPY'].reindex(idx).ffill()
    rv20 = (np.log(spy / spy.shift(1)).rolling(20).std() * math.sqrt(252))
    cur = _conn().cursor(); pc = {}

    cum_pnl = 0.0          # realized SPY P&L
    pos = None             # open straddle: {eask, cost, edate, exp, K, c, p, mv}
    funded = skipped = 0
    min_boxx_after = 1e12
    combined = []
    for d_ts in idx:
        d = d_ts.date()
        ung_nav = float(ung.loc[d_ts, 'nav'])
        boxx_usd = float(ung.loc[d_ts, 'boxx_usd'])
        unreal = 0.0
        # mark / exit open straddle
        if pos is not None:
            c, p = pos['c'], pos['p']
            if d in c and d in p:
                mid = c[d][0] + p[d][0]; bid = c[d][1] + p[d][1]
                pos['mv'] = pos['cost'] * (mid / pos['eask'])
                ret = mid / pos['eask'] - 1; held = (d - pos['edate']).days
                if ret >= PT or ret <= -STOP or held >= MAXHOLD or d >= pos['exp']:
                    cum_pnl += pos['cost'] * (bid / pos['eask']) - pos['cost']
                    pos = None
            elif d >= pos['exp']:
                cum_pnl += pos['cost'] * (pos['mv'] / max(pos['cost'], 1e-9)) - pos['cost']
                pos = None
        # entry decision (GREEN + BOXX available)
        if pos is None and float(vix.loc[d_ts]) <= 16 and not math.isnan(rv20.loc[d_ts]):
            S = float(spy.loc[d_ts])
            pe = pick_entry(cur, d, S, 38, 52)
            if pe:
                exp, K, dte = pe
                c, p = straddle_path(cur, exp, K, d, pc)
                if d in c and d in p:
                    eask = c[d][2] + p[d][2]; emid = c[d][0] + p[d][0]
                    iv = emid / (0.7979 * S * math.sqrt(dte / 365)) if dte > 0 else 0
                    if iv >= float(rv20.loc[d_ts]) and eask > 0:   # not-cheap GREEN
                        combined_nav_now = ung_nav + cum_pnl
                        want = alloc * combined_nav_now
                        deploy = min(want, max(0.0, boxx_usd))     # fund ONLY from BOXX
                        if deploy >= 0.5 * want and deploy > 1000:  # enough dry powder
                            pos = {'eask': eask, 'cost': deploy, 'edate': d, 'exp': exp, 'K': K,
                                   'c': c, 'p': p, 'mv': deploy * (emid / eask)}
                            funded += 1
                            min_boxx_after = min(min_boxx_after, boxx_usd - deploy)
                        else:
                            skipped += 1
        if pos is not None:
            unreal = pos['mv'] - pos['cost']
        combined.append(ung_nav + cum_pnl + unreal)

    ung_nav_s = ung['nav'].values
    cu, su, mu = metr(ung_nav_s, idx)
    cc, sc, mc = metr(combined, idx)
    print(f"\n=== COMBINED: UNG wheel + SPY vega (BOXX-funded, alloc {alloc:.0%}/trade, {idx[0].date()}→{idx[-1].date()}) ===\n")
    print(f"  {'sleeve':<28}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}")
    print(f"  {'UNG wheel alone':<28}{cu:>7.1%}{su:>8.2f}{mu:>8.1%}")
    print(f"  {'UNG + SPY (BOXX-funded)':<28}{cc:>7.1%}{sc:>8.2f}{mc:>8.1%}")
    print(f"\n  Δ from adding SPY overlay:  CAGR {cc-cu:+.1%}   Sharpe {sc-su:+.2f}   MaxDD {mc-mu:+.1%}")
    print(f"  SPY setups funded: {funded}  | skipped (BOXX-constrained): {skipped}")
    print(f"  realized SPY P&L: ${cum_pnl:,.0f}  | min BOXX left after a deploy: ${min_boxx_after:,.0f}"
          if funded else "  (no SPY setups in window)")
    print(f"  avg BOXX dry powder: ${ung['boxx_usd'].mean():,.0f}  (vs ~{alloc:.0%}-of-NAV straddle need)")
    print("DONE", flush=True)


if __name__ == '__main__':
    a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
    main(a)
