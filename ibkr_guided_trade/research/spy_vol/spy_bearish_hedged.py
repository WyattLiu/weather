"""Bearish-HEDGED structure vs the symmetric straddle, triggered by the liquidity SNAP-DOWN composite.

Core idea: long puts (bearish snap-down bet) + a cheap NEAR-DTE call (wrong-way hedge — costs little when
SPY drops, but cushions if SPY rips up). Sweep put/call DTE, strike, and ratio; compare every structure to
the symmetric ATM straddle on the SAME trigger days. Two triggers: FULL composite (incl. VIX) and VIX-FREE
(credit + bond-vol + rates only — independent of the vol we trade). All legs priced with a consistent
Black-Scholes model (VIX=IV), mid-to-mid, so the RELATIVE comparison is fair (absolute is idealized).

  venv/bin/python research/spy_vol/spy_bearish_hedged.py
"""
import math
import numpy as np
import pandas as pd

R, HOLD = 0.045, 10   # risk-free; hold 10 trading days (the snap-down horizon)
TK = {'SPY': 'SPY', 'VIX': '^VIX', 'HYG': 'HYG', 'LQD': 'LQD', 'TLT': 'TLT', 'TNX': '^TNX', 'MOVE': '^MOVE'}


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, dte_td, sigma, kind):
    T = dte_td / 252.0
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if kind == 'C' else (K - S))
    d1 = (math.log(S / K) + (R + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == 'C':
        return S * N(d1) - K * math.exp(-R * T) * N(d2)
    return K * math.exp(-R * T) * N(-d2) - S * N(-d1)


def pull():
    import yfinance as yf
    out = {}
    for n, tk in TK.items():
        s = yf.download(tk, start='2015-01-01', progress=False)['Close']
        if hasattr(s, 'columns'):
            s = s.iloc[:, 0]
        out[n] = s
    return pd.DataFrame(out)


def composites(df):
    f = pd.DataFrame(index=df.index)
    f['credit'] = -(df['HYG'] / df['LQD']).pct_change(5) * 100
    f['hyg'] = -df['HYG'].pct_change(5) * 100
    f['move_chg'] = df['MOVE'].pct_change(5) * 100
    f['move_lvl'] = df['MOVE']
    f['rates'] = df['TNX'].diff(5)
    f['vix_lvl'] = df['VIX']
    f['vix_chg'] = df['VIX'] - df['VIX'].shift(5)
    z = lambda s: (s - s.rolling(252).mean()) / s.rolling(252).std()
    full = f[['hyg', 'move_chg', 'move_lvl', 'rates', 'vix_lvl', 'vix_chg']].apply(z).mean(axis=1)
    vixfree = f[['credit', 'hyg', 'move_chg', 'move_lvl', 'rates']].apply(z).mean(axis=1)
    return full, vixfree


# structures: list of legs (qty, dte_td, moneyness, kind). moneyness: put<0 = OTM below; call>0 = OTM above
STRUCTS = {
    'STRADDLE 1C1P atm45 (base)': [(1, 45, 0.0, 'C'), (1, 45, 0.0, 'P')],
    'puts only atm45':            [(1, 45, 0.0, 'P')],
    'puts only 3%otm 45':         [(1, 45, -0.03, 'P')],
    'bear 1P45 +1C14 (3%otm)':    [(1, 45, 0.0, 'P'), (1, 14, 0.03, 'C')],
    'bear 1P45 +0.5C14 (3%otm)':  [(1, 45, 0.0, 'P'), (0.5, 14, 0.03, 'C')],
    'bear 1P45 +1C14 (5%otm)':    [(1, 45, 0.0, 'P'), (1, 14, 0.05, 'C')],
    'bear 1P45 +1C21 (3%otm)':    [(1, 45, 0.0, 'P'), (1, 21, 0.03, 'C')],
    'bear 2P45 +1C14 (3%otm)':    [(2, 45, 0.0, 'P'), (1, 14, 0.03, 'C')],
    'bear 1P30 +1C10 (3%otm)':    [(1, 30, 0.0, 'P'), (1, 10, 0.03, 'C')],
    'bear 1P45(3otm)+1C14(4otm)': [(1, 45, -0.03, 'P'), (1, 14, 0.04, 'C')],
    'put spread 45 (atm/7otm)':   [(1, 45, 0.0, 'P'), (-1, 45, -0.07, 'P')],
}


def sim(struct, i, spy, vix):
    S0 = spy[i]; sig0 = vix[i] / 100.0
    entry = exit_ = 0.0
    for qty, dte, mny, kind in struct:
        K = S0 * (1 + mny)
        entry += qty * bs(S0, K, dte, sig0, kind)
        eff = min(HOLD, dte)
        j = i + eff
        if j >= len(spy):
            return None
        if dte <= HOLD:                       # leg expired → intrinsic at its expiry-day spot
            Se = spy[i + dte]
            exit_ += qty * (max(0.0, Se - K) if kind == 'C' else max(0.0, K - Se))
        else:
            exit_ += qty * bs(spy[j], K, dte - HOLD, vix[j] / 100.0, kind)
    if entry <= 0:
        return None
    return exit_ / entry - 1, spy[i + HOLD] / S0 - 1


def run(name, trig, spy, vix):
    idx = np.where(trig)[0]
    idx = idx[idx + HOLD < len(spy)]
    print(f"\n=== trigger: {name}  (n={len(idx)} entry days) ===")
    print(f"  {'structure':<30}{'avg':>8}{'win%':>6}{'ret|↓':>8}{'ret|↑':>8}{'Sharpe':>8}")
    print('  ' + '-' * 68)
    base = None
    for sname, st in STRUCTS.items():
        rs, mv = [], []
        for i in idx:
            r = sim(st, i, spy, vix)
            if r:
                rs.append(r[0]); mv.append(r[1])
        rs = np.array(rs); mv = np.array(mv)
        if not len(rs):
            continue
        dn = rs[mv < 0]; up = rs[mv > 0]
        sh = rs.mean() / rs.std() * math.sqrt(252 / HOLD) if rs.std() > 0 else 0
        tag = '  <= base' if 'base' in sname else ''
        print(f"  {sname:<30}{rs.mean():>+7.1%}{(rs>0).mean()*100:>5.0f}%"
              f"{(dn.mean() if len(dn) else 0):>+8.1%}{(up.mean() if len(up) else 0):>+8.1%}{sh:>8.2f}{tag}")


def main():
    print("pulling cross-asset…")
    df = pull().dropna(subset=['SPY', 'VIX'])
    full, vixfree = composites(df)
    spy = df['SPY'].values; vix = df['VIX'].values
    thr_f = np.nanquantile(full.values, 0.80)
    thr_v = np.nanquantile(vixfree.values, 0.80)
    trig_full = (full.values >= thr_f)
    trig_vf = (vixfree.values >= thr_v)
    trig_all = np.ones(len(spy), bool)
    print(f"  full-composite 80th pct = {thr_f:+.2f} | vix-free 80th pct = {thr_v:+.2f}")
    run('ALL DAYS (no trigger, baseline)', trig_all, spy, vix)
    run('FULL composite top-quintile', trig_full, spy, vix)
    run('VIX-FREE composite top-quintile', trig_vf, spy, vix)
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
