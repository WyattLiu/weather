"""Tune cross-asset LIQUIDITY factors for SNAP-DOWNS (the left tail), not average returns. A snap-down
= SPY drops sharply within a short window. For each liquidity factor we measure P(snap-down) when the
factor is elevated vs the base rate — i.e. does dollar-up / credit-widening / bond-vol-rising actually
precede sudden equity drops? Builds a tuned composite and prints lift + the current reading.

  venv/bin/python research/spy_vol/spy_snapdown_signals.py
"""
import pandas as pd

TK = {'SPY': 'SPY', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'HYG': 'HYG', 'LQD': 'LQD',
      'TLT': 'TLT', 'TNX': '^TNX', 'MOVE': '^MOVE', 'GLD': 'GLD'}


def pull():
    import yfinance as yf
    out = {}
    for n, tk in TK.items():
        try:
            s = yf.download(tk, start='2015-01-01', progress=False)['Close']
            if hasattr(s, 'columns'):
                s = s.iloc[:, 0]
            out[n] = s
        except Exception as e:
            print(f"  {n} FAIL {repr(e)[:40]}")
    return pd.DataFrame(out)


def main():
    print("pulling cross-asset…")
    df = pull()
    spy, vix = df['SPY'], df['VIX']
    f = pd.DataFrame(index=df.index)
    f['dxy_5d'] = df['DXY'].pct_change(5) * 100
    f['dxy_20d'] = df['DXY'].pct_change(20) * 100
    f['credit_5d'] = -(df['HYG'] / df['LQD']).pct_change(5) * 100        # HY underperform IG
    f['hyg_5d'] = -df['HYG'].pct_change(5) * 100                          # HY falling
    f['move_lvl'] = df['MOVE']
    f['move_chg5'] = df['MOVE'].pct_change(5) * 100                       # bond vol RISING
    f['rates_5d'] = df['TNX'].diff(5)
    f['gold_5d'] = df['GLD'].pct_change(5) * 100
    f['vix_lvl'] = vix
    f['vix_chg5'] = vix - vix.shift(5)

    # SNAP-DOWN target: worst forward cumulative return over next 10 trading days < -3%
    fwd_min10 = pd.concat([spy.shift(-k) / spy - 1 for k in range(1, 11)], axis=1).min(axis=1)
    snap = (fwd_min10 < -0.03).astype(float)                              # sudden >3% drop ahead
    base = snap.mean()
    print(f"\n=== SNAP-DOWN base rate (>3% drop within 10d): {base:.1%} of days ===\n")
    print(f"  {'factor':<12}{'P(snap|hi)':>11}{'P(snap|lo)':>11}{'lift hi/base':>13}")
    print('  ' + '-' * 47)
    lifts = {}
    for c in [x for x in f.columns]:
        s = pd.DataFrame({'x': f[c], 'snap': snap}).dropna()
        hi = s[s['x'] >= s['x'].quantile(0.8)]['snap'].mean()             # top quintile of the factor
        lo = s[s['x'] <= s['x'].quantile(0.2)]['snap'].mean()
        lifts[c] = hi / base if base else 0
        print(f"  {c:<12}{hi:>10.1%}{lo:>11.1%}{hi/base:>12.2f}x")

    # tuned composite = mean z of the factors that LIFT snap-downs (lift>1.1), signed so + = stress
    good = [c for c, l in lifts.items() if l > 1.1]
    print(f"\n  factors that lift snap-downs (>1.1x): {good}")
    fz = f[good].apply(lambda s: (s - s.rolling(252).mean()) / s.rolling(252).std())
    comp = fz.mean(axis=1)
    s = pd.DataFrame({'comp': comp, 'snap': snap}).dropna()
    print("\n  === TUNED composite (mean-z of those) → P(snap-down) by quintile ===")
    qs = s['comp'].quantile([0, .2, .4, .6, .8, 1.0]).values
    for i in range(5):
        lo, hi = qs[i], qs[i + 1]
        x = s[(s['comp'] >= lo) & (s['comp'] <= hi)] if i == 4 else s[(s['comp'] >= lo) & (s['comp'] < hi)]
        print(f"    quintile {i+1} [{lo:>+5.2f},{hi:>+5.2f}] n={len(x):>4}  P(snap)={x['snap'].mean():>5.1%}  "
              f"({x['snap'].mean()/base:.2f}x base)")
    # current
    print(f"\n  === CURRENT (latest) ===  date {f.index[-1].date()} SPY {spy.iloc[-1]:.0f} VIX {vix.iloc[-1]:.1f}")
    cz = comp.iloc[-1]
    print(f"    tuned composite z = {cz:+.2f}  → " +
          ("ELEVATED snap-down risk" if cz > 0.5 else "low/neutral" if cz < 0.3 else "mild"))
    for c in good:
        print(f"      {c:<12} {f[c].iloc[-1]:+.2f}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
