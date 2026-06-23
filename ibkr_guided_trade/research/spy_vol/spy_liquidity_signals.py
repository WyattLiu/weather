"""Cross-asset LIQUIDITY / risk-off signal set — does it LEAD equity stress?

Pulls DXY, credit (HYG/LQD), Treasuries (TLT/IEF/^TNX), bond vol (^MOVE), gold (GLD), VIX, SPY (yfinance,
2015-2026) and builds liquidity-stress features, then tests whether they PREDICT forward SPY return /
VIX change. The thesis: dollar up + credit underperforming + bond vol up + gold bid = liquidity crunch
→ equities fall. Prints predictive power per signal, a composite, and the CURRENT reading.

  venv/bin/python research/spy_vol/spy_liquidity_signals.py
"""
import numpy as np
import pandas as pd

TICKERS = {'SPY': 'SPY', 'VIX': '^VIX', 'DXY': 'DX-Y.NYB', 'HYG': 'HYG', 'LQD': 'LQD',
           'TLT': 'TLT', 'IEF': 'IEF', 'TNX': '^TNX', 'MOVE': '^MOVE', 'GLD': 'GLD'}


def pull():
    import yfinance as yf
    out = {}
    for name, tk in TICKERS.items():
        try:
            s = yf.download(tk, start='2015-01-01', progress=False)['Close']
            if hasattr(s, 'columns'):
                s = s.iloc[:, 0]
            if len(s) > 100:
                out[name] = s
                print(f"  {name:5} ({tk:10}) {len(s)} days  {s.index[0].date()}→{s.index[-1].date()}")
            else:
                print(f"  {name:5} ({tk}) — too few rows, skip")
        except Exception as e:
            print(f"  {name:5} ({tk}) — FAIL {repr(e)[:50]}")
    return pd.DataFrame(out).dropna(how='all')


def zlast(s, win=252):
    return (s - s.rolling(win).mean()) / s.rolling(win).std()


def main():
    print("Pulling cross-asset data (yfinance)…")
    df = pull()
    spy, vix = df['SPY'], df['VIX']
    f = pd.DataFrame(index=df.index)
    # --- liquidity-stress features (higher = more stress) ---
    if 'DXY' in df:  f['dxy_5d'] = df['DXY'].pct_change(5) * 100              # USD up = tightening
    if {'HYG', 'LQD'} <= set(df): f['credit'] = -(df['HYG'] / df['LQD']).pct_change(5) * 100  # HY underperform IG
    if {'HYG', 'TLT'} <= set(df): f['hyg_tlt'] = -(df['HYG'] / df['TLT']).pct_change(5) * 100  # credit vs duration
    if 'MOVE' in df: f['move'] = zlast(df['MOVE'])                            # bond vol level
    elif 'TLT' in df: f['move'] = zlast(np.log(df['TLT']/df['TLT'].shift(1)).rolling(20).std())  # proxy
    if 'TNX' in df:  f['rates_5d'] = df['TNX'].diff(5)                        # yields rising fast
    if 'GLD' in df:  f['gold_5d'] = df['GLD'].pct_change(5) * 100            # safe-haven bid
    # composite = mean z of all stress features
    fz = f.apply(zlast)
    f['LIQ_STRESS'] = fz.mean(axis=1)
    # forward targets
    fwd_spy5 = spy.shift(-5) / spy - 1
    fwd_spy10 = spy.shift(-10) / spy - 1
    fwd_vix10 = vix.shift(-10) - vix

    print("\n=== Does each liquidity signal LEAD equity stress? (corr with forward moves) ===")
    print(f"  {'signal':<12}{'→fwd5d SPY':>12}{'→fwd10d SPY':>13}{'→fwd10d VIX':>13}")
    for c in [x for x in f.columns]:
        s = f[c]
        print(f"  {c:<12}{s.corr(fwd_spy5):>+12.2f}{s.corr(fwd_spy10):>+13.2f}{s.corr(fwd_vix10):>+13.2f}")

    print("\n=== Composite LIQ_STRESS quartiles → forward SPY 10d (does high stress = SPY falls?) ===")
    s = pd.DataFrame({'liq': f['LIQ_STRESS'], 'fwd': fwd_spy10, 'fvix': fwd_vix10}).dropna()
    qs = s['liq'].quantile([0, .25, .5, .75, 1.0]).values
    for i in range(4):
        lo, hi = qs[i], qs[i+1]
        x = s[(s['liq'] >= lo) & (s['liq'] <= hi)] if i == 3 else s[(s['liq'] >= lo) & (s['liq'] < hi)]
        print(f"    LIQ_STRESS [{lo:>+5.2f},{hi:>+5.2f}] n={len(x):>4}  fwd-10d SPY {x['fwd'].mean():>+6.2%}  "
              f"down% {(x['fwd']<0).mean()*100:>3.0f}%  fwd-VIX {x['fvix'].mean():>+5.1f}")

    print("\n=== CURRENT reading (latest) ===")
    last = f.iloc[-1]
    print(f"  date {f.index[-1].date()} | SPY {spy.iloc[-1]:.2f} | VIX {vix.iloc[-1]:.1f}")
    for c in f.columns:
        v = last[c]
        print(f"    {c:<12} {v:+.2f}" + ("  <-- composite (>+0.5 = elevated stress)" if c == 'LIQ_STRESS' else ""))
    # was it elevated this week / around 06-22?
    recent = f['LIQ_STRESS'].dropna().tail(8)
    print("\n  LIQ_STRESS last 8 days:")
    for d, v in recent.items():
        print(f"    {d.date()}  {v:+.2f}")
    print("DONE", flush=True)


if __name__ == '__main__':
    main()
