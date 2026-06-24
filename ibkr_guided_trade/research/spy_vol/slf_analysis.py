"""SLF (Sun Life) bearish setup: technicals + liquidity + options OI + GEX (dealer gamma).
GEX>0 = dealers long gamma (pin/dampen); GEX<0 = short gamma (moves amplify — selloffs accelerate).

  venv/bin/python research/spy_vol/slf_analysis.py [TICKER]
"""
import sys
import math
import numpy as np
import pandas as pd
import yfinance as yf

TICK = sys.argv[1] if len(sys.argv) > 1 else 'SLF'


def Nd(x):
    return math.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def gamma(S, K, T, sig):
    if T <= 0 or sig <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (0.045 + sig * sig / 2) * T) / (sig * math.sqrt(T))
    return Nd(d1) / (S * sig * math.sqrt(T))


def main():
    t = yf.Ticker(TICK)
    px = t.history(period='2y')['Close']
    if len(px) < 60:
        print(f"  no price data for {TICK}"); return
    S = float(px.iloc[-1])
    vol = t.history(period='3mo')['Volume'].mean()
    sma = {n: px.rolling(n).mean().iloc[-1] for n in (20, 50, 200)}
    r = px.pct_change()
    rsi = 100 - 100 / (1 + (r.clip(lower=0).rolling(14).mean() / -r.clip(upper=0).rolling(14).mean()).iloc[-1])
    hi52 = px.rolling(252).max().iloc[-1]
    rv20 = r.rolling(20).std().iloc[-1] * math.sqrt(252)
    print(f"=== {TICK}  spot ${S:.2f} ===")
    print(f"  trend: 20d ${sma[20]:.2f} {'>' if S>sma[20] else '<'} | 50d ${sma[50]:.2f} {'>' if S>sma[50] else '<'} | 200d ${sma[200]:.2f} {'>' if S>sma[200] else '<'} spot")
    print(f"  RSI14 {rsi:.0f} | dist 52w-high {S/hi52-1:+.1%} | 20d mom {px.iloc[-1]/px.iloc[-21]-1:+.1%} | 60d {px.iloc[-1]/px.iloc[-61]-1:+.1%}")
    print(f"  realized vol(20d) {rv20:.0%} | avg vol {vol:,.0f}/day (~${vol*S/1e6:.0f}M) | "
          f"{'THIN' if vol*S<2e7 else 'ok'} liquidity")

    exps = t.options
    if not exps:
        print("\n  NO LISTED OPTIONS — can't do OI/GEX (trade via shares/CFD or skip)."); return
    print(f"\n  option expiries available: {len(exps)} (front {exps[:4]})")
    rows = []
    tot_gex = 0.0
    oi_by_strike = {}
    for e in exps[:5]:
        try:
            ch = t.option_chain(e)
        except Exception:
            continue
        T = max((pd.Timestamp(e) - pd.Timestamp.now()).days, 1) / 365
        for df, kind, sign in ((ch.calls, 'C', +1), (ch.puts, 'P', -1)):
            for _, o in df.iterrows():
                K = float(o['strike']); oi = float(o.get('openInterest') or 0); iv = float(o.get('impliedVolatility') or 0)
                if oi <= 0 or iv <= 0:
                    continue
                g = gamma(S, K, T, iv)
                gex = sign * g * oi * 100 * S * S * 0.01   # $ gamma per 1% move; dealers long calls / short puts
                tot_gex += gex
                oi_by_strike.setdefault(round(K, 1), [0, 0])[0 if kind == 'C' else 1] += oi
                rows.append((e, kind, K, oi, iv, gex))
    if not rows:
        print("  options listed but OI/IV empty (illiquid chain)."); return
    print(f"\n  TOTAL GEX ≈ {tot_gex/1e6:+,.1f} $M/1% → dealers {'LONG gamma (pin/dampen moves)' if tot_gex>0 else 'SHORT gamma (moves AMPLIFY — bearish accelerant)'}")
    # biggest OI walls
    walls = sorted(oi_by_strike.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
    print("  biggest OI strikes (call_OI / put_OI):")
    for K, (c, p) in sorted(walls):
        rel = (K / S - 1) * 100
        print(f"    ${K:<6} ({rel:+4.0f}%)  C {int(c):>6} / P {int(p):>6}  {'<-- put wall (support/magnet)' if p>c*1.5 else '<-- call wall (resistance/cap)' if c>p*1.5 else ''}")
    # gamma-flip: cumulative GEX vs strike (rough)
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
