"""Historical dealer GAMMA EXPOSURE (GEX) from real OI — for regime/decision signals.

GEX(day) = Σ_strikes  γ(K) · OI(K) · 100 · spot² · 0.01 · sign
  sign: +1 for calls, −1 for puts (naive dealer-long-calls / short-puts convention).
  γ from Black-Scholes using the ung_iv_surface IV; OI from ung_options_oi (real).

Interpretation:
  • net GEX > 0  → dealers LONG gamma → they fade moves → vol SUPPRESSED, pinning.
  • net GEX < 0  → dealers SHORT gamma → they chase moves → vol AMPLIFIED, swing-prone.
  • gamma-flip = spot level where net GEX crosses zero.

This is the decision/regime layer that ties to swing-day risk: we VALIDATE that
negative-GEX days actually realize bigger moves, then use it to size down / hedge into
negative-GEX regimes and sell premium into positive-GEX (pinning) regimes.
"""
import os
import math
import psycopg2
import numpy as np
import pandas as pd

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
OUT = os.path.join(CACHE, 'ung_gex_history.csv')


def _gamma(S, K, T, sigma):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) / (S * sigma * math.sqrt(T))


def compute():
    conn = psycopg2.connect(**DB)
    # join daily OI (raw strike) to the IV surface (strike_real = raw) for IV + spot
    q = """
      SELECT o.trade_date, o.option_right, o.open_interest,
             s.strike_real, s.spot_real, s.iv, (s.expiration - s.date) dte
      FROM ung_options_oi o
      JOIN ung_iv_surface s
        ON o.trade_date = s.date AND o.expiration = s.expiration
       AND o.option_right = s.option_right
       AND abs(o.strike - s.strike_real) < 0.001
      WHERE s.iv > 0 AND (s.expiration - s.date) BETWEEN 1 AND 90
    """
    df = pd.read_sql(q, conn); conn.close()
    if not len(df):
        print("no OI×IV join rows yet"); return None
    df['T'] = df['dte'] / 365.0
    df['gamma'] = [
        _gamma(float(s), float(k), float(t), float(iv))
        for s, k, t, iv in zip(df['spot_real'], df['strike_real'], df['T'], df['iv'])]
    sign = np.where(df['option_right'] == 'C', 1.0, -1.0)
    df['gex'] = (df['gamma'] * df['open_interest'] * 100 *
                 df['spot_real'].astype(float) ** 2 * 0.01 * sign)
    g = df.groupby('trade_date').agg(net_gex=('gex', 'sum'),
                                     spot=('spot_real', 'first'),
                                     total_oi=('open_interest', 'sum')).reset_index()
    g['net_gex_mm'] = (g['net_gex'] / 1e6).round(2)     # $mm per 1% move
    g['regime'] = np.where(g['net_gex'] < 0, 'NEG_GEX_swing', 'POS_GEX_pin')
    g.to_csv(OUT, index=False)
    return g


def validate(g):
    """Do NEGATIVE-GEX days actually realize bigger UNG moves? (the whole premise)"""
    u = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'), index_col=0,
                    parse_dates=True)['UNG'].dropna()
    u.index = u.index.normalize()                    # strip intraday time for date join
    nxt = (u.shift(-1) / u - 1).abs() * 100          # next-day |move|
    gg = g.copy(); gg['trade_date'] = pd.to_datetime(gg['trade_date']).dt.normalize()
    gg = gg.set_index('trade_date').join(nxt.rename('next_move')).dropna(subset=['next_move'])
    neg = gg[gg['net_gex'] < 0]['next_move']
    pos = gg[gg['net_gex'] >= 0]['next_move']
    print(f"\nGEX history: {len(g)} days  ({g['trade_date'].min()} → {g['trade_date'].max()})")
    print(f"  net GEX range: {g['net_gex_mm'].min():.1f} … {g['net_gex_mm'].max():.1f} $mm/1%")
    print(f"  NEG-GEX days: {100*(g['net_gex']<0).mean():.0f}%")
    print("\nVALIDATION — next-day |move|:")
    print(f"  after NEG-GEX (swing): mean {neg.mean():.2f}%  p90 {neg.quantile(.9):.2f}%  (n={len(neg)})")
    print(f"  after POS-GEX (pin):   mean {pos.mean():.2f}%  p90 {pos.quantile(.9):.2f}%  (n={len(pos)})")
    lift = neg.mean() / max(pos.mean(), 1e-9) - 1
    print(f"  → NEG-GEX days move {100*lift:+.0f}% more than POS-GEX "
          f"{'(premise CONFIRMED)' if lift > 0.05 else '(premise weak — reconsider)'}")


if __name__ == '__main__':
    g = compute()
    if g is not None:
        validate(g)
