"""Intraday microstructure miner — learns WHEN and WHAT DAY the strategy can actually
execute well, from the real intraday bid/ask path (PG ung_options_history).

Answers operational questions the daily engine is blind to:
  - Which HOUR of the session has the tightest spreads (best fills)?
  - Which DAY OF WEEK is best to roll/open (Wed vs Thu vs Fri)? Nat-gas has the EIA
    storage print Thursday 10:30 ET — does Thursday widen/tighten around it?
  - The (day x hour) fill-quality grid → concrete "roll Thursday 13:00" guidance.

This is the foundation for an execution-aware engine: time entries/rolls into the
tight-spread windows the data identifies, avoid the wide ones.
"""
import psycopg2
import pandas as pd

DB = {'host': '192.168.1.172', 'port': 5432, 'database': 'market_scanner',
      'user': 'postgres', 'password': 'shinobi2025'}
DOW = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}


def _load():
    """Pull two-sided near-money quotes with relative spread + dow + hour."""
    conn = psycopg2.connect(**DB)
    q = """
      WITH q AS (
        SELECT trade_date, bar_time, expiration, strike, option_right, underlying_price,
               MAX(CASE WHEN data_type='BID' THEN close END) AS bid,
               MAX(CASE WHEN data_type='ASK' THEN close END) AS ask
        FROM ung_options_history
        GROUP BY trade_date, bar_time, expiration, strike, option_right, underlying_price)
      SELECT trade_date, bar_time, expiration, strike, option_right, underlying_price, bid, ask
      FROM q WHERE bid > 0 AND ask > bid"""
    df = pd.read_sql(q, conn)
    conn.close()
    df['bar_time'] = pd.to_datetime(df['bar_time'])
    df['mid'] = (df['bid'] + df['ask']) / 2
    df['rel_spread'] = (df['ask'] - df['bid']) / df['mid'] * 100
    df['dow'] = df['bar_time'].dt.dayofweek
    df['hour'] = df['bar_time'].dt.hour
    df['dte'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(df['trade_date'])).dt.days
    # near-money only (the orders that act)
    df['moneyness'] = (df['strike'] - df['underlying_price']).abs() / df['underlying_price']
    return df[df['moneyness'] <= 0.10]


def main():
    df = _load()
    print(f"Intraday microstructure — {len(df):,} near-money two-sided quotes, "
          f"{df['trade_date'].nunique()} days ({df['trade_date'].min()}→{df['trade_date'].max()})\n")

    print("=== SPREAD by HOUR of session (tighter = better fills) ===")
    byh = df.groupby('hour')['rel_spread'].agg(['median', 'count'])
    for h, r in byh.iterrows():
        bar = '█' * int(r['median'] / 2)
        print(f"  {h:02d}:00  {r['median']:5.1f}%  {bar}")

    print("\n=== SPREAD by DAY OF WEEK (which day to roll/open) ===")
    byd = df.groupby('dow')['rel_spread'].agg(['median', 'count'])
    best = byd['median'].idxmin()
    for d, r in byd.iterrows():
        tag = '  ← tightest' if d == best else ('  (EIA storage print 10:30)' if d == 3 else '')
        print(f"  {DOW.get(d, d):3}  {r['median']:5.1f}%  (n={int(r['count']):,}){tag}")

    print("\n=== FILL-QUALITY GRID: median spread by DAY x HOUR (find the sweet spot) ===")
    grid = df.pivot_table('rel_spread', 'dow', 'hour', 'median')
    grid.index = [DOW.get(i, i) for i in grid.index]
    pd.set_option('display.width', 160, 'display.float_format', lambda x: f'{x:4.0f}')
    print(grid.round(0).to_string())
    # the single best (day,hour) cell
    flat = df.groupby(['dow', 'hour'])['rel_spread'].median()
    bd, bh = flat.idxmin()
    print(f"\n  TIGHTEST window: {DOW.get(bd)} {bh:02d}:00  ({flat.min():.1f}% spread) "
          f"→ work rolls/opens here.  WIDEST: avoid {DOW.get(flat.idxmax()[0])} {flat.idxmax()[1]:02d}:00 ({flat.max():.1f}%).")

    # Thursday EIA-print microstructure: spread before vs after 10:30
    thu = df[df['dow'] == 3]
    if len(thu):
        pre = thu[thu['hour'] < 11]['rel_spread'].median()
        post = thu[thu['hour'] >= 11]['rel_spread'].median()
        print(f"\n  THURSDAY EIA print: pre-11:00 spread {pre:.1f}% vs post {post:.1f}% "
              f"→ {'wait until after the print' if post < pre else 'tighter before the print'}")


if __name__ == '__main__':
    main()
