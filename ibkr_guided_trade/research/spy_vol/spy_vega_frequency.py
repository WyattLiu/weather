"""How REPEATABLE is the low-VIX vega-scrape? Frequency + year-by-year, non-overlapping.

One position at a time (45 DTE straddle): enter on the first eligible day (VIX<thr & 10d-std<1.2),
hold to exit (first-of +30%/-40%/VIX+3/30d), then look for the next entry. Counts how often the
setup actually fires per year — the binding constraint on the strategy is regime frequency, not edge.

  venv/bin/python research/spy_vol/spy_vega_frequency.py
"""
import os
import sys
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
from spy_vega_study import _conn, pick_entry, run_trade

SPY_CSV = os.path.join(THIS, 'cache', 'spy_vix_daily.csv')


def campaign(vthr, spv, vix, v10, vix_path, cur):
    elig = (vix < vthr) & (v10 < 1.2)
    trades = []; open_until = None
    for d_ts, row in spv.iterrows():
        if open_until and d_ts <= open_until:
            continue
        if not elig.loc[d_ts]:
            continue
        pe = pick_entry(cur, d_ts.date(), float(row['SPY']), 40, 50)
        if not pe:
            continue
        tr = run_trade(cur, d_ts.date(), float(row['SPY']), pe[0], pe[1], 30, 0.30, 0.40, 3.0,
                       float(row['VIX']), vix_path)
        if not tr:
            continue
        open_until = d_ts + pd.Timedelta(days=tr['held'])
        trades.append({'year': d_ts.year, 'ret': tr['ret'], 'held': tr['held']})
    return pd.DataFrame(trades)


def main():
    spv = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True); spv.index = spv.index.normalize()
    vix = spv['VIX']; v10 = vix.rolling(10).std()
    vix_path = {pd.Timestamp(k).date(): v for k, v in zip(spv.index, spv['VIX'])}
    span_yrs = (spv.index[-1] - spv.index[0]).days / 365.25
    conn = _conn(); cur = conn.cursor()
    for vthr in (14, 15):
        df = campaign(vthr, spv, vix, v10, vix_path, cur)
        n = len(df)
        print(f"\n=== VIX<{vthr} & consolidated — one-at-a-time ({span_yrs:.1f} yrs) ===")
        if n:
            util = df['held'].sum() / (span_yrs * 365) * 100
            print(f"  {n} trades = {n/span_yrs:.1f}/yr | win {(df['ret']>0).mean()*100:.0f}% | "
                  f"avg {df['ret'].mean():+.1%} | capital deployed ~{util:.0f}% of the time")
            for y, r in df.groupby('year')['ret'].agg(['count', 'mean', 'sum']).iterrows():
                print(f"    {y}: {int(r['count'])} trades, avg {r['mean']:+.1%}, summed {r['sum']:+.1%}")
            for y in sorted(set(spv.index.year)):
                if y not in df['year'].values:
                    print(f"    {y}: 0 trades (no clean setup)")
    conn.close()
    print("\nDONE", flush=True)


if __name__ == '__main__':
    main()
