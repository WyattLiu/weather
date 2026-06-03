"""Day-by-day strategy introspection.

Replays a single strategy and emits:
- Per-day NAV, dd, regime, share count, key Greeks
- Every trade with P&L attribution
- Top winning and losing days
- Worst drawdown episodes with decomposition
- Year-by-year breakdown

Usage:
  cd /home/wyatt/weather/ibkr_guided_trade
  ../venv/bin/python backtest/probe_strategy.py champion_target_25
  ../venv/bin/python backtest/probe_strategy.py champion_target_25 --start 2022-01 --end 2022-12
  ../venv/bin/python backtest/probe_strategy.py champion_target_25 --trades-csv /tmp/trades.csv
"""
import os
import sys
import argparse
import pandas as pd
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z, compute_historical_z, regime


def probe(strategy_name, start=None, end=None, trades_csv=None, daily_csv=None, top_n=15):
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
    df = pd.read_csv(os.path.join(CACHE_DIR, 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    if start: df = df.loc[start:]
    if end: df = df.loc[:end]

    if strategy_name not in STRATEGIES:
        print(f"Unknown strategy '{strategy_name}'. Available:")
        for n in sorted(STRATEGIES): print(f"  {n}")
        return

    print(f"=== PROBE: {strategy_name} ({df.index[0].date()} → {df.index[-1].date()}) ===\n")
    strat = STRATEGIES[strategy_name]
    hist, trades = run_strategy_simple(df, strat, 48000, 6200)
    hist = hist.set_index(pd.to_datetime(hist['date']))
    trades['date'] = pd.to_datetime(trades['date'])
    initial = 48000 + 6200 * df['UNG'].iloc[0]

    # Summary
    final = hist.iloc[-1]['nav']
    fret = (final/initial - 1) * 100
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    ann = (1+fret/100)**(1/yrs)*100 - 100
    rets = hist['nav'].pct_change().dropna()
    sh = rets.mean() / (rets.std()+1e-9) * math.sqrt(252)
    peak = hist['nav'].cummax()
    dd = (hist['nav'] - peak) / peak * 100
    mdd = dd.min()
    print(f"NAV:        ${initial:,.0f} → ${final:,.0f}  (+{fret:.1f}%)")
    print(f"Annualized: {ann:+.2f}% over {yrs:.2f} yrs")
    print(f"Sharpe:     {sh:+.3f}")
    print(f"Max DD:     {mdd:.1f}% (trough {dd.idxmin().date()})")
    print(f"Trades:     {len(trades)} events  Days: {len(hist)}")

    # Year-by-year
    print("\n=== YEAR-BY-YEAR ===")
    for yr in sorted(hist.index.year.unique()):
        ydf = hist[hist.index.year == yr]
        if len(ydf) > 1:
            ystart = ydf['nav'].iloc[0]; yend = ydf['nav'].iloc[-1]
            yret = (yend/ystart - 1) * 100
            yrets = ydf['nav'].pct_change().dropna()
            ysh = yrets.mean()/(yrets.std()+1e-9)*math.sqrt(252) if len(yrets) else 0
            ymdd = ((ydf['nav'] - ydf['nav'].cummax())/ydf['nav'].cummax()*100).min()
            print(f"  {yr}: ${yend-ystart:+,.0f} ({yret:+.1f}%) Sh={ysh:+.2f} MDD={ymdd:.1f}%  {len(ydf)}d")

    # Trade type breakdown
    print("\n=== TRADE TYPES (top 10 by count) ===")
    tt = trades.groupby('type').agg(count=('type','size'), pnl_sum=('pnl','sum'), pnl_mean=('pnl','mean'))
    tt = tt.sort_values('count', ascending=False).head(10)
    print(tt.to_string())

    # Top winning days (largest single-day NAV gains)
    daily_pnl = hist['nav'].diff().dropna()
    print(f"\n=== TOP {top_n} WIN DAYS ===")
    for d, pnl in daily_pnl.nlargest(top_n).items():
        nav = hist.loc[d,'nav']
        ung = df.loc[d, 'UNG'] if d in df.index else None
        # Trades that day
        td = trades[trades['date'] == d]
        types = td['type'].value_counts().to_dict()
        z = compute_historical_z(df.loc[d], use_surprise=True) if d in df.index else 0
        r = regime(z)
        print(f"  {d.date()}  +${pnl:,.0f}  NAV=${nav:,.0f}  UNG=${ung:.2f}  {r:<14}  {len(td)} trades {dict(list(types.items())[:3])}")

    print(f"\n=== TOP {top_n} LOSS DAYS ===")
    for d, pnl in daily_pnl.nsmallest(top_n).items():
        nav = hist.loc[d,'nav']
        ung = df.loc[d, 'UNG'] if d in df.index else None
        td = trades[trades['date'] == d]
        types = td['type'].value_counts().to_dict()
        z = compute_historical_z(df.loc[d], use_surprise=True) if d in df.index else 0
        r = regime(z)
        print(f"  {d.date()}  ${pnl:,.0f}  NAV=${nav:,.0f}  UNG=${ung:.2f}  {r:<14}  {len(td)} trades {dict(list(types.items())[:3])}")

    # Drawdown episodes (>3% DD)
    print("\n=== DRAWDOWN EPISODES (>3%) ===")
    in_dd = dd < -3
    episodes = []
    s_idx = None
    for d, flag in in_dd.items():
        if flag and s_idx is None: s_idx = d
        elif not flag and s_idx is not None:
            sub = dd.loc[s_idx:d]
            episodes.append((s_idx, sub.idxmin(), d, sub.min(), (d-s_idx).days))
            s_idx = None
    if s_idx is not None:
        sub = dd.loc[s_idx:]
        episodes.append((s_idx, sub.idxmin(), hist.index[-1], sub.min(), (hist.index[-1]-s_idx).days))
    for e in sorted(episodes, key=lambda x: x[3])[:5]:
        print(f"  start={e[0].date()}  trough={e[1].date()}  end={e[2].date()}  DD={e[3]:.1f}%  duration={e[4]}d")

    # Top P&L trades
    if 'pnl' in trades.columns:
        valid = trades[trades['pnl'].abs() > 0.01]
        if len(valid) > 0:
            print(f"\n=== TOP {top_n} P&L SINGLE TRADES ===")
            for _, t in valid.nlargest(top_n, 'pnl').iterrows():
                print(f"  {pd.to_datetime(t['date']).date()}  {t['type']:<24} pnl=${t['pnl']:+,.0f}  qty={t.get('qty','?')}")
            print(f"\n=== BOTTOM {top_n} P&L SINGLE TRADES ===")
            for _, t in valid.nsmallest(top_n, 'pnl').iterrows():
                print(f"  {pd.to_datetime(t['date']).date()}  {t['type']:<24} pnl=${t['pnl']:+,.0f}  qty={t.get('qty','?')}")

    # Optional outputs
    if trades_csv:
        trades.to_csv(trades_csv, index=False)
        print(f"\nTrades written: {trades_csv}")
    if daily_csv:
        out = hist.copy()
        out['daily_pnl'] = daily_pnl
        out['drawdown_pct'] = dd
        out.to_csv(daily_csv)
        print(f"Daily series written: {daily_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('strategy', help='strategy name (see STRATEGIES)')
    p.add_argument('--start', default=None)
    p.add_argument('--end', default=None)
    p.add_argument('--trades-csv', default=None)
    p.add_argument('--daily-csv', default=None)
    p.add_argument('--top', type=int, default=15)
    args = p.parse_args()
    probe(args.strategy, args.start, args.end, args.trades_csv, args.daily_csv, args.top)
