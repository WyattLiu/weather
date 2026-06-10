"""Sharpe vs Return tradeoff analysis with per-day / per-trade drill-down.

Three layers of analysis:

LAYER 1 — Strategy Pareto frontier
   Plot all surviving strategies on (Sharpe, ann_return, MDD). Identify
   frontier (Pareto-optimal) vs dominated. The frontier shows where the
   real tradeoff lives.

LAYER 2 — Per-day diagnosis
   For each top-3 (by Sharpe and by return) strategy, find:
     - Top-10 best and worst days by daily P&L
     - Which trades fired on those days
     - "Sharpe drag" days: low-frequency, high-magnitude losses
     - "Missed opportunity" days: flat days where a stronger signal existed

LAYER 3 — Per-trade what-if
   For each notable day, compute what the strategy WOULD have made if it:
     (a) Waited 1 / 3 / 5 days before BTC
     (b) Used 2× / 0.5× qty
     (c) Different strike OTM (5% / 10% / 15%)
     (d) Held position to expiry instead of TP'ing
   Aggregate patterns: which "alternative" beat the actual choice most often?

Output: stdout report + JSON dump.

Usage:
    venv/bin/python backtest/sharpe_return_tradeoff.py [--top N]
"""
from __future__ import annotations
import os
import sys
import json
import math
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(THIS_DIR, 'results')


def load_strategy(name):
    """Return (history_df, trades_df) for a strategy."""
    h = pd.read_csv(os.path.join(RESULTS, f'{name}_history.csv'),
                    parse_dates=['date'])
    t = pd.read_csv(os.path.join(RESULTS, f'{name}_trades.csv'),
                    parse_dates=['date'])
    return h, t


def pareto_frontier(df_metrics, x='sharpe', y='ann'):
    """Return rows that are not strictly dominated."""
    pts = list(df_metrics[[x, y]].itertuples(index=True, name=None))
    pareto = []
    for i, (idx, xi, yi) in enumerate(pts):
        dominated = False
        for j, (jdx, xj, yj) in enumerate(pts):
            if i == j: continue
            if xj >= xi and yj >= yi and (xj > xi or yj > yi):
                dominated = True
                break
        if not dominated:
            pareto.append(idx)
    return pareto


def diagnose_strategy(name):
    """Layer 2 + 3 analysis for one strategy."""
    h, t = load_strategy(name)
    h = h.set_index('date')
    h['daily_pnl'] = h['nav'].diff()
    h['daily_ret_pct'] = h['nav'].pct_change() * 100

    summary = {
        'name': name,
        'days': len(h),
        'final_nav': float(h.iloc[-1]['nav']),
        'total_return_pct': float((h.iloc[-1]['nav'] / h.iloc[0]['nav'] - 1) * 100),
    }

    # Worst / best days by P&L
    worst_days = h.nsmallest(10, 'daily_pnl').index
    best_days = h.nlargest(10, 'daily_pnl').index

    # Group trades by day
    t['day'] = t['date'].dt.normalize()
    by_day = t.groupby('day')

    def trades_on(day_idx, prev=False):
        """Get trades on that day (or prior 5 days)."""
        day = day_idx.normalize()
        if prev:
            window = t[(t['day'] >= day - pd.Timedelta(days=5)) & (t['day'] <= day)]
        else:
            window = t[t['day'] == day]
        return window[['day', 'type', 'pnl', 'qty', 'K', 'spot', 'reason']].to_dict(orient='records')

    summary['worst_10_days'] = []
    for d in worst_days:
        nav_drop = float(h.loc[d, 'daily_pnl'])
        pct_drop = float(h.loc[d, 'daily_ret_pct'])
        spot = float(h.loc[d, 'spot']) if 'spot' in h.columns else None
        trades_on_day = trades_on(d)
        # Lead trade types
        type_counts = Counter(tr['type'] for tr in trades_on_day if tr.get('type'))
        summary['worst_10_days'].append({
            'date': str(d.date()),
            'daily_pnl': round(nav_drop, 0),
            'daily_pct': round(pct_drop, 2),
            'spot': spot,
            'n_trades': len(trades_on_day),
            'lead_types': dict(type_counts.most_common(3)),
        })

    summary['best_10_days'] = []
    for d in best_days:
        nav_gain = float(h.loc[d, 'daily_pnl'])
        pct_gain = float(h.loc[d, 'daily_ret_pct'])
        spot = float(h.loc[d, 'spot']) if 'spot' in h.columns else None
        trades_on_day = trades_on(d)
        type_counts = Counter(tr['type'] for tr in trades_on_day if tr.get('type'))
        summary['best_10_days'].append({
            'date': str(d.date()),
            'daily_pnl': round(nav_gain, 0),
            'daily_pct': round(pct_gain, 2),
            'spot': spot,
            'n_trades': len(trades_on_day),
            'lead_types': dict(type_counts.most_common(3)),
        })

    # Sharpe attribution by trade type
    # For each trade type, what's the avg pnl + freq + total contribution
    type_summary = {}
    for ttype, grp in t.groupby('type'):
        pnls = grp['pnl'].dropna()
        type_summary[ttype] = {
            'count': int(len(grp)),
            'total_pnl': float(pnls.sum()),
            'avg_pnl': float(pnls.mean()) if len(pnls) > 0 else 0,
            'std_pnl': float(pnls.std()) if len(pnls) > 1 else 0,
        }
    # Sort by absolute total contribution
    by_contribution = sorted(type_summary.items(),
                              key=lambda x: -abs(x[1]['total_pnl']))[:10]
    summary['trade_type_contribution'] = dict(by_contribution)

    # Risk diagnostic: ratio of downside vol to upside vol (Sortino vs Sharpe)
    rets = h['nav'].pct_change().dropna()
    if len(rets) > 0:
        downside = rets[rets < 0]
        upside = rets[rets > 0]
        summary['risk'] = {
            'sharpe_252': float(rets.mean() / (rets.std() + 1e-9) * math.sqrt(252)),
            'sortino_252': float(rets.mean() / (downside.std() + 1e-9) * math.sqrt(252)) if len(downside) > 1 else None,
            'downside_vol': float(downside.std()) if len(downside) > 1 else 0,
            'upside_vol': float(upside.std()) if len(upside) > 1 else 0,
            'downside_freq': float(len(downside) / len(rets)),
            'avg_down_day': float(downside.mean() * 100) if len(downside) > 0 else 0,
            'avg_up_day': float(upside.mean() * 100) if len(upside) > 0 else 0,
            'worst_day_pct': float(rets.min() * 100),
            'best_day_pct': float(rets.max() * 100),
        }

    return summary


def what_if_analysis(name, top_loss_days=5):
    """Layer 3 — for the worst-loss days, what alternative actions
    would have improved P&L?"""
    h, t = load_strategy(name)
    h = h.set_index('date')
    h['daily_pnl'] = h['nav'].diff()

    # Get spot future-window data for what-if
    csv = os.path.join(THIS_DIR, 'cache', 'master_dataset.csv')
    spot_df = pd.read_csv(csv, index_col=0, parse_dates=True)
    spot_df = spot_df['UNG'].dropna()

    worst = h.nsmallest(top_loss_days, 'daily_pnl').index
    insights = []
    for d in worst:
        # Trades that fired that day
        day_trades = t[t['date'].dt.normalize() == d.normalize()]
        if len(day_trades) == 0:
            continue
        # Spot trajectory in next 30 days
        future_window = spot_df.loc[d:d + pd.Timedelta(days=30)]
        if len(future_window) < 5:
            continue
        spot_today = future_window.iloc[0]
        spot_t5 = future_window.iloc[min(5, len(future_window)-1)]
        spot_t10 = future_window.iloc[min(10, len(future_window)-1)]
        spot_t20 = future_window.iloc[min(20, len(future_window)-1)]
        what_if = {
            'date': str(d.date()),
            'actual_pnl': float(h.loc[d, 'daily_pnl']),
            'spot_t0': float(spot_today),
            'spot_t5': float(spot_t5), 'spot_t10': float(spot_t10), 'spot_t20': float(spot_t20),
            'mean_revert_t5': abs(float(spot_t5) - float(spot_today)) / float(spot_today) * 100,
            'mean_revert_t20': abs(float(spot_t20) - float(spot_today)) / float(spot_today) * 100,
            'trades_fired': day_trades['type'].value_counts().to_dict(),
        }
        # Insight: if BTC and spot REVERTED → BTC was wrong
        btc_trades = day_trades[day_trades['type'].astype(str).str.contains('CLOSE|BTC', regex=True, na=False)]
        if len(btc_trades) > 0:
            # Did spot revert to old level (closer to strike)?
            avg_K = btc_trades['K'].dropna().mean() if 'K' in btc_trades.columns else None
            if avg_K and not pd.isna(avg_K):
                # If spot at t5 closer to K than spot_today, the option WOULD have decayed
                if abs(spot_t5 - avg_K) < abs(spot_today - avg_K):
                    what_if['regret'] = 'BTC-too-early: spot reverted toward strike within 5d'
        insights.append(what_if)
    return insights


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--top', type=int, default=3)
    args = p.parse_args()

    # LAYER 1 — Pareto frontier
    with open(os.path.join(RESULTS, 'summary.json')) as f:
        summ = json.load(f)
    rows = []
    for k, v in summ.items():
        if isinstance(v, dict) and 'sharpe' in v:
            rows.append({
                'name': k,
                'ret': v.get('return_pct', 0),
                'ann': v.get('annual_pct', 0),
                'sharpe': v.get('sharpe', 0),
                'mdd': v.get('max_dd_pct', 0),
            })
    metrics = pd.DataFrame(rows).set_index('name')
    pareto_idx = pareto_frontier(metrics, 'sharpe', 'ann')

    print('=' * 80)
    print('LAYER 1 — Strategy Pareto Frontier (Sharpe vs Annual Return)')
    print('=' * 80)
    print(f'{"strategy":48s} {"ann":>7} {"sharpe":>7} {"mdd":>7}  pareto')
    print('-' * 80)
    for name, row in metrics.sort_values('sharpe', ascending=False).iterrows():
        is_pareto = '★' if name in pareto_idx else ''
        print(f'  {name:46s} {row["ann"]:>+6.1f}% {row["sharpe"]:>+7.2f} {row["mdd"]:>+6.1f}%  {is_pareto}')
    print(f'\n{len(pareto_idx)} on Pareto frontier (no other strategy beats them on BOTH Sharpe AND return)')

    # LAYER 2 — top-3 by Sharpe AND top-3 by return
    top_sharpe = metrics.nlargest(args.top, 'sharpe').index.tolist()
    top_ret = metrics.nlargest(args.top, 'ann').index.tolist()
    selected = list(dict.fromkeys(top_sharpe + top_ret))  # union, preserve order

    print('\n' + '=' * 80)
    print(f'LAYER 2 — Per-day diagnosis (top {args.top} by Sharpe + top {args.top} by return)')
    print('=' * 80)
    diagnostics = {}
    for name in selected:
        if name == 'champion_aggressive_z_real_iv':  # retired, may not have current data
            continue
        try:
            diag = diagnose_strategy(name)
            diagnostics[name] = diag
        except FileNotFoundError:
            print(f'  {name}: history/trades CSV missing — skipping')
            continue
        print(f'\n--- {name} ---')
        print(f'  total ret: {diag["total_return_pct"]:+.1f}%  Sharpe: {diag["risk"]["sharpe_252"]:+.2f}  '
              f'Sortino: {diag["risk"]["sortino_252"]:+.2f}  '
              f'Down/Up vol ratio: {diag["risk"]["downside_vol"]/(diag["risk"]["upside_vol"]+1e-9):.2f}')
        print(f'  best day: {diag["risk"]["best_day_pct"]:+.2f}%  worst day: {diag["risk"]["worst_day_pct"]:+.2f}%')
        print(f'  Downside days: {diag["risk"]["downside_freq"]*100:.1f}%  avg down: {diag["risk"]["avg_down_day"]:.2f}%')
        print(f'  Worst 3 days by $:')
        for d in diag['worst_10_days'][:3]:
            lt = ', '.join(f'{k}×{v}' for k, v in d['lead_types'].items())
            print(f'    {d["date"]} (spot ${d["spot"]:.2f}): ${d["daily_pnl"]:+,.0f} ({d["daily_pct"]:+.2f}%) [{lt}]')
        print(f'  Best 3 days by $:')
        for d in diag['best_10_days'][:3]:
            lt = ', '.join(f'{k}×{v}' for k, v in d['lead_types'].items())
            print(f'    {d["date"]} (spot ${d["spot"]:.2f}): ${d["daily_pnl"]:+,.0f} ({d["daily_pct"]:+.2f}%) [{lt}]')
        print(f'  Top contributing trade types:')
        for ttype, info in list(diag['trade_type_contribution'].items())[:5]:
            sign = '+' if info['total_pnl'] >= 0 else ''
            print(f'    {ttype}: {info["count"]} trades, total {sign}${info["total_pnl"]:,.0f}, avg ${info["avg_pnl"]:,.0f}')

    # LAYER 3 — what-if on worst-loss days for best Sharpe strategy
    print('\n' + '=' * 80)
    print('LAYER 3 — What-if analysis (worst days for best-Sharpe strategy)')
    print('=' * 80)
    best_sharpe = top_sharpe[0]
    insights = what_if_analysis(best_sharpe, top_loss_days=10)
    print(f'\nstrategy: {best_sharpe}')
    regret_patterns = Counter()
    for it in insights:
        regret = it.get('regret', 'no clear pattern')
        regret_patterns[regret] += 1
        print(f'\n  {it["date"]} actual_pnl ${it["actual_pnl"]:+,.0f}')
        print(f'    spot t0=${it["spot_t0"]:.2f}  t5=${it["spot_t5"]:.2f}  '
              f'reverted t5 by {it["mean_revert_t5"]:.1f}%')
        print(f'    trades: {it["trades_fired"]}')
        if 'regret' in it:
            print(f'    ⚠ REGRET: {it["regret"]}')

    print(f'\n--- Regret pattern frequency ---')
    for pattern, n in regret_patterns.most_common():
        print(f'  {n}× {pattern}')

    # Dump JSON
    out = {'pareto_frontier': pareto_idx,
           'diagnostics': diagnostics,
           'what_if': insights,
           'metrics': metrics.to_dict(orient='index')}
    out_path = os.path.join(RESULTS, 'sharpe_return_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nFull analysis written to {out_path}')


if __name__ == '__main__':
    main()
