"""Day-by-day strategy evolution analyzer.

For a given strategy, walks through every trading day and identifies:
1. What the strategy ACTUALLY did
2. What it COULD HAVE done (counterfactuals)
3. Missed opportunities (no-trade days with rich premium)
4. Bad-decision days (trades that lost vs alternatives)
5. Pattern clusters (what types of days hurt us systematically)

Outputs an evolution report with specific suggested mechanic changes.

Usage:
    venv/bin/python backtest/evolve_strategy.py champion_premium_harvest
    venv/bin/python backtest/evolve_strategy.py champion_premium_harvest --top 20 --output /tmp/evolve.json
"""
import os
import sys
import math
import json
import argparse
import pandas as pd
from collections import Counter, defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from replay_engine import (
    run_strategy_simple, STRATEGIES, precompute_factor_z,
    compute_historical_z, regime, detect_anomaly, falling_knife,
    detect_grind_down,
)


def analyze(strategy_name, top_n=20):
    df = pd.read_csv(os.path.join(THIS_DIR, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    strat = STRATEGIES[strategy_name]
    hist, trades = run_strategy_simple(df, strat, 100000, 0)
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.set_index('date')
    trades['date'] = pd.to_datetime(trades['date'])

    # Daily P&L
    hist['daily_pnl'] = hist['nav'].diff()
    hist['daily_pct'] = hist['nav'].pct_change() * 100

    # Annotate each day with regime, anomaly, etc.
    enriched = []
    for d in hist.index:
        if d not in df.index:
            continue
        row = df.loc[d]
        z = compute_historical_z(row, use_surprise=True)
        anom = detect_anomaly(row)
        knife = falling_knife(row)
        grind = detect_grind_down(row)
        day_trades = trades[trades['date'] == d]
        enriched.append({
            'date': d, 'nav': hist.loc[d, 'nav'],
            'daily_pnl': hist.loc[d, 'daily_pnl'], 'daily_pct': hist.loc[d, 'daily_pct'],
            'shares': int(hist.loc[d, 'shares']) if pd.notna(hist.loc[d, 'shares']) else 0,
            'ung': float(df.loc[d, 'UNG']),
            'z': z, 'regime': regime(z),
            'anomaly': anom, 'knife': knife, 'grind': grind,
            'rv30': float(df.loc[d, 'rv_30']) if 'rv_30' in df.columns and pd.notna(df.loc[d, 'rv_30']) else None,
            'n_trades': len(day_trades),
            'trade_types': Counter(day_trades['type'].tolist()) if len(day_trades) else {},
            'trade_pnl': float(day_trades['pnl'].sum()) if len(day_trades) else 0.0,
        })

    # WORST loss days
    worst = sorted(enriched, key=lambda x: x['daily_pnl'])[:top_n]
    # BEST gain days
    best = sorted(enriched, key=lambda x: -x['daily_pnl'])[:top_n]
    # NO-TRADE days with EXTREME conditions (potential missed opportunities)
    no_trade = [e for e in enriched if e['n_trades'] == 0]
    high_iv_missed = [e for e in no_trade if e['rv30'] and e['rv30'] > 0.70]
    extreme_z_missed = [e for e in no_trade if abs(e['z']) > 1.0]

    # Pattern clusters: what regimes drive losses?
    loss_by_regime = defaultdict(list)
    loss_by_anomaly = defaultdict(list)
    for e in enriched:
        if e['daily_pnl'] and e['daily_pnl'] < 0:
            loss_by_regime[e['regime']].append(e['daily_pnl'])
            loss_by_anomaly[e['anomaly']].append(e['daily_pnl'])

    gain_by_regime = defaultdict(list)
    for e in enriched:
        if e['daily_pnl'] and e['daily_pnl'] > 0:
            gain_by_regime[e['regime']].append(e['daily_pnl'])

    # Roll-up summaries
    regime_summary = {}
    for r in set(list(loss_by_regime.keys()) + list(gain_by_regime.keys())):
        losses = loss_by_regime.get(r, [])
        gains = gain_by_regime.get(r, [])
        regime_summary[r] = {
            'n_loss_days': len(losses), 'n_gain_days': len(gains),
            'avg_loss': sum(losses)/len(losses) if losses else 0,
            'avg_gain': sum(gains)/len(gains) if gains else 0,
            'sum_loss': sum(losses),
            'sum_gain': sum(gains),
            'net': sum(gains) + sum(losses),
        }

    # Suggested evolutions based on patterns
    suggestions = []
    # Pattern 1: NEUTRAL days with losses (z noise, sizing too aggressive?)
    if 'NEUTRAL' in regime_summary and regime_summary['NEUTRAL']['avg_loss'] < -1000:
        suggestions.append({
            'pattern': 'NEUTRAL_regime_losses',
            'evidence': f"NEUTRAL avg loss ${regime_summary['NEUTRAL']['avg_loss']:.0f}/loss-day, "
                        f"sum ${regime_summary['NEUTRAL']['sum_loss']:.0f} across {regime_summary['NEUTRAL']['n_loss_days']} days",
            'suggestion': 'Reduce put_qty during NEUTRAL z (less leverage in noise zone)',
        })
    # Pattern 2: ANOMALY losses despite anomaly_standdown availability
    anom_losses = sum(loss_by_anomaly.get('ANOMALY_DOWN', []))
    if abs(anom_losses) > 10000:
        suggestions.append({
            'pattern': 'ANOMALY_DOWN_losses',
            'evidence': f"ANOMALY_DOWN sum loss ${anom_losses:.0f} — anomaly_standdown not enabled or not catching",
            'suggestion': 'Test anomaly_standdown_DOWN_only flag (skip puts during down anomaly, allow during up)',
        })
    # Pattern 3: High-vol no-trade days
    if len(high_iv_missed) > 30:
        sum_pct = sum(abs(e['daily_pct']) for e in high_iv_missed) / max(len(high_iv_missed), 1)
        suggestions.append({
            'pattern': 'high_IV_no_trade',
            'evidence': f"{len(high_iv_missed)} days with rv30>0.70 and 0 trades fired (premium-rich days missed)",
            'suggestion': 'Add IV-percentile trigger: force-open puts when rv30 > 0.70 (currently waiting for entry_cadence)',
        })
    # Pattern 4: Big-loss days clustered in grind
    grind_loss_days = [e for e in worst if e['grind']]
    if len(grind_loss_days) > 3:
        suggestions.append({
            'pattern': 'grind_concentration',
            'evidence': f"{len(grind_loss_days)} of worst {top_n} loss days had grind_down=True",
            'suggestion': 'Add grind_down → close existing puts at TP threshold (don\'t wait for normal tp_50)',
        })
    # Pattern 5: Big winners on PUT_TP — can we accelerate?
    big_gain_put_tp = [e for e in best if any('PUT_TP' in str(t) for t in e['trade_types'])]
    if big_gain_put_tp:
        suggestions.append({
            'pattern': 'put_tp_concentration',
            'evidence': f"{len(big_gain_put_tp)}/{top_n} best days had PUT_TP fire",
            'suggestion': 'Lower tp_pct from 50→30 — exit earlier, redeploy capital faster',
        })

    return {
        'strategy': strategy_name,
        'total_days': len(enriched),
        'regime_summary': regime_summary,
        'worst_days': [{k: (str(v) if hasattr(v, 'isoformat') else v) for k, v in e.items() if k not in ('trade_types',)} | {'trade_types': dict(e['trade_types'])} for e in worst[:10]],
        'best_days': [{k: (str(v) if hasattr(v, 'isoformat') else v) for k, v in e.items() if k not in ('trade_types',)} | {'trade_types': dict(e['trade_types'])} for e in best[:10]],
        'high_iv_missed_count': len(high_iv_missed),
        'extreme_z_missed_count': len(extreme_z_missed),
        'evolution_suggestions': suggestions,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('strategy')
    p.add_argument('--top', type=int, default=20)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.strategy not in STRATEGIES:
        print(f'Unknown strategy: {args.strategy}')
        print(f'Available: {", ".join(sorted(STRATEGIES.keys()))}')
        sys.exit(1)

    result = analyze(args.strategy, args.top)

    print(f"\n=== Day-by-day evolution: {args.strategy} ===")
    print(f"Total days analyzed: {result['total_days']}")
    print(f"\nRegime P&L summary:")
    for r, s in sorted(result['regime_summary'].items()):
        print(f"  {r:<15} loss_days={s['n_loss_days']:>3} avg=${s['avg_loss']:>+8.0f}  "
              f"gain_days={s['n_gain_days']:>3} avg=${s['avg_gain']:>+8.0f}  "
              f"net=${s['net']:>+10.0f}")

    print(f"\nMissed opportunities:")
    print(f"  High-IV no-trade days: {result['high_iv_missed_count']}")
    print(f"  Extreme-z no-trade days: {result['extreme_z_missed_count']}")

    print(f"\nWorst {len(result['worst_days'])} loss days:")
    for e in result['worst_days']:
        tt = list(e['trade_types'].keys())[:3]
        print(f"  {e['date'][:10]}  ${e['daily_pnl']:>+8.0f}  UNG ${e['ung']:.2f}  "
              f"{e['regime']:<14} anom={e['anomaly']:<12} knife={e['knife']} grind={e['grind']}  trades:{tt}")

    print(f"\nBest {len(result['best_days'])} gain days:")
    for e in result['best_days']:
        tt = list(e['trade_types'].keys())[:3]
        print(f"  {e['date'][:10]}  ${e['daily_pnl']:>+8.0f}  UNG ${e['ung']:.2f}  "
              f"{e['regime']:<14} anom={e['anomaly']:<12}  trades:{tt}")

    print(f"\n=== EVOLUTION SUGGESTIONS ===")
    for s in result['evolution_suggestions']:
        print(f"\n[{s['pattern']}]")
        print(f"  Evidence:   {s['evidence']}")
        print(f"  Suggestion: {s['suggestion']}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nDetailed JSON: {args.output}")


if __name__ == '__main__':
    main()
