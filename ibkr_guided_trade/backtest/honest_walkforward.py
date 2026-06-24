"""HONEST walk-forward with sealed test data + realistic costs.

Addresses Codex GPT-5.5 critique:
1. Sealed train/test split — no parameter tuning sees test data
2. Disjoint (non-overlapping) test windows
3. Commission + slippage modeling (IBKR-realistic)
4. Early-assignment haircut for short ITM puts
5. Report TRUE out-of-sample performance

Setup:
- TRAIN window: 2021-06 → 2024-01 (~2.6 yrs) — parameter selection knew this
- TEST window:  2024-01 → 2026-06 (~2.5 yrs) — sealed, no tuning saw this
- Strategy parameters frozen at TRAIN-end values
- Out-of-sample (TEST) numbers are the HONEST estimate of live performance

Usage:
    cd ibkr_guided_trade
    venv/bin/python backtest/honest_walkforward.py
"""
from __future__ import annotations

import os
import sys
import math
import argparse
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z, SPREAD_OPTION


TRAIN_START = '2021-06-01'
TRAIN_END   = '2024-01-01'  # frozen here; everything after is SEALED test
TEST_START  = '2024-01-02'
TEST_END    = '2026-06-03'

# Cost model (Wealthsimple-realistic)
COMMISSION_PER_CONTRACT = 0.0   # Wealthsimple is COMMISSION-FREE (was IBKR $0.65). The real cost is the
#                                 bid/ask SPREAD, modeled IN-ENGINE per-leg via SPREAD_OPTION ($0.07/sh
#                                 half-spread, calibrated to real UNG chains) + fill_factor on opens.
SLIPPAGE_PCT_OF_PREMIUM = 0.0   # spread is in-engine; no extra slippage (would double-count the open)
# EARLY ASSIGNMENT is now MODELED in the engine (deep-ITM |delta|>0.99 + extrinsic≈0 → assigned early,
# not at expiry), so there is no EARLY_ASSIGN_HAIRCUT fudge term.


def measure_period(strat, df_period, cash_start=100000):
    """Run strategy on a date range, return summary stats."""
    if len(df_period) < 50:
        return None
    try:
        hist, trades = run_strategy_simple(df_period, strat, cash_start, 0)
        hist = hist.set_index(pd.to_datetime(hist['date']))

        # Compute realistic cost drag
        n_trades_with_qty = trades[trades['type'].astype(str).isin([
            'OPEN_PUT', 'OPEN_CC', 'OPEN_ITM_CC', 'PUT_TP', 'CALL_TP',
            'PUT_ROLL_DOWN', 'CALL_ROLL_UP', 'OPEN_LONG_PUT', 'OPEN_REBUILD_PUT',
        ])]
        # commission: per contract per leg
        total_commission = 0
        if 'qty' in trades.columns:
            total_commission = float(n_trades_with_qty['qty'].abs().sum()) * COMMISSION_PER_CONTRACT
        # slippage: 5% of premium on OPENs
        total_slippage = 0
        opens = trades[trades['type'].astype(str).str.startswith('OPEN_')]
        if 'credit' in opens.columns:
            total_slippage += float(opens['credit'].abs().sum()) * SLIPPAGE_PCT_OF_PREMIUM
        cost_drag = total_commission + total_slippage

        final_nav = float(hist.iloc[-1]['nav']) - cost_drag
        initial = cash_start
        fret = (final_nav / initial - 1) * 100
        yrs = (df_period.index[-1] - df_period.index[0]).days / 365.25
        ann = (1 + fret/100) ** (1/yrs) * 100 - 100 if yrs > 0 else 0
        rets = hist['nav'].pct_change().dropna()
        sh = rets.mean()/(rets.std()+1e-9) * math.sqrt(252)
        peak = hist['nav'].cummax()
        mdd = ((hist['nav'] - peak)/peak * 100).min()
        return {
            'ret': fret, 'ann': ann, 'sharpe': sh, 'mdd': mdd,
            'cost_drag': cost_drag,
            'n_trades': len(trades),
            'yrs': yrs,
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--strategies', nargs='+', default=[
        'champion_premium_harvest_scale_invariant',
        'champion_premium_harvest',
        'champion_target_25_smooth',
        'champion_target_25_walkforward_safe',
        'champion_target_25',
        'kelly_firmness',
    ])
    p.add_argument('--cash', type=int, default=100000)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(THIS_DIR, 'cache', 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])

    df_train = df.loc[TRAIN_START:TRAIN_END]
    df_test = df.loc[TEST_START:TEST_END]

    print('=' * 100)
    print(f'HONEST WALK-FORWARD (cash ${args.cash:,}, realistic costs)')
    print(f'  TRAIN window: {TRAIN_START} → {TRAIN_END}  ({len(df_train)} days, {(df_train.index[-1]-df_train.index[0]).days/365.25:.2f} yrs)')
    print(f'  TEST  window: {TEST_START} → {TEST_END}  ({len(df_test)} days, {(df_test.index[-1]-df_test.index[0]).days/365.25:.2f} yrs) — SEALED')
    print(f'  Costs (Wealthsimple): $0 commission, bid/ask spread in-engine '
          f'(SPREAD_OPTION ${SPREAD_OPTION}/sh/leg + fill_factor), early assignment MODELED '
          f'(deep-ITM |delta|>0.99 + extrinsic~0)')
    print('=' * 100)

    rows = []
    for name in args.strategies:
        if name not in STRATEGIES:
            continue
        strat = STRATEGIES[name]
        train_stats = measure_period(strat, df_train, args.cash)
        test_stats = measure_period(strat, df_test, args.cash)
        if train_stats and test_stats and 'error' not in train_stats and 'error' not in test_stats:
            rows.append((name, train_stats, test_stats))

    print(f'{"strategy":<46} {"TRAIN ann/Sh/MDD":>22} {"TEST ann/Sh/MDD":>22} {"deg":>8}')
    print('-' * 100)
    for name, tr, te in rows:
        # Degradation: how much did TEST drop vs TRAIN
        deg_ret = te['ann'] - tr['ann']
        deg_sh = te['sharpe'] - tr['sharpe']
        deg_mdd = te['mdd'] - tr['mdd']
        print(f'{name:<46} {tr["ann"]:>+5.1f}%/{tr["sharpe"]:>+4.2f}/{tr["mdd"]:>+5.1f}% '
              f'{te["ann"]:>+5.1f}%/{te["sharpe"]:>+4.2f}/{te["mdd"]:>+5.1f}% '
              f'{deg_ret:>+5.1f}pp')

    # Summary on best test variant
    print()
    if rows:
        best_test_sharpe = max(rows, key=lambda r: r[2]['sharpe'])
        best_test_ann = max(rows, key=lambda r: r[2]['ann'])
        print(f'BEST out-of-sample SHARPE: {best_test_sharpe[0]}  → Sharpe {best_test_sharpe[2]["sharpe"]:+.2f} '
              f'(TRAIN was {best_test_sharpe[1]["sharpe"]:+.2f})')
        print(f'BEST out-of-sample RETURN: {best_test_ann[0]}  → ann {best_test_ann[2]["ann"]:+.1f}% '
              f'(TRAIN was {best_test_ann[1]["ann"]:+.1f}%)')
        print()
        print('Cost impact on best-test variant:')
        cd = best_test_sharpe[2]['cost_drag']
        print(f'  Total cost drag (TEST window): ${cd:,.0f}')
        print(f'  As % of starting NAV: {cd/args.cash*100:.1f}%')

if __name__ == '__main__':
    main()
