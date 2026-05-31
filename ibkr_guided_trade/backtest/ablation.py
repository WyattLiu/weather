"""Ablation runner — measures TRUE marginal contribution of each kernel.

For a given base strategy, runs N variants each with one feature disabled.
The diff in final NAV between base and variant = real marginal P&L of
that feature. This is the right answer to "what does kernel X contribute?",
NOT the realized per-fire P&L (which can mislead — see
[[feedback_attribution_counterfactual]]).
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay_engine import (  # type: ignore
    STRATEGIES, run_strategy_simple, precompute_factor_z, CACHE_DIR, RESULTS_DIR,
)


# Kernels we can ablate (set to disabling value)
ABLATIONS = {
    'tp_50':              False,   # disable take-profit
    'roll_down':          False,   # disable rolldown
    'bearish_stack':      False,   # disable long-put tail hedge
    'boxx':               False,   # disable BOXX yield parking
    'roll_up_calls':      False,   # disable call roll-up
    'aggressive_itm_cc':  None,    # disable aggressive ITM CC (clear flag)
    'elevator_close':     False,   # disable elevator close
    'use_surprise_z':     False,   # use raw z instead of surprise
    'regime_skip_puts':   None,    # always sell puts regardless of regime
}


def ablate(params: dict, key: str) -> dict:
    """Return a copy of params with one feature disabled."""
    out = dict(params)
    if key == 'aggressive_itm_cc':
        out.pop('aggressive_itm_cc_z', None)
        out.pop('itm_cc_pct', None)
    elif key == 'regime_skip_puts':
        out.pop('regime_skip_puts_z', None)
    else:
        if key in out:
            out[key] = ABLATIONS[key]
    return out


def run_ablation(base_strategy: str, initial_cash: int = 48000,
                 initial_shares: int = 6200) -> pd.DataFrame:
    """Run base strategy + one variant per ablation. Return diff table."""
    if base_strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {base_strategy}")
    df = pd.read_csv(os.path.join(CACHE_DIR, 'master_dataset.csv'),
                     parse_dates=['Date'], index_col=0)
    df = precompute_factor_z(df).dropna(subset=['UNG'])

    initial_nav = initial_cash + initial_shares * df['UNG'].iloc[0]

    base_params = STRATEGIES[base_strategy]
    print(f"Ablating: {base_strategy}")
    print(f"Initial NAV: ${initial_nav:,.0f}")
    print()

    # Base
    hist, _ = run_strategy_simple(df, base_params, initial_cash, initial_shares)
    base_final = float(hist.iloc[-1]['nav'])
    print(f"  base                  ${base_final:>11,.0f}  (baseline)")

    rows = [{'variant': 'base', 'final_nav': base_final, 'delta': 0.0,
             'delta_pct': 0.0, 'pct_of_initial': 0.0}]

    for key in ABLATIONS:
        # Skip if feature not in base
        if key == 'aggressive_itm_cc':
            if 'aggressive_itm_cc_z' not in base_params:
                continue
        elif key == 'regime_skip_puts':
            if 'regime_skip_puts_z' not in base_params:
                continue
        elif key not in base_params:
            continue
        # Skip if already at disabled value
        if key in base_params and base_params[key] == ABLATIONS.get(key):
            continue

        ablated = ablate(base_params, key)
        hist_a, _ = run_strategy_simple(df, ablated, initial_cash, initial_shares)
        final_a = float(hist_a.iloc[-1]['nav'])
        delta = base_final - final_a   # positive = kernel ADDED value
        delta_pct = delta / abs(base_final) * 100 if base_final else 0
        ipct = delta / initial_nav * 100
        sign = '+' if delta > 0 else ''
        print(f"  no_{key:<18} ${final_a:>11,.0f}  Δ ${sign}{delta:>10,.0f}  "
              f"({sign}{delta_pct:5.1f}% of NAV, {sign}{ipct:5.1f}% of initial)")
        rows.append({
            'variant': f'no_{key}',
            'final_nav': final_a,
            'delta': delta,
            'delta_pct': delta_pct,
            'pct_of_initial': ipct,
        })

    df_out = pd.DataFrame(rows)
    return df_out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--strategy', default='elevator_close_surprise',
                   help='Base strategy to ablate (default: elevator_close_surprise)')
    args = p.parse_args()
    out = run_ablation(args.strategy)
    out_path = os.path.join(RESULTS_DIR, f'ablation_{args.strategy}.csv')
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
