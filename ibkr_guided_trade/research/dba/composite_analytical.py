"""Composite UNG×DBA: analytical solution using known returns.

Why analytical: my simple wheel engine can't simulate the UNG production
kernel's 40+ protections (HH-storm, vol_aware_sizing, dd_trim, etc.) so
naive UNG simulations bankrupt in 2022. Production kernel walk-forward
results are KNOWN (per [[project_champion_strategies_v2]]):
  UNG production kernel: ~16% ann return, Sharpe ~1.0, MDD ~-24% worst window

DBA wheel results from sweep_dba.py are MEASURED on real DBA prices
with same engine — no simulation gap since DBA doesn't have 2022-style
disasters.

This composes them analytically using measured correlation (≈0).
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wheel_backtest import run_wheel

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

# Production UNG kernel — known walk-forward results
UNG_PROD = {
    'ann_ret': 0.16,    # +16% per [[feedback_sample_bias_rolling_window]]
    'vol_ann': 0.16,    # implies Sharpe = 1.0
    'sharpe': 1.0,
    'mdd_worst_12m': -0.24,
}


def main():
    # Measure DBA wheel at multiple strike/DTE combos
    print('[analytical] measuring DBA wheel @ key configurations...')
    dba_configs = [
        ('60d_2OTM_aggressive', 60, 0.02),
        ('60d_5OTM_kernel_rec', 60, 0.05),
        ('45d_3OTM_balanced', 45, 0.03),
        ('90d_2OTM_yield', 90, 0.02),
    ]
    dba_results = []
    dba_curves = {}
    for label, dte, otm in dba_configs:
        curve, _ = run_wheel('DBA', dte_target=dte, otm_pct=otm)
        ret = curve['nav'].pct_change().dropna()
        ann = ret.mean() * 252
        vol = ret.std() * math.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        mdd = (curve['nav'] / curve['nav'].cummax() - 1).min()
        dba_results.append({
            'config': label, 'dte': dte, 'otm': otm,
            'ann_ret': round(ann, 4), 'vol': round(vol, 4),
            'sharpe': round(sharpe, 3), 'mdd': round(mdd, 4),
        })
        dba_curves[label] = ret
        print(f'  {label:>26s}  ann={ann:+.2%}  σ={vol:.2%}  '
              f'sharpe={sharpe:+.3f}  mdd={mdd:.2%}')

    # ─── Analytical compositing ─────────────────────────────────────────
    # σ_blend² = w_a²σ_a² + w_b²σ_b² + 2 w_a w_b ρ σ_a σ_b
    # μ_blend = w_a μ_a + w_b μ_b
    # Sharpe_blend = μ_blend / σ_blend
    #
    # ρ_UNG_DBA ≈ 0 (measured +0.004 in step 2). With orthogonality the
    # cross term vanishes and adding a stream of similar Sharpe boosts
    # the portfolio Sharpe meaningfully.

    rho = 0.00  # measured ~0 in step 2
    print(f'\nUNG×DBA correlation: ρ = {rho:.3f} (measured ≈0)')
    print(f'\nProduction UNG kernel baseline:')
    print(f'  ann={UNG_PROD["ann_ret"]:+.2%}  σ={UNG_PROD["vol_ann"]:.2%}  '
          f'sharpe={UNG_PROD["sharpe"]:+.3f}')

    print(f'\n=== Composite scenarios (UNG production × DBA wheel) ===')
    print(f'{"DBA config":>26s}  {"weight":>10s}  {"ann":>7s}  {"σ":>6s}  {"sharpe":>7s}  {"Δ vs UNG":>10s}')
    print('-' * 80)
    rows = []
    for d in dba_results:
        for w_dba in (0.20, 0.30, 0.40):
            w_ung = 1 - w_dba
            mu = w_ung * UNG_PROD['ann_ret'] + w_dba * d['ann_ret']
            var = (w_ung**2 * UNG_PROD['vol_ann']**2
                   + w_dba**2 * d['vol']**2
                   + 2 * w_ung * w_dba * rho * UNG_PROD['vol_ann'] * d['vol'])
            sigma = math.sqrt(max(var, 0))
            sharpe = mu / sigma if sigma > 0 else 0
            delta = sharpe - UNG_PROD['sharpe']
            print(f'{d["config"]:>26s}  {w_ung*100:.0f}/{w_dba*100:.0f}  '
                  f'  {mu:+.2%}  {sigma:.2%}  {sharpe:+.3f}  {delta:+.3f}')
            rows.append({
                'dba_config': d['config'], 'w_ung': w_ung, 'w_dba': w_dba,
                'ann_ret': round(mu, 4), 'vol': round(sigma, 4),
                'sharpe': round(sharpe, 3), 'sharpe_lift_vs_ung': round(delta, 3),
            })

    # Find best Sharpe
    best = max(rows, key=lambda r: r['sharpe'])
    print(f'\n🏆 BEST risk-adjusted composite:')
    print(f'   {best["dba_config"]}  @  {best["w_ung"]*100:.0f}/{best["w_dba"]*100:.0f}')
    print(f'   ann return: {best["ann_ret"]:+.2%}')
    print(f'   Sharpe: {best["sharpe"]:+.3f} (vs UNG-alone {UNG_PROD["sharpe"]:+.3f}, '
          f'+{best["sharpe_lift_vs_ung"]:.2f})')
    print(f'   Return-lift: {(best["ann_ret"] - UNG_PROD["ann_ret"])*100:+.1f}pp')

    out = {
        'baseline_ung_production': UNG_PROD,
        'dba_configs_measured': dba_results,
        'correlation_assumption': rho,
        'composite_grid': rows,
        'best_by_sharpe': best,
    }
    with open(os.path.join(CACHE, 'composite_analytical.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n→ {CACHE}/composite_analytical.json')


if __name__ == '__main__':
    main()
