"""DBA wheel parameter sweep — find best (DTE, OTM) combo.

Goal: data-driven answer to "what DBA strikes/expiries should I sell"?
Output: ranked grid of Sharpe / return / MDD across the parameter space.
"""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wheel_backtest import run_wheel, summarize

DTE_GRID = [21, 30, 45, 60, 90]
OTM_GRID = [0.02, 0.03, 0.05, 0.075, 0.10]


def main():
    rows = []
    for dte in DTE_GRID:
        for otm in OTM_GRID:
            try:
                curve, trades = run_wheel('DBA', start='2015-01-01',
                                          dte_target=dte, otm_pct=otm)
                s = summarize(curve, f'DBA_dte{dte}_otm{int(otm*100)}')
                s['dte'] = dte
                s['otm'] = otm
                s['n_trades'] = len(trades)
                rows.append(s)
                print(f'  DTE={dte:>3d} OTM={otm:.0%}  '
                      f'ann={s["ann_ret"]:+.2%}  sharpe={s["sharpe"]:+.3f}  '
                      f'mdd={s["mdd"]:.2%}  n={s["n_trades"]}')
            except Exception as e:
                print(f'  DTE={dte} OTM={otm}: FAILED ({e})')

    df = pd.DataFrame(rows)
    df = df.sort_values('sharpe', ascending=False)
    print('\n=== Top 5 by Sharpe ===')
    print(df.head(5)[['dte', 'otm', 'ann_ret', 'sharpe', 'mdd', 'n_trades']].to_string(index=False))
    print('\n=== Top 5 by Return ===')
    print(df.sort_values('ann_ret', ascending=False)
            .head(5)[['dte', 'otm', 'ann_ret', 'sharpe', 'mdd', 'n_trades']].to_string(index=False))
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cache', 'wheel_sweep_dba.csv'), index=False)


if __name__ == '__main__':
    main()
