"""Consolidated frontier under the CORRECTED engine (2026-06-16):
real-fill default, SPREAD_OPTION $0.07 (measured), feature-indentation bugfix,
NaN guard, de-duped cost model. Runs each validated kernel full-sample AND on the
sealed OOS test window, then draws the return/risk trade-off frontier.

Output: results/frontier_consolidated.csv + results/frontier.png
"""
import os
import sys
import math
import numpy as np
import pandas as pd

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from replay_engine import run_strategy_simple, STRATEGIES, precompute_factor_z

CACHE = os.path.join(THIS, 'cache')
RESULTS = os.path.join(THIS, 'results')
TEST_START = '2024-01-02'   # sealed OOS window (matches honest_walkforward)

# curated frontier set: label -> strategy key
FRONTIER = [
    ('Champion (live, gen-8)', 'champion_kold15_ivrank_kbh'),
    ('KOLD15+IVrank (gen-2)', 'champion_kold15_ivrank'),
    ('Premium Harvest (SI)', 'premium_harvest_scale_invariant'),
    ('Premium Harvest', 'premium_harvest'),
    ('Target-25 Smooth', 'target_25_smooth'),
    ('Router-safe (gen-11)', 'g11_router_safe'),
    ('PutRatio-2x (gen-11 C3)', 'g11_putratio_big'),
    ('ITM-put (gen-11 A)', 'g11_itmput_conv'),
    ('Bwd-derisk on router (g12)', 'g12_bwd_on_router'),
    ('Gen-10 book55 (rejected)', 'g10_book55'),
]


def metrics(nav):
    nav = nav.dropna()
    if len(nav) < 30:
        return None
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
    sh = r.mean() / (r.std() + 1e-12) * math.sqrt(252)
    mdd = ((nav / nav.cummax() - 1).min()) * 100
    return ann, sh, mdd


def main():
    df = pd.read_csv(os.path.join(CACHE, 'master_dataset.csv'),
                     index_col=0, parse_dates=True)
    df = precompute_factor_z(df).dropna(subset=['UNG'])
    df_test = df.loc[TEST_START:]
    ic, ish = 48000, 6200

    rows = []
    for label, key in FRONTIER:
        if key not in STRATEGIES:
            print(f'  SKIP {key} (absent)')
            continue
        h_full, _ = run_strategy_simple(df, STRATEGIES[key], ic, ish)
        h_full = h_full.set_index(pd.to_datetime(h_full['date']))
        mf = metrics(h_full['nav'])
        # OOS: fresh run on the sealed test window only
        h_oos, _ = run_strategy_simple(df_test, STRATEGIES[key], 100000, 0)
        h_oos = h_oos.set_index(pd.to_datetime(h_oos['date']))
        mo = metrics(h_oos['nav'])
        if not mf or not mo:
            continue
        rows.append({'label': label, 'key': key,
                     'full_ann': mf[0], 'full_sharpe': mf[1], 'full_mdd': mf[2],
                     'oos_ann': mo[0], 'oos_sharpe': mo[1], 'oos_mdd': mo[2]})
        print(f'  {label:30} FULL {mf[0]:+5.1f}%/{mf[1]:.2f}/{mf[2]:.0f}  '
              f'OOS {mo[0]:+5.1f}%/{mo[1]:.2f}/{mo[2]:.0f}')
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS, 'frontier_consolidated.csv'), index=False)

    # ---- FRONTIER PLOT: return vs risk (MDD), OOS panel + full-sample panel ----
    res = res.reset_index(drop=True)
    res['num'] = res.index + 1
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
    for ax, (xa, ya, sa, ttl) in zip(axes, [
            ('oos_mdd', 'oos_ann', 'oos_sharpe', 'SEALED OOS (2024-01 → 2026-06) — the honest frontier'),
            ('full_mdd', 'full_ann', 'full_sharpe', 'Full-sample (2021-06 → 2026-06)')]):
        sc = ax.scatter(res[xa], res[ya], c=res[sa], s=420, cmap='viridis',
                        edgecolors='black', linewidths=0.8, zorder=3)
        for _, r in res.iterrows():
            hl = r['key'] == 'champion_kold15_ivrank_kbh'
            ax.annotate(str(int(r['num'])), (r[xa], r[ya]), fontsize=9,
                        ha='center', va='center', zorder=4,
                        color='white', fontweight='bold')
            if hl:
                ax.scatter([r[xa]], [r[ya]], s=720, facecolors='none',
                           edgecolors='crimson', linewidths=2.2, zorder=2)
        ax.set_xlabel('Max Drawdown (%)  ← riskier')
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title(ttl, fontsize=10)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Sharpe', fontsize=8)
    # numbered legend with OOS metrics
    leg = '\n'.join(
        f"{int(r['num'])}. {r['label']}  —  OOS {r['oos_ann']:+.0f}%/Sh {r['oos_sharpe']:.2f}/DD {r['oos_mdd']:.0f}%"
        + ("   ★LIVE" if r['key'] == 'champion_kold15_ivrank_kbh' else "")
        for _, r in res.iterrows())
    fig.text(0.5, -0.02, leg, ha='center', va='top', fontsize=8.5,
             family='monospace', bbox=dict(boxstyle='round', fc='#f6f8fa', ec='#d0d7de'))
    fig.suptitle('UNG Kernel Frontier — corrected engine (real fills, $0.07 spread, '
                 'feature bugfix, de-duped costs)\n(★ red ring = live champion; '
                 'tight cluster = kernels are near-equivalent; only rejected gen-10 sits apart)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    out = os.path.join(RESULTS, 'frontier.png')
    fig.savefig(out, bbox_inches='tight')
    print(f'\nSaved frontier graph → {out}')


if __name__ == '__main__':
    main()
