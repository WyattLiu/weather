"""Empirical UNG-kernel × DBA-wheel composite backtest.

Unlike composite_analytical.py (closed-form Sharpe math) this blends the
ACTUAL day-by-day NAV curves:
  - UNG leg: production kernel history from backtest/results/
    (champion_premium_harvest_scale_invariant_history.csv — the real
    replay with all 40+ protections)
  - DBA leg: wheel_backtest at the sweep-optimal 60d / 2% OTM
  - Weights: daily regime allocation (ENSO + drought + UNG surge-z),
    same rules as composite_edge.allocate(), normalized to w_ung+w_dba+w_boxx=1
  - BOXX leg: 4.74%/yr on the idle weight

Compares: UNG-kernel-only vs static 60/40 vs regime-gated composite.

Run AFTER replay_engine has produced fresh results:
    venv/bin/python research/dba/composite_empirical.py
    venv/bin/python research/dba/composite_empirical.py --ung-strategy champion_premium_harvest_scale_invariant_hh_storm
"""
import os
import sys
import math
import json
import argparse
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
RESULTS_DIR = os.path.join(ROOT, 'backtest', 'results')
CACHE = os.path.join(THIS_DIR, 'cache')
sys.path.insert(0, THIS_DIR)

from wheel_backtest import run_wheel  # noqa: E402


def load_ung_kernel(strategy):
    p = os.path.join(RESULTS_DIR, f'{strategy}_history.csv')
    if not os.path.exists(p):
        raise SystemExit(f'missing {p} — run replay_engine.py first')
    h = pd.read_csv(p)
    date_col = 'date' if 'date' in h.columns else h.columns[0]
    h[date_col] = pd.to_datetime(h[date_col]).dt.normalize()  # strip tz-artifact times
    h = h.set_index(date_col)
    return h['nav'].astype(float)


def stats(ret, label):
    ret = ret.dropna()
    if len(ret) < 100 or ret.std() == 0:
        return {'strategy': label, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0}
    nav = (1 + ret).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / years) - 1
    sharpe = ret.mean() / ret.std() * math.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    # worst rolling 12mo window ([[feedback_walk_forward_truth]])
    w12 = nav.pct_change(252).dropna()
    worst12 = w12.min() if len(w12) else float('nan')
    return {'strategy': label, 'ann_ret': round(ann, 4),
            'sharpe': round(sharpe, 3), 'mdd': round(mdd, 4),
            'worst_12mo': round(worst12, 4) if worst12 == worst12 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ung-strategy',
                    default='champion_premium_harvest_scale_invariant')
    args = ap.parse_args()

    print(f'[composite-emp] UNG leg: {args.ung_strategy}')
    ung_nav = load_ung_kernel(args.ung_strategy)
    ung_ret = ung_nav.pct_change().fillna(0)

    print('[composite-emp] DBA leg: wheel 60d / 2% OTM...')
    dba_curve, _ = run_wheel('DBA', start=str(ung_nav.index[0].date()),
                             dte_target=60, otm_pct=0.02)
    dba_ret = dba_curve['nav'].pct_change().fillna(0)

    idx = ung_ret.index.intersection(dba_ret.index)
    print(f'[composite-emp] overlap: {len(idx)} days '
          f'({idx[0].date()} → {idx[-1].date()})')
    ung_ret = ung_ret.reindex(idx).fillna(0)
    dba_ret = dba_ret.reindex(idx).fillna(0)
    boxx_ret = pd.Series(0.0474 / 252, index=idx)

    # Regime weights (same logic family as composite_edge.allocate)
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    ung_z = ((panel['UNG'] - panel['UNG'].rolling(20).mean())
             / panel['UNG'].rolling(20).std()).abs()
    enso = (panel['oni'] / 2.0).clip(-1, 1)
    drought = (panel['dsci_z'] / 2.0).clip(-1, 1)
    dba_edge = (0.6 * enso + 0.4 * drought).clip(-1, 1)

    ung_active = (ung_z > 0.6).reindex(idx).ffill().fillna(False)
    dba_active = (dba_edge > 0.4).reindex(idx).ffill().fillna(False)

    w_ung = pd.Series(0.30, index=idx)   # idle default
    w_dba = pd.Series(0.0, index=idx)
    w_ung[ung_active & ~dba_active] = 0.85
    w_ung[~ung_active & dba_active] = 0.45
    w_dba[~ung_active & dba_active] = 0.40
    w_ung[ung_active & dba_active] = 0.50
    w_dba[ung_active & dba_active] = 0.30
    # lag 1d — no lookahead
    w_ung, w_dba = w_ung.shift(1).fillna(0.3), w_dba.shift(1).fillna(0.0)
    w_boxx = (1 - w_ung - w_dba).clip(0, 1)

    composite = w_ung * ung_ret + w_dba * dba_ret + w_boxx * boxx_ret
    static = 0.6 * ung_ret + 0.4 * dba_ret

    rows = [
        stats(ung_ret, f'UNG kernel only ({args.ung_strategy[:30]})'),
        stats(dba_ret, 'DBA wheel only (60d 2% OTM)'),
        stats(static, 'static 60/40'),
        stats(composite, 'regime-gated composite'),
    ]
    df = pd.DataFrame(rows)
    print(f'\n=== EMPIRICAL COMPOSITE ({idx[0].date()} → {idx[-1].date()}) ===')
    print(df.to_string(index=False))

    corr = pd.DataFrame({'u': ung_ret, 'd': dba_ret}).corr().iloc[0, 1]
    print(f'\nUNG-kernel × DBA-wheel daily correlation: {corr:+.4f}')
    print(f'Avg weights: UNG {w_ung.mean():.0%} / DBA {w_dba.mean():.0%} / '
          f'BOXX {w_boxx.mean():.0%}')

    out = {'summary': rows, 'correlation': float(corr),
           'ung_strategy': args.ung_strategy,
           'avg_weights': {'ung': float(w_ung.mean()),
                           'dba': float(w_dba.mean()),
                           'boxx': float(w_boxx.mean())}}
    with open(os.path.join(CACHE, 'composite_empirical.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'→ {CACHE}/composite_empirical.json')


if __name__ == '__main__':
    main()
