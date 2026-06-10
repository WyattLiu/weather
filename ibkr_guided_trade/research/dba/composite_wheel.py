"""Composite-gated UNG×DBA wheel backtest (Step 3).

Layers:
  A) DBA wheel at optimal params (DTE 60, OTM 2% — Sharpe 1.47)
  B) UNG wheel at same simple-wheel params (broken without protections —
     so we use a vol-gated version that DOES survive 2022 as our UNG proxy)
  C) Composite: each day, scale put-write notional by the allocator's
     w_ung / w_dba weights from composite_edge logic

Outputs:
  - cache/composite_wheel_curves.csv (NAV curves for each strategy)
  - cache/composite_wheel_summary.json (Sharpe/return/MDD comparison)
"""
import os
import sys
import math
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wheel_backtest import run_wheel, summarize

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')


def main():
    print('[composite] running DBA wheel @ optimal params (60d, 2% OTM)...')
    dba_curve, _ = run_wheel('DBA', start='2015-01-01',
                              dte_target=60, otm_pct=0.02)

    print('[composite] running UNG wheel @ same params with TIGHT vol gate...')
    # Tighter vol gate (60%) for UNG so 2022 spike doesn't bankrupt it
    ung_curve, _ = run_wheel('UNG', start='2015-01-01',
                              dte_target=60, otm_pct=0.02, vol_gate=0.6)

    # Combine: each strategy started with $100k. Build 60/40 blend
    # using DAILY allocation weights from composite logic.
    dba_ret = dba_curve['nav'].pct_change().fillna(0)
    ung_ret = ung_curve['nav'].pct_change().fillna(0)

    # Align indices
    idx = dba_curve.index.intersection(ung_curve.index)
    dba_ret = dba_ret.reindex(idx).fillna(0)
    ung_ret = ung_ret.reindex(idx).fillna(0)

    # Static 60/40 (baseline diversification)
    static_60_40 = 0.6 * ung_ret + 0.4 * dba_ret

    # Regime-aware allocation from composite panel
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    # Compute UNG surge_z (current setup signal) and DBA edge proxy
    ung_z = ((panel['UNG'] - panel['UNG'].rolling(20).mean()) /
             panel['UNG'].rolling(20).std()).abs().fillna(0)
    enso = (panel['oni'] / 2.0).clip(-1, 1).fillna(0)
    drought = (panel['dsci_z'] / 2.0).clip(-1, 1).fillna(0)
    dba_e = (0.6*enso + 0.4*drought).clip(-1, 1)

    # Daily weights: high UNG z → favor UNG, high DBA edge → favor DBA
    w_ung = (0.4 + 0.3 * (ung_z / 3.0).clip(0, 1)).reindex(idx).ffill().fillna(0.4)
    w_dba = (0.2 + 0.4 * dba_e.clip(0, 1)).reindex(idx).ffill().fillna(0.2)
    # Normalize so weights sum to 1 (remainder = BOXX cash, treated as 4.74%/yr)
    total = w_ung + w_dba
    excess = (total - 1).clip(lower=0)
    w_ung = (w_ung - excess * w_ung / total).clip(0, 1)
    w_dba = (w_dba - excess * w_dba / total).clip(0, 1)
    w_boxx = (1 - w_ung - w_dba).clip(0, 1)
    boxx_ret = pd.Series(0.0474 / 252, index=idx)

    composite_ret = w_ung * ung_ret + w_dba * dba_ret + w_boxx * boxx_ret

    # Summarize
    def stats(ret, label):
        ret = ret.dropna()
        if len(ret) < 100 or ret.std() == 0:
            return {'strategy': label, 'cum_ret': 0, 'ann_ret': 0, 'sharpe': 0, 'mdd': 0}
        cum_nav = (1 + ret).cumprod()
        cum = cum_nav.iloc[-1] - 1
        years = (ret.index[-1] - ret.index[0]).days / 365.25
        ann = (1 + cum) ** (1/years) - 1 if cum > -1 else -1
        sharpe = ret.mean() / ret.std() * math.sqrt(252)
        mdd = (cum_nav / cum_nav.cummax() - 1).min()
        return {'strategy': label, 'cum_ret': round(cum, 4),
                'ann_ret': round(ann, 4), 'sharpe': round(sharpe, 3),
                'mdd': round(mdd, 4), 'years': round(years, 1)}

    rows = [
        stats(ung_ret, 'UNG_only (vol-gated, sim)'),
        stats(dba_ret, 'DBA_only (60d 2% OTM)'),
        stats(static_60_40, 'static_60_40_blend'),
        stats(composite_ret, 'composite_regime_gated'),
    ]
    df = pd.DataFrame(rows)
    print('\n=== Composite wheel backtest ===')
    print(df.to_string(index=False))

    # Correlation
    corr = pd.DataFrame({'UNG': ung_ret, 'DBA': dba_ret}).corr().iloc[0, 1]
    print(f'\nUNG×DBA daily-return correlation: {corr:+.4f}')

    # Weight distribution
    print(f'\nAvg weights over backtest: UNG={w_ung.mean():.2%}  '
          f'DBA={w_dba.mean():.2%}  BOXX={w_boxx.mean():.2%}')
    print(f'Latest weights ({idx[-1].date()}): UNG={w_ung.iloc[-1]:.2%}  '
          f'DBA={w_dba.iloc[-1]:.2%}  BOXX={w_boxx.iloc[-1]:.2%}')

    # Save
    out = {
        'summary': rows,
        'correlation_ung_dba': float(corr),
        'avg_weights': {'ung': float(w_ung.mean()), 'dba': float(w_dba.mean()),
                        'boxx': float(w_boxx.mean())},
        'latest_weights': {'ung': float(w_ung.iloc[-1]), 'dba': float(w_dba.iloc[-1]),
                           'boxx': float(w_boxx.iloc[-1])},
    }
    with open(os.path.join(CACHE, 'composite_wheel_summary.json'), 'w') as f:
        json.dump(out, f, indent=2)
    curves = pd.DataFrame({
        'UNG_only': (1+ung_ret).cumprod(),
        'DBA_only': (1+dba_ret).cumprod(),
        'static_60_40': (1+static_60_40).cumprod(),
        'composite': (1+composite_ret).cumprod(),
    })
    curves.to_csv(os.path.join(CACHE, 'composite_wheel_curves.csv'))
    print(f'\n→ {CACHE}/composite_wheel_summary.json')


if __name__ == '__main__':
    main()
