"""Factor-enhanced DBA wheel — wire significant factors into sizing/strike.

Evidence from factor_scan.py (quintile fwd-63d spreads):
  oni        -5.7%  t=-2.9 p=.004 n=230  (19yr — most robust)
  ng_trend   -6.8%  t=-3.2 p=.004 n=54   (5yr only — one macro cycle)
  dxy_trend  -6.7%  t=-2.6 p=.020 n=54   (5yr only)
  seasonality: Dec/Jan/May/Oct strong; Jun/Sep/Nov weak (n≈20/mo)

Variants (all vs baseline 60d/2% OTM, 2015-2026):
  baseline      no signal
  oni_tilt      oni<0 → 1.3x size; oni>+0.5 → 0.5x size + 5% OTM
  seasonal      weak months (6,9,11) → 0.5x; strong (12,1,5,10) → 1.3x
  macro         dxy_trend>0 AND ng_trend>0 → 0.5x size (headwind)
  combo         multiplicative blend of all three, clipped [0.3, 1.6]

Sizing-only modifications keep the strategy identity (premium harvest)
intact — we tilt exposure, never flip direction.
"""
import os
import sys
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(THIS_DIR, 'cache')
sys.path.insert(0, THIS_DIR)

from wheel_backtest import run_wheel, summarize  # noqa: E402

START = '2015-01-01'


def load_signals():
    f = pd.read_csv(os.path.join(CACHE, 'dba_factor_panel.csv'),
                    index_col=0, parse_dates=True)
    # lag ALL signals 1 day — no lookahead
    return f[['oni', 'dxy_trend', 'ng_trend', 'month']].shift(1)


SIG = load_signals()
WEAK_MONTHS = {6, 9, 11}
STRONG_MONTHS = {12, 1, 5, 10}


def _row(date):
    try:
        return SIG.loc[date]
    except KeyError:
        return None


def sig_oni(date):
    r = _row(date)
    if r is None or pd.isna(r['oni']):
        return {}
    if r['oni'] < 0:
        return {'size_mult': 1.3, 'otm_pct': 0.02}
    if r['oni'] > 0.5:
        return {'size_mult': 0.5, 'otm_pct': 0.05}
    return {}


def sig_seasonal(date):
    r = _row(date)
    if r is None:
        return {}
    m = int(r['month']) if not pd.isna(r['month']) else date.month
    if m in WEAK_MONTHS:
        return {'size_mult': 0.5}
    if m in STRONG_MONTHS:
        return {'size_mult': 1.3}
    return {}


def sig_macro(date):
    r = _row(date)
    if r is None or pd.isna(r['dxy_trend']) or pd.isna(r['ng_trend']):
        return {}
    if r['dxy_trend'] > 0 and r['ng_trend'] > 0:
        return {'size_mult': 0.5}
    return {}


def sig_combo(date):
    r = _row(date)
    if r is None:
        return {}
    mult = 1.0
    otm = 0.02
    if not pd.isna(r['oni']):
        if r['oni'] < 0:
            mult *= 1.3
        elif r['oni'] > 0.5:
            mult *= 0.5
            otm = 0.05
    m = int(r['month']) if not pd.isna(r['month']) else date.month
    if m in WEAK_MONTHS:
        mult *= 0.6
    elif m in STRONG_MONTHS:
        mult *= 1.25
    if (not pd.isna(r['dxy_trend']) and not pd.isna(r['ng_trend'])
            and r['dxy_trend'] > 0 and r['ng_trend'] > 0):
        mult *= 0.6
    return {'size_mult': max(0.3, min(1.6, mult)), 'otm_pct': otm}


def main():
    variants = {
        'baseline 60d/2%': None,
        'oni_tilt': sig_oni,
        'seasonal': sig_seasonal,
        'macro (dxy+ng)': sig_macro,
        'combo': sig_combo,
    }
    rows = []
    for name, fn in variants.items():
        curve, trades = run_wheel('DBA', start=START, dte_target=60,
                                  otm_pct=0.02, signal_fn=fn)
        s = summarize(curve, name)
        s['n_trades'] = len(trades)
        rows.append(s)
        print(f'  {name:<18} ann={s["ann_ret"]:+.2%}  sharpe={s["sharpe"]:+.3f}  '
              f'mdd={s["mdd"]:.2%}  trades={len(trades)}')
    df = pd.DataFrame(rows).sort_values('sharpe', ascending=False)
    print('\n=== FACTOR-ENHANCED DBA WHEEL (2015-2026) ===')
    print(df.to_string(index=False))
    df.to_csv(os.path.join(CACHE, 'factor_wheel_results.csv'), index=False)


if __name__ == '__main__':
    main()
