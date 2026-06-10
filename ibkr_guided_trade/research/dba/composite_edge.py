"""Composite UNG × DBA edge allocator.

Hypothesis:
  - UNG edge fires on 1-30d weather (HDD/CDD, HH basis, storage surprise)
  - DBA edge fires on 3-12mo weather (ENSO, drought, growing-season)
  - When UNG has no setup AND DBA does → divert capital to DBA puts
  - When both fire → split by edge magnitude
  - When neither → park in BOXX

This script:
  1. Computes daily edge scores for UNG and DBA from the master_panel
  2. Backtests the allocation rule against UNG-only baseline
  3. Outputs a current allocation recommendation for live use

Run:
    venv/bin/python research/dba/composite_edge.py
"""
import os
import sys
import json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, 'cache')


def _load_enso_outlook():
    """Read latest CPC ENSO probability outlook (optional)."""
    path = os.path.join(CACHE, 'enso_outlook.json')
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def compute_edges(panel):
    """Return DataFrame with ung_edge and dba_edge scores in [-1, 1]."""
    df = panel.copy()

    # UNG edge: surge z-score (price vs 20d mean / 20d sd) + flat regime check
    ma20 = df['UNG'].rolling(20).mean()
    sd20 = df['UNG'].rolling(20).std()
    df['ung_surge_z'] = ((df['UNG'] - ma20) / sd20.replace(0, np.nan)).fillna(0.0)
    # UNG edge magnitude: |surge_z| > 1 = strong setup
    df['ung_edge'] = df['ung_surge_z'].clip(-3, 3) / 3.0  # normalize to [-1,1]

    # DBA edge: composite of ENSO trajectory + drought
    # ENSO score: -1 to +1 based on ONI (strong nino = +1, strong nina = -1)
    df['enso_score'] = (df['oni'] / 2.0).clip(-1, 1)
    # Drought score: dsci_z clipped (drought is bullish ag)
    df['drought_score'] = (df['dsci_z'] / 2.0).clip(-1, 1)
    # Trajectory bonus: ONI rising fast → conviction
    df['oni_trajectory'] = (df['oni_delta_3m'] / 1.0).clip(-0.5, 0.5)

    # DBA edge: ENSO is primary; drought + trajectory add conviction.
    # Cap at [-1, 1].
    df['dba_edge'] = (0.6 * df['enso_score']
                      + 0.25 * df['drought_score']
                      + 0.15 * df['oni_trajectory']).clip(-1, 1)
    return df


def allocate(row):
    """Return dict {ung_pct, dba_pct, boxx_pct} from edges."""
    ung = abs(row.get('ung_edge', 0))
    dba = max(0, row.get('dba_edge', 0))  # only LONG DBA bias counts (we sell puts)
    # Thresholds: must clear noise floor to deploy capital
    UNG_FLOOR = 0.2  # |surge_z| > 0.6 (=0.2*3)
    DBA_FLOOR = 0.2  # ENSO score > 0.4 → weak Niño at minimum
    ung_active = ung > UNG_FLOOR
    dba_active = dba > DBA_FLOOR

    if ung_active and not dba_active:
        return {'ung': 0.85, 'dba': 0.0, 'boxx': 0.15}
    if dba_active and not ung_active:
        return {'ung': 0.45, 'dba': 0.40, 'boxx': 0.15}
    if ung_active and dba_active:
        # Split by relative magnitude
        total = ung + dba
        return {'ung': 0.7 * ung / total, 'dba': 0.7 * dba / total, 'boxx': 0.30}
    # Neither: park in BOXX, keep small UNG grind
    return {'ung': 0.30, 'dba': 0.0, 'boxx': 0.70}


def backtest(panel):
    """Side-by-side: composite vs UNG-only vs UNG+DBA equal-weight."""
    df = compute_edges(panel).dropna(subset=['UNG', 'DBA', 'oni', 'dsci_z'])
    df['ung_ret'] = df['UNG'].pct_change()
    df['dba_ret'] = df['DBA'].pct_change()
    df['boxx_ret'] = 0.0474 / 252  # 4.74% standing yield

    # Lag allocation by 1d (avoid lookahead)
    allocs = df.apply(allocate, axis=1)
    df['w_ung'] = allocs.apply(lambda x: x['ung']).shift(1)
    df['w_dba'] = allocs.apply(lambda x: x['dba']).shift(1)
    df['w_boxx'] = allocs.apply(lambda x: x['boxx']).shift(1)

    df['composite_ret'] = (df['w_ung'] * df['ung_ret']
                           + df['w_dba'] * df['dba_ret']
                           + df['w_boxx'] * df['boxx_ret'])
    df['ung_only_ret'] = df['ung_ret']
    df['equal_ret'] = 0.5 * df['ung_ret'] + 0.5 * df['dba_ret']

    df = df.dropna(subset=['composite_ret'])

    def summarize(ret, name):
        cum = (1 + ret).prod() - 1
        ann = (1 + ret).prod() ** (252 / len(ret)) - 1
        sharpe = ret.mean() / ret.std() * np.sqrt(252)
        mdd = (1 + ret).cumprod().div((1 + ret).cumprod().cummax()).sub(1).min()
        return {'strategy': name, 'cum_ret': cum, 'ann_ret': ann, 'sharpe': sharpe, 'mdd': mdd}

    rows = [
        summarize(df['composite_ret'], 'composite'),
        summarize(df['ung_only_ret'], 'ung_only'),
        summarize(df['equal_ret'], 'equal_weight'),
    ]
    summary = pd.DataFrame(rows)
    return df, summary


def main():
    panel = pd.read_csv(os.path.join(CACHE, 'master_panel.csv'),
                        index_col=0, parse_dates=True)
    df, summary = backtest(panel)

    print('=== Backtest comparison (held weights, no friction) ===\n')
    print(summary.to_string(index=False))

    # Allocation regime distribution
    print('\n=== Allocation regime distribution (last 5y) ===')
    recent = df.tail(252*5)
    bucket = recent[['w_ung', 'w_dba', 'w_boxx']].copy()
    bucket['regime'] = bucket.apply(
        lambda r: ('ung_only' if r['w_ung'] >= 0.7 and r['w_dba'] < 0.1 else
                   'dba_only' if r['w_dba'] >= 0.3 and r['w_ung'] < 0.5 else
                   'both' if r['w_dba'] >= 0.1 and r['w_ung'] >= 0.3 else
                   'idle_boxx'), axis=1)
    print(bucket['regime'].value_counts())

    # Forward signal from CPC plume — bump latest dba_edge for live recommendation
    # (backtest section above uses historical-only signals; this is live-only)
    outlook = _load_enso_outlook()
    fwd_pct = None
    fwd_bump = 0.0
    djf_pct = None
    if outlook:
        s = outlook.get('summary', {})
        fwd_pct = s.get('peak_el_nino_pct', 0)
        djf_pct = s.get('djf_el_nino_pct', 0)
        # Bump scaled by peak probability: 80% → +0.2, 95% → +0.3
        if fwd_pct >= 80:
            fwd_bump = min(0.35, (fwd_pct - 60) / 100)
            # Apply bump only to last row's edge + reallocate
            df.loc[df.index[-1], 'dba_edge'] = min(1.0, df.loc[df.index[-1], 'dba_edge'] + fwd_bump)
            new_alloc = allocate(df.iloc[-1])
            df.loc[df.index[-1], 'w_ung'] = new_alloc['ung']
            df.loc[df.index[-1], 'w_dba'] = new_alloc['dba']
            df.loc[df.index[-1], 'w_boxx'] = new_alloc['boxx']

    # Current state
    latest = df.iloc[-1]
    print(f'\n=== Current allocation recommendation ({df.index[-1].date()}) ===')
    print(f'  ung_edge   = {latest["ung_edge"]:+.3f}  (surge_z={latest["ung_surge_z"]:+.2f})')
    print(f'  dba_edge   = {latest["dba_edge"]:+.3f}  (enso={latest["enso_score"]:+.2f}, '
          f'drought={latest["drought_score"]:+.2f}, traj={latest["oni_trajectory"]:+.2f})')
    if fwd_pct is not None:
        print(f'  CPC forward: peak El Niño = {fwd_pct}% (issued {outlook.get("issue_date")})')
    print(f'  allocation = UNG {latest["w_ung"]:.0%}  DBA {latest["w_dba"]:.0%}  '
          f'BOXX {latest["w_boxx"]:.0%}')

    # Dump for kernel consumption
    out = {
        'as_of': str(df.index[-1].date()),
        'ung_edge': float(latest['ung_edge']),
        'dba_edge': float(latest['dba_edge']),
        'oni': float(latest['oni']),
        'dsci_z': float(latest['dsci_z']),
        'cpc_outlook': {
            'issue_date': outlook.get('issue_date') if outlook else None,
            'status': outlook.get('status') if outlook else None,
            'peak_el_nino_pct': fwd_pct,
            'djf_el_nino_pct': djf_pct,
            'forward_bump_applied': round(fwd_bump, 2),
        },
        'allocation': {
            'ung': float(latest['w_ung']),
            'dba': float(latest['w_dba']),
            'boxx': float(latest['w_boxx']),
        },
        'backtest': summary.to_dict('records'),
    }
    with open(os.path.join(CACHE, 'composite_state.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\n→ {CACHE}/composite_state.json')


if __name__ == '__main__':
    main()
