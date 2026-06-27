"""Allocation scheme sweep: UNG production kernel × DBA wheel.

Per [[composite_empirical]]: forecast-gating (ENSO) underperforms static.
This sweeps NON-forecast schemes — ratios and risk-based rebalancing:

  1. Static grid: 90/10 → 30/70
  2. Inverse-vol (risk parity): w_i ∝ 1/σ_i(63d) — balances effective
     dollar-risk (delta-dollar exposure shows up as realized leg vol)
  3. Vol-target: risk-parity mix scaled so portfolio vol ≈ target;
     surplus parked in BOXX
  4. DD-responsive: risk-parity + halve a leg's weight while it's >5%
     below its 63d NAV peak (cheap trend filter, no forecast)

All weights lagged 1 day. Monthly rebalance (21 trading days) to keep
turnover realistic.

Run: venv/bin/python research/dba/allocation_sweep.py
"""
import os
import sys
import math
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
RESULTS_DIR = os.path.join(ROOT, 'backtest', 'results')
CACHE = os.path.join(THIS_DIR, 'cache')
sys.path.insert(0, THIS_DIR)

from wheel_backtest import run_wheel  # noqa: E402

BOXX_DAILY = 0.0474 / 252
REBAL_DAYS = 21
VOL_WINDOW = 63


def stats(ret, label):
    ret = ret.dropna()
    if len(ret) < 100 or ret.std() == 0:
        return None
    nav = (1 + ret).cumprod()
    years = (ret.index[-1] - ret.index[0]).days / 365.25
    ann = nav.iloc[-1] ** (1 / years) - 1
    sharpe = ret.mean() / ret.std() * math.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()
    w12 = nav.pct_change(252).dropna()
    return {'scheme': label, 'ann_ret': round(ann, 4),
            'sharpe': round(sharpe, 3), 'mdd': round(mdd, 4),
            'worst_12mo': round(w12.min(), 4) if len(w12) else None}


def monthly(series_daily):
    """Hold each rebalance-day value constant for REBAL_DAYS."""
    out = series_daily.copy()
    keep = series_daily.iloc[::REBAL_DAYS].reindex(series_daily.index).ffill()
    out.loc[:] = keep
    return out


def main():
    h = pd.read_csv(os.path.join(
        RESULTS_DIR, 'champion_premium_harvest_scale_invariant_history.csv'))
    h['date'] = pd.to_datetime(h['date']).dt.normalize()
    ung_nav = h.set_index('date')['nav'].astype(float)
    ung_ret = ung_nav.pct_change().fillna(0)

    dba_curve, _ = run_wheel('DBA', start=str(ung_nav.index[0].date()),
                             dte_target=60, otm_pct=0.02)
    dba_ret = dba_curve['nav'].pct_change().fillna(0)

    idx = ung_ret.index.intersection(dba_ret.index)
    u = ung_ret.reindex(idx).fillna(0)
    d = dba_ret.reindex(idx).fillna(0)
    bx = pd.Series(BOXX_DAILY, index=idx)

    rows = []

    # ── 1. static grid ──────────────────────────────────────────────
    for w_u in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        w_d = 1 - w_u
        rows.append(stats(w_u * u + w_d * d, f'static {int(w_u*100)}/{int(w_d*100)}'))

    # ── rolling leg vols (effective dollar-risk proxies) ────────────
    vol_u = u.rolling(VOL_WINDOW).std().shift(1)
    vol_d = d.rolling(VOL_WINDOW).std().shift(1)

    # ── 2. inverse-vol risk parity ──────────────────────────────────
    iu, idv = 1 / vol_u, 1 / vol_d
    w_u_rp = monthly((iu / (iu + idv)).clip(0.2, 0.9))
    w_d_rp = 1 - w_u_rp
    rows.append(stats(w_u_rp * u + w_d_rp * d, 'inverse-vol risk parity'))

    # ── 3. vol-target (rp mix scaled to target, surplus → BOXX) ────
    for tgt_ann in (0.10, 0.14, 0.18):
        tgt_d = tgt_ann / math.sqrt(252)
        mix = w_u_rp * u + w_d_rp * d
        mix_vol = mix.rolling(VOL_WINDOW).std().shift(1)
        scale = monthly((tgt_d / mix_vol).clip(0.3, 1.5))
        port = (scale.clip(upper=1.0) * mix
                + (1 - scale.clip(upper=1.0)) * bx
                + (scale - scale.clip(upper=1.0)) * mix)  # mild leverage ≤1.5x
        rows.append(stats(port, f'vol-target {int(tgt_ann*100)}% (rp base)'))

    # ── 4. dd-responsive risk parity ─────────────────────────────────
    nav_u = (1 + u).cumprod()
    nav_d = (1 + d).cumprod()
    dd_u = (nav_u / nav_u.rolling(VOL_WINDOW, min_periods=10).max() - 1).shift(1)
    dd_d = (nav_d / nav_d.rolling(VOL_WINDOW, min_periods=10).max() - 1).shift(1)
    wu = w_u_rp.where(dd_u > -0.05, w_u_rp * 0.5)
    wd = w_d_rp.where(dd_d > -0.05, w_d_rp * 0.5)
    w_bx = (1 - wu - wd).clip(0, 1)
    rows.append(stats(wu * u + wd * d + w_bx * bx, 'dd-responsive rp (+BOXX idle)'))

    # baselines
    rows.append(stats(u, 'UNG kernel only'))
    rows.append(stats(d, 'DBA wheel only'))

    df = pd.DataFrame([r for r in rows if r])
    df = df.sort_values('sharpe', ascending=False)
    print(f'\n=== ALLOCATION SWEEP ({idx[0].date()} → {idx[-1].date()}, '
          f'monthly rebalance) ===')
    print(df.to_string(index=False))

    df.to_csv(os.path.join(CACHE, 'allocation_sweep.csv'), index=False)
    print(f'\n→ {CACHE}/allocation_sweep.csv')

    # weight diagnostics for the rp scheme
    print(f'\nrisk-parity weight on UNG: mean {w_u_rp.mean():.0%}, '
          f'min {w_u_rp.min():.0%}, max {w_u_rp.max():.0%}')
    print(f'leg vol (ann): UNG {vol_u.mean()*math.sqrt(252):.1%}, '
          f'DBA {vol_d.mean()*math.sqrt(252):.1%}')


if __name__ == '__main__':
    main()
