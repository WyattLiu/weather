#!/usr/bin/env python3
"""
Parameter-sweep backtest wrapper (CENTRAL_PHILOSOPHY validation gate).

Purpose: every cron-proposed parameter change must find proof from the past
via backtesting before commit. This wrapper runs the UNG wheel backtest with
baseline params + a set of overrides, reports key metrics side-by-side, and
gives a PASS/FAIL verdict vs the strategic objective constraints:

  - Sharpe: override must be within 0.20 of baseline (don't degrade)
  - Max DD: override must NOT breach -10% if baseline didn't
  - CAGR:   override must be within 2 percentage points of baseline

Usage:
  # CLI form (overrides WheelBacktest class attrs)
  python param_sweep.py --set IV=0.45 --set TARGET_DELTA_P=0.25 [--label x]

  # JSON form
  python param_sweep.py --json overrides.json

  # Importable form (returns metrics dict, used by autonomous cron loop)
  from param_sweep import run_sweep
  result = run_sweep({'TARGET_DELTA_P': 0.25}, label='tighter_strike')
  if result['verdict'] == 'PASS': ...

Only class-level constants on WheelBacktest are overrideable in v1:
  IV, RFIR, FRIC, TARGET_DTE, TARGET_DELTA_P, TARGET_DELTA_C,
  ROLL_PCT, REBAL_DAYS

leverage_map and other regime/compute_regime params are out of v1 scope.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Imports below intentionally after warnings filter so pandas/numpy don't
# emit deprecation warnings during their initialization. # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

# Import the existing backtest infra
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ung_wheel_backtest_v2 import (  # type: ignore  # noqa: E402
    WheelBacktest, compute_regime, compute_metrics,
)


# ── Allowed overrides (whitelist; prevents typos from silently no-op'ing) ─
ALLOWED_PARAMS = {
    'IV', 'RFIR', 'FRIC',
    'TARGET_DTE', 'TARGET_DELTA_P', 'TARGET_DELTA_C',
    'ROLL_PCT', 'REBAL_DAYS',
}

# Strategic objective thresholds (CENTRAL_PHILOSOPHY.md)
SHARPE_DEGRADE_MAX = 0.20      # override Sharpe ≥ baseline - 0.20
MAX_DD_HARD_FLOOR = -0.10      # never accept a breach if baseline didn't
CAGR_DEGRADE_MAX = 0.02        # override CAGR ≥ baseline - 2 pp


# ── Cached price + regime so sweep runs aren't I/O bound ─────────────────
_data_cache = {'prices': None, 'regimes': None, 'leverages': None}


def _fetch_data(years: int = 10):
    """Fetch UNG price history and compute regime — cached for re-runs."""
    if _data_cache['prices'] is not None:
        return (_data_cache['prices'],
                _data_cache['regimes'],
                _data_cache['leverages'])

    ticker = yf.Ticker('UNG')
    hist = ticker.history(period='max')
    if hist.index.tz is not None:
        hist.index = hist.index.tz_localize(None)
    prices = hist['Close'].sort_index()
    prices = prices[prices > 0].dropna()
    cutoff = prices.index[-1] - pd.DateOffset(years=years)
    prices = prices[prices.index >= cutoff]
    prices.index = prices.index.normalize()

    _, regimes, leverages = compute_regime(prices, lookback=60, window=252)
    regimes_d = regimes.reindex(prices.index, method='ffill').fillna('fair').to_dict()
    leverages_d = leverages.reindex(prices.index, method='ffill').fillna(0.6).to_dict()

    _data_cache.update(prices=prices, regimes=regimes_d, leverages=leverages_d)
    return prices, regimes_d, leverages_d


def _run_one(label: str, overrides: dict, capital: float = 100_000):
    """Run a single backtest with the given overrides; return metrics dict."""
    prices, regimes, leverages = _fetch_data()

    bad = set(overrides) - ALLOWED_PARAMS
    if bad:
        raise ValueError(f"Disallowed override keys: {sorted(bad)}. "
                         f"Allowed: {sorted(ALLOWED_PARAMS)}")

    # Build override subclass so we don't mutate the original class
    cls = type(f'WheelBacktest__{label}',
               (WheelBacktest,),
               {k: v for k, v in overrides.items()})

    bt = cls(prices, regimes, leverages, start_capital=capital)
    bt.run()

    nav = pd.Series([n for _, n in bt.nav_history],
                    index=pd.DatetimeIndex([d for d, _ in bt.nav_history]))
    m = compute_metrics(nav, start_capital=capital)

    # Income metrics: weekly premium proxy (use realized total_premium / weeks)
    weeks = max(1, len(nav) / 5.0)
    avg_weekly_prem = bt.total_premium / weeks
    median_monthly_prem = float(np.median(
        list(bt.monthly_premium.values()) or [0.0]))

    return {
        'label': label,
        'overrides': overrides,
        'cagr': m['cagr'],
        'sharpe': m['sharpe'],
        'sortino': m['sortino'],
        'max_dd': m['max_drawdown'],
        'final_nav': m['final_nav'],
        'total_premium': bt.total_premium,
        'avg_weekly_prem': avg_weekly_prem,
        'median_monthly_prem': median_monthly_prem,
        'trade_count': bt.trade_count,
        'n_years': m['n_years'],
    }


def _verdict(baseline: dict, override: dict) -> tuple[str, list[str]]:
    """Apply strategic-objective gates; return (PASS|FAIL, reasons)."""
    reasons = []
    ok = True

    # Sharpe: don't degrade more than SHARPE_DEGRADE_MAX
    delta_sharpe = override['sharpe'] - baseline['sharpe']
    if delta_sharpe < -SHARPE_DEGRADE_MAX:
        ok = False
        reasons.append(
            f"Sharpe degraded by {delta_sharpe:+.2f} (cap: -{SHARPE_DEGRADE_MAX:.2f})"
        )
    else:
        reasons.append(f"Sharpe Δ {delta_sharpe:+.2f} (OK)")

    # Max DD: never breach -10% if baseline didn't (or worsen breach significantly)
    if baseline['max_dd'] >= MAX_DD_HARD_FLOOR and override['max_dd'] < MAX_DD_HARD_FLOOR:
        ok = False
        reasons.append(
            f"Max DD breached {MAX_DD_HARD_FLOOR:.0%}: "
            f"{override['max_dd']:.1%} (baseline {baseline['max_dd']:.1%})"
        )
    elif override['max_dd'] < baseline['max_dd'] - 0.03:
        ok = False
        reasons.append(
            f"Max DD worsened by {(override['max_dd']-baseline['max_dd']):.1%}"
        )
    else:
        reasons.append(
            f"Max DD {override['max_dd']:.1%} (baseline {baseline['max_dd']:.1%}, OK)"
        )

    # CAGR: don't degrade more than 2 pp
    delta_cagr = override['cagr'] - baseline['cagr']
    if delta_cagr < -CAGR_DEGRADE_MAX:
        ok = False
        reasons.append(
            f"CAGR degraded by {delta_cagr*100:+.1f}pp (cap: -{CAGR_DEGRADE_MAX*100:.0f}pp)"
        )
    else:
        reasons.append(f"CAGR Δ {delta_cagr*100:+.1f}pp (OK)")

    return ('PASS' if ok else 'FAIL'), reasons


def run_sweep(overrides: dict, *, label: str = 'override',
              capital: float = 100_000, print_report: bool = True) -> dict:
    """Run baseline and override; report side-by-side; return result dict.

    Returns:
      {
        'baseline': {...metrics...},
        'override': {...metrics...},
        'verdict':  'PASS' | 'FAIL',
        'reasons':  [str, ...],
      }
    """
    baseline = _run_one('baseline', {}, capital=capital)
    over = _run_one(label, overrides, capital=capital)
    verdict, reasons = _verdict(baseline, over)

    if print_report:
        print('=' * 78)
        print(f'  Parameter sweep: {label}')
        print(f'  Overrides: {overrides}')
        print('=' * 78)
        rows = [
            ('CAGR',                'cagr',                lambda v: f'{v*100:+.1f}%'),
            ('Sharpe',              'sharpe',              lambda v: f'{v:.2f}'),
            ('Sortino',             'sortino',             lambda v: f'{v:.2f}'),
            ('Max DD',              'max_dd',              lambda v: f'{v*100:.1f}%'),
            ('Final NAV',           'final_nav',           lambda v: f'${v:,.0f}'),
            ('Total premium',       'total_premium',       lambda v: f'${v:,.0f}'),
            ('Avg weekly prem',     'avg_weekly_prem',     lambda v: f'${v:,.0f}'),
            ('Median monthly prem', 'median_monthly_prem', lambda v: f'${v:,.0f}'),
            ('Trade count',         'trade_count',         lambda v: f'{v:,d}'),
        ]
        print(f'{"Metric":24s} {"Baseline":>14s} {"Override":>14s}  Δ')
        print('-' * 78)
        for name, key, fmt in rows:
            b = baseline[key]
            o = over[key]
            try:
                delta = o - b
                d_str = f'{delta:+.4g}'
            except Exception:
                d_str = ''
            print(f'{name:24s} {fmt(b):>14s} {fmt(o):>14s}  {d_str}')
        print()
        print(f'Verdict: {verdict}')
        for r in reasons:
            print(f'  - {r}')
        print('=' * 78)

    return {
        'baseline': baseline,
        'override': over,
        'verdict':  verdict,
        'reasons':  reasons,
    }


def _parse_set(items: list[str]) -> dict:
    """Parse --set KEY=VAL pairs; coerce to int/float when possible."""
    out = {}
    for it in items or []:
        if '=' not in it:
            raise ValueError(f'--set expects KEY=VAL, got: {it!r}')
        k, v = it.split('=', 1)
        k = k.strip()
        v = v.strip()
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                continue
        else:
            out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='UNG param-sweep validation gate')
    p.add_argument('--set', action='append', default=[],
                   help='Override KEY=VAL (repeatable)')
    p.add_argument('--json', type=Path,
                   help='Load overrides from JSON file (dict at root)')
    p.add_argument('--label', default='override',
                   help='Short label for the override run (default: "override")')
    p.add_argument('--capital', type=float, default=100_000)
    p.add_argument('--exit-fail', action='store_true',
                   help='Exit nonzero on FAIL verdict (for CI/cron gating)')
    args = p.parse_args(argv)

    overrides = _parse_set(args.set)
    if args.json:
        overrides.update(json.loads(args.json.read_text()))
    if not overrides:
        print('No overrides supplied. Running baseline vs baseline (sanity check).')

    result = run_sweep(overrides, label=args.label, capital=args.capital)
    if args.exit_fail and result['verdict'] != 'PASS':
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
