"""Stale-strategy retirement lifecycle.

Runs after each backtest cycle. Marks (not deletes) underperforming
strategies as STALE when better variants accumulate. Keeps active set
at target size (default 20). Stale strategies remain in code/git but
are excluded from cycle runs.

Rules:
- Reads latest cycle results JSON.
- Considers a strategy "active" if not in the stale registry.
- If active count > target + threshold, retires (active - target) worst
  by Sharpe (excluding always-keep baselines like naive_atm).
- Logs every retirement with date, names, reason — never deletes.

State file: backtest/strategy_lifecycle.json
- stale: list of strategy names currently STALE
- retirement_log: [{date, retired: [names], reason, cycle_id}]
- always_keep: list of names that can NEVER be retired (baselines)

Usage:
    venv/bin/python backtest/retire_stale.py
    # Hooked into backtest/run_cycle.sh after cycle commits
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
STATE_FILE = THIS_DIR / 'strategy_lifecycle.json'
RESULTS_DIR = THIS_DIR / 'results'

# Strategies that anchor the harness for reference; never retire these.
DEFAULT_ALWAYS_KEEP = {
    'naive_atm',                  # sanity baseline (negative-Sharpe)
    'kelly_firmness',             # high-return / low-Sharpe reference
    'otm_managed',                # minimal-features reference
    'champion_20pct_plus_floor',  # protected-family reference
}

TARGET_ACTIVE = 20
RETIREMENT_THRESHOLD = 3  # retire when active > target + this many


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        'stale': [],
        'retirement_log': [],
        'always_keep': sorted(DEFAULT_ALWAYS_KEEP),
        'target_active': TARGET_ACTIVE,
    }


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _latest_results() -> dict | None:
    """Find the most recent cycle results.

    Cycles write to backtest/results/summary.json (dict by strategy name).
    Cycle ID derived from log directory's newest cycle_*.log.
    """
    summary = RESULTS_DIR / 'summary.json'
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text())
    except Exception:
        return None
    # Cycle id from newest log
    log_dir = RESULTS_DIR / 'logs'
    cycle_id = 'unknown'
    if log_dir.exists():
        logs = sorted(log_dir.glob('cycle_*.log'))
        if logs:
            cycle_id = logs[-1].stem
    return {'cycle_id': cycle_id, 'data': data}


def evaluate_and_retire(dry_run: bool = False) -> dict:
    """Run one retirement evaluation. Returns summary dict.

    Logic:
    - Load state + latest cycle results
    - Filter strategies: active = not stale AND not always_keep
    - If active count > target + threshold, retire (active - target) worst
      by Sharpe (lowest-Sharpe first)
    - Always_keep strategies are never candidates for retirement
    """
    state = _load_state()
    stale = set(state.get('stale', []))
    always_keep = set(state.get('always_keep', list(DEFAULT_ALWAYS_KEEP)))
    target = int(state.get('target_active', TARGET_ACTIVE))

    latest = _latest_results()
    if not latest:
        return {'action': 'NOOP', 'reason': 'no cycle results found'}

    results = latest['data']
    if isinstance(results, dict):
        # Filter out top-level scalar keys (aggregate stats); keep strategy dicts
        rows = [{'name': k, **v} for k, v in results.items()
                if isinstance(v, dict) and 'sharpe' in v]
    elif isinstance(results, list):
        rows = results
    else:
        return {'action': 'NOOP', 'reason': 'unsupported results format'}

    active = [r for r in rows
              if r.get('name') not in stale
              and r.get('name') not in always_keep]
    n_active = len(active)

    if n_active <= target + RETIREMENT_THRESHOLD:
        return {
            'action': 'NOOP',
            'reason': f'active={n_active} ≤ target+threshold={target+RETIREMENT_THRESHOLD}',
            'active_count': n_active,
            'stale_count': len(stale),
        }

    # Need to retire: pick (n_active - target) lowest-Sharpe
    to_retire_count = n_active - target
    by_sharpe = sorted(active, key=lambda r: float(r.get('sharpe', 0) or 0))
    to_retire = [r['name'] for r in by_sharpe[:to_retire_count]]

    log_entry = {
        'date': date.today().isoformat(),
        'cycle_id': latest['cycle_id'],
        'retired': to_retire,
        'retired_with_sharpe': {r['name']: r.get('sharpe', 0) for r in by_sharpe[:to_retire_count]},
        'reason': (f'active count {n_active} exceeded target+threshold '
                   f'{target+RETIREMENT_THRESHOLD}; retired {to_retire_count} lowest-Sharpe'),
        'still_active_count': n_active - to_retire_count,
    }

    if dry_run:
        return {'action': 'DRY_RUN', **log_entry}

    # Persist
    stale.update(to_retire)
    state['stale'] = sorted(stale)
    state['retirement_log'].append(log_entry)
    _save_state(state)

    return {'action': 'RETIRED', **log_entry}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without persisting')
    parser.add_argument('--show', action='store_true',
                        help='Show current stale registry and exit')
    args = parser.parse_args()

    if args.show:
        s = _load_state()
        print(f'Stale ({len(s["stale"])}): {", ".join(s["stale"]) or "(none)"}')
        print(f'Always-keep ({len(s["always_keep"])}): {", ".join(s["always_keep"])}')
        print(f'Target active: {s.get("target_active", TARGET_ACTIVE)}')
        print(f'\nRetirement log ({len(s["retirement_log"])}):')
        for e in s['retirement_log'][-10:]:
            print(f'  {e["date"]} [{e["cycle_id"]}] retired {e["retired"]}')
        sys.exit(0)

    result = evaluate_and_retire(dry_run=args.dry_run)
    print(json.dumps(result, indent=2, default=str))
