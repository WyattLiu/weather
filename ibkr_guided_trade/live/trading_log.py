"""Append-only JSONL log of every trading action — planned, attempted, filled.

This is the forward-data store we've been missing for backtest validation.
Every cron firing writes one or more entries describing:
- the kernel verdict at that moment
- the action planned
- whether it was paper or live
- if live: order id, fills, eventual outcome

Format (one JSON per line):
    {
      "ts": "2026-06-05T11:30:00Z",
      "kernel": "champion_premium_harvest_scale_invariant",
      "spot": 11.79,
      "nav": 121670,
      "surge_z": 1.21,
      "positions_snapshot": {"shares": 4100, "short_calls": 41, "short_puts": 53},
      "verdict_best_play": {"order_type": "PUT_SHORT_MIX", "ev": 234, "legs": [...]},
      "mode": "paper" | "live",
      "action_taken": "planned_only" | "submitted" | "skipped_existing" | "failed",
      "order_ids": ["abc-123", ...],     # ws external_ids if submitted
      "notes": "...",
    }
"""
from __future__ import annotations
import os
import json
from datetime import datetime, timezone

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, 'trading_actions.jsonl')


def log_action(entry: dict) -> None:
    """Append one structured entry. Timestamp auto-added if missing."""
    if 'ts' not in entry:
        entry['ts'] = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    line = json.dumps(entry, default=str)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')


def tail(n: int = 20) -> list:
    """Return last n entries (most recent first)."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = f.readlines()
    out = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    out.reverse()
    return out


def summary() -> dict:
    """Quick metrics on the log: count by mode and action type."""
    if not os.path.exists(LOG_PATH):
        return {'total': 0}
    from collections import Counter
    modes = Counter()
    actions = Counter()
    order_types = Counter()
    with open(LOG_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
                modes[e.get('mode', 'unknown')] += 1
                actions[e.get('action_taken', 'unknown')] += 1
                vbp = e.get('verdict_best_play') or {}
                order_types[vbp.get('order_type', 'NONE')] += 1
            except Exception:
                continue
    return {
        'total': sum(modes.values()),
        'by_mode': dict(modes),
        'by_action': dict(actions),
        'by_order_type': dict(order_types),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        print(json.dumps(summary(), indent=2))
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        for e in tail(n):
            print(json.dumps(e, indent=2))
