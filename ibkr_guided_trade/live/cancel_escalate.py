"""Cancel + escalate logic for unfilled orders.

On each cycle, before submitting NEW orders:
  1. Read trading_actions.jsonl — find orders submitted in last 24h
  2. For each, check WS open-orders list
  3. If still pending AND age > escalation_threshold_min:
       - Cancel
       - Tag in log: 'escalation_pending' — next cycle will resubmit at
         a more aggressive ladder tier

Escalation ladder (tier index → name → max age before escalating):
  tier 0 → passive    →  90 min wait (best premium)
  tier 1 → near-mid   →  60 min wait
  tier 2 → mid        →  30 min wait
  tier 3 → cross      →  no escalation (already crossed the spread)

If tier 3 sits unfilled for 60+ min, mark as 'GAVE_UP' (something changed:
spread widened beyond expected, market moved, etc.) — don't resubmit.
"""
from __future__ import annotations
import os
import sys
import json
import time
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT); sys.path.insert(0, THIS_DIR)

from trading_log import LOG_PATH, log_action
from ws_safe import safe_call, cancel_order_safe

ESCALATION_AGE_MIN = {0: 90, 1: 60, 2: 30}  # by tier
ABANDON_AGE_MIN = 60  # tier 3 (cross) max age before abandoning


def load_recent_submissions(hours: int = 24) -> List[dict]:
    """Pull recent submitted-order entries from trading log."""
    if not os.path.exists(LOG_PATH):
        return []
    cutoff_ts = time.time() - hours * 3600
    out = []
    with open(LOG_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get('action_taken') not in ('submitted', 'submitted_partial'):
                    continue
                # Parse ts back to epoch
                from datetime import datetime
                t = datetime.strptime(e['ts'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
                if t < cutoff_ts: continue
                e['_submitted_ts'] = t
                out.append(e)
            except Exception:
                continue
    return out


def check_and_escalate(live: bool = False) -> List[dict]:
    """Find pending orders past their tier deadline; cancel and tag for resubmit.
    Returns list of escalation events."""
    from ws_sdk import WSClient
    try:
        ws = WSClient()
    except Exception as e:
        log_action({'action_taken': 'escalate_skipped', 'notes': f'WS init failed: {e}'})
        return []

    # Fetch current open orders once
    try:
        open_orders = safe_call(ws.list_open_orders, verify=False) or []
    except Exception as e:
        log_action({'action_taken': 'escalate_skipped', 'notes': f'open orders fetch failed: {e}'})
        return []

    open_by_ext = {}
    for o in open_orders:
        ext = getattr(o, 'external_id', None) or getattr(o, 'id', None)
        if ext:
            open_by_ext[ext] = o

    events = []
    recent = load_recent_submissions(hours=24)
    for entry in recent:
        for sub in entry.get('submitted_orders', []) or []:
            ext = sub.get('external_id')
            if not ext or ext == '?':
                continue
            if ext not in open_by_ext:
                continue  # already filled or cancelled
            tier_kind = sub.get('tier', 'passive')
            tier_idx = {'passive': 0, 'near-mid': 1, 'mid': 2, 'cross': 3}.get(tier_kind, 0)
            age_min = (time.time() - entry['_submitted_ts']) / 60
            threshold = ESCALATION_AGE_MIN.get(tier_idx, ABANDON_AGE_MIN)
            if age_min < threshold:
                continue
            # Past threshold — escalate or abandon
            if tier_idx >= 3:
                # cross tier sat too long → abandon
                if age_min < ABANDON_AGE_MIN:
                    continue
                if live:
                    ok = cancel_order_safe(ws, ext)
                else:
                    ok = True  # simulate
                events.append({
                    'event': 'abandoned', 'ext_id': ext,
                    'tier': tier_kind, 'age_min': round(age_min, 1),
                    'cancelled': ok, 'mode': 'live' if live else 'paper',
                    'symbol': sub.get('symbol_human', sub.get('symbol', '?')),
                })
            else:
                # Cancel, then mark for resubmit at next tier next cycle
                next_tier = ['near-mid', 'mid', 'cross'][tier_idx]
                if live:
                    ok = cancel_order_safe(ws, ext)
                else:
                    ok = True
                events.append({
                    'event': 'escalation', 'ext_id': ext,
                    'from_tier': tier_kind, 'to_tier': next_tier,
                    'age_min': round(age_min, 1),
                    'cancelled': ok, 'mode': 'live' if live else 'paper',
                    'symbol': sub.get('symbol_human', sub.get('symbol', '?')),
                })
    if events:
        log_action({
            'action_taken': 'escalation_sweep',
            'mode': 'live' if live else 'paper',
            'events': events,
            'notes': f'{len(events)} order(s) escalated/abandoned',
        })
    return events


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--live', action='store_true')
    args = p.parse_args()
    live = args.live and os.environ.get('KERNEL_LIVE') == '1'
    evs = check_and_escalate(live=live)
    if not evs:
        print('no escalations needed')
    else:
        for e in evs:
            print(f'[{e["event"]}] {e["symbol"]} tier={e.get("from_tier", e.get("tier"))} → {e.get("to_tier","ABANDON")}  age {e["age_min"]}min')
