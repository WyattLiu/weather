"""Daily summary digest from trading_actions.jsonl.

Aggregates:
  - Actions attempted/submitted/filled in last 24h
  - Best-play decision history (what kernel said, what we did)
  - Outstanding orders + age
  - Realized credits + NAV trajectory

Usage:
  python live/daily_digest.py           # last 24h, stdout
  python live/daily_digest.py --hours 168  # last week
  python live/daily_digest.py --markdown   # markdown formatted (for email/Slack)
"""
from __future__ import annotations
import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from collections import Counter, defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
from trading_log import LOG_PATH


def load_entries(hours: int = 24) -> list:
    if not os.path.exists(LOG_PATH):
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out = []
    with open(LOG_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
                t = datetime.strptime(e['ts'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
                if t < cutoff: continue
                e['_t'] = t
                out.append(e)
            except Exception:
                continue
    return out


def digest(hours: int = 24, markdown: bool = False) -> str:
    entries = load_entries(hours)
    if not entries:
        return f'No trading activity in last {hours}h.'

    actions = Counter(e.get('action_taken', '?') for e in entries)
    modes = Counter(e.get('mode', '?') for e in entries)
    best_play_kinds = Counter(
        (e.get('verdict_best_play') or {}).get('order_type', '?') for e in entries
    )
    submitted = [e for e in entries if e.get('action_taken') in ('submitted', 'submitted_partial')]
    consult = [e for e in entries if e.get('action_taken') == 'consult_required']
    escalations = [e for e in entries if e.get('action_taken') == 'escalation_sweep']

    # NAV trajectory
    navs = [(e['_t'], e.get('nav')) for e in entries if e.get('nav') is not None]
    spots = [(e['_t'], e.get('spot')) for e in entries if e.get('spot') is not None]
    first_nav = navs[0][1] if navs else None
    last_nav = navs[-1][1] if navs else None
    first_spot = spots[0][1] if spots else None
    last_spot = spots[-1][1] if spots else None

    fmt = '## ' if markdown else '=== '
    end = ' ##' if markdown else ' ==='
    bold_l, bold_r = ('**', '**') if markdown else ('', '')
    lines = []
    lines.append(f'{fmt}Trading Digest — last {hours}h{end}')
    lines.append('')
    lines.append(f'{bold_l}Cycles run:{bold_r} {len(entries)}')
    lines.append(f'{bold_l}Modes:{bold_r} {dict(modes)}')
    lines.append(f'{bold_l}Actions:{bold_r} {dict(actions)}')
    lines.append('')
    lines.append(f'{bold_l}Kernel best-play distribution:{bold_r}')
    for k, n in best_play_kinds.most_common():
        lines.append(f'  - {k}: {n}')
    lines.append('')
    if last_spot and first_spot:
        delta_spot = (last_spot - first_spot) / first_spot * 100
        lines.append(f'{bold_l}UNG spot:{bold_r} ${first_spot:.2f} → ${last_spot:.2f} ({delta_spot:+.2f}%)')
    if last_nav and first_nav:
        delta_nav = last_nav - first_nav
        delta_nav_pct = delta_nav / first_nav * 100
        lines.append(f'{bold_l}NAV:{bold_r} ${first_nav:,.0f} → ${last_nav:,.0f} ({delta_nav:+,.0f} / {delta_nav_pct:+.2f}%)')
    lines.append('')

    if submitted:
        lines.append(f'{bold_l}Orders submitted ({len(submitted)}):{bold_r}')
        for e in submitted[-10:]:
            for so in e.get('submitted_orders', []) or []:
                lines.append(f'  - {e["ts"]}: {so.get("side", "?")} {so.get("qty", "?")}× '
                             f'{so.get("symbol_human", so.get("symbol","?"))} @ ${so.get("limit_price", "?")} '
                             f'(ext {so.get("external_id", "?")})')
        lines.append('')

    if consult:
        lines.append(f'{bold_l}⚠️ Consult-required actions ({len(consult)}):{bold_r}')
        for e in consult:
            lines.append(f'  - {e["ts"]}: {e.get("notes", "")[:150]}')
        lines.append('')

    if escalations:
        lines.append(f'{bold_l}Escalation events ({len(escalations)}):{bold_r}')
        for e in escalations:
            evs = e.get('events') or []
            for ev in evs:
                lines.append(f'  - {ev.get("event","?")}: {ev.get("symbol", "?")} '
                             f'{ev.get("from_tier","")} → {ev.get("to_tier","ABANDON")}  age {ev.get("age_min","?")}min')
        lines.append('')

    # Recent best-play rationale (1 line per unique kind)
    lines.append(f'{bold_l}Recent best-play rationale samples:{bold_r}')
    seen_kinds = set()
    for e in reversed(entries):
        vbp = e.get('verdict_best_play') or {}
        k = vbp.get('order_type')
        if k and k not in seen_kinds:
            seen_kinds.add(k)
            rat = vbp.get('rationale', '')[:160]
            lines.append(f'  - [{k}]: {rat}')
            if len(seen_kinds) >= 4: break

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hours', type=int, default=24)
    p.add_argument('--markdown', action='store_true')
    args = p.parse_args()
    print(digest(hours=args.hours, markdown=args.markdown))


if __name__ == '__main__':
    main()
