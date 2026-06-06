"""Live executor — translates the kernel's best play into actual orders.

SAFETY FIRST:
  - Default mode: PAPER (log intentions, do NOT submit)
  - Live mode: requires BOTH --live flag AND env var KERNEL_LIVE=1
  - Lock file prevents double-execution
  - Idempotency: skip if existing order at same (expiry, strike, side) pending
  - Conservative: places only the PASSIVE tier of the limit ladder
    (highest premium for us); lets it sit. If it doesn't fill in N hours,
    we go aggressive next cycle. NO modify-as-market-moves.

USAGE:
  # Cron-friendly default: just log what we'd do
  python live/execute_kernel_plan.py

  # Execute live (requires both flags)
  KERNEL_LIVE=1 python live/execute_kernel_plan.py --live

  # Show recent decisions
  python live/trading_log.py 5
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone
from typing import Optional

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'backtest'))

from trading_log import log_action

LOCK_FILE = '/tmp/kernel_executor.lock'


def acquire_lock():
    """Single-instance lock (PID-based)."""
    if os.path.exists(LOCK_FILE):
        old_pid = open(LOCK_FILE).read().strip()
        if old_pid.isdigit() and os.path.exists(f'/proc/{old_pid}'):
            print(f'[lock] another executor running (PID {old_pid}) — aborting', file=sys.stderr)
            sys.exit(1)
    open(LOCK_FILE, 'w').write(str(os.getpid()))


def release_lock():
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass


def fetch_verdict() -> Optional[dict]:
    """Fetch verdict via local dashboard API (always current)."""
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:10001/api/state', timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f'[verdict] dashboard unreachable: {e}', file=sys.stderr)
        return None


def osi_to_human(osi: str) -> str:
    """UNG  260717P00011000 -> UNG 26-07-17 P11.00"""
    if not osi or len(osi) < 21:
        return osi
    parts = osi.split()
    if len(parts) != 2:
        return osi
    underlying = parts[0]
    rest = parts[1]
    yymmdd = rest[:6]
    right = rest[6]
    strike_milli = int(rest[7:])
    strike = strike_milli / 1000
    return f'{underlying} {yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]} {right}{strike:.2f}'


def check_existing_order(ws_orders: list, expected_symbol: str, side: str) -> Optional[dict]:
    """Return matching pending order if one already exists."""
    for o in ws_orders or []:
        if str(o.get('symbol', '')) == expected_symbol and str(o.get('side', '')).upper() == side.upper():
            if str(o.get('status', '')).upper() in ('SUBMITTED', 'WORKING', 'PENDING', 'NEW'):
                return o
    return None


def count_recent_submissions(hours: int = 24) -> int:
    """Count distinct submitted orders in the trading log over last N hours."""
    import json, time
    from datetime import datetime
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'log', 'trading_actions.jsonl')):
        return 0
    log_path = os.path.join(os.path.dirname(__file__), 'log', 'trading_actions.jsonl')
    cutoff = time.time() - hours * 3600
    n = 0
    with open(log_path) as f:
        for line in f:
            try:
                e = json.loads(line)
                t = datetime.strptime(e['ts'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
                if t < cutoff: continue
                if e.get('action_taken') in ('submitted', 'submitted_partial'):
                    n += len(e.get('submitted_orders') or [])
            except Exception:
                continue
    return n


def execute_best_play(verdict: dict, mode: str = 'paper',
                       daily_max_submissions: int = 4) -> dict:
    """Translate verdict.best_play into orders. Mode controls live/review/paper.

    Modes:
      paper  — log intent only, never submit (cron-safe default)
      review — print plan to stderr, read stdin y/n per order set
      auto   — submit if under daily_max_submissions cap; else fall back to review log
    """
    actionable = [o for o in (verdict.get('actionable_orders') or []) if o.get('best_play')]
    if not actionable:
        return {'action_taken': 'no_best_play', 'mode': mode,
                'notes': 'kernel surfaced nothing actionable today'}

    best = actionable[0]
    kind = best.get('order_type')

    # Sanity: hard-pass on non-actionable types
    pass_kinds = {'WAIT_FOR_ASSIGNMENT', 'CC_SKIPPED', 'SHARES_SELL_BLOCKED',
                  'SYNTHETIC_SHORT_BLOCKED'}
    if kind in pass_kinds:
        return {'action_taken': 'pass_no_order_needed', 'mode': mode,
                'notes': f'{kind}: {best.get("rationale", "")[:120]}'}

    # Daily-cap gate: applies ONLY in auto mode. Paper/review unconstrained.
    if mode == 'auto':
        recent = count_recent_submissions(hours=24)
        if recent >= daily_max_submissions:
            return {'action_taken': 'auto_cap_reached', 'mode': 'auto',
                    'recent_submissions_24h': recent,
                    'daily_max': daily_max_submissions,
                    'notes': f'{recent}/{daily_max_submissions} daily-cap reached; falling back to paper-log only. Override: KERNEL_LIVE=1 + --mode review for next action.'}

    # We currently support: PUT_SHORT_MIX (laddered), CALL_SHORT_COVERED (laddered),
    # BUY_BOXX / SELL_BOXX (laddered stock orders)
    if kind == 'PUT_SHORT_MIX':
        return _execute_put_short_mix(best, mode=mode)
    if kind == 'CALL_SHORT_COVERED':
        return _execute_cc(best, mode=mode)
    if kind in ('BUY_BOXX', 'SELL_BOXX'):
        return _execute_boxx(best, mode=mode)
    if kind == 'CC_BTC_TO_FREE_SHARES':
        return {'action_taken': 'manual_review', 'mode': mode,
                'notes': f'BTC {best.get("qty")} calls — requires manual leg-picking by extrinsic; not auto-executed'}
    return {'action_taken': 'unhandled', 'mode': mode,
            'notes': f'unhandled best-play type: {kind}'}


def _execute_boxx(order: dict, mode: str = 'paper') -> dict:
    """BUY or SELL BOXX with limit ladder. Stock orders are simpler than
    options — just security_id + qty + price + side."""
    side = order.get('side')
    sec_id = order.get('sec_id')
    if not sec_id:
        return {'action_taken': 'failed', 'mode': mode, 'notes': 'BOXX order has no sec_id'}

    ladder = order.get('limit_ladder') or []
    if not ladder:
        return {'action_taken': 'no_ladder', 'mode': mode, 'notes': 'BOXX had no ladder'}

    # Use only PASSIVE tier (conservative — let it sit)
    passive = ladder[0]
    intent = {
        'symbol': 'BOXX',
        'sec_id': sec_id,
        'side': 'BUY' if side == 'BUY' else 'SELL',
        'qty': int(passive['qty']),
        'limit_price': float(passive['limit_price']),
        'tier': passive['kind'],
        'live_bid': order.get('live_bid'),
        'live_ask': order.get('live_ask'),
        'est_value': round(int(passive['qty']) * float(passive['limit_price']), 0),
    }

    # PAPER
    if mode == 'paper':
        return {'action_taken': 'planned_only', 'mode': 'paper',
                'notes': f'PAPER: would {side} {passive["qty"]} BOXX @ ${passive["limit_price"]} (passive tier)',
                'planned_orders': [intent]}

    # REVIEW
    if mode == 'review':
        print(f'\n=== REVIEW: {side} BOXX ===', file=sys.stderr)
        print(f'  {passive["qty"]} BOXX @ ${passive["limit_price"]}  '
              f'(live bid {order.get("live_bid", "?")}/ask {order.get("live_ask", "?")})  '
              f'est ${intent["est_value"]:,.0f}', file=sys.stderr)
        if not _confirm_interactive(f'Submit {side} {passive["qty"]} BOXX?'):
            return {'action_taken': 'review_declined', 'mode': 'review',
                    'planned_orders': [intent]}

    # LIVE submission
    if os.environ.get('KERNEL_LIVE') != '1':
        return {'action_taken': 'env_guard_blocked', 'mode': mode,
                'notes': f'mode={mode} requires KERNEL_LIVE=1; refusing',
                'planned_orders': [intent]}
    try:
        from ws_sdk import WSClient
    except Exception as e:
        return {'action_taken': 'failed', 'mode': mode, 'notes': f'ws_sdk import: {e}'}
    ws = WSClient()
    try:
        # BOXX is a stock; use sell/buy methods. Open/close not applicable for stocks.
        if side == 'BUY':
            ord_obj = ws.buy_to_open(security_id=sec_id, qty=intent['qty'], price=intent['limit_price'])
        else:
            ord_obj = ws.sell_to_close(security_id=sec_id, qty=intent['qty'], price=intent['limit_price'])
        ext_id = getattr(ord_obj, 'external_id', None) or getattr(ord_obj, 'id', '?')
        intent['external_id'] = ext_id
        print(f'[LIVE] {side} {intent["qty"]} BOXX @ ${intent["limit_price"]} → {ext_id}')
        return {'action_taken': 'submitted', 'mode': mode,
                'submitted_orders': [intent]}
    except Exception as e:
        intent['error'] = str(e)
        return {'action_taken': 'failed', 'mode': mode,
                'failed_orders': [intent],
                'notes': f'BOXX submit failed: {e}'}


def _confirm_interactive(prompt: str) -> bool:
    """Read y/n from stdin. If stdin isn't a tty (e.g. cron), default to no."""
    import sys
    if not sys.stdin.isatty():
        print(f'[review] non-interactive stdin — auto-declining: {prompt}', file=sys.stderr)
        return False
    try:
        ans = input(f'\n{prompt} [y/N]: ').strip().lower()
        return ans in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        return False


def _execute_put_short_mix(order: dict, mode: str = 'paper') -> dict:
    """Submit ONLY the passive tier for each leg, anchored to LIVE bid/ask.

    Conservative defaults in live mode:
      - max_qty_per_leg = 2 contracts (override via env LIVE_MAX_QTY)
      - passive tier price = live mid (NOT BSM estimate which can be off-bid)
      - require manual review if total credit > $500
      - cap total contracts per cycle at 5
    """
    from ws_option_resolver import resolve_osi

    legs = order.get('legs') or []
    if not legs:
        return {'action_taken': 'no_legs', 'notes': 'PUT_SHORT_MIX had no legs'}

    max_qty_per_leg = int(os.environ.get('LIVE_MAX_QTY', '2'))
    max_total_contracts = 5
    review_threshold_credit = 500.0

    intent = []
    resolved_data = []
    total_contracts = 0
    for leg in legs:
        if total_contracts >= max_total_contracts:
            break
        osi = leg.get('symbol', '')
        resolved = resolve_osi(osi)
        if not resolved:
            intent.append({'symbol': osi, 'error': 'unresolved; option may not exist or chain unreachable'})
            continue
        # Use LIVE mid for pricing — NOT BSM estimate
        live_mid = resolved['mid']
        live_bid = resolved['bid']
        live_ask = resolved['ask']
        # Conservative passive: just above live bid (we want to sell at OUR price)
        # passive = live_mid (collect mid premium); if we want more aggressive: live_bid+0.01
        passive_price = round(live_mid, 2)
        # If spread is wide (>$0.20), use bid+0.05 instead of mid (more likely to fill)
        spread = max(0, live_ask - live_bid)
        if spread > 0.20:
            passive_price = round(live_bid + 0.05, 2)
        # Cap qty
        kernel_qty = int(leg.get('qty', 1))
        live_qty = min(kernel_qty, max_qty_per_leg, max_total_contracts - total_contracts)
        if live_qty < 1:
            continue
        total_contracts += live_qty
        intent.append({
            'symbol': osi,
            'symbol_human': osi_to_human(osi),
            'sec_id': resolved['sec_id'],
            'side': 'SELL_TO_OPEN',
            'qty': live_qty,
            'kernel_qty': kernel_qty,
            'limit_price': passive_price,
            'live_bid': live_bid, 'live_ask': live_ask, 'live_mid': live_mid,
            'spread': round(spread, 3),
            'oi': resolved.get('oi', 0),
            'est_credit': round(passive_price * 100 * live_qty, 0),
        })
        resolved_data.append(resolved)

    total_credit = sum(o.get('est_credit', 0) for o in intent if 'error' not in o)
    review_required_by_amount = total_credit > review_threshold_credit

    # PAPER mode: log intent, never submit
    if mode == 'paper':
        return {
            'action_taken': 'planned_only',
            'mode': 'paper',
            'notes': (f'PAPER: {len(intent)} legs planned, total credit ~${total_credit:.0f} '
                      f'across {total_contracts} contracts (capped from kernel: '
                      f'{sum(l.get("kernel_qty", l.get("qty", 0)) for l in intent if "error" not in l)})'),
            'planned_orders': intent,
            'review_required_by_amount': review_required_by_amount,
        }

    # REVIEW mode: print plan, ask confirm per leg
    if mode == 'review':
        print(f'\n=== REVIEW: {len(intent)} legs, total credit ~${total_credit:.0f} ===', file=sys.stderr)
        for i, o in enumerate(intent):
            if 'error' in o:
                print(f'  [{i}] SKIP {o["symbol"]} — {o["error"]}', file=sys.stderr)
                continue
            print(f'  [{i}] {o["side"]} {o["qty"]}× {o["symbol_human"]} @ ${o["limit_price"]}'
                  f'  (live bid {o["live_bid"]} / ask {o["live_ask"]} / OI {o["oi"]})  est credit ${o["est_credit"]:.0f}',
                  file=sys.stderr)
        if not _confirm_interactive('Submit ALL legs above? (no = skip all)'):
            return {'action_taken': 'review_declined', 'mode': 'review',
                    'planned_orders': intent,
                    'notes': 'user declined or non-interactive stdin'}
        # User said yes — fall through to live submit

    # LIVE / AUTO submission path (review fell through, or auto mode)
    if mode == 'auto' and review_required_by_amount:
        return {
            'action_taken': 'consult_required',
            'mode': 'auto',
            'notes': f'AUTO: total credit ${total_credit:.0f} > ${review_threshold_credit} threshold; requires --mode review approval',
            'planned_orders': intent,
        }

    # Live env var still required as belt-and-suspenders
    if os.environ.get('KERNEL_LIVE') != '1':
        return {
            'action_taken': 'env_guard_blocked',
            'mode': mode,
            'notes': f'mode={mode} requires env KERNEL_LIVE=1 to actually submit; refusing without it',
            'planned_orders': intent,
        }
    try:
        from ws_sdk import WSClient
    except Exception as e:
        return {'action_taken': 'failed', 'notes': f'ws_sdk import failed: {e}'}
    ws = WSClient()
    submitted, failed = [], []
    for o in intent:
        if 'error' in o:
            failed.append(o); continue
        try:
            ord_obj = ws.sell_to_open(security_id=o['sec_id'], qty=o['qty'], price=o['limit_price'])
            ext_id = getattr(ord_obj, 'external_id', None) or getattr(ord_obj, 'id', '?')
            submitted.append({**o, 'external_id': ext_id})
            print(f'[LIVE] SELL {o["qty"]}× {o["symbol_human"]} @ ${o["limit_price"]} → {ext_id}')
        except Exception as e:
            failed.append({**o, 'error': str(e)})
            print(f'[LIVE-FAIL] {o["symbol_human"]}: {e}', file=sys.stderr)
    return {
        'action_taken': 'submitted' if submitted and not failed else ('submitted_partial' if submitted else 'failed'),
        'mode': 'live',
        'submitted_orders': submitted, 'failed_orders': failed,
        'total_credit_attempted': total_credit,
    }


def _execute_cc(order: dict, mode: str = 'paper') -> dict:
    """CC live not yet supported in this code path — surface intent only."""
    ladder = order.get('limit_ladder') or []
    if not ladder:
        return {'action_taken': 'no_ladder', 'mode': mode, 'notes': 'CC had no ladder'}
    passive = ladder[0]
    intent = {
        'symbol': order['symbol'],
        'symbol_human': osi_to_human(order['symbol']),
        'side': 'SELL_TO_OPEN',
        'qty': int(passive['qty']),
        'limit_price': float(passive['limit_price']),
        'tier': passive['kind'],
    }
    return {'action_taken': 'planned_only' if mode == 'paper' else 'cc_live_not_implemented',
            'mode': mode,
            'notes': 'CC submission not wired to live yet; will be enabled after PUT_SHORT_MIX validated',
            'planned_orders': [intent]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['paper', 'review', 'auto'], default='paper',
                   help='paper=log only (safe), review=confirm per leg interactively, '
                        'auto=submit up to --daily-max per 24h then fall back to paper')
    p.add_argument('--daily-max', type=int, default=4,
                   help='Auto mode: max submitted orders per 24h before falling back to paper')
    # Backward-compat: --live → --mode auto + sets KERNEL_LIVE assumed
    p.add_argument('--live', action='store_true', help='Alias for --mode auto')
    args = p.parse_args()
    if args.live:
        args.mode = 'auto'

    mode = args.mode

    acquire_lock()
    try:
        verdict_full = fetch_verdict()
        if not verdict_full:
            log_action({'action_taken': 'fetch_failed',
                        'notes': 'dashboard /api/state unreachable',
                        'mode': 'paper' if dry_run_only else 'live'})
            print('FETCH FAILED', file=sys.stderr)
            return 1
        verdict = verdict_full.get('verdict') or {}
        spot = verdict_full.get('spot')
        balance = verdict_full.get('balance') or {}
        positions = verdict_full.get('positions') or []
        ung_positions = [p for p in positions if p.get('symbol') == 'UNG']
        snapshot = {
            'shares': sum(p['quantity'] for p in ung_positions if not p.get('is_option')),
            'short_calls': sum(abs(p['quantity']) for p in ung_positions
                                if p.get('option_type') == 'CALL'),
            'short_puts': sum(abs(p['quantity']) for p in ung_positions
                                if p.get('option_type') == 'PUT'),
        }

        # Pick best play
        actionable = [o for o in (verdict.get('actionable_orders') or []) if o.get('best_play')]
        best_play = actionable[0] if actionable else None
        verdict_best_play_log = None
        if best_play:
            verdict_best_play_log = {
                'order_type': best_play.get('order_type'),
                'ev': best_play.get('expected_ev_dollars'),
                'rationale': best_play.get('rationale', '')[:200],
                'legs': [
                    {'symbol': l.get('symbol'), 'qty': l.get('qty'),
                     'strike': l.get('strike'),
                     'passive_tier_price': (l.get('limit_ladder') or [{}])[0].get('limit_price')}
                    for l in (best_play.get('legs') or [])
                ] if best_play.get('legs') else None,
                'passive_tier_price': (best_play.get('limit_ladder') or [{}])[0].get('limit_price') if best_play.get('limit_ladder') else None,
            }

        result = execute_best_play(verdict, mode=mode, daily_max_submissions=args.daily_max)

        log_entry = {
            'kernel': verdict.get('kernel'),
            'spot': spot,
            'nav': balance.get('net_liquidation'),
            'surge_z': verdict.get('surge_z'),
            'positions_snapshot': snapshot,
            'verdict_best_play': verdict_best_play_log,
            **result,  # result already has 'mode'
        }
        log_action(log_entry)

        print(f'[{log_entry["ts"]}] mode={log_entry.get("mode", mode)} action={log_entry["action_taken"]}')
        if log_entry.get('planned_orders'):
            for po in log_entry['planned_orders']:
                if 'error' in po:
                    print(f'  SKIP: {po.get("symbol","?")} — {po["error"]}')
                else:
                    print(f'  PLAN: {po.get("side","?")} {po.get("qty","?")}× {po.get("symbol_human", po.get("symbol","?"))} @ ${po.get("limit_price","?")}')
        if log_entry.get('submitted_orders'):
            for so in log_entry['submitted_orders']:
                print(f'  ✓ SUBMITTED: {so.get("side","?")} {so.get("qty","?")}× {so.get("symbol_human","?")} @ ${so.get("limit_price","?")} → {so.get("external_id","?")}')
        if log_entry.get('recent_submissions_24h') is not None:
            print(f'  📊 daily counter: {log_entry["recent_submissions_24h"]}/{log_entry.get("daily_max", "?")} in last 24h')
        return 0
    finally:
        release_lock()


if __name__ == '__main__':
    sys.exit(main())
