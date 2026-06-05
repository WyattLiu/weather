"""Manual CLI — same underlying code as automated executor, eyes-open mode.

Use this when you want to learn what the engine would do, do it step by
step, or do a one-off trade outside the kernel's recommendation.

Subcommands:
  plan        Show what the kernel WOULD do this cycle (no submission)
  submit      Run executor in --mode review (confirm per leg)
  auto        Run executor in --mode auto (cap=N, defaults to 4)
  orders      List current open WS orders
  cancel <ext_id>   Cancel a specific order (with verification)
  sell-put <expiry> <strike> <qty> <price>     Manual single-leg put-sell
  positions   Show current UNG positions
  digest      Daily digest (last 24h)
  log [N]     Tail last N trading-log entries (default 10)

Examples:
  python live/manual.py plan
  python live/manual.py orders
  python live/manual.py submit                              # interactive
  KERNEL_LIVE=1 python live/manual.py auto --daily-max 4   # auto cap 4/day
  KERNEL_LIVE=1 python live/manual.py sell-put 2026-07-17 11 1 0.38
"""
from __future__ import annotations
import os
import sys
import json
import argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT); sys.path.insert(0, THIS_DIR)


def cmd_plan(args):
    """Show what the kernel would do — paper mode equivalent."""
    from execute_kernel_plan import fetch_verdict, execute_best_play
    v = fetch_verdict()
    if not v: print('fetch failed'); return 1
    verdict = v.get('verdict') or {}
    print(f"spot ${v.get('spot')}  surge_z {verdict.get('surge_z'):.2f}")
    print(f"kernel: {verdict.get('kernel')}")
    print(f"shares {verdict.get('current_shares')} / short_calls {verdict.get('current_short_calls')} / short_puts {verdict.get('current_short_puts')}")
    print()
    actionable = [o for o in (verdict.get('actionable_orders') or [])]
    best = [o for o in actionable if o.get('best_play')]
    print('=== ALL ACTIONABLE ORDERS (best_play first) ===')
    for o in best + [o for o in actionable if not o.get('best_play')]:
        star = ' ⭐' if o.get('best_play') else ''
        print(f"[{o['order_type']}] EV ${o.get('expected_ev_dollars', 0):+.0f} score {o.get('ranked_score', 0):+.0f}{star}")
        print(f"  rationale: {(o.get('rationale') or '')[:200]}")
        if o.get('legs'):
            for l in o['legs']:
                print(f"  leg: {l.get('qty')}× ${l.get('strike')} ({l.get('effective_otm_pct')}% OTM) ~${l.get('est_premium_per')}")
        print()


def cmd_submit(args):
    """Same as executor --mode review (interactive)."""
    from execute_kernel_plan import main as exec_main
    sys.argv = ['execute_kernel_plan.py', '--mode', 'review']
    return exec_main()


def cmd_auto(args):
    from execute_kernel_plan import main as exec_main
    sys.argv = ['execute_kernel_plan.py', '--mode', 'auto', '--daily-max', str(args.daily_max)]
    return exec_main()


def cmd_orders(args):
    from ws_sdk import WSClient
    from ws_safe import safe_call
    try:
        ws = WSClient()
        orders = safe_call(ws.list_open_orders, verify=False) or []
    except Exception as e:
        print(f'ERROR: {e}'); return 1
    if not orders:
        print('No open orders.')
        return 0
    print(f'=== {len(orders)} OPEN ORDERS ===')
    for o in orders:
        ext = getattr(o, 'external_id', None) or getattr(o, 'id', '?')
        sym = getattr(o, 'symbol', '?')
        side = getattr(o, 'side', '?')
        qty = getattr(o, 'quantity', '?')
        price = getattr(o, 'price', '?')
        status = getattr(o, 'status', '?')
        print(f'  [{ext}] {side} {qty}× {sym} @ ${price}  [{status}]')


def cmd_cancel(args):
    from ws_sdk import WSClient
    from ws_safe import cancel_order_safe
    ws = WSClient()
    if not args.live and os.environ.get('KERNEL_LIVE') != '1':
        print(f'DRY-RUN: would cancel {args.ext_id} (no env KERNEL_LIVE=1)')
        return 0
    ok = cancel_order_safe(ws, args.ext_id)
    print(f'cancel {"OK" if ok else "FAILED (still in open list?)"}')


def cmd_sell_put(args):
    """Manual single-leg put-sell. Build OSI, resolve, confirm, submit."""
    from ws_option_resolver import resolve_osi
    yymmdd = args.expiry.replace('-', '')[2:]  # 2026-07-17 → 260717
    strike_milli = int(args.strike * 1000)
    osi = f'UNG   {yymmdd}P{strike_milli:08d}'
    print(f'OSI: {osi}')
    r = resolve_osi(osi)
    if not r:
        print(f'ERROR: could not resolve {osi}'); return 1
    print(f'live bid {r["bid"]}  ask {r["ask"]}  mid {r["mid"]}  OI {r["oi"]}')
    print(f'PLAN: SELL_TO_OPEN {args.qty}× @ ${args.price}')
    if os.environ.get('KERNEL_LIVE') != '1':
        print('DRY-RUN: KERNEL_LIVE=1 not set; refusing to submit')
        return 0
    ans = input(f'Submit? [y/N]: ').strip().lower()
    if ans not in ('y', 'yes'):
        print('declined'); return 0
    from ws_sdk import WSClient
    from ws_safe import submit_and_verify
    ws = WSClient()
    r2 = submit_and_verify(ws, 'SELL_TO_OPEN', r['sec_id'], args.qty, args.price)
    print(f'result: {r2}')


def cmd_positions(args):
    """Show current UNG positions via dashboard."""
    from execute_kernel_plan import fetch_verdict
    v = fetch_verdict()
    if not v: print('fetch failed'); return 1
    ung = [p for p in (v.get('positions') or []) if p.get('symbol') == 'UNG']
    shares = sum(p['quantity'] for p in ung if not p.get('is_option'))
    calls = [p for p in ung if p.get('option_type') == 'CALL']
    puts = [p for p in ung if p.get('option_type') == 'PUT']
    print(f'SHARES: {shares}')
    print(f'\nSHORT CALLS ({len(calls)} contracts):')
    for p in sorted(calls, key=lambda x: x['expiry']):
        print(f"  {abs(p['quantity']):>3}× ${p['strike']:.2f} {p['expiry']}  mkt ${p['market_value']:+.0f}")
    print(f'\nSHORT PUTS ({len(puts)} contracts):')
    for p in sorted(puts, key=lambda x: x['expiry']):
        print(f"  {abs(p['quantity']):>3}× ${p['strike']:.2f} {p['expiry']}  mkt ${p['market_value']:+.0f}")


def cmd_digest(args):
    from daily_digest import digest
    print(digest(hours=args.hours))


def cmd_log(args):
    from trading_log import tail
    for e in tail(args.n):
        spot = e.get('spot'); sz = e.get('surge_z'); act = e.get('action_taken')
        vbp = e.get('verdict_best_play') or {}; bp = vbp.get('order_type', '?')
        print(f'{e["ts"]}  spot=${spot}  sz={sz:+.2f}  best_play={bp}  action={act}  mode={e.get("mode","?")}')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan', help='Show kernel best play (no submission)')
    sub.add_parser('submit', help='Interactive --mode review')
    a = sub.add_parser('auto', help='Auto-submit with daily cap')
    a.add_argument('--daily-max', type=int, default=4)
    sub.add_parser('orders', help='List open WS orders')
    c = sub.add_parser('cancel', help='Cancel a specific order')
    c.add_argument('ext_id'); c.add_argument('--live', action='store_true')
    sp = sub.add_parser('sell-put', help='Manual single-leg put-sell')
    sp.add_argument('expiry', help='YYYY-MM-DD'); sp.add_argument('strike', type=float)
    sp.add_argument('qty', type=int); sp.add_argument('price', type=float)
    sub.add_parser('positions', help='Show UNG positions')
    d = sub.add_parser('digest'); d.add_argument('--hours', type=int, default=24)
    l = sub.add_parser('log'); l.add_argument('n', type=int, nargs='?', default=10)
    args = p.parse_args()
    fn = {'plan': cmd_plan, 'submit': cmd_submit, 'auto': cmd_auto,
          'orders': cmd_orders, 'cancel': cmd_cancel, 'sell-put': cmd_sell_put,
          'positions': cmd_positions, 'digest': cmd_digest, 'log': cmd_log}[args.cmd]
    return fn(args) or 0


if __name__ == '__main__':
    sys.exit(main())
