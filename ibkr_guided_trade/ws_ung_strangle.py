#!/usr/bin/env python3
"""
UNG Share-Backed Strangle Ladder

Sells 1 covered call + 1 cash-secured put per cycle at the forward ATM strike.
Each cycle re-detects ATM — if UNG spikes between pairs the next pair captures
the elevated premium at the new strike automatically.

Sell side only: each leg placed at mid+$0.01, repriced on timeout.
Both legs run in parallel threads; script waits for both before next pair.

Usage:
    python ws_ung_strangle.py                           # dry run
    python ws_ung_strangle.py --place                   # execute
    python ws_ung_strangle.py --place --pairs 3
    python ws_ung_strangle.py --place --expiry 2026-06-19
    python ws_ung_strangle.py --cancel                  # cancel resting UNG option orders
    python ws_ung_strangle.py --status                  # show current positions
"""

import argparse
import os
import sys
import time
import threading
from datetime import datetime

LOCK_FILE = '/tmp/ws_ung_strangle.lock'


def acquire_lock():
    if os.path.exists(LOCK_FILE):
        old_pid = open(LOCK_FILE).read().strip()
        if os.path.exists(f'/proc/{old_pid}'):
            print(f'ERROR: already running (PID {old_pid}). Kill it first or run --cancel.')
            sys.exit(1)
    open(LOCK_FILE, 'w').write(str(os.getpid()))


def release_lock():
    try:
        os.remove(LOCK_FILE)
    except FileNotFoundError:
        pass

sys.path.insert(0, '.')
from ws_sdk import (
    WSClient,
    OrderStatus,
    OrderTimeout,
    OrderRejected,
    KNOWN_SECURITIES,
    QUERY_OPTION_CHAIN,
    graphql_query,
)

UNG_SEC_ID     = KNOWN_SECURITIES.get('UNG', 'sec-s-32f0b46791214cbcbee9486e40232ea4')
DEFAULT_EXPIRY = '2026-05-15'
DEFAULT_PAIRS  = 5
MAX_PUTS       = 10   # hard cap on total short puts regardless of --puts arg
POLL_INTERVAL  = 15    # seconds between fill checks
ORDER_TIMEOUT  = 180   # seconds before repricing
MAX_REPRICE    = 5     # max reprice attempts per leg before giving up

_lock = threading.Lock()


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    with _lock:
        print(f'[{ts}] {msg}', flush=True)


# ── chain ──────────────────────────────────────────────────────────────────────

def fetch_chain(session, expiry, opt_type):
    """Return [{strike, sec_id, bid, ask}] sorted by strike."""
    data = graphql_query(session, 'FetchOptionChain', QUERY_OPTION_CHAIN, {
        'id': UNG_SEC_ID, 'expiryDate': expiry, 'optionType': opt_type,
        'realTimeQuote': True, 'includeGreeks': False, 'first': 60,
    })
    out = []
    for e in (data or {}).get('security', {}).get('optionChain', {}).get('edges', []):
        n = e.get('node', {})
        d = n.get('optionDetails', {})
        q = n.get('quoteV2', {}) or {}
        K = float(d.get('strikePrice', 0) or 0)
        if K <= 0:
            continue
        out.append({
            'strike': K,
            'sec_id': n.get('id', ''),
            'bid':    float(q.get('bid', 0) or 0),
            'ask':    float(q.get('ask', 0) or 0),
        })
    return sorted(out, key=lambda x: x['strike'])


def find_atm(calls, puts):
    """
    Forward ATM via put-call parity: strike where |call_mid - put_mid| is min.
    Returns (atm_K, fwd_price, call_data, put_data) or (None,)*4.
    """
    pm = {p['strike']: p for p in puts}
    best = None
    for c in calls:
        K  = c['strike']
        p  = pm.get(K)
        if not p:
            continue
        cm = (c['bid'] + c['ask']) / 2
        pm_ = (p['bid'] + p['ask']) / 2
        if cm <= 0 or pm_ <= 0:
            continue
        diff = abs(cm - pm_)
        if best is None or diff < best[0]:
            best = (diff, K, cm, pm_, c, p)
    if not best:
        return None, None, None, None
    _, K, cm, pm_, c, p = best
    fwd = round(K + cm - pm_, 2)
    return K, fwd, c, p


# ── positions (via SDK, margin-account filtered) ─────────────────────────────

def get_ung_positions(ws: WSClient):
    """Return (shares, short_calls, short_puts, avg_cost).

    Uses :meth:`WSClient.list_positions` which is already scoped to the
    margin account — crypto holdings on the separate WS crypto account
    cannot leak in.
    """
    shares = sc = sp = 0
    avg_cost = 0.0
    for pos in ws.list_positions():
        # Shares: stock.symbol == 'UNG' and not an option
        if not pos.is_option:
            if pos.symbol != 'UNG':
                continue
            shares = int(pos.quantity)
            avg_cost = float(pos.average_price)
            continue
        # Options: underlying must be UNG
        if pos.underlying_symbol != 'UNG':
            continue
        qty = abs(int(pos.quantity))
        if pos.option_type == 'CALL':
            sc += qty
        elif pos.option_type == 'PUT':
            sp += qty
    return shares, sc, sp, avg_cost


# ── single-leg execution (SDK-backed fill detection) ─────────────────────────

def run_leg(ws: WSClient, expiry, opt_type, strike, label, results, key):
    """
    Thread target. Places 1 contract sell at ``strike``, waits for fill
    via :func:`wait_for_order` against ``soOrdersExtendedOrder`` — no
    more activity-feed polling and no more stale/phantom fills.

    On timeout: cancels, waits for cancel to settle, reprices at fresh mid.
    Stores result in ``results[key] = {'filled': bool, 'price': float}``.
    """
    for attempt in range(1, MAX_REPRICE + 1):
        chain = fetch_chain(ws.session, expiry, opt_type)
        leg   = next((x for x in chain if x['strike'] == strike), None)

        if not leg or leg['bid'] <= 0:
            log(f'  {label} no quote for ${strike:.2f} — retry in 15s')
            time.sleep(15)
            continue

        mid   = round((leg['bid'] + leg['ask']) / 2, 2)
        price = round(mid + 0.01, 2)
        log(f'  {label} ${strike:.2f}  bid={leg["bid"]:.2f} ask={leg["ask"]:.2f} -> ${price:.2f} (try {attempt})')

        try:
            placed = ws.sell_to_open(leg['sec_id'], 1, price)
        except OrderRejected as exc:
            log(f'  {label} order rejected: {exc}')
            results[key] = {'filled': False, 'price': price}
            return

        order_id = placed.external_id
        if not order_id:
            log(f'  {label} no order_id — aborting')
            results[key] = {'filled': False, 'price': price}
            return

        # Canonical poll against soOrdersExtendedOrder — no activity-feed lag.
        try:
            final = ws.wait_for_order(order_id, timeout=ORDER_TIMEOUT, poll_interval=POLL_INTERVAL)
        except OrderTimeout:
            log(f'  {label} timeout (attempt {attempt}) — cancelling...')
            ws.cancel(order_id)
            time.sleep(2)   # let cancel settle before next place
            continue

        if final.is_filled:
            fill_px = float(final.average_filled_price) if final.average_filled_price else price
            log(f'  {label} FILLED @${fill_px:.2f} ✓')
            results[key] = {'filled': True, 'price': fill_px}
            return

        if final.status == OrderStatus.CANCELLED:
            log(f'  {label} cancelled by broker — stopping')
            results[key] = {'filled': False, 'price': price}
            return
        if final.status == OrderStatus.REJECTED:
            log(f'  {label} rejected: {final.rejection_cause or "unknown"}')
            results[key] = {'filled': False, 'price': price}
            return
        # Else (EXPIRED, unknown terminal) — try again
        log(f'  {label} terminal status {final.status.value} — repricing')
        time.sleep(2)

    log(f'  {label} gave up after {MAX_REPRICE} attempts')
    results[key] = {'filled': False, 'price': 0.0}


# ── independent ladder per leg ────────────────────────────────────────────────

def run_ladder(expiry, opt_type, total, label, dry_run=False):
    """
    Independent ladder: sell `total` contracts of `opt_type`, 1 at a time.
    Creates its own WSClient (thread-safe). Re-detects forward ATM each
    contract so a UNG spike automatically shifts to the new higher
    strike on the next fill.
    """
    ws     = WSClient()
    filled = 0

    for i in range(1, total + 1):
        # Fresh ATM each contract
        calls = fetch_chain(ws.session, expiry, 'CALL')
        puts  = fetch_chain(ws.session, expiry, 'PUT')
        atm_K, fwd, c_data, p_data = find_atm(calls, puts)
        if atm_K is None or c_data is None or p_data is None:
            log(f'  {label} {i}/{total}: no ATM found — skipping')
            continue

        leg = c_data if opt_type == 'CALL' else p_data
        mid = round((leg['bid'] + leg['ask']) / 2, 2)
        log(f'\n[{label} {i}/{total}]  fwd=${fwd:.2f}  ${atm_K:.2f}  '
            f'bid={leg["bid"]:.2f} ask={leg["ask"]:.2f}  mid={mid:.2f}')

        if dry_run:
            log(f'  [DRY RUN] would sell @${round(mid + 0.01, 2):.2f}')
            filled += 1
            continue

        results = {}
        run_leg(ws, expiry, opt_type, atm_K, f'{label}-{i}', results, 'leg')
        if results.get('leg', {}).get('filled'):
            filled += 1

        time.sleep(1)

    log(f'\n{label} ladder done: {filled}/{total} filled')


# ── cancel helpers ────────────────────────────────────────────────────────────

def cancel_ung_options(ws: WSClient):
    """Cancel all resting UNG option orders.

    Uses :meth:`WSClient.list_open_orders` (margin-account filtered,
    crypto/recurring excluded) rather than shelling out to
    ``ws_trading.py open-orders`` and parsing stdout. Each returned
    order is already verified against the canonical extended-order
    endpoint so we never try to cancel an order that's already in a
    terminal state.
    """
    open_orders = ws.list_open_orders()
    ung_orders = [
        o for o in open_orders
        if o.security_id.startswith('sec-o-')
        and 'UNG' in (o.raw.get('assetSymbol') or '')
    ]
    if not ung_orders:
        print('No resting UNG option orders found.')
        return
    print(f'Cancelling {len(ung_orders)} UNG option orders...')
    for o in ung_orders:
        ok = ws.cancel(o.external_id)
        print(f'  {o.external_id[-12:]}: {"OK" if ok else "FAILED"}')
        time.sleep(0.2)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    acquire_lock()
    import atexit; atexit.register(release_lock)

    ap = argparse.ArgumentParser(description='UNG Share-Backed Strangle Ladder')
    ap.add_argument('--place',  action='store_true', help='Place orders (default: dry run)')
    ap.add_argument('--calls',  type=int, default=DEFAULT_PAIRS, help='Number of calls to sell')
    ap.add_argument('--puts',   type=int, default=DEFAULT_PAIRS, help='Number of puts to sell')
    ap.add_argument('--expiry', default=DEFAULT_EXPIRY, help='Expiry YYYY-MM-DD')
    ap.add_argument('--cancel', action='store_true', help='Cancel resting UNG option orders')
    ap.add_argument('--status', action='store_true', help='Show current UNG positions only')
    args = ap.parse_args()

    ws = WSClient()

    if args.cancel:
        cancel_ung_options(ws)
        return

    shares, sc, sp, avg_cost = get_ung_positions(ws)
    max_calls   = shares // 100
    call_room   = max(0, max_calls - sc)
    put_room    = max(0, MAX_PUTS - sp)

    print('\nUNG Position')
    print(f'  Shares     : {shares}  avg cost ${avg_cost:.4f}')
    print(f'  Short calls: {sc}  (covered cap {max_calls}, room for {call_room} more)')
    print(f'  Short puts : {sp}  (hard cap {MAX_PUTS}, room for {put_room} more)')

    if args.status:
        return

    n_calls = min(args.calls, call_room)
    n_puts  = min(args.puts, put_room)
    if n_calls < args.calls:
        print(f'\nWARNING: only room for {call_room} more covered calls — capping calls at {n_calls}')
    if n_puts < args.puts:
        print(f'\nWARNING: put cap {MAX_PUTS} reached — capping puts at {n_puts}')
    if n_calls <= 0 and n_puts <= 0:
        print('\nNothing to place.')
        return

    dry = not args.place
    if dry:
        print(f'\nDRY RUN — {n_calls} calls + {n_puts} puts on {args.expiry}. Add --place to execute.\n')
    else:
        print('\nCancelling any resting UNG option orders...')
        cancel_ung_options(ws)
        time.sleep(2)
        print(f'\nLaunching independent ladders: {n_calls} calls + {n_puts} puts on {args.expiry}\n')

    threads = []
    if n_calls > 0:
        threads.append(threading.Thread(
            target=run_ladder,
            args=(args.expiry, 'CALL', n_calls, 'CALL', dry),
            daemon=True,
        ))
    if n_puts > 0:
        threads.append(threading.Thread(
            target=run_ladder,
            args=(args.expiry, 'PUT', n_puts, 'PUT', dry),
            daemon=True,
        ))

    for i, t in enumerate(threads):
        t.start()
        if i < len(threads) - 1:
            time.sleep(2)   # stagger start so first quotes don't collide

    for t in threads:
        t.join()

    print('\n=== ALL DONE ===')


if __name__ == '__main__':
    main()
