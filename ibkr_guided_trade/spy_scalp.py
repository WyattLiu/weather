#!/usr/bin/env python3
"""
SPY Straddle Scalper — Accumulate + scalp simultaneously

Dynamic forward ATM: each cycle re-detects the forward ATM strike per expiry
and places buy ladders there. Sell ladders placed on existing positions only.

Usage:
    python spy_scalp.py                    # Analyze only
    python spy_scalp.py --place            # Place all orders
    python spy_scalp.py --place --loop     # Place + refresh every N seconds
    python spy_scalp.py --place --loop --interval 60  # Custom refresh interval
    python spy_scalp.py --cancel           # Cancel all resting SPY orders
"""

import argparse
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')
from ws_trading import (
    get_session, graphql_query, QUERY_OPTION_CHAIN, QUERY_FETCH_POSITIONS,
    place_multileg_order, cancel_order, fetch_multileg_order,
    load_config, load_cookies, extract_identity_from_cookies,
    KNOWN_SECURITIES, DEFAULT_ACCOUNT_ID,
)

SPY_SEC_ID = KNOWN_SECURITIES.get('SPY', 'sec-s-27167ecbd81140fe9cdc02535f43174d')

# Expiries to trade — buy_ladder: how many buy orders (0 = sell only)
EXPIRIES = [
    {'expiry': '2026-04-17', 'buy_ladder': 0, 'label': 'Apr17'},  # 32d — sell only
    {'expiry': '2026-04-30', 'buy_ladder': 0, 'label': 'Apr30'},  # 45d — sell only
    {'expiry': '2026-05-15', 'buy_ladder': 0, 'label': 'May15'},  # 60d — sell only
    {'expiry': '2026-05-29', 'buy_ladder': 0, 'label': 'May29'},  # 74d — sell only
]

# Position limits
MAX_STRADDLES = 10  # HARD CAP — no new buys if total straddles >= this

# Order settings
SELL_LADDER = 2   # sell orders per held strike (capped by position)
MIN_HOLD = 0      # sell everything — pure scalp all expiries
SELL_STEP = 0.10  # sell ladder increment between rungs


def get_positions(session):
    """Get current SPY option positions as {(expiry, strike, type): {qty, avg_cost}}"""
    config = load_config()
    cookies = load_cookies()
    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)

    if not identity_id:
        return {}

    pos_data = graphql_query(session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, {
        "identityId": identity_id, "currency": "CAD", "first": 50,
        "aggregated": True, "currencyOverride": "MARKET",
        "sort": "TODAY_GAIN", "includeSecurity": True,
        "includeAccountData": True, "includeOneDayReturnsBaseline": True,
    })

    positions = {}
    if not pos_data:
        return positions

    edges = (pos_data.get('identity', {})
             .get('financials', {})
             .get('current', {})
             .get('positions', {})
             .get('edges', []))

    for edge in edges:
        node = edge.get('node', {})
        sec = node.get('security', {}) or {}
        opt = sec.get('optionDetails', {}) or {}
        if not opt:
            continue

        underlying = (opt.get('underlyingSecurity', {}) or {}).get('stock', {}) or {}
        symbol = underlying.get('symbol', '')
        if symbol != 'SPY':
            continue

        strike = float(opt.get('strikePrice', 0))
        opt_type = opt.get('optionType', '')  # CALL or PUT
        expiry = opt.get('expiryDate', '')
        qty = int(float(node.get('quantity', 0)))

        avg_price = node.get('marketAveragePrice', node.get('averagePrice', {}))
        avg_cost_total = float(avg_price.get('amount', 0)) if avg_price else 0
        # marketAveragePrice is per-contract (×100), divide to get per-share for limit comparison
        avg_cost = avg_cost_total / 100 if avg_cost_total else 0

        if strike > 0 and opt_type and expiry:
            positions[(expiry, strike, opt_type)] = {'qty': qty, 'avg_cost': avg_cost}

    return positions


def get_straddle_position(positions, expiry, strike):
    """Return (straddle_qty, avg_cost_per_straddle) for this strike/expiry"""
    call_info = positions.get((expiry, strike, 'CALL'), {})
    put_info = positions.get((expiry, strike, 'PUT'), {})
    call_qty = call_info.get('qty', 0)
    put_qty = put_info.get('qty', 0)
    straddles = min(call_qty, put_qty)

    # Cost per straddle = call avg cost + put avg cost (per-share, so * 100 each for total)
    # But for limit price comparison, we need per-share cost
    call_cost = call_info.get('avg_cost', 0)
    put_cost = put_info.get('avg_cost', 0)
    straddle_cost = call_cost + put_cost

    return straddles, straddle_cost


def fetch_full_chain(session, expiry, opt_type):
    """Fetch full option chain for an expiry/type. Returns list of {strike, sec_id, bid, ask}"""
    data = graphql_query(session, 'FetchOptionChain', QUERY_OPTION_CHAIN, {
        'id': SPY_SEC_ID, 'expiryDate': expiry, 'optionType': opt_type,
        'realTimeQuote': True, 'includeGreeks': False, 'first': 80,
    })
    results = []
    if not data:
        return results
    for edge in data.get('security', {}).get('optionChain', {}).get('edges', []):
        n = edge.get('node', {})
        d = n.get('optionDetails', {})
        q = n.get('quoteV2', {})
        results.append({
            'strike': float(d.get('strikePrice', 0)),
            'sec_id': n.get('id', ''),
            'bid': float(q.get('bid', 0) or 0),
            'ask': float(q.get('ask', 0) or 0),
        })
    return results


def find_forward_atm(calls, puts):
    """Find forward ATM strike: where |call_mid - put_mid| is smallest.
    Returns (strike, call_data, put_data) or None."""
    put_map = {p['strike']: p for p in puts}
    best = None
    for c in calls:
        K = c['strike']
        p = put_map.get(K)
        if not p:
            continue
        c_mid = (c['bid'] + c['ask']) / 2
        p_mid = (p['bid'] + p['ask']) / 2
        if c_mid <= 0 or p_mid <= 0:
            continue
        diff = abs(c_mid - p_mid)
        if best is None or diff < best[0]:
            best = (diff, K, c, p)
    if best:
        return best[1], best[2], best[3]
    return None


def place_straddle(session, call_id, put_id, price, action='BUY', open_close='OPEN'):
    """Place a straddle order. action: BUY or SELL"""
    order_type = 'BUY_QUANTITY' if action == 'BUY' else 'SELL_QUANTITY'
    effective_price = price if action == 'BUY' else -price

    legs = [
        {'securityId': put_id, 'orderType': order_type, 'openClose': open_close},
        {'securityId': call_id, 'orderType': order_type, 'openClose': open_close},
    ]
    r = place_multileg_order(session, legs, effective_price)
    errs = r.get('result', {}).get('soOrdersCreateOrderExecution', {}).get('errors', [])
    return not errs, errs


def cancel_all_spy_orders(session):
    """Cancel all resting SPY option orders"""
    import subprocess
    result = subprocess.run(
        ['python', 'ws_trading.py', 'open-orders'],
        capture_output=True, text=True, cwd='/home/wyatt/ibkr_guided_trade'
    )

    batch_ids = []
    for line in result.stdout.split('\n'):
        if 'SPY' in line and 'order-batch-' in line:
            parts = line.strip().split()
            bid = parts[-1]
            if bid.startswith('order-batch-'):
                batch_ids.append(bid)

    cancelled = 0
    for bid in batch_ids:
        data = fetch_multileg_order(session, bid)
        legs = data.get('soOrdersMultilegOrder', {}).get('legs', []) if data else []
        if legs:
            ext_id = legs[0].get('externalId', '')
            if ext_id:
                r = cancel_order(session, ext_id)
                errs = r.get('orderServiceCancelOrder', {}).get('errors', [])
                if not errs:
                    cancelled += 1

    return cancelled, len(batch_ids)


def run_cycle(session, place=False):
    """Run one scalp cycle: find forward ATM, place buys there, sell existing positions"""
    ts = datetime.now().strftime('%H:%M:%S')
    positions = get_positions(session)

    print(f'\n[{ts}] SCALP CYCLE')
    print('=' * 80)

    total_buy = 0
    total_sell = 0
    all_orders = []

    # Count total straddles held across all expiries
    all_strikes = {}
    for (exp, K, otype), info in positions.items():
        key = (exp, K)
        if key not in all_strikes:
            all_strikes[key] = {'CALL': 0, 'PUT': 0}
        all_strikes[key][otype] = info.get('qty', 0)
    total_held = sum(min(v['CALL'], v['PUT']) for v in all_strikes.values())
    buys_allowed = total_held < MAX_STRADDLES

    if not buys_allowed:
        print(f'\n  *** POSITION CAP: {total_held}/{MAX_STRADDLES} straddles — NO NEW BUYS ***')
    else:
        print(f'\n  Position: {total_held}/{MAX_STRADDLES} straddles')

    for exp_cfg in EXPIRIES:
        expiry = exp_cfg['expiry']
        exp_label = exp_cfg['label']

        # Fetch full chains once per expiry
        calls = fetch_full_chain(session, expiry, 'CALL')
        puts = fetch_full_chain(session, expiry, 'PUT')
        call_map = {c['strike']: c for c in calls}
        put_map = {p['strike']: p for p in puts}

        if not calls or not puts:
            print(f'\n  {exp_label}: NO CHAIN DATA — skipping')
            continue

        # Find forward ATM
        atm_result = find_forward_atm(calls, puts)
        if not atm_result:
            print(f'\n  {exp_label}: CANNOT FIND FORWARD ATM — skipping')
            continue

        atm_strike, atm_call, atm_put = atm_result
        atm_bid = atm_call['bid'] + atm_put['bid']
        atm_ask = atm_call['ask'] + atm_put['ask']
        atm_mid = round((atm_bid + atm_ask) / 2, 2)
        fwd = round(atm_strike + (atm_call['bid'] + atm_call['ask']) / 2
                     - (atm_put['bid'] + atm_put['ask']) / 2, 2)

        label = f'{exp_label} ${atm_strike:.0f}'
        print(f'\n  {label} [fwd ${fwd:.1f}] | bid ${atm_bid:.2f} mid ${atm_mid:.2f} ask ${atm_ask:.2f}')

        # BUY SIDE: penny ladder at forward ATM (gated by position cap)
        buy_count = exp_cfg.get('buy_ladder', 0) if buys_allowed else 0
        buy_prices = []
        for i in range(buy_count):
            price = round(atm_mid - i * 0.01, 2)
            buy_prices.append(price)

        if buy_prices:
            print(f'    BUY:  {" / ".join(f"${p:.2f}" for p in buy_prices)}')
        for price in buy_prices:
            all_orders.append({
                'call_id': atm_call['sec_id'], 'put_id': atm_put['sec_id'],
                'price': price, 'action': 'BUY', 'open_close': 'OPEN',
                'label': f'{label} BUY @${price:.2f}',
            })
            total_buy += 1

        # SELL SIDE: check all held strikes for this expiry
        # Find all SPY option strikes we hold for this expiry
        expiry_strikes = set()
        for (exp, K, otype), info in positions.items():
            if exp == expiry and info.get('qty', 0) > 0:
                expiry_strikes.add(K)

        for K in sorted(expiry_strikes):
            held, cost = get_straddle_position(positions, expiry, K)
            if held <= 0:
                continue

            c = call_map.get(K)
            p = put_map.get(K)
            if not c or not p:
                continue

            strad_bid = c['bid'] + p['bid']
            strad_ask = c['ask'] + p['ask']
            strad_mid = round((strad_bid + strad_ask) / 2, 2)

            sellable = max(0, held - MIN_HOLD)
            sell_base = strad_bid
            k_label = f'{exp_label} ${K:.0f}'

            # Cost floor: don't sell below avg cost
            if cost > 0 and sell_base < cost:
                print(f'    {k_label} SELL: — ({held}x held, mid-ask ${sell_base:.2f} < cost ${cost:.2f}, NO LOSS SELL)')
                continue

            if sellable > 0:
                sell_qty = min(SELL_LADDER, sellable)
                sell_prices = []
                for i in range(sell_qty):
                    price = round(sell_base + i * SELL_STEP, 2)
                    sell_prices.append(price)

                cost_str = f', cost ${cost:.2f}' if cost > 0 else ''
                print(f'    {k_label} SELL: {" / ".join(f"${sp:.2f}" for sp in sell_prices)} ({sellable}x sellable, keeping {MIN_HOLD}x{cost_str})')
                for price in sell_prices:
                    all_orders.append({
                        'call_id': c['sec_id'], 'put_id': p['sec_id'],
                        'price': price, 'action': 'SELL', 'open_close': 'CLOSE',
                        'label': f'{k_label} SELL @${price:.2f}',
                    })
                    total_sell += 1
            else:
                print(f'    {k_label} SELL: — (held {held}x, keeping {MIN_HOLD}x)')

    print(f'\n  TOTAL: {total_buy} buy + {total_sell} sell = {total_buy + total_sell} orders')

    if not place:
        print('\n  [ANALYZE ONLY] Re-run with --place to submit')
        return

    # Cancel existing SPY orders first
    print('\n  Cancelling existing orders...')
    cancelled, attempted = cancel_all_spy_orders(session)
    print(f'  Cancelled {cancelled}/{attempted}')
    time.sleep(1)

    # Place all orders
    print(f'\n  Placing {len(all_orders)} orders...')
    placed = 0
    for order in all_orders:
        ok, errs = place_straddle(
            session, order['call_id'], order['put_id'],
            order['price'], order['action'], order['open_close']
        )
        status = 'OK' if ok else errs[0].get('message', '') if errs else 'FAIL'
        print(f'    {order["label"]} -> {status}')
        if ok:
            placed += 1
        time.sleep(0.3)

    print(f'\n  Placed: {placed}/{len(all_orders)}')


def main():
    parser = argparse.ArgumentParser(description='SPY Straddle Scalper')
    parser.add_argument('--place', action='store_true', help='Place orders (default: analyze only)')
    parser.add_argument('--cancel', action='store_true', help='Cancel all resting SPY orders')
    parser.add_argument('--loop', action='store_true', help='Continuously refresh orders')
    parser.add_argument('--interval', type=int, default=90, help='Loop interval in seconds (default: 90)')
    args = parser.parse_args()

    session = get_session()

    if args.cancel:
        print('Cancelling all SPY orders...')
        cancelled, attempted = cancel_all_spy_orders(session)
        print(f'Cancelled {cancelled}/{attempted}')
        return

    if args.loop and args.place:
        print(f'SCALP LOOP — refreshing every {args.interval}s')
        print('Press Ctrl+C to stop')
        try:
            while True:
                run_cycle(session, place=True)
                print(f'\n  Sleeping {args.interval}s...')
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('\nStopped.')
    else:
        run_cycle(session, place=args.place)


if __name__ == '__main__':
    main()
