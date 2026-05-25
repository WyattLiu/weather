#!/usr/bin/env python3
"""
Roll $690 straddle -> $691 straddle via 4-leg orders.

Single atomic order: sell $690 call + sell $690 put + buy $691 call + buy $691 put
Starts at max credit, creeps down $0.01 each cycle until fill, then replaces.

Usage:
    python ws_swap.py                  # Run (default 16 rolls)
    python ws_swap.py --max 1          # Just 1 roll
    python ws_swap.py --poll 3         # Poll every 3s (default)
"""

import argparse
import time

import ws_trading as ws

# Security IDs - Mar 27 expiry
SEC_690_PUT = 'sec-o-2aaf91808df640829038d52e9f094863'
SEC_690_CALL = 'sec-o-af2739cc6e4942caae8e2b34341bdbeb'
SEC_691_PUT = 'sec-o-111a6598a27043ce9be8c8132f7773e4'
SEC_691_CALL = 'sec-o-eb8267e1f28d45c191caa9ad2a1877ff'

ROLL_LEGS = [
    {'openClose': 'CLOSE', 'orderType': 'SELL_QUANTITY', 'quantity': 1, 'securityId': SEC_690_PUT},
    {'openClose': 'CLOSE', 'orderType': 'SELL_QUANTITY', 'quantity': 1, 'securityId': SEC_690_CALL},
    {'openClose': 'OPEN', 'orderType': 'BUY_QUANTITY', 'quantity': 1, 'securityId': SEC_691_PUT},
    {'openClose': 'OPEN', 'orderType': 'BUY_QUANTITY', 'quantity': 1, 'securityId': SEC_691_CALL},
]


def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def get_roll_quotes(session):
    """Get quotes for both strikes and compute credit range."""
    spy_id = ws.KNOWN_SECURITIES['SPY']
    prices = {}
    for strike in [690, 691]:
        for opt_type in ['CALL', 'PUT']:
            data = ws.graphql_query(session, 'FetchOptionChain', ws.QUERY_OPTION_CHAIN, {
                'id': spy_id, 'expiryDate': '2026-03-27',
                'optionType': opt_type, 'realTimeQuote': True, 'includeGreeks': False
            })
            for edge in data.get('security', {}).get('optionChain', {}).get('edges', []):
                node = edge.get('node', {})
                s = float(node.get('optionDetails', {}).get('strikePrice', 0))
                if s == strike:
                    q = node.get('quoteV2', {})
                    prices[(strike, opt_type)] = {
                        'bid': float(q.get('bid', 0)),
                        'ask': float(q.get('ask', 0)),
                    }

    sell_bid = prices[(690, 'CALL')]['bid'] + prices[(690, 'PUT')]['bid']
    sell_ask = prices[(690, 'CALL')]['ask'] + prices[(690, 'PUT')]['ask']
    buy_bid = prices[(691, 'CALL')]['bid'] + prices[(691, 'PUT')]['bid']
    buy_ask = prices[(691, 'CALL')]['ask'] + prices[(691, 'PUT')]['ask']

    max_credit = round(sell_ask - buy_bid, 2)
    mid_credit = round((sell_bid + sell_ask) / 2 - (buy_bid + buy_ask) / 2, 2)
    min_credit = round(sell_bid - buy_ask, 2)

    return {
        'sell_bid': round(sell_bid, 2), 'sell_ask': round(sell_ask, 2),
        'buy_bid': round(buy_bid, 2), 'buy_ask': round(buy_ask, 2),
        'max_credit': max_credit, 'mid_credit': mid_credit, 'min_credit': min_credit,
    }


def place_roll(session, credit):
    """Place 4-leg roll order at given credit. Returns leg ext IDs for cancellation."""
    log(f'  Placing ROLL at ${credit:.2f} credit...')
    result = ws.place_multileg_order(session, ROLL_LEGS, -credit)
    errors = result.get('result', {}).get('soOrdersCreateOrderExecution', {}).get('errors')
    if errors:
        log(f'  ROLL ERROR: {errors}')
        return []
    ext_id = result.get('order_id', '')
    orders = result.get('result', {}).get('soOrdersCreateOrderExecution', {}).get('orders', [])
    leg_ids = [f'{ext_id}-leg-{i+1}' for i in range(len(orders))]
    log(f'  Placed: {ext_id} ({len(orders)} legs)')
    return leg_ids


def cancel_legs(session, leg_ids):
    """Cancel order by leg external IDs."""
    for lid in leg_ids:
        ws.cancel_order(session, lid)


def count_completed(session):
    """Count completed multileg orders in activity feed."""
    data = ws.graphql_query(session, 'FetchActivityFeedItems', ws.QUERY_FETCH_ACTIVITIES, {
        'first': 50, 'orderBy': 'OCCURRED_AT_DESC'
    })
    count = 0
    for edge in data.get('activityFeedItems', {}).get('edges', []):
        act = edge.get('node', {})
        if 'MULTILEG' in (act.get('subType') or '') and act.get('unifiedStatus') == 'COMPLETED':
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='Roll $690 -> $691 straddle')
    parser.add_argument('--max', type=int, default=16, help='Max rolls (default 16)')
    parser.add_argument('--poll', type=int, default=3, help='Poll interval seconds (default 3)')
    parser.add_argument('--start', type=float, default=None, help='Starting credit (overrides max)')
    parser.add_argument('--floor', type=float, default=None, help='Min credit floor (overrides min)')
    args = parser.parse_args()

    log('=== STRADDLE ROLL: $690 -> $691 ===')
    session = ws.get_session()
    log('Session ready.')

    # Get quotes and credit range
    q = get_roll_quotes(session)
    log(f'Sell $690: bid=${q["sell_bid"]:.2f} ask=${q["sell_ask"]:.2f}')
    log(f'Buy  $691: bid=${q["buy_bid"]:.2f} ask=${q["buy_ask"]:.2f}')
    log(f'Credit range: max=${q["max_credit"]:.2f} mid=${q["mid_credit"]:.2f} min=${q["min_credit"]:.2f}')

    # Start at max credit (or override)
    credit = args.start if args.start is not None else q['max_credit']
    min_credit = args.floor if args.floor is not None else q['min_credit']
    rolls_done = 0
    total_credit = 0.0

    # Baseline completed count
    baseline_completed = count_completed(session)
    log(f'Baseline completed orders: {baseline_completed}')

    # Place first order
    log(f'Starting at ${credit:.2f} credit, creeping down to ${min_credit:.2f}')
    leg_ids = place_roll(session, credit)

    log(f'Polling every {args.poll}s... Ctrl+C to stop')
    log('=' * 60)

    try:
        cycle = 0
        while rolls_done < args.max:
            time.sleep(args.poll)
            cycle += 1

            # Check for new fills
            cur_completed = count_completed(session)
            new_fills = cur_completed - baseline_completed - rolls_done

            if new_fills > 0:
                rolls_done += new_fills
                total_credit += credit * new_fills
                log(f'>>> FILLED at ${credit:.2f} credit! (roll {rolls_done}/{args.max}, total credit: ${total_credit:.2f})')

                if rolls_done >= args.max:
                    break

                # Reset to starting credit
                credit = args.start if args.start is not None else get_roll_quotes(session)['max_credit']
                log(f'  Resetting credit to ${credit:.2f}')
                leg_ids = place_roll(session, credit)

            # Every 3 cycles, creep credit down
            elif cycle % 3 == 0:
                if credit > min_credit and leg_ids:
                    old_credit = credit
                    cancel_legs(session, leg_ids)
                    time.sleep(0.5)
                    credit = round(credit - 0.01, 2)
                    if credit < min_credit:
                        credit = min_credit
                    log(f'  Creep ${old_credit:.2f} -> ${credit:.2f} (min=${min_credit:.2f})')
                    leg_ids = place_roll(session, credit)
                else:
                    log(f'  [wait] credit=${credit:.2f} | rolls={rolls_done}/{args.max} | earned=${total_credit:.2f}')

            # Every 10 cycles, refresh quotes
            if cycle % 10 == 0:
                q = get_roll_quotes(session)
                new_min = q['min_credit']
                log(f'  [REFRESH] max=${q["max_credit"]:.2f} mid=${q["mid_credit"]:.2f} min=${new_min:.2f}')
                # Only update min_credit if no floor override
                if args.floor is None:
                    min_credit = new_min

    except KeyboardInterrupt:
        log('Stopping...')
        if leg_ids:
            cancel_legs(session, leg_ids)
            log('  Cancelled open roll order')

    log('=' * 60)
    log(f'DONE: {rolls_done} rolls, total credit earned: ${total_credit:.2f} (${total_credit * 100:.0f} per contract)')


if __name__ == '__main__':
    main()
