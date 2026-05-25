#!/usr/bin/env python3
"""
SPY Straddle Ladder — Reusable entry tool

Samples straddle pricing over a few minutes, finds the true forward ATM
(delta-neutral strike), analyzes fill likelihood from bid/ask fluctuations,
and places a ladder of straddle orders accordingly.

Usage:
    python spy_ladder.py                          # Analyze only, 4 straddles, May 15
    python spy_ladder.py --place                  # Analyze + place orders
    python spy_ladder.py --qty 6 --place          # 6 straddles
    python spy_ladder.py --expiry 2026-04-30      # Different expiry
    python spy_ladder.py --samples 10 --interval 10  # 10 samples, 10s apart
    python spy_ladder.py --strike 673             # Override strike (skip ATM detection)
"""

import argparse
import math
import sys
import time
from datetime import datetime

sys.path.insert(0, '.')
from ws_trading import (
    get_session, graphql_query, QUERY_OPTION_CHAIN,
    place_multileg_order, KNOWN_SECURITIES
)

SPY_SEC_ID = KNOWN_SECURITIES.get('SPY', 'sec-s-27167ecbd81140fe9cdc02535f43174d')
DEFAULT_EXPIRY = '2026-05-15'


def fetch_straddle_strip(session, expiry, strikes=None):
    """Fetch call+put chains, return {strike: {call: {...}, put: {...}}} for near-ATM strikes."""
    chain = {}
    spot = 0

    for opt_type in ('CALL', 'PUT'):
        data = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
            "id": SPY_SEC_ID,
            "expiryDate": expiry,
            "optionType": opt_type,
            "realTimeQuote": True,
            "includeGreeks": True,
            "first": 80,
        })
        if not data:
            continue

        for edge in data.get('security', {}).get('optionChain', {}).get('edges', []):
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            quote = node.get('quoteV2', {})
            greeks = details.get('greekSymbols', {}) or {}
            strike = float(details.get('strikePrice', 0))

            if not spot:
                spot = float(quote.get('underlyingSpot', 0) or 0)

            # Filter to near-ATM if we know spot
            if strikes:
                if strike not in strikes:
                    continue
            elif spot > 0 and abs(strike - spot) > spot * 0.03:
                continue

            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0

            entry = {
                'sec_id': node.get('id', ''),
                'bid': bid, 'ask': ask, 'mid': mid,
                'delta': float(greeks.get('delta', 0) or 0),
                'gamma': float(greeks.get('gamma', 0) or 0),
                'theta': float(greeks.get('theta', 0) or 0),
                'vega': float(greeks.get('vega', 0) or 0),
                'iv': float(greeks.get('impliedVolatility', 0) or 0),
            }
            chain.setdefault(strike, {})[opt_type.lower()] = entry

    return chain, spot


def find_forward_atm(chain):
    """Find the delta-neutral strike using put-call parity."""
    best = None
    best_diff = float('inf')

    for strike, sides in sorted(chain.items()):
        c = sides.get('call', {})
        p = sides.get('put', {})
        if c.get('mid', 0) <= 0 or p.get('mid', 0) <= 0:
            continue

        # Forward ATM: strike where |call_mid - put_mid| is smallest
        diff = abs(c['mid'] - p['mid'])
        fwd = strike + c['mid'] - p['mid']
        strad_delta = c.get('delta', 0) + p.get('delta', 0)

        if diff < best_diff:
            best_diff = diff
            best = {
                'strike': strike,
                'forward': fwd,
                'strad_bid': c['bid'] + p['bid'],
                'strad_mid': c['mid'] + p['mid'],
                'strad_ask': c['ask'] + p['ask'],
                'call_mid': c['mid'],
                'put_mid': p['mid'],
                'delta': strad_delta,
                'gamma': c.get('gamma', 0) + p.get('gamma', 0),
                'theta': c.get('theta', 0) + p.get('theta', 0),
                'vega': c.get('vega', 0) + p.get('vega', 0),
                'iv': (c.get('iv', 0) + p.get('iv', 0)) / 2,
                'call_sec_id': c.get('sec_id', ''),
                'put_sec_id': p.get('sec_id', ''),
            }

    return best


def sample_pricing(session, expiry, num_samples, interval, strike_override=None):
    """Sample straddle pricing over time, return list of snapshots."""
    samples = []
    strikes_filter = {strike_override} if strike_override else None

    for i in range(num_samples):
        ts = datetime.now().strftime('%H:%M:%S')
        chain, spot = fetch_straddle_strip(session, expiry, strikes=strikes_filter)

        if strike_override:
            # Get data for specific strike
            sides = chain.get(strike_override, {})
            c = sides.get('call', {})
            p = sides.get('put', {})
            if c.get('mid', 0) > 0 and p.get('mid', 0) > 0:
                fwd_data = {
                    'strike': strike_override,
                    'forward': strike_override + c['mid'] - p['mid'],
                    'strad_bid': c['bid'] + p['bid'],
                    'strad_mid': c['mid'] + p['mid'],
                    'strad_ask': c['ask'] + p['ask'],
                    'delta': c.get('delta', 0) + p.get('delta', 0),
                    'iv': (c.get('iv', 0) + p.get('iv', 0)) / 2,
                    'call_sec_id': c.get('sec_id', ''),
                    'put_sec_id': p.get('sec_id', ''),
                    'gamma': c.get('gamma', 0) + p.get('gamma', 0),
                    'theta': c.get('theta', 0) + p.get('theta', 0),
                    'vega': c.get('vega', 0) + p.get('vega', 0),
                }
            else:
                fwd_data = None
        else:
            fwd_data = find_forward_atm(chain)

        if fwd_data:
            samples.append({
                'time': ts, 'spot': spot,
                **fwd_data,
            })
            sym = '+' if fwd_data['delta'] >= 0 else ''
            print(f"  [{ts}] spot ${spot:.2f} | fwd ${fwd_data['forward']:.2f} | "
                  f"${fwd_data['strike']:.0f} strad: "
                  f"bid ${fwd_data['strad_bid']:.2f} / mid ${fwd_data['strad_mid']:.2f} / ask ${fwd_data['strad_ask']:.2f} | "
                  f"delta {sym}{fwd_data['delta']:.3f} | IV {fwd_data['iv']*100:.1f}%")
        else:
            print(f"  [{ts}] No data")

        if i < num_samples - 1:
            time.sleep(interval)

    return samples


def analyze_samples(samples):
    """Analyze pricing fluctuations and compute fill probabilities."""
    if not samples:
        return None

    bids = [s['strad_bid'] for s in samples]
    mids = [s['strad_mid'] for s in samples]
    asks = [s['strad_ask'] for s in samples]
    deltas = [s['delta'] for s in samples]
    forwards = [s['forward'] for s in samples]
    spots = [s['spot'] for s in samples]

    analysis = {
        'strike': samples[-1]['strike'],
        'spot_range': (min(spots), max(spots)),
        'spot_last': spots[-1],
        'forward_range': (min(forwards), max(forwards)),
        'forward_last': forwards[-1],
        'bid_range': (min(bids), max(bids)),
        'mid_range': (min(mids), max(mids)),
        'ask_range': (min(asks), max(asks)),
        'bid_last': bids[-1],
        'mid_last': mids[-1],
        'ask_last': asks[-1],
        'spread': asks[-1] - bids[-1],
        'delta_range': (min(deltas), max(deltas)),
        'delta_last': deltas[-1],
        'iv': samples[-1]['iv'],
        'gamma': samples[-1].get('gamma', 0),
        'theta': samples[-1].get('theta', 0),
        'vega': samples[-1].get('vega', 0),
        'call_sec_id': samples[-1].get('call_sec_id', ''),
        'put_sec_id': samples[-1].get('put_sec_id', ''),
        'mid_volatility': max(mids) - min(mids),
        'bid_volatility': max(bids) - min(bids),
    }

    return analysis


def compute_ladder(analysis, qty):
    """Compute ladder prices based on observed fluctuations."""
    bid = analysis['bid_last']
    mid = analysis['mid_last']
    ask = analysis['ask_last']
    spread = analysis['spread']
    mid_vol = analysis['mid_volatility']

    # The spread tells us the market maker's edge
    # Our ladder should live between bid+$0.01 and mid
    # If mid fluctuates, we can sometimes get filled below the current mid

    ladder = []

    if qty <= 0:
        return ladder

    # Strategy: distribute orders from (mid - spread/3) up to mid
    # More aggressive (closer to mid) orders fill faster
    # Bottom orders only fill on brief dips

    # Effective range: from bid + $0.01 to mid
    low = round(bid + 0.01, 2)
    high = round(mid, 2)

    # If mid fluctuates, we can go slightly below current mid
    if mid_vol > 0.02:
        # Mid has been moving — place some orders where mid dipped to
        low = round(min(analysis['mid_range'][0] - 0.01, bid + 0.01), 2)

    if qty == 1:
        # Single order: just below mid
        ladder.append(round(mid - 0.01, 2))
    elif qty == 2:
        ladder.append(round(mid - 0.02, 2))
        ladder.append(round(mid - 0.01, 2))
    else:
        # Distribute: bottom 1/3 near low, middle 1/3 in range, top 1/3 near mid
        step = (high - low) / max(qty - 1, 1)

        # But cap step at $0.01 — penny increments are ideal
        if step < 0.01:
            # Range too tight for penny ladder — cluster near mid
            for i in range(qty):
                price = round(mid - (qty - 1 - i) * 0.01, 2)
                ladder.append(price)
        else:
            for i in range(qty):
                price = round(low + i * step, 2)
                ladder.append(price)

    # Assign fill likelihood based on position in the range
    result = []
    for price in ladder:
        if price >= mid:
            pct = 95
        elif price >= mid - 0.02:
            pct = 80
        elif price >= (bid + mid) / 2:
            pct = 50
        elif price > bid + 0.02:
            pct = 25
        else:
            pct = 10

        # Boost likelihood if we've seen mid dip to this level
        if mid_vol > 0 and price >= analysis['mid_range'][0]:
            pct = min(pct + 15, 95)

        result.append({
            'price': price,
            'fill_pct': pct,
            'vs_mid': price - mid,
        })

    return result


def place_straddle_order(session, call_sec_id, put_sec_id, price, qty=1):
    """Place a single straddle order."""
    legs = [
        {"securityId": put_sec_id, "orderType": "BUY_QUANTITY", "openClose": "OPEN"},
        {"securityId": call_sec_id, "orderType": "BUY_QUANTITY", "openClose": "OPEN"},
    ]
    result = place_multileg_order(session, legs, price, quantity_multiplier=qty)

    order_result = result.get('result', {})
    exec_data = order_result.get('soOrdersCreateOrderExecution', {})
    errors = exec_data.get('errors', [])

    if errors:
        err_msg = '; '.join(f"{e.get('code')}: {e.get('message')}" for e in errors)
        return False, err_msg, result.get('order_id', '')
    else:
        return True, 'OK', result.get('order_id', '')


def main():
    parser = argparse.ArgumentParser(description='SPY Straddle Ladder Tool')
    parser.add_argument('--expiry', default=DEFAULT_EXPIRY, help=f'Expiry date YYYY-MM-DD (default: {DEFAULT_EXPIRY})')
    parser.add_argument('--qty', type=int, default=4, help='Number of straddles (default: 4)')
    parser.add_argument('--samples', type=int, default=8, help='Number of price samples (default: 8)')
    parser.add_argument('--interval', type=int, default=15, help='Seconds between samples (default: 15)')
    parser.add_argument('--strike', type=float, default=None, help='Override strike (skip ATM detection)')
    parser.add_argument('--place', action='store_true', help='Actually place orders (default: analyze only)')
    parser.add_argument('--skip-confirm', action='store_true', help='Skip order confirmation prompt')
    args = parser.parse_args()

    session = get_session()
    expiry = args.expiry
    dte = (datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).days

    print("=" * 75)
    print(f"SPY STRADDLE LADDER — {expiry} ({dte} DTE)")
    print(f"Sampling {args.samples}x every {args.interval}s = ~{args.samples * args.interval // 60}m{args.samples * args.interval % 60}s")
    print("=" * 75)
    print()

    # Phase 1: Sample pricing
    print("PHASE 1: Sampling pricing...")
    samples = sample_pricing(session, expiry, args.samples, args.interval, args.strike)

    if not samples:
        print("No samples collected. Check market hours / expiry date.")
        return

    # Phase 2: Analyze
    print()
    print("PHASE 2: Analysis")
    print("-" * 75)
    analysis = analyze_samples(samples)

    if not analysis:
        print("Analysis failed.")
        return

    K = analysis['strike']
    print(f"  Forward ATM Strike: ${K:.0f}")
    print(f"  Forward Price:      ${analysis['forward_last']:.2f} (range: ${analysis['forward_range'][0]:.2f} - ${analysis['forward_range'][1]:.2f})")
    print(f"  SPY Spot:           ${analysis['spot_last']:.2f} (range: ${analysis['spot_range'][0]:.2f} - ${analysis['spot_range'][1]:.2f})")
    print(f"  Delta:              {analysis['delta_last']:+.4f} (range: {analysis['delta_range'][0]:+.4f} to {analysis['delta_range'][1]:+.4f})")
    print(f"  IV:                 {analysis['iv']*100:.1f}%")
    print(f"  Greeks/strad:       G {analysis['gamma']:.4f}  V ${analysis['vega']*100:.0f}/1%IV  Th -${abs(analysis['theta'])*100:.0f}/day")
    print()
    print(f"  Straddle Bid:       ${analysis['bid_last']:.2f} (range: ${analysis['bid_range'][0]:.2f} - ${analysis['bid_range'][1]:.2f})")
    print(f"  Straddle Mid:       ${analysis['mid_last']:.2f} (range: ${analysis['mid_range'][0]:.2f} - ${analysis['mid_range'][1]:.2f})")
    print(f"  Straddle Ask:       ${analysis['ask_last']:.2f} (range: ${analysis['ask_range'][0]:.2f} - ${analysis['ask_range'][1]:.2f})")
    print(f"  Spread:             ${analysis['spread']:.2f}")
    print(f"  Mid Fluctuation:    ${analysis['mid_volatility']:.2f} over sampling period")

    # Check if strike is truly delta-neutral
    if abs(analysis['delta_last']) > 0.05:
        print()
        print(f"  WARNING: Delta {analysis['delta_last']:+.4f} is non-trivial.")
        # Suggest neighboring strike
        if analysis['delta_last'] > 0:
            print(f"  Consider ${K+1:.0f} for more neutral delta (current strike is slightly bullish)")
        else:
            print(f"  Consider ${K-1:.0f} for more neutral delta (current strike is slightly bearish)")

    # Phase 3: Compute ladder
    print()
    print(f"PHASE 3: Ladder ({args.qty} straddles)")
    print("-" * 75)
    ladder = compute_ladder(analysis, args.qty)

    total_cost = 0
    print(f"  {'#':>3}  {'Price':>8}  {'vs Mid':>8}  {'Fill %':>7}  {'Cost':>10}")
    print(f"  {'-'*45}")
    for i, entry in enumerate(ladder):
        cost = entry['price'] * 100
        total_cost += cost
        print(f"  {i+1:>3}  ${entry['price']:.2f}  {entry['vs_mid']:+.2f}    {entry['fill_pct']:>3}%   ${cost:>9,.2f}")

    print(f"  {'-'*45}")
    print(f"  Total (if all fill): ${total_cost:,.2f}")
    print(f"  Greeks (if all fill): G +{analysis['gamma'] * args.qty:.1f}  V +${analysis['vega'] * 100 * args.qty:,.0f}/1%IV  Th -${abs(analysis['theta']) * 100 * args.qty:,.0f}/day")

    if not args.place:
        print()
        print("  [ANALYZE ONLY] Re-run with --place to submit orders")
        print(f"  Example: python spy_ladder.py --qty {args.qty} --expiry {expiry} --place")
        return

    # Phase 4: Place orders
    print()
    print("PHASE 4: Placing orders")
    print("-" * 75)

    if not args.skip_confirm:
        confirm = input(f"\n  Place {len(ladder)} straddle orders at ${K:.0f}? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("  Cancelled.")
            return

    call_sec_id = analysis['call_sec_id']
    put_sec_id = analysis['put_sec_id']

    if not call_sec_id or not put_sec_id:
        print("  ERROR: Missing security IDs. Cannot place orders.")
        return

    placed = 0
    order_ids = []
    for i, entry in enumerate(ladder):
        ok, msg, oid = place_straddle_order(session, call_sec_id, put_sec_id, entry['price'])
        status = "OK" if ok else f"FAILED: {msg}"
        print(f"  #{i+1} ${entry['price']:.2f} → {status} [{oid}]")
        if ok:
            placed += 1
            order_ids.append(oid)
        time.sleep(0.5)  # Brief pause between orders

    print()
    print(f"  Placed: {placed}/{len(ladder)} orders")
    if order_ids:
        print(f"  Order IDs: {', '.join(order_ids)}")
    print()
    print(f"  Monitor: python ws_trading.py open-orders")
    print(f"  Cancel:  python ws_trading.py cancel <order-id>")


if __name__ == '__main__':
    main()
