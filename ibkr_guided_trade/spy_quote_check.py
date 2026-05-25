#!/usr/bin/env python3
"""
SPY Straddle Quote Checker — Cross-reference WS vs IBKR

Usage:
    python spy_quote_check.py                     # Check all held positions
    python spy_quote_check.py 667 668 669         # Check specific strikes
    python spy_quote_check.py --expiry 2026-04-30 667 668  # Specific expiry
    python spy_quote_check.py --all               # All strikes 660-675, all expiries
"""

import argparse
import sys

sys.path.insert(0, '.')
from ws_trading import (
    get_session, graphql_query, QUERY_OPTION_CHAIN, QUERY_FETCH_POSITIONS,
    load_config, load_cookies, extract_identity_from_cookies, KNOWN_SECURITIES,
)
from ib_insync import Option, Stock
from modules.common import connect

SPY_SEC_ID = KNOWN_SECURITIES.get('SPY', 'sec-s-27167ecbd81140fe9cdc02535f43174d')

# Expiries to check by default
DEFAULT_EXPIRIES = [
    ('2026-04-17', '20260417', 'Apr17'),
    ('2026-04-30', '20260430', 'Apr30'),
    ('2026-05-15', '20260515', 'May15'),
    ('2026-05-29', '20260529', 'May29'),
]


def get_ws_quotes(session, ws_expiry, strikes):
    """Get WS quotes for a list of strikes at one expiry"""
    results = {}
    for opt_type in ['CALL', 'PUT']:
        data = graphql_query(session, 'FetchOptionChain', QUERY_OPTION_CHAIN, {
            'id': SPY_SEC_ID, 'expiryDate': ws_expiry, 'optionType': opt_type,
            'realTimeQuote': True, 'includeGreeks': False, 'first': 80,
        })
        for edge in data.get('security', {}).get('optionChain', {}).get('edges', []):
            n = edge.get('node', {})
            d = n.get('optionDetails', {})
            K = float(d.get('strikePrice', 0))
            if K in strikes:
                q = n.get('quoteV2', {})
                bid = float(q.get('bid', 0) or 0)
                ask = float(q.get('ask', 0) or 0)
                if K not in results:
                    results[K] = {}
                results[K][opt_type] = {'bid': bid, 'ask': ask}
    return results


def get_ibkr_quotes(ib, ibkr_expiry, strikes):
    """Get IBKR quotes for a list of strikes at one expiry"""
    contracts = []
    for K in strikes:
        for right in ['C', 'P']:
            c = Option('SPY', ibkr_expiry, K, right, 'SMART')
            contracts.append(c)

    qualified = ib.qualifyContracts(*contracts)
    qualified = [c for c in qualified if c.conId > 0]
    tickers = ib.reqTickers(*qualified)
    ib.sleep(2)

    results = {}
    for t in tickers:
        K = t.contract.strike
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        opt_type = 'CALL' if t.contract.right == 'C' else 'PUT'
        if K not in results:
            results[K] = {}
        results[K][opt_type] = {'bid': bid, 'ask': ask}
    return results


def get_held_positions(session):
    """Get current SPY option positions"""
    config = load_config()
    cookies = load_cookies()
    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)

    pos_data = graphql_query(session, 'FetchIdentityPositions', QUERY_FETCH_POSITIONS, {
        'identityId': identity_id, 'currency': 'CAD', 'first': 50,
        'aggregated': True, 'currencyOverride': 'MARKET',
        'sort': 'TODAY_GAIN', 'includeSecurity': True,
        'includeAccountData': True, 'includeOneDayReturnsBaseline': True,
    })

    positions = {}
    edges = (pos_data.get('identity', {}).get('financials', {}).get('current', {})
             .get('positions', {}).get('edges', []))

    for edge in edges:
        node = edge.get('node', {})
        sec = node.get('security', {}) or {}
        opt = sec.get('optionDetails', {}) or {}
        if not opt:
            continue
        underlying = (opt.get('underlyingSecurity', {}) or {}).get('stock', {}) or {}
        if underlying.get('symbol', '') != 'SPY':
            continue
        strike = float(opt.get('strikePrice', 0))
        expiry = opt.get('expiryDate', '')
        qty = int(float(node.get('quantity', 0)))
        avg_price = node.get('marketAveragePrice', node.get('averagePrice', {}))
        avg_cost = float(avg_price.get('amount', 0)) if avg_price else 0

        key = (expiry, strike)
        if key not in positions:
            positions[key] = {'qty': 0, 'cost': 0}
        positions[key]['qty'] = max(positions[key]['qty'], qty)
        positions[key]['cost'] += avg_cost

    return positions


def print_comparison(ws_quotes, ibkr_quotes, strikes, label, positions=None):
    """Print side-by-side comparison"""
    print(f'\n{"="*90}')
    print(f'  {label}')
    print(f'{"="*90}')
    print(f'  {"Strike":>6} | {"WS Bid":>8} {"WS Mid":>8} {"WS Ask":>8} | {"IB Bid":>8} {"IB Mid":>8} {"IB Ask":>8} | {"Diff":>6} | Pos')
    print(f'  {"------":>6}-+-{"--------":>8}-{"--------":>8}-{"--------":>8}-+-{"--------":>8}-{"--------":>8}-{"--------":>8}-+-{"------":>6}-+----')

    for K in sorted(strikes):
        ws = ws_quotes.get(K, {})
        ib = ibkr_quotes.get(K, {})

        ws_c = ws.get('CALL', {})
        ws_p = ws.get('PUT', {})
        ib_c = ib.get('CALL', {})
        ib_p = ib.get('PUT', {})

        ws_bid = ws_c.get('bid', 0) + ws_p.get('bid', 0)
        ws_ask = ws_c.get('ask', 0) + ws_p.get('ask', 0)
        ws_mid = round((ws_bid + ws_ask) / 2, 2) if ws_bid and ws_ask else 0

        ib_bid = ib_c.get('bid', 0) + ib_p.get('bid', 0)
        ib_ask = ib_c.get('ask', 0) + ib_p.get('ask', 0)
        ib_mid = round((ib_bid + ib_ask) / 2, 2) if ib_bid and ib_ask else 0

        diff = round(ws_mid - ib_mid, 2) if ws_mid and ib_mid else 0

        # Position info
        pos_str = ''
        if positions:
            for (exp, s), info in positions.items():
                if s == K and label.split()[0] in exp.replace('-', ''):
                    pos_str = f'{info["qty"]}x @${info["cost"]/100:.2f}'

        diff_str = f'{diff:+.2f}' if diff else '  n/a'
        print(f'  ${K:5.0f} | ${ws_bid:7.2f} ${ws_mid:7.2f} ${ws_ask:7.2f} | ${ib_bid:7.2f} ${ib_mid:7.2f} ${ib_ask:7.2f} | {diff_str:>6} | {pos_str}')

    # Summary of call/put balance at forward ATM
    best_K = None
    best_diff = None
    for K in sorted(strikes):
        ib = ibkr_quotes.get(K, {})
        c = ib.get('CALL', {})
        p = ib.get('PUT', {})
        if not c or not p:
            continue
        c_mid = (c['bid'] + c['ask']) / 2
        p_mid = (p['bid'] + p['ask']) / 2
        if c_mid <= 0 or p_mid <= 0:
            continue
        d = abs(c_mid - p_mid)
        if best_diff is None or d < best_diff:
            best_diff = d
            best_K = K

    if best_K:
        ib = ibkr_quotes[best_K]
        c_mid = (ib['CALL']['bid'] + ib['CALL']['ask']) / 2
        p_mid = (ib['PUT']['bid'] + ib['PUT']['ask']) / 2
        fwd = round(best_K + c_mid - p_mid, 2)
        print(f'\n  Forward ATM: ${best_K:.0f} (fwd ${fwd:.2f}) | C ${c_mid:.2f} vs P ${p_mid:.2f} (diff ${c_mid-p_mid:+.2f})')


def main():
    parser = argparse.ArgumentParser(description='SPY Straddle Quote Checker')
    parser.add_argument('strikes', nargs='*', type=float, help='Strikes to check')
    parser.add_argument('--expiry', type=str, default=None, help='Specific expiry (YYYY-MM-DD)')
    parser.add_argument('--all', action='store_true', help='Check all strikes 660-675')
    args = parser.parse_args()

    print('Connecting...')
    session = get_session()
    ib = connect(client_id=99)
    spy = Stock('SPY', 'SMART', 'USD')
    ib.qualifyContracts(spy)
    [ticker] = ib.reqTickers(spy)
    spot = ticker.marketPrice()
    print(f'SPY spot: ${spot:.2f}')

    positions = get_held_positions(session)

    # Determine which expiries and strikes to check
    if args.expiry:
        expiries = [(e[0], e[1], e[2]) for e in DEFAULT_EXPIRIES if e[0] == args.expiry]
        if not expiries:
            ibkr_exp = args.expiry.replace('-', '')
            expiries = [(args.expiry, ibkr_exp, args.expiry)]
    elif args.strikes or args.all:
        expiries = DEFAULT_EXPIRIES
    else:
        # Auto-detect from positions
        held_expiries = set()
        for (exp, K) in positions:
            held_expiries.add(exp)
        expiries = [(e[0], e[1], e[2]) for e in DEFAULT_EXPIRIES if e[0] in held_expiries]
        if not expiries:
            expiries = DEFAULT_EXPIRIES[:2]

    if args.all:
        strikes = set(range(int(spot) - 7, int(spot) + 8))
    elif args.strikes:
        strikes = set(args.strikes)
    else:
        # Auto from positions + nearby
        strikes = set()
        for (exp, K) in positions:
            strikes.add(K)
        # Add nearby ATM
        for K in range(int(spot) - 3, int(spot) + 4):
            strikes.add(float(K))

    strikes = sorted(strikes)
    print(f'Checking {len(strikes)} strikes across {len(expiries)} expiries')

    for ws_exp, ibkr_exp, label in expiries:
        ws_quotes = get_ws_quotes(session, ws_exp, strikes)
        ibkr_quotes = get_ibkr_quotes(ib, ibkr_exp, strikes)
        print_comparison(ws_quotes, ibkr_quotes, strikes, f'{label} ({ws_exp})', positions)

    ib.disconnect()
    print('\nDone.')


if __name__ == '__main__':
    main()
