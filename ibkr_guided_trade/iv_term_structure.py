#!/usr/bin/env python3
"""
IV Term Structure Analyzer
Fetches option chain for a stock to analyze IV by DTE
"""

import sys
from datetime import datetime, timedelta
from ws_trading import (
    get_session, graphql_query, KNOWN_SECURITIES,
    QUERY_OPTION_EXPIRATION_DATES, QUERY_OPTION_CHAIN
)


def analyze_iv_term_structure(symbol: str = 'AMD'):
    """Analyze IV term structure for a symbol"""

    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known: {list(KNOWN_SECURITIES.keys())}")
        return

    session = get_session()
    today = datetime.now()

    # Fetch expiries
    print(f"\nFetching {symbol} option expiries...")
    data = graphql_query(session, "FetchOptionExpirationDates", QUERY_OPTION_EXPIRATION_DATES, {
        "securityId": security_id,
        "minDate": today.strftime('%Y-%m-%d'),
        "maxDate": (today + timedelta(days=60)).strftime('%Y-%m-%d')
    })

    if not data:
        print("Failed to fetch expiries")
        return

    exp_data = data.get('security', {}).get('optionExpirationDates', {})
    expiries = exp_data.get('expirationDates', [])

    print(f"Found {len(expiries)} expiries: {expiries[:8]}...")

    # Get quote for current price
    quote_data = data.get('security', {}).get('quoteV2', {})
    spot = float(quote_data.get('last', 0) or quote_data.get('bid', 0) or 100)
    print(f"\n{symbol} Spot Price: ${spot:.2f}")

    print(f"\n{'='*80}")
    print(f"IV TERM STRUCTURE - {symbol}")
    print(f"{'='*80}")
    print(f"\n{'Expiry':<12} {'DTE':>5} {'ATM Put IV':>12} {'ATM Call IV':>12} {'Avg IV':>10}")
    print("-" * 60)

    iv_by_dte = {}

    for expiry in expiries[:8]:  # First 8 expiries
        exp_date = datetime.strptime(expiry, '%Y-%m-%d')
        dte = (exp_date - today).days

        # Fetch put and call chains
        puts = {}
        calls = {}

        for opt_type in ['PUT', 'CALL']:
            chain_data = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
                "id": security_id,
                "expiryDate": expiry,
                "optionType": opt_type,
                "realTimeQuote": True,
                "includeGreeks": True
            })

            if not chain_data:
                continue

            chain = chain_data.get('security', {}).get('optionChain', {})
            edges = chain.get('edges', [])

            for edge in edges:
                node = edge.get('node', {})
                details = node.get('optionDetails', {})
                greeks = details.get('greekSymbols', {}) or {}

                strike = float(details.get('strikePrice', 0))
                iv = float(greeks.get('impliedVolatility', 0) or 0)

                if opt_type == 'PUT':
                    puts[strike] = iv
                else:
                    calls[strike] = iv

        # Find ATM strike (closest to spot)
        all_strikes = sorted(set(puts.keys()) | set(calls.keys()))
        if not all_strikes:
            continue

        atm_strike = min(all_strikes, key=lambda x: abs(x - spot))

        atm_put_iv = puts.get(atm_strike, 0)
        atm_call_iv = calls.get(atm_strike, 0)

        # Average of put and call IV
        if atm_put_iv > 0 and atm_call_iv > 0:
            avg_iv = (atm_put_iv + atm_call_iv) / 2
        elif atm_put_iv > 0:
            avg_iv = atm_put_iv
        elif atm_call_iv > 0:
            avg_iv = atm_call_iv
        else:
            avg_iv = 0

        iv_by_dte[dte] = avg_iv

        print(f"{expiry:<12} {dte:>5} {atm_put_iv*100:>11.1f}% {atm_call_iv*100:>11.1f}% {avg_iv*100:>9.1f}%")

    print(f"\n{'='*80}")
    print("SUMMARY - Baseline IV by DTE (no earnings premium)")
    print(f"{'='*80}")

    # Calculate baseline
    print("\nCopy these values to intc_rc_analyzer.py:")
    print("\nself.baseline_iv = {")
    for dte in sorted(iv_by_dte.keys()):
        print(f"    {dte}: {iv_by_dte[dte]:.2f},")
    print("}")

    return iv_by_dte


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AMD'
    analyze_iv_term_structure(symbol)
