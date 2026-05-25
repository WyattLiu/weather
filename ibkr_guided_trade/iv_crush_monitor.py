#!/usr/bin/env python3
"""
IV Crush Monitor - Track IV throughout the day after earnings

Usage:
    python iv_crush_monitor.py INTC 52 55          # Monitor specific strikes
    python iv_crush_monitor.py INTC 52 55 --log    # Log to file
    python iv_crush_monitor.py INTC 52 55 --once   # Single snapshot
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

# Try IBKR first, fall back to Wealthsimple
try:
    from modules.trading import connect
    from ib_insync import Option
    USE_IBKR = True
except:
    from ws_trading import get_session, graphql_query, KNOWN_SECURITIES, QUERY_OPTION_CHAIN
    USE_IBKR = False


def get_iv_ibkr(symbol, expiry, strikes):
    """Get IV from IBKR"""
    ib = connect()

    results = {}

    for strike in strikes:
        for right in ['P', 'C']:
            opt = Option(symbol, expiry, strike, right, 'SMART')
            ib.qualifyContracts(opt)
            ib.reqMktData(opt, '', False, False)

    ib.sleep(3)

    for strike in strikes:
        for right in ['P', 'C']:
            opt = Option(symbol, expiry, strike, right, 'SMART')
            ticker = ib.ticker(opt)

            key = f"{expiry}_{strike}{right}"
            results[key] = {
                'strike': strike,
                'right': right,
                'expiry': expiry,
                'bid': ticker.bid if ticker.bid else 0,
                'ask': ticker.ask if ticker.ask else 0,
                'mid': (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else 0,
                'iv': ticker.modelGreeks.impliedVol if ticker.modelGreeks else 0,
                'delta': ticker.modelGreeks.delta if ticker.modelGreeks else 0,
            }

    ib.disconnect()
    return results


def get_iv_ws(symbol, expiry, strikes):
    """Get IV from Wealthsimple"""
    session = get_session()
    security_id = KNOWN_SECURITIES.get(symbol)

    if not security_id:
        print(f"Unknown symbol: {symbol}")
        return {}

    results = {}

    for opt_type in ['PUT', 'CALL']:
        data = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
            "id": security_id,
            "expiryDate": expiry,
            "optionType": opt_type,
            "realTimeQuote": True,
            "includeGreeks": True
        })

        if not data:
            continue

        chain = data.get('security', {}).get('optionChain', {})
        edges = chain.get('edges', [])

        for edge in edges:
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            quote = node.get('quoteV2', {})
            greeks = details.get('greekSymbols', {}) or {}

            strike = float(details.get('strikePrice', 0))

            if strike in strikes:
                right = 'P' if opt_type == 'PUT' else 'C'
                key = f"{expiry}_{strike}{right}"

                results[key] = {
                    'strike': strike,
                    'right': right,
                    'expiry': expiry,
                    'bid': float(quote.get('bid', 0) or 0),
                    'ask': float(quote.get('ask', 0) or 0),
                    'mid': (float(quote.get('bid', 0) or 0) + float(quote.get('ask', 0) or 0)) / 2,
                    'iv': float(greeks.get('impliedVolatility', 0) or 0),
                    'delta': float(greeks.get('delta', 0) or 0),
                }

    return results


def monitor_iv(symbol, put_strike, call_strike, expiries, interval=300, log_file=None):
    """Monitor IV at regular intervals"""

    strikes = [put_strike, call_strike]

    print(f"\n{'='*80}")
    print(f"IV CRUSH MONITOR - {symbol}")
    print(f"{'='*80}")
    print(f"Strikes: ${put_strike} Put, ${call_strike} Call")
    print(f"Expiries: {expiries}")
    print(f"Interval: {interval} seconds")
    print(f"Using: {'IBKR' if USE_IBKR else 'Wealthsimple'}")
    print(f"{'='*80}\n")

    # Header
    header = f"{'Time':<12}"
    for expiry in expiries:
        header += f" | {expiry} Put IV | {expiry} Call IV | {expiry} Put $ | {expiry} Call $"
    print(header)
    print("-" * len(header))

    log_data = []

    while True:
        try:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S")

            row = f"{timestamp:<12}"
            row_data = {'time': timestamp}

            for expiry in expiries:
                if USE_IBKR:
                    results = get_iv_ibkr(symbol, expiry, strikes)
                else:
                    results = get_iv_ws(symbol, expiry, strikes)

                put_key = f"{expiry}_{put_strike}P"
                call_key = f"{expiry}_{call_strike}C"

                put_iv = results.get(put_key, {}).get('iv', 0) * 100
                call_iv = results.get(call_key, {}).get('iv', 0) * 100
                put_mid = results.get(put_key, {}).get('mid', 0)
                call_mid = results.get(call_key, {}).get('mid', 0)

                row += f" |    {put_iv:>6.1f}% |     {call_iv:>6.1f}% |    ${put_mid:>5.2f} |     ${call_mid:>5.2f}"

                row_data[f'{expiry}_put_iv'] = put_iv
                row_data[f'{expiry}_call_iv'] = call_iv
                row_data[f'{expiry}_put_mid'] = put_mid
                row_data[f'{expiry}_call_mid'] = call_mid

            print(row)
            log_data.append(row_data)

            if log_file:
                with open(log_file, 'a') as f:
                    f.write(row + '\n')

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

    return log_data


def single_snapshot(symbol, put_strike, call_strike, expiries):
    """Take a single IV snapshot"""

    strikes = [put_strike, call_strike]

    print(f"\n{'='*80}")
    print(f"IV SNAPSHOT - {symbol} @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    for expiry in expiries:
        if USE_IBKR:
            results = get_iv_ibkr(symbol, expiry, strikes)
        else:
            results = get_iv_ws(symbol, expiry, strikes)

        print(f"Expiry: {expiry}")
        print("-" * 60)

        for strike in strikes:
            for right in ['P', 'C']:
                key = f"{expiry}_{strike}{right}"
                data = results.get(key, {})

                iv = data.get('iv', 0) * 100
                mid = data.get('mid', 0)
                bid = data.get('bid', 0)
                ask = data.get('ask', 0)
                delta = data.get('delta', 0)

                opt_type = "Put" if right == 'P' else "Call"
                print(f"  ${strike} {opt_type}: IV={iv:>6.1f}%  Mid=${mid:>5.2f}  Bid=${bid:.2f}/Ask=${ask:.2f}  Delta={delta:>+.2f}")

        print()


def main():
    parser = argparse.ArgumentParser(description='IV Crush Monitor')
    parser.add_argument('symbol', help='Stock symbol')
    parser.add_argument('put_strike', type=float, help='Put strike to monitor')
    parser.add_argument('call_strike', type=float, help='Call strike to monitor')
    parser.add_argument('--expiries', nargs='+', help='Expiries to monitor (YYYY-MM-DD)')
    parser.add_argument('--interval', type=int, default=300, help='Seconds between checks (default: 300)')
    parser.add_argument('--log', type=str, help='Log file path')
    parser.add_argument('--once', action='store_true', help='Single snapshot only')

    args = parser.parse_args()

    # Default expiries: today and next week
    if not args.expiries:
        from datetime import timedelta
        today = datetime.now()
        # Find next Friday
        days_until_friday = (4 - today.weekday()) % 7
        if days_until_friday == 0 and today.hour >= 16:
            days_until_friday = 7
        this_friday = today + timedelta(days=days_until_friday)
        next_friday = this_friday + timedelta(days=7)

        args.expiries = [
            this_friday.strftime('%Y-%m-%d'),
            next_friday.strftime('%Y-%m-%d'),
        ]

    if args.once:
        single_snapshot(args.symbol, args.put_strike, args.call_strike, args.expiries)
    else:
        monitor_iv(args.symbol, args.put_strike, args.call_strike, args.expiries,
                   args.interval, args.log)


if __name__ == '__main__':
    main()
