#!/usr/bin/env python3
"""
INTC Reverse Calendar Analyzer & Trader
Uses Wealthsimple for option chain data and order placement

Usage:
    python intc_rc_analyzer.py analyze              # Analyze best strikes
    python intc_rc_analyzer.py analyze --dte 7 14   # Compare 7DTE/14DTE
    python intc_rc_analyzer.py place                # Place the trade (interactive)
"""

import argparse
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm

# Import from ws_trading
from ws_trading import (
    get_session, graphql_query, KNOWN_SECURITIES,
    QUERY_OPTION_EXPIRATION_DATES, QUERY_OPTION_CHAIN
)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes option pricing"""
    if T <= 0.0001:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class INTCReverseCalendarAnalyzer:
    def __init__(self, symbol: str = 'INTC'):
        self.symbol = symbol
        self.security_id = KNOWN_SECURITIES.get(symbol)
        if not self.security_id:
            raise ValueError(f"Unknown symbol: {symbol}")

        self.session = get_session()
        self.r = 0.045  # Risk-free rate
        self.spot = None
        self.expiries = []
        self.chains = {}  # {expiry: {strike: {type: option_data}}}

        # Calibrated IV model - UPDATED after INTC trade post-mortem (Jan 23, 2026)
        # Pre-earnings IV by DTE: 0DTE: ~210%, 7DTE: ~95%
        # Actual post-earnings: 7DTE crushed to ~60-65% (not 52% as modeled)
        #
        # CONSERVATIVE estimates - better to underestimate profit than overestimate
        # Post-earnings, IV crushes to roughly:
        #   - Near-term: ~58-62% (was 52%, actual was ~60%)
        #   - Longer-term: ~52-55%
        self.post_earnings_iv = {
            0: 0.50,   # 0DTE expires at intrinsic
            7: 0.60,   # 7DTE post-crush: ~60% (UPDATED from 52%)
            14: 0.57,  # 14DTE post-crush: ~57% (UPDATED from 50%)
            21: 0.54,  # 21DTE post-crush: ~54% (UPDATED from 48%)
            28: 0.52,  # 28DTE post-crush: ~52%
            35: 0.50,  # 35DTE: ~50%
            56: 0.48,  # 56DTE: ~48% baseline
        }

    def get_post_earnings_iv(self, dte: int) -> float:
        """Estimate post-earnings IV for a given DTE"""
        # Interpolate from calibrated values
        dtes = sorted(self.post_earnings_iv.keys())

        if dte <= dtes[0]:
            return self.post_earnings_iv[dtes[0]]
        if dte >= dtes[-1]:
            return self.post_earnings_iv[dtes[-1]]

        # Linear interpolation
        for i in range(len(dtes) - 1):
            if dtes[i] <= dte <= dtes[i + 1]:
                lower_dte, upper_dte = dtes[i], dtes[i + 1]
                lower_iv = self.post_earnings_iv[lower_dte]
                upper_iv = self.post_earnings_iv[upper_dte]
                ratio = (dte - lower_dte) / (upper_dte - lower_dte)
                return lower_iv + ratio * (upper_iv - lower_iv)

        return 0.45  # Default

    def fetch_expiries(self) -> list:
        """Fetch available expiration dates"""
        today = datetime.now().strftime('%Y-%m-%d')
        max_date = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')

        data = graphql_query(self.session, "FetchOptionExpirationDates", QUERY_OPTION_EXPIRATION_DATES, {
            "securityId": self.security_id,
            "minDate": today,
            "maxDate": max_date
        })

        if not data:
            return []

        exp_data = data.get('security', {}).get('optionExpirationDates', {})
        self.expiries = exp_data.get('expirationDates', [])
        return self.expiries

    def fetch_chain(self, expiry: str, option_type: str) -> dict:
        """Fetch option chain for a specific expiry and type"""
        data = graphql_query(self.session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
            "id": self.security_id,
            "expiryDate": expiry,
            "optionType": option_type.upper(),
            "realTimeQuote": True,
            "includeGreeks": True
        })

        if not data:
            return {}

        chain = data.get('security', {}).get('optionChain', {})
        edges = chain.get('edges', [])

        result = {}
        for edge in edges:
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            quote = node.get('quoteV2', {})
            greeks = details.get('greekSymbols', {}) or {}

            strike = float(details.get('strikePrice', 0))
            result[strike] = {
                'id': node.get('id', ''),
                'bid': float(quote.get('bid', 0) or 0),
                'ask': float(quote.get('ask', 0) or 0),
                'last': float(quote.get('last', 0) or 0),
                'mid': (float(quote.get('bid', 0) or 0) + float(quote.get('ask', 0) or 0)) / 2,
                'oi': int(quote.get('openInterest', 0) or 0),
                'iv': float(greeks.get('impliedVolatility', 0) or 0),
                'delta': float(greeks.get('delta', 0) or 0),
                'theta': float(greeks.get('theta', 0) or 0),
            }

        return result

    def fetch_all_chains(self, expiries: list):
        """Fetch chains for multiple expiries"""
        for expiry in expiries:
            self.chains[expiry] = {
                'PUT': self.fetch_chain(expiry, 'PUT'),
                'CALL': self.fetch_chain(expiry, 'CALL')
            }
            print(f"  Fetched {expiry}: {len(self.chains[expiry]['PUT'])} puts, {len(self.chains[expiry]['CALL'])} calls")

    def get_spot_price(self) -> float:
        """Estimate spot price from ATM options"""
        if not self.chains:
            return 54.45  # Default

        # Use the first expiry's chain to estimate spot
        first_expiry = list(self.chains.keys())[0]
        puts = self.chains[first_expiry]['PUT']
        calls = self.chains[first_expiry]['CALL']

        # Find strike where put and call are closest in price
        best_strike = None
        min_diff = float('inf')

        for strike in puts:
            if strike in calls:
                put_mid = puts[strike]['mid']
                call_mid = calls[strike]['mid']
                diff = abs(put_mid - call_mid)
                if diff < min_diff:
                    min_diff = diff
                    best_strike = strike

        self.spot = best_strike if best_strike else 54.45
        return self.spot

    def calculate_rc_pnl(self, short_expiry: str, long_expiry: str,
                         put_strike: float, call_strike: float,
                         spot_exit: float, iv_post: float = None) -> dict:
        """
        Calculate Reverse Calendar P&L

        Reverse Calendar: Buy short-dated, Sell long-dated (same or different strikes)
        """
        short_puts = self.chains.get(short_expiry, {}).get('PUT', {})
        short_calls = self.chains.get(short_expiry, {}).get('CALL', {})
        long_puts = self.chains.get(long_expiry, {}).get('PUT', {})
        long_calls = self.chains.get(long_expiry, {}).get('CALL', {})

        if put_strike not in short_puts or put_strike not in long_puts:
            return None
        if call_strike not in short_calls or call_strike not in long_calls:
            return None

        # Entry prices (buy at ask, sell at bid)
        entry_buy_put = short_puts[put_strike]['ask']
        entry_sell_put = long_puts[put_strike]['bid']
        entry_buy_call = short_calls[call_strike]['ask']
        entry_sell_call = long_calls[call_strike]['bid']

        if entry_buy_put == 0 or entry_sell_put == 0 or entry_buy_call == 0 or entry_sell_call == 0:
            return None

        net_credit = (entry_sell_put - entry_buy_put) + (entry_sell_call - entry_buy_call)

        # Calculate DTE
        short_date = datetime.strptime(short_expiry, '%Y-%m-%d')
        long_date = datetime.strptime(long_expiry, '%Y-%m-%d')
        today = datetime.now()

        short_dte = max(0, (short_date - today).days)
        long_dte = max(0, (long_date - today).days)

        # Use DTE-based post-IV model if not specified
        if iv_post is None:
            # Post-earnings IV depends on remaining DTE after short expires
            remaining_dte = long_dte - short_dte
            iv_post = self.get_post_earnings_iv(remaining_dte)

        # Exit values (assuming short-dated expires, long-dated has residual)
        # Short-dated: intrinsic only (or 0 if OTM)
        exit_short_put = max(0, put_strike - spot_exit)
        exit_short_call = max(0, spot_exit - call_strike)

        # Long-dated: use Black-Scholes with post-IV
        T_long_exit = max(0.001, (long_dte - short_dte) / 365)
        exit_long_put = bs_price(spot_exit, put_strike, T_long_exit, self.r, iv_post, 'put')
        exit_long_call = bs_price(spot_exit, call_strike, T_long_exit, self.r, iv_post, 'call')

        # P&L
        pnl_put = (exit_short_put - entry_buy_put) + (entry_sell_put - exit_long_put)
        pnl_call = (exit_short_call - entry_buy_call) + (entry_sell_call - exit_long_call)
        total_pnl = pnl_put + pnl_call

        return {
            'put_strike': put_strike,
            'call_strike': call_strike,
            'short_expiry': short_expiry,
            'long_expiry': long_expiry,
            'short_dte': short_dte,
            'long_dte': long_dte,
            'entry_buy_put': entry_buy_put,
            'entry_sell_put': entry_sell_put,
            'entry_buy_call': entry_buy_call,
            'entry_sell_call': entry_sell_call,
            'net_credit': net_credit,
            'spot_exit': spot_exit,
            'iv_post': iv_post,
            'exit_short_put': exit_short_put,
            'exit_short_call': exit_short_call,
            'exit_long_put': exit_long_put,
            'exit_long_call': exit_long_call,
            'pnl_put': pnl_put,
            'pnl_call': pnl_call,
            'total_pnl': total_pnl,
            'put_id_short': short_puts[put_strike]['id'],
            'put_id_long': long_puts[put_strike]['id'],
            'call_id_short': short_calls[call_strike]['id'],
            'call_id_long': long_calls[call_strike]['id'],
            'put_iv_short': short_puts[put_strike]['iv'],
            'put_iv_long': long_puts[put_strike]['iv'],
            'call_iv_short': short_calls[call_strike]['iv'],
            'call_iv_long': long_calls[call_strike]['iv'],
        }

    def analyze_all_strikes(self, short_expiry: str, long_expiry: str,
                           iv_post: float = 0.50) -> list:
        """Analyze all valid strike combinations"""
        if not self.spot:
            self.get_spot_price()

        results = []

        short_puts = self.chains.get(short_expiry, {}).get('PUT', {})
        short_calls = self.chains.get(short_expiry, {}).get('CALL', {})

        # Get put strikes below spot, call strikes above spot
        put_strikes = sorted([s for s in short_puts.keys() if s < self.spot and s > self.spot * 0.85])
        call_strikes = sorted([s for s in short_calls.keys() if s > self.spot and s < self.spot * 1.15])

        for put_strike in put_strikes:
            for call_strike in call_strikes:
                # Calculate P&L at different spot moves
                pnls = {}
                for move in [-15, -12, -10, -8, -5, 0, 5, 8, 10, 12, 15]:
                    spot_exit = self.spot * (1 + move / 100)
                    result = self.calculate_rc_pnl(short_expiry, long_expiry,
                                                   put_strike, call_strike,
                                                   spot_exit, iv_post)
                    if result:
                        pnls[move] = result['total_pnl']

                if pnls:
                    base_result = self.calculate_rc_pnl(short_expiry, long_expiry,
                                                        put_strike, call_strike,
                                                        self.spot, iv_post)
                    if base_result:
                        base_result['pnl_by_move'] = pnls
                        # Calculate expected value (normal dist, std=10%)
                        moves = list(pnls.keys())
                        probs = np.exp(-np.array(moves)**2 / (2 * 10**2))
                        probs = probs / probs.sum()
                        ev = sum(pnls[m] * p for m, p in zip(moves, probs))
                        base_result['expected_value'] = ev
                        base_result['max_loss'] = min(pnls.values())
                        base_result['max_profit'] = max(pnls.values())
                        results.append(base_result)

        # Sort by expected value
        results.sort(key=lambda x: x['expected_value'], reverse=True)
        return results

    def print_analysis(self, results: list, top_n: int = 10):
        """Print analysis results"""
        if not results:
            print("No valid combinations found")
            return

        print(f"\n{'='*100}")
        print(f"TOP {top_n} REVERSE CALENDAR COMBINATIONS (by Expected Value)")
        print(f"{'='*100}")
        print(f"Spot: ${self.spot:.2f}")
        print()

        print(f"{'Put':>6} {'Call':>6} {'Credit':>8} {'EV':>8} {'MaxLoss':>8} "
              f"{'0%':>7} {'-8%':>7} {'+8%':>7} {'-12%':>7} {'+12%':>7} "
              f"{'IV Short':>10} {'IV Long':>10}")
        print("-" * 100)

        for i, r in enumerate(results[:top_n]):
            pnl = r['pnl_by_move']
            iv_short = (r['put_iv_short'] + r['call_iv_short']) / 2 * 100
            iv_long = (r['put_iv_long'] + r['call_iv_long']) / 2 * 100

            print(f"${r['put_strike']:>5.0f} ${r['call_strike']:>5.0f} "
                  f"${r['net_credit']:>+7.2f} ${r['expected_value']:>+7.2f} ${r['max_loss']:>+7.2f} "
                  f"${pnl.get(0, 0):>+6.2f} ${pnl.get(-8, 0):>+6.2f} ${pnl.get(8, 0):>+6.2f} "
                  f"${pnl.get(-12, 0):>+6.2f} ${pnl.get(12, 0):>+6.2f} "
                  f"{iv_short:>9.1f}% {iv_long:>9.1f}%")

        print()

        # Print best option details
        best = results[0]
        print(f"\n{'='*100}")
        print("BEST COMBINATION DETAILS")
        print(f"{'='*100}")
        print(f"\nStrike Combo: ${best['put_strike']:.0f} Put / ${best['call_strike']:.0f} Call")
        print(f"Expiries: {best['short_expiry']} ({best['short_dte']}DTE) / {best['long_expiry']} ({best['long_dte']}DTE)")
        print()
        print("ENTRY:")
        print(f"  BUY  {best['short_expiry']} ${best['put_strike']:.0f} Put  @ ${best['entry_buy_put']:.2f}  (ID: {best['put_id_short']})")
        print(f"  SELL {best['long_expiry']} ${best['put_strike']:.0f} Put  @ ${best['entry_sell_put']:.2f}  (ID: {best['put_id_long']})")
        print(f"  BUY  {best['short_expiry']} ${best['call_strike']:.0f} Call @ ${best['entry_buy_call']:.2f}  (ID: {best['call_id_short']})")
        print(f"  SELL {best['long_expiry']} ${best['call_strike']:.0f} Call @ ${best['entry_sell_call']:.2f}  (ID: {best['call_id_long']})")
        print(f"\n  Net Credit: ${best['net_credit']:.2f} (${best['net_credit']*100:.0f} per combo)")
        print(f"  Expected Value: ${best['expected_value']:.2f}")
        print(f"  Max Loss: ${best['max_loss']:.2f}")

        return best


def cmd_analyze(args):
    """Analyze reverse calendar combinations"""
    print(f"\n{'='*100}")
    print("INTC REVERSE CALENDAR ANALYZER")
    print(f"{'='*100}")

    analyzer = INTCReverseCalendarAnalyzer('INTC')

    # Fetch expiries
    print("\nFetching expiration dates...")
    expiries = analyzer.fetch_expiries()

    if len(expiries) < 2:
        print("Not enough expiration dates available")
        return

    print(f"Available expiries: {expiries[:6]}")

    # Determine which expiries to use
    if args.dte:
        # User specified DTE
        short_dte, long_dte = args.dte
        today = datetime.now()

        # Find closest expiries to target DTE
        def find_closest_expiry(target_dte):
            best = None
            min_diff = float('inf')
            for exp in expiries:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                dte = (exp_date - today).days
                if abs(dte - target_dte) < min_diff:
                    min_diff = abs(dte - target_dte)
                    best = exp
            return best

        short_expiry = find_closest_expiry(short_dte)
        long_expiry = find_closest_expiry(long_dte)
    else:
        # Default: use first two expiries (0DTE/7DTE or closest)
        short_expiry = expiries[0]
        long_expiry = expiries[1] if len(expiries) > 1 else expiries[0]

    print(f"\nAnalyzing: {short_expiry} / {long_expiry}")

    # Fetch chains
    print("\nFetching option chains...")
    analyzer.fetch_all_chains([short_expiry, long_expiry])

    # Get spot
    spot = analyzer.get_spot_price()
    print(f"\nEstimated spot price: ${spot:.2f}")

    # Analyze
    print("\nAnalyzing all strike combinations...")
    results = analyzer.analyze_all_strikes(short_expiry, long_expiry, iv_post=args.iv_post)

    if not results:
        print("No valid combinations found")
        return

    # Print results
    best = analyzer.print_analysis(results, top_n=args.top)

    # Compare different DTE combos if requested
    if args.compare_dte and len(expiries) >= 3:
        print(f"\n{'='*100}")
        print("COMPARING DIFFERENT DTE COMBINATIONS")
        print(f"{'='*100}")

        combos = [
            (expiries[0], expiries[1], "0DTE/7DTE"),
            (expiries[1], expiries[2], "7DTE/14DTE"),
        ]

        for short_exp, long_exp, label in combos:
            if short_exp in analyzer.chains and long_exp in analyzer.chains:
                continue  # Already fetched
            analyzer.fetch_all_chains([short_exp, long_exp])

        for short_exp, long_exp, label in combos:
            print(f"\n{label} ({short_exp} / {long_exp}):")
            results = analyzer.analyze_all_strikes(short_exp, long_exp, iv_post=args.iv_post)
            if results:
                best = results[0]
                print(f"  Best: ${best['put_strike']:.0f}P/${best['call_strike']:.0f}C")
                print(f"  Credit: ${best['net_credit']:.2f}, EV: ${best['expected_value']:.2f}, MaxLoss: ${best['max_loss']:.2f}")

    return best


def cmd_place(args):
    """Interactive trade placement"""
    print(f"\n{'='*100}")
    print("REVERSE CALENDAR TRADE PLACEMENT")
    print(f"{'='*100}")

    # First analyze to get best combination
    analyzer = INTCReverseCalendarAnalyzer('INTC')

    print("\nFetching data...")
    expiries = analyzer.fetch_expiries()

    if len(expiries) < 2:
        print("Not enough expiration dates")
        return

    short_expiry = expiries[0]
    long_expiry = expiries[1]

    analyzer.fetch_all_chains([short_expiry, long_expiry])
    spot = analyzer.get_spot_price()

    print(f"\nSpot: ${spot:.2f}")
    print(f"Expiries: {short_expiry} / {long_expiry}")

    results = analyzer.analyze_all_strikes(short_expiry, long_expiry, iv_post=0.50)

    if not results:
        print("No valid combinations")
        return

    # Show top 5
    print("\nTop 5 combinations:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. ${r['put_strike']:.0f}P/${r['call_strike']:.0f}C - "
              f"Credit: ${r['net_credit']:.2f}, EV: ${r['expected_value']:.2f}")

    # Let user choose
    choice = input("\nSelect combination (1-5) or 'q' to quit: ").strip()

    if choice.lower() == 'q':
        return

    try:
        idx = int(choice) - 1
        selected = results[idx]
    except (ValueError, IndexError):
        print("Invalid choice")
        return

    # Confirm trade
    print(f"\n{'='*60}")
    print("TRADE CONFIRMATION")
    print(f"{'='*60}")
    print(f"\nReverse Calendar: ${selected['put_strike']:.0f}P / ${selected['call_strike']:.0f}C")
    print(f"\nLEG 1: BUY  {selected['short_expiry']} ${selected['put_strike']:.0f} Put  @ ${selected['entry_buy_put']:.2f}")
    print(f"LEG 2: SELL {selected['long_expiry']} ${selected['put_strike']:.0f} Put  @ ${selected['entry_sell_put']:.2f}")
    print(f"LEG 3: BUY  {selected['short_expiry']} ${selected['call_strike']:.0f} Call @ ${selected['entry_buy_call']:.2f}")
    print(f"LEG 4: SELL {selected['long_expiry']} ${selected['call_strike']:.0f} Call @ ${selected['entry_sell_call']:.2f}")
    print(f"\nNet Credit: ${selected['net_credit']:.2f}")

    qty = input("\nEnter quantity (contracts per leg): ").strip()
    try:
        qty = int(qty)
    except ValueError:
        print("Invalid quantity")
        return

    total_credit = selected['net_credit'] * qty * 100
    print(f"\nTotal Credit: ${total_credit:.2f}")

    confirm = input("\nProceed with trade? (yes/no): ").strip().lower()

    if confirm != 'yes':
        print("Trade cancelled")
        return

    # Place orders
    print("\nPlacing orders...")
    print("\nTo place these orders manually, run:")
    print("\n  # Leg 1: Buy short-dated put")
    print(f"  python ws_trading.py buy-opt {selected['put_id_short']} {qty} {selected['entry_buy_put']:.2f}")
    print("\n  # Leg 2: Sell long-dated put")
    print(f"  python ws_trading.py sell-opt {selected['put_id_long']} {qty} {selected['entry_sell_put']:.2f}")
    print("\n  # Leg 3: Buy short-dated call")
    print(f"  python ws_trading.py buy-opt {selected['call_id_short']} {qty} {selected['entry_buy_call']:.2f}")
    print("\n  # Leg 4: Sell long-dated call")
    print(f"  python ws_trading.py sell-opt {selected['call_id_long']} {qty} {selected['entry_sell_call']:.2f}")

    # Optionally execute
    execute = input("\nExecute orders now? (yes/no): ").strip().lower()

    if execute == 'yes':
        from ws_trading import place_order, DEFAULT_ACCOUNT_ID

        orders = [
            ('BUY', selected['put_id_short'], qty, selected['entry_buy_put'], 'OPEN'),
            ('SELL', selected['put_id_long'], qty, selected['entry_sell_put'], 'OPEN'),
            ('BUY', selected['call_id_short'], qty, selected['entry_buy_call'], 'OPEN'),
            ('SELL', selected['call_id_long'], qty, selected['entry_sell_call'], 'OPEN'),
        ]

        session = analyzer.session

        for action, sec_id, q, price, oc in orders:
            order_type = 'BUY_QUANTITY' if action == 'BUY' else 'SELL_QUANTITY'

            print(f"\n  Placing {action} {q}x {sec_id} @ ${price:.2f}...")

            result = place_order(
                session=session,
                account_id=DEFAULT_ACCOUNT_ID,
                security_id=sec_id,
                order_type=order_type,
                order_sub_type='LIMIT',
                quantity=q,
                limit_price=price,
                time_in_force='DAY'
            )

            if result:
                print(f"    Order placed: {result.get('id', 'Unknown')}")
            else:
                print("    Order FAILED")

        print("\n✓ All orders submitted")


def main():
    parser = argparse.ArgumentParser(description='INTC Reverse Calendar Analyzer')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze strike combinations')
    analyze_parser.add_argument('--dte', nargs=2, type=int, metavar=('SHORT', 'LONG'),
                               help='Target DTEs (e.g., --dte 7 14)')
    analyze_parser.add_argument('--iv-post', type=float, default=0.50,
                               help='Expected post-earnings IV (default: 0.50)')
    analyze_parser.add_argument('--top', type=int, default=10,
                               help='Show top N combinations (default: 10)')
    analyze_parser.add_argument('--compare-dte', action='store_true',
                               help='Compare different DTE combinations')

    # Place command
    subparsers.add_parser('place', help='Place trade interactively')

    args = parser.parse_args()

    if args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'place':
        cmd_place(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
