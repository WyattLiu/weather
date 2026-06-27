#!/usr/bin/env python3
"""
STX Reverse Calendar Analyzer & Trader
Uses Wealthsimple for option chain data and order placement

STX (Seagate) characteristics:
- Current price: ~$358
- Daily std dev: 3.54%
- Volatile stock with large swings
- No imminent earnings (as of Jan 2026)

Usage:
    python stx_rc_analyzer.py analyze              # Analyze best strikes
    python stx_rc_analyzer.py analyze --dte 25 53  # Compare Feb 20 / Mar 20
    python stx_rc_analyzer.py place                # Place the trade (interactive)
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


class STXReverseCalendarAnalyzer:
    def __init__(self, symbol: str = 'STX'):
        self.symbol = symbol
        self.security_id = KNOWN_SECURITIES.get(symbol)
        if not self.security_id:
            raise ValueError(f"Unknown symbol: {symbol}")

        self.session = get_session()
        self.r = 0.045  # Risk-free rate
        self.spot = None
        self.expiries = []
        self.chains = {}  # {expiry: {strike: {type: option_data}}}

        # STX IV model - calibrated for high-beta tech/storage stock
        # STX has ~70-80% IV typically, higher than INTC
        # For reverse calendar without earnings catalyst, IV crush is smaller
        # We're betting on directional move OR time decay differential
        #
        # STX IV term structure (approximate):
        #   - Near-term (4 DTE): ~120-130% (elevated)
        #   - Medium-term (25 DTE): ~75-80%
        #   - Longer-term (53 DTE): ~70%
        self.iv_by_dte = {
            0: 1.20,   # 0DTE: very elevated
            4: 1.25,   # 4DTE: elevated
            7: 1.10,   # 7DTE: still elevated
            14: 0.90,  # 14DTE: moderating
            25: 0.78,  # 25DTE: ~78%
            35: 0.73,  # 35DTE: ~73%
            53: 0.70,  # 53DTE: ~70% baseline
            80: 0.68,  # 80DTE: ~68%
        }

        # Support levels for STX (from historical analysis)
        self.support_levels = [
            277.65,  # Dec 17 low
            275.39,  # Dec 31 low
            258.67,  # Dec 3 low
        ]

    def get_iv_for_dte(self, dte: int) -> float:
        """Estimate IV for a given DTE based on term structure"""
        dtes = sorted(self.iv_by_dte.keys())

        if dte <= dtes[0]:
            return self.iv_by_dte[dtes[0]]
        if dte >= dtes[-1]:
            return self.iv_by_dte[dtes[-1]]

        # Linear interpolation
        for i in range(len(dtes) - 1):
            if dtes[i] <= dte <= dtes[i + 1]:
                lower_dte, upper_dte = dtes[i], dtes[i + 1]
                lower_iv = self.iv_by_dte[lower_dte]
                upper_iv = self.iv_by_dte[upper_dte]
                ratio = (dte - lower_dte) / (upper_dte - lower_dte)
                return lower_iv + ratio * (upper_iv - lower_iv)

        return 0.75  # Default

    def fetch_expiries(self) -> list:
        """Fetch available expiration dates"""
        today = datetime.now().strftime('%Y-%m-%d')
        max_date = (datetime.now() + timedelta(days=120)).strftime('%Y-%m-%d')

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
            return 358.0  # Default for STX

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

        self.spot = best_strike if best_strike else 358.0
        return self.spot

    def calculate_rc_pnl(self, short_expiry: str, long_expiry: str,
                         put_strike: float, call_strike: float,
                         spot_exit: float, iv_exit: float = None) -> dict:
        """
        Calculate Reverse Calendar P&L

        Reverse Calendar: Buy short-dated, Sell long-dated (same or different strikes)
        For STX without earnings: profits from big moves or IV differential
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

        # Use DTE-based IV model if not specified
        if iv_exit is None:
            # At short expiry, estimate long option's IV
            remaining_dte = long_dte - short_dte
            iv_exit = self.get_iv_for_dte(remaining_dte)

        # Exit values (assuming short-dated expires, long-dated has residual)
        # Short-dated: intrinsic only (or 0 if OTM)
        exit_short_put = max(0, put_strike - spot_exit)
        exit_short_call = max(0, spot_exit - call_strike)

        # Long-dated: use Black-Scholes with estimated IV
        T_long_exit = max(0.001, (long_dte - short_dte) / 365)
        exit_long_put = bs_price(spot_exit, put_strike, T_long_exit, self.r, iv_exit, 'put')
        exit_long_call = bs_price(spot_exit, call_strike, T_long_exit, self.r, iv_exit, 'call')

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
            'iv_exit': iv_exit,
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
            'put_oi_short': short_puts[put_strike]['oi'],
            'call_oi_short': short_calls[call_strike]['oi'],
        }

    def analyze_all_strikes(self, short_expiry: str, long_expiry: str,
                           iv_exit: float = None, min_oi: int = 10) -> list:
        """Analyze all valid strike combinations"""
        if not self.spot:
            self.get_spot_price()

        results = []

        short_puts = self.chains.get(short_expiry, {}).get('PUT', {})
        short_calls = self.chains.get(short_expiry, {}).get('CALL', {})

        # For STX at ~$358, look at strikes in reasonable range
        # Put strikes: 10-25% below spot
        # Call strikes: 0-15% above spot
        put_strikes = sorted([s for s in short_puts.keys()
                              if s < self.spot * 0.98 and s > self.spot * 0.75
                              and short_puts[s]['oi'] >= min_oi])
        call_strikes = sorted([s for s in short_calls.keys()
                               if s > self.spot * 0.98 and s < self.spot * 1.15
                               and short_calls[s]['oi'] >= min_oi])

        print(f"  Analyzing {len(put_strikes)} put strikes x {len(call_strikes)} call strikes...")

        for put_strike in put_strikes:
            for call_strike in call_strikes:
                # Calculate P&L at different spot moves
                # STX is volatile - consider larger moves
                pnls = {}
                for move in [-20, -15, -12, -10, -8, -5, 0, 5, 8, 10, 12, 15, 20]:
                    spot_exit = self.spot * (1 + move / 100)
                    result = self.calculate_rc_pnl(short_expiry, long_expiry,
                                                   put_strike, call_strike,
                                                   spot_exit, iv_exit)
                    if result:
                        pnls[move] = result['total_pnl']

                if pnls:
                    base_result = self.calculate_rc_pnl(short_expiry, long_expiry,
                                                        put_strike, call_strike,
                                                        self.spot, iv_exit)
                    if base_result:
                        base_result['pnl_by_move'] = pnls

                        # Calculate expected value
                        # STX daily std = 3.54%, so for 25 DTE, expected move ~17.7%
                        # Use slightly wider distribution
                        moves = list(pnls.keys())
                        std_move = 12  # Assume ~12% expected move for analysis
                        probs = np.exp(-np.array(moves)**2 / (2 * std_move**2))
                        probs = probs / probs.sum()
                        ev = sum(pnls[m] * p for m, p in zip(moves, probs))

                        base_result['expected_value'] = ev
                        base_result['max_loss'] = min(pnls.values())
                        base_result['max_profit'] = max(pnls.values())
                        base_result['pnl_flat'] = pnls.get(0, 0)
                        base_result['pnl_down_10'] = pnls.get(-10, 0)
                        base_result['pnl_up_10'] = pnls.get(10, 0)
                        results.append(base_result)

        # Sort by expected value
        results.sort(key=lambda x: x['expected_value'], reverse=True)
        return results

    def print_analysis(self, results: list, top_n: int = 10):
        """Print analysis results"""
        if not results:
            print("No valid combinations found")
            return

        print(f"\n{'='*120}")
        print(f"TOP {top_n} REVERSE CALENDAR COMBINATIONS (by Expected Value)")
        print(f"{'='*120}")
        print(f"Spot: ${self.spot:.2f}")
        print(f"Support levels: ${self.support_levels[0]:.0f}, ${self.support_levels[1]:.0f}, ${self.support_levels[2]:.0f}")
        print()

        print(f"{'Put':>7} {'Call':>7} {'Credit':>9} {'EV':>9} {'Flat':>8} "
              f"{'-10%':>8} {'+10%':>8} {'MaxLoss':>9} {'MaxProf':>9} "
              f"{'IVshort':>8} {'IVlong':>8} {'OI':>8}")
        print("-" * 120)

        for i, r in enumerate(results[:top_n]):
            pnl = r['pnl_by_move']
            iv_short = (r['put_iv_short'] + r['call_iv_short']) / 2 * 100
            iv_long = (r['put_iv_long'] + r['call_iv_long']) / 2 * 100
            total_oi = r['put_oi_short'] + r['call_oi_short']

            print(f"${r['put_strike']:>6.0f} ${r['call_strike']:>6.0f} "
                  f"${r['net_credit']:>+8.2f} ${r['expected_value']:>+8.2f} ${r['pnl_flat']:>+7.2f} "
                  f"${r['pnl_down_10']:>+7.2f} ${r['pnl_up_10']:>+7.2f} "
                  f"${r['max_loss']:>+8.2f} ${r['max_profit']:>+8.2f} "
                  f"{iv_short:>7.0f}% {iv_long:>7.0f}% {total_oi:>8}")

        print()

        # Print best option details
        best = results[0]
        print(f"\n{'='*120}")
        print("BEST COMBINATION DETAILS")
        print(f"{'='*120}")
        print(f"\nStrike Combo: ${best['put_strike']:.0f} Put / ${best['call_strike']:.0f} Call")
        print(f"Expiries: {best['short_expiry']} ({best['short_dte']}DTE) / {best['long_expiry']} ({best['long_dte']}DTE)")
        print()
        print("ENTRY:")
        print(f"  BUY  {best['short_expiry']} ${best['put_strike']:.0f} Put  @ ${best['entry_buy_put']:.2f}")
        print(f"  SELL {best['long_expiry']} ${best['put_strike']:.0f} Put  @ ${best['entry_sell_put']:.2f}")
        print(f"  BUY  {best['short_expiry']} ${best['call_strike']:.0f} Call @ ${best['entry_buy_call']:.2f}")
        print(f"  SELL {best['long_expiry']} ${best['call_strike']:.0f} Call @ ${best['entry_sell_call']:.2f}")
        print(f"\n  Net Credit: ${best['net_credit']:.2f} (${best['net_credit']*100:.0f} per combo)")
        print(f"  Expected Value: ${best['expected_value']:.2f}")
        print(f"  Max Loss (flat): ${best['max_loss']:.2f}")
        print(f"  Max Profit (big move): ${best['max_profit']:.2f}")

        # P&L table
        print("\n  P&L BY SPOT MOVE:")
        print(f"  {'-'*60}")
        for move in sorted(best['pnl_by_move'].keys()):
            spot_at = self.spot * (1 + move / 100)
            pnl = best['pnl_by_move'][move]
            bar = '█' * max(0, int((pnl + 20) / 2)) if pnl > -20 else ''
            print(f"  {move:+4}% (${spot_at:>7.2f}): ${pnl:>+8.2f}  {bar}")

        return best


def cmd_analyze(args):
    """Analyze reverse calendar combinations"""
    print(f"\n{'='*120}")
    print("STX REVERSE CALENDAR ANALYZER")
    print(f"{'='*120}")

    analyzer = STXReverseCalendarAnalyzer('STX')

    # Fetch expiries
    print("\nFetching expiration dates...")
    expiries = analyzer.fetch_expiries()

    if len(expiries) < 2:
        print("Not enough expiration dates available")
        return

    print(f"Available expiries: {expiries[:8]}")

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
        # Default: Feb 20 / Mar 20 (25 DTE / 53 DTE approximately)
        today = datetime.now()

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

        short_expiry = find_closest_expiry(25)  # ~Feb 20
        long_expiry = find_closest_expiry(53)   # ~Mar 20

    print(f"\nAnalyzing: {short_expiry} / {long_expiry}")

    # Fetch chains
    print("\nFetching option chains...")
    analyzer.fetch_all_chains([short_expiry, long_expiry])

    # Get spot
    spot = analyzer.get_spot_price()
    print(f"\nEstimated spot price: ${spot:.2f}")

    # Analyze
    print("\nAnalyzing all strike combinations...")
    iv_exit = args.iv_exit if args.iv_exit else None
    results = analyzer.analyze_all_strikes(short_expiry, long_expiry,
                                           iv_exit=iv_exit,
                                           min_oi=args.min_oi)

    if not results:
        print("No valid combinations found")
        return

    # Print results
    best = analyzer.print_analysis(results, top_n=args.top)

    # IBKR command
    if best:
        print(f"\n{'='*120}")
        print("IBKR COMMAND")
        print(f"{'='*120}")
        short_fmt = short_expiry.replace('-', '')
        long_fmt = long_expiry.replace('-', '')
        print(f"\n  python ibkr_trading.py rc STX {short_fmt} {long_fmt} {int(best['put_strike'])} {int(best['call_strike'])} --dry-run")
        print("\n  # Or with specific credit:")
        print(f"  python ibkr_trading.py rc STX {short_fmt} {long_fmt} {int(best['put_strike'])} {int(best['call_strike'])} --credit {best['net_credit']:.2f}")

    return best


def cmd_place(args):
    """Interactive trade placement"""
    print(f"\n{'='*120}")
    print("STX REVERSE CALENDAR TRADE PLACEMENT")
    print(f"{'='*120}")

    # First analyze to get best combination
    analyzer = STXReverseCalendarAnalyzer('STX')

    print("\nFetching data...")
    expiries = analyzer.fetch_expiries()

    if len(expiries) < 2:
        print("Not enough expiration dates")
        return

    # Default to ~25 DTE / ~53 DTE
    today = datetime.now()

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

    short_expiry = find_closest_expiry(25)
    long_expiry = find_closest_expiry(53)

    analyzer.fetch_all_chains([short_expiry, long_expiry])
    spot = analyzer.get_spot_price()

    print(f"\nSpot: ${spot:.2f}")
    print(f"Expiries: {short_expiry} / {long_expiry}")

    results = analyzer.analyze_all_strikes(short_expiry, long_expiry, min_oi=10)

    if not results:
        print("No valid combinations")
        return

    # Show top 5
    print("\nTop 5 combinations:")
    for i, r in enumerate(results[:5]):
        print(f"  {i+1}. ${r['put_strike']:.0f}P/${r['call_strike']:.0f}C - "
              f"Credit: ${r['net_credit']:.2f}, EV: ${r['expected_value']:.2f}, "
              f"Flat: ${r['pnl_flat']:.2f}")

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
    print(f"\n{'='*80}")
    print("TRADE CONFIRMATION")
    print(f"{'='*80}")
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

    # Show IBKR command
    short_fmt = selected['short_expiry'].replace('-', '')
    long_fmt = selected['long_expiry'].replace('-', '')
    print("\nIBKR Command:")
    print(f"  python ibkr_trading.py rc STX {short_fmt} {long_fmt} {int(selected['put_strike'])} {int(selected['call_strike'])} --credit {selected['net_credit']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='STX Reverse Calendar Analyzer')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze strike combinations')
    analyze_parser.add_argument('--dte', nargs=2, type=int, metavar=('SHORT', 'LONG'),
                               help='Target DTEs (e.g., --dte 25 53)')
    analyze_parser.add_argument('--iv-exit', type=float, default=None,
                               help='Expected exit IV (default: auto from term structure)')
    analyze_parser.add_argument('--top', type=int, default=10,
                               help='Show top N combinations (default: 10)')
    analyze_parser.add_argument('--min-oi', type=int, default=10,
                               help='Minimum open interest (default: 10)')

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
