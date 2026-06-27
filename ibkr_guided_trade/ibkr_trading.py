#!/usr/bin/env python3
"""
IBKR Trading System - 3 Module Architecture

Modules:
  1. Trading  - Direct trading commands (status, orders, options scanning)
  2. Analysis - Account analysis, Greek exposures, risk metrics
  3. Algo     - Algorithmic trading with oscillation strategies

LESSONS LEARNED:
================
1. CLIENT ID MATTERS: Each ib_insync connection uses a client ID. If you use random
   client IDs, each reconnection creates a "new" client that can't see orders placed
   by previous connections. This leads to duplicate orders stacking up on IBKR's server.

2. SOLUTION - FIXED CLIENT ID: Use a consistent client ID (e.g., 50) for trading
   operations. This way, reconnections can see and manage previously placed orders.

3. VIEW ALL ORDERS: Use `ib.reqAllOpenOrders()` to fetch ALL orders from IBKR server,
   not just orders from the current client connection.

4. CANCEL ALL ORDERS: Use `ib.reqGlobalCancel()` to cancel ALL orders regardless of
   which client placed them.

Usage:
    # Trading Commands
    python ibkr_trading.py status              # Show open orders and positions
    python ibkr_trading.py snapshot            # Full account P&L snapshot
    python ibkr_trading.py cancel-all          # Cancel all open orders
    python ibkr_trading.py quote UNG           # Get stock quote
    python ibkr_trading.py opt-chain UNG 15    # Options chain ~15 DTE
    python ibkr_trading.py scan-puts UNG       # Scan 10-45 DTE for best puts
    python ibkr_trading.py sell-put UNG 20260116 11 0.35  # Sell put
    python ibkr_trading.py straddle SPY              # Straddle pricing near fwd ATM
    python ibkr_trading.py straddle SPY 30           # Target ~30 DTE
    python ibkr_trading.py buy-straddle SPY 20260403 570 12.50  # Buy 1 straddle
    python ibkr_trading.py buy-straddle SPY 20260403 570 12.50 --qty 5  # Buy 5

    # Analysis Commands
    python ibkr_trading.py greeks              # Portfolio-wide Greek exposure
    python ibkr_trading.py greeks UNG          # Greeks for specific symbol
    python ibkr_trading.py risk                # Risk analysis
    python ibkr_trading.py what-if UNG 10 12   # What-if for selling 10 puts @ $12

    # Scanner Commands
    python ibkr_trading.py scan                      # Full 3-phase S&P 500 scan
    python ibkr_trading.py scan --deep INTC,TSLA     # Deep dive specific tickers
    python ibkr_trading.py scan --quick              # Scan curated ~50 stocks
    python ibkr_trading.py scan --quick --portfolio  # Scan + portfolio optimization
    python ibkr_trading.py scan --phase1             # Phase 1 only
    python ibkr_trading.py scan --cached             # Use cached Phase 1

    # Algo Commands
    python ibkr_trading.py algo list           # List all strategies
    python ibkr_trading.py algo add UNG        # Create oscillation strategy
    python ibkr_trading.py algo show 1         # Show strategy details
    python ibkr_trading.py algo run            # Execute pending orders
    python ibkr_trading.py algo sync           # Sync order statuses
    python ibkr_trading.py algo pause 1        # Pause strategy
    python ibkr_trading.py algo resume 1       # Resume strategy
    python ibkr_trading.py algo cancel 1       # Cancel strategy
"""

import argparse

from modules import trading, analysis, algo, scanner


def main():
    parser = argparse.ArgumentParser(
        description='IBKR Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    Show open orders and positions
  %(prog)s greeks                    Portfolio Greeks exposure
  %(prog)s algo add UNG --buy-pct 2  Create buy-low strategy for UNG
  %(prog)s algo run                  Execute all active strategies
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # ==================== TRADING COMMANDS ====================

    # status
    subparsers.add_parser('status', help='Show open orders and positions')

    # snapshot
    subparsers.add_parser('snapshot', help='Full account P&L snapshot')

    # cancel-all
    subparsers.add_parser('cancel-all', help='Cancel all open orders')

    # modify
    p = subparsers.add_parser('modify', help='Modify an open order price')
    p.add_argument('symbol', help='Symbol to match (e.g., ZSL, UNG)')
    p.add_argument('new_price', type=float, help='New limit price')
    p.add_argument('--old-price', type=float, default=None, help='Current price to match (optional)')

    # quote
    p = subparsers.add_parser('quote', help='Get stock quote')
    p.add_argument('symbol', help='Stock symbol')

    # opt-chain
    p = subparsers.add_parser('opt-chain', help='Show options chain')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('dte', type=int, help='Target days to expiration')

    # scan-puts
    p = subparsers.add_parser('scan-puts', help='Scan multiple expirations for best put opportunities')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('--min-dte', type=int, default=10, help='Min days to expiration (default: 10)')
    p.add_argument('--max-dte', type=int, default=45, help='Max days to expiration (default: 45)')

    # scan-calls
    p = subparsers.add_parser('scan-calls', help='Scan multiple expirations for covered call opportunities')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('--min-dte', type=int, default=10, help='Min days to expiration (default: 10)')
    p.add_argument('--max-dte', type=int, default=45, help='Max days to expiration (default: 45)')

    # sell-put
    p = subparsers.add_parser('sell-put', help='Sell a put option')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('expiry', help='Expiry YYYYMMDD')
    p.add_argument('strike', type=float, help='Strike price')
    p.add_argument('price', type=float, help='Limit price')
    p.add_argument('--qty', type=int, default=1, help='Quantity (default: 1)')

    # spread
    p = subparsers.add_parser('spread', help='Place a vertical spread (bear call or bull put)')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('expiry', help='Expiry YYYYMMDD')
    p.add_argument('right', choices=['C', 'P'], help='C for calls (bear spread), P for puts (bull spread)')
    p.add_argument('short_strike', type=float, help='Strike to sell')
    p.add_argument('long_strike', type=float, help='Strike to buy')
    p.add_argument('--credit', type=float, default=None, help='Target credit (default: use mid)')
    p.add_argument('--qty', type=int, default=1, help='Quantity (default: 1)')
    p.add_argument('--use-mid', action='store_true', default=True, help='Use mid price (default)')
    p.add_argument('--aggressive', action='store_true', help='Start above mid price for better fill opportunity')
    p.add_argument('--dry-run', action='store_true', help='Show spread details without placing order')
    p.add_argument('--close', action='store_true', help='Close an existing spread (auto-detects direction from positions)')
    p.add_argument('--open-debit', action='store_true', help='Open a DEBIT spread (buy short_strike, sell long_strike)')
    p.add_argument('--debit', type=float, default=None, help='Target debit for closing or open-debit (default: use natural)')

    # straddle
    p = subparsers.add_parser('straddle', help='Show straddle pricing near forward ATM')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('dte', type=int, nargs='?', default=25, help='Target DTE (default: 25)')

    # buy-straddle
    p = subparsers.add_parser('buy-straddle', help='Buy a straddle (long call + long put)')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('expiry', help='Expiry YYYYMMDD')
    p.add_argument('strike', type=float, help='Strike price')
    p.add_argument('price', type=float, help='Limit price (total debit per straddle)')
    p.add_argument('--qty', type=int, default=1, help='Number of straddles (default: 1)')
    p.add_argument('--dry-run', action='store_true', help='Show details without placing order')

    # rc (reverse calendar) - 4-leg combo
    p = subparsers.add_parser('rc', help='Place a 4-leg reverse calendar (buy short-dated, sell long-dated)')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('short_expiry', help='Short-dated expiry YYYYMMDD (buy)')
    p.add_argument('long_expiry', help='Long-dated expiry YYYYMMDD (sell)')
    p.add_argument('put_strike', type=float, help='Put strike price')
    p.add_argument('call_strike', type=float, help='Call strike price')
    p.add_argument('--credit', type=float, default=None, help='Target credit when opening (default: use mid)')
    p.add_argument('--debit', type=float, default=None, help='Target debit when closing (default: use mid)')
    p.add_argument('--qty', type=int, default=1, help='Quantity (default: 1)')
    p.add_argument('--close', action='store_true', help='Close existing position (reverses all legs)')
    p.add_argument('--dry-run', action='store_true', help='Show details without placing order')

    # ==================== ANALYSIS COMMANDS ====================

    # greeks
    p = subparsers.add_parser('greeks', help='Portfolio Greek exposure')
    p.add_argument('symbol', nargs='?', default=None, help='Optional: specific symbol')

    # risk
    subparsers.add_parser('risk', help='Risk analysis')

    # what-if
    p = subparsers.add_parser('what-if', help='What-if analysis for new position')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('qty', type=int, help='Number of contracts')
    p.add_argument('strike', type=float, help='Strike price')

    # ==================== SCANNER COMMANDS ====================

    scan_parser = subparsers.add_parser('scan', help='Market scanner for credit spread opportunities')
    scan_parser.add_argument('--deep', type=str, default=None, help='Skip Phase 1, deep-dive specific tickers (comma-separated)')
    scan_parser.add_argument('--phase1', action='store_true', help='Phase 1 only')
    scan_parser.add_argument('--cached', action='store_true', help='Load Phase 1 from cache')
    scan_parser.add_argument('--quick', action='store_true', help='Use curated ~50 stock watchlist')
    scan_parser.add_argument('--min-drop', type=float, default=10, help='Min drop/rise%% threshold (default: 10)')
    scan_parser.add_argument('--min-edge', type=float, default=5, help='Min IV-RV edge%% (default: 5)')
    scan_parser.add_argument('--dte', type=int, default=35, help='Target DTE (default: 35)')
    scan_parser.add_argument('--top', type=int, default=10, help='Number of deep dives (default: 10)')
    scan_parser.add_argument('--portfolio', action='store_true', help='Run Markowitz portfolio optimization after scan')
    scan_parser.add_argument('--capital', type=float, default=10000, help='Total capital for portfolio optimization (default: 10000)')

    # ==================== ALGO COMMANDS ====================

    algo_parser = subparsers.add_parser('algo', help='Algorithmic trading commands')
    algo_sub = algo_parser.add_subparsers(dest='algo_command', help='Algo subcommands')

    # algo list
    algo_sub.add_parser('list', help='List all strategies')

    # algo add
    p = algo_sub.add_parser('add', help='Add oscillation strategy')
    p.add_argument('symbol', help='Stock symbol')
    p.add_argument('--buy-pct', type=float, default=2.0, help='Buy X%% below current price (default: 2)')
    p.add_argument('--qty', type=int, default=10, help='Shares per order (default: 10)')
    p.add_argument('--max-shares', type=int, default=100, help='Max shares to accumulate (default: 100)')
    p.add_argument('--sell-pct', type=float, default=3.0, help='Sell X%% above avg cost (default: 3)')
    p.add_argument('--sell-profit', type=float, default=0, help='Sell at fixed $ profit per share (default: 0)')

    # algo show
    p = algo_sub.add_parser('show', help='Show strategy details')
    p.add_argument('id', type=int, help='Strategy ID')

    # algo run
    algo_sub.add_parser('run', help='Execute pending orders for active strategies')

    # algo sync
    algo_sub.add_parser('sync', help='Sync order statuses with IBKR')

    # algo pause
    p = algo_sub.add_parser('pause', help='Pause a strategy')
    p.add_argument('id', type=int, help='Strategy ID')

    # algo resume
    p = algo_sub.add_parser('resume', help='Resume a paused strategy')
    p.add_argument('id', type=int, help='Strategy ID')

    # algo cancel
    p = algo_sub.add_parser('cancel', help='Cancel a strategy and its orders')
    p.add_argument('id', type=int, help='Strategy ID')

    # ==================== DISPATCH ====================

    args = parser.parse_args()

    # Trading commands
    if args.command == 'status':
        trading.cmd_status(args)
    elif args.command == 'snapshot':
        trading.cmd_snapshot(args)
    elif args.command == 'cancel-all':
        trading.cmd_cancel_all(args)
    elif args.command == 'modify':
        trading.cmd_modify_order(args)
    elif args.command == 'quote':
        trading.cmd_quote(args)
    elif args.command == 'opt-chain':
        trading.cmd_opt_chain(args)
    elif args.command == 'scan-puts':
        trading.cmd_scan_puts(args)
    elif args.command == 'scan-calls':
        trading.cmd_scan_calls(args)
    elif args.command == 'sell-put':
        trading.cmd_sell_put(args)
    elif args.command == 'spread':
        trading.cmd_spread(args)
    elif args.command == 'rc':
        trading.cmd_reverse_calendar(args)
    elif args.command == 'straddle':
        trading.cmd_straddle(args)
    elif args.command == 'buy-straddle':
        trading.cmd_buy_straddle(args)

    # Analysis commands
    elif args.command == 'greeks':
        analysis.cmd_greeks(args)
    elif args.command == 'risk':
        analysis.cmd_risk(args)
    elif args.command == 'what-if':
        analysis.cmd_whatif(args)

    # Scanner commands
    elif args.command == 'scan':
        scanner.cmd_scan(args)

    # Algo commands
    elif args.command == 'algo':
        if args.algo_command == 'list':
            algo.cmd_algo_list(args)
        elif args.algo_command == 'add':
            algo.cmd_algo_add(args)
        elif args.algo_command == 'show':
            algo.cmd_algo_show(args)
        elif args.algo_command == 'run':
            algo.cmd_algo_run(args)
        elif args.algo_command == 'sync':
            algo.cmd_algo_sync(args)
        elif args.algo_command == 'pause':
            algo.cmd_algo_pause(args)
        elif args.algo_command == 'resume':
            algo.cmd_algo_resume(args)
        elif args.algo_command == 'cancel':
            algo.cmd_algo_cancel(args)
        else:
            algo_parser.print_help()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
