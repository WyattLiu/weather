"""
Module 2: Account Analysis & Greek Exposures

Commands for analyzing portfolio risk and Greek exposures.
"""
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportPossiblyUnboundVariable=false

from ib_insync import Stock
from collections import defaultdict

from .common import connect, get_timestamp, DEFAULT_ACCOUNT


def cmd_greeks(args):
    """Show portfolio-wide Greek exposure summary"""
    ib = connect()

    print(f"{'='*60}")
    print(f"PORTFOLIO GREEKS - {get_timestamp()}")
    print(f"{'='*60}")

    ib.sleep(3)

    # Get all positions
    positions = ib.positions()

    if not positions:
        print("\nNo positions found.")
        ib.disconnect()
        return

    # Filter for options only (or specific symbol if provided)
    option_positions = []
    for p in positions:
        if p.contract.secType == 'OPT':
            if args.symbol is None or p.contract.symbol == args.symbol.upper():
                option_positions.append(p)

    if not option_positions:
        msg = "No option positions found"
        if args.symbol:
            msg += f" for {args.symbol.upper()}"
        print(f"\n{msg}.")
        ib.disconnect()
        return

    # Request market data for all options to get Greeks
    for p in option_positions:
        ib.qualifyContracts(p.contract)
        ib.reqMktData(p.contract, '', False, False)

    ib.sleep(4)

    # Aggregate Greeks
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0

    by_underlying = defaultdict(lambda: {
        'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0,
        'contracts': 0, 'positions': []
    })

    print(f"\n{'Symbol':<25} {'Qty':>6} {'Delta':>8} {'Gamma':>8} {'Theta':>8} {'Vega':>8}")
    print("-" * 70)

    for p in option_positions:
        ticker = ib.ticker(p.contract)
        symbol = p.contract.localSymbol or p.contract.symbol
        qty = p.position
        underlying = p.contract.symbol

        if ticker.modelGreeks:
            g = ticker.modelGreeks
            # Greeks are per share, multiply by position qty and 100 (contract multiplier)
            pos_delta = (g.delta or 0) * qty * 100
            pos_gamma = (g.gamma or 0) * qty * 100
            pos_theta = (g.theta or 0) * qty * 100
            pos_vega = (g.vega or 0) * qty * 100

            total_delta += pos_delta
            total_gamma += pos_gamma
            total_theta += pos_theta
            total_vega += pos_vega

            by_underlying[underlying]['delta'] += pos_delta
            by_underlying[underlying]['gamma'] += pos_gamma
            by_underlying[underlying]['theta'] += pos_theta
            by_underlying[underlying]['vega'] += pos_vega
            by_underlying[underlying]['contracts'] += abs(qty)
            by_underlying[underlying]['positions'].append({
                'symbol': symbol,
                'qty': qty,
                'delta': pos_delta,
                'theta': pos_theta
            })

            print(f"{symbol:<25} {qty:>6.0f} {pos_delta:>8.2f} {pos_gamma:>8.4f} {pos_theta:>8.2f} {pos_vega:>8.2f}")
        else:
            print(f"{symbol:<25} {qty:>6.0f} {'N/A':>8} {'N/A':>8} {'N/A':>8} {'N/A':>8}")

        ib.cancelMktData(p.contract)

    # Summary
    print(f"\n{'='*60}")
    print("PORTFOLIO TOTALS")
    print(f"{'='*60}")
    print(f"\n  Total Delta:  {total_delta:>10.2f}  (equivalent to {total_delta:.0f} shares)")
    print(f"  Total Gamma:  {total_gamma:>10.4f}")
    print(f"  Total Theta:  {total_theta:>+10.2f}  (${abs(total_theta):.2f}/day {'in your favor' if total_theta > 0 else 'against you'})")
    print(f"  Total Vega:   {total_vega:>10.2f}")

    # By underlying
    if len(by_underlying) > 1 or args.symbol is None:
        print(f"\n{'='*60}")
        print("BY UNDERLYING")
        print(f"{'='*60}")
        for underlying, data in sorted(by_underlying.items()):
            print(f"\n  {underlying}:")
            print(f"    Delta: {data['delta']:>8.2f}  Theta: {data['theta']:>+8.2f}  Contracts: {data['contracts']}")

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    if total_delta < 0:
        print(f"\n  📉 You are SHORT {abs(total_delta):.0f} delta (bearish bias)")
        print(f"     If underlying rises $1, you lose ~${abs(total_delta):.0f}")
    else:
        print(f"\n  📈 You are LONG {total_delta:.0f} delta (bullish bias)")
        print(f"     If underlying rises $1, you gain ~${total_delta:.0f}")

    if total_theta > 0:
        print(f"\n  ⏰ Time decay works FOR you: +${total_theta:.2f}/day")
    else:
        print(f"\n  ⏰ Time decay works AGAINST you: ${total_theta:.2f}/day")

    ib.disconnect()


def cmd_risk(args):
    """Risk analysis - margin utilization, concentration, max loss"""
    ib = connect()

    print(f"{'='*60}")
    print(f"RISK ANALYSIS - {get_timestamp()}")
    print(f"{'='*60}")

    ib.sleep(3)

    # Get account values
    account = DEFAULT_ACCOUNT
    acct_values = {v.tag: v.value for v in ib.accountValues() if v.account == account}

    net_liq = float(acct_values.get('NetLiquidation', 0))
    maint_margin = float(acct_values.get('MaintMarginReq', 0))
    float(acct_values.get('InitMarginReq', 0))
    float(acct_values.get('AvailableFunds', 0))

    # Margin utilization
    margin_util = (maint_margin / net_liq * 100) if net_liq > 0 else 0

    print("\n--- MARGIN UTILIZATION ---")
    print(f"  Net Liquidation:    ${net_liq:>12,.2f}")
    print(f"  Maintenance Margin: ${maint_margin:>12,.2f}")
    print(f"  Utilization:        {margin_util:>12.1f}%")

    # Warning levels
    if margin_util > 80:
        print("  ⚠️  DANGER: Margin utilization very high!")
    elif margin_util > 60:
        print("  ⚠️  WARNING: Margin utilization elevated")
    elif margin_util > 40:
        print("  ℹ️  Margin utilization moderate")
    else:
        print("  ✅ Margin utilization healthy")

    # Get positions for concentration analysis
    positions = ib.positions()
    option_positions = [p for p in positions if p.contract.secType == 'OPT']

    if option_positions:
        print("\n--- POSITION CONCENTRATION ---")

        # Group by underlying
        by_underlying = defaultdict(list)
        for p in option_positions:
            by_underlying[p.contract.symbol].append(p)

        # Calculate max loss for each underlying (if assigned)
        print(f"\n  {'Underlying':<10} {'Contracts':>10} {'Max Loss (assigned)':>20}")
        print(f"  {'-'*45}")

        total_max_loss = 0
        for underlying, pos_list in sorted(by_underlying.items()):
            total_contracts = sum(abs(p.position) for p in pos_list)

            # Calculate max loss - for short puts, it's strike * 100 * qty
            max_loss = 0
            for p in pos_list:
                if p.position < 0:  # Short position
                    if p.contract.right == 'P':  # Short put
                        # Max loss = (strike - premium received) * 100 * qty
                        # Simplified: strike * 100 * qty (assuming worst case)
                        max_loss += p.contract.strike * 100 * abs(p.position)
                    elif p.contract.right == 'C':  # Short call
                        # Theoretically unlimited, but use a proxy
                        max_loss += p.contract.strike * 100 * abs(p.position) * 0.5

            total_max_loss += max_loss
            print(f"  {underlying:<10} {total_contracts:>10.0f} ${max_loss:>18,.0f}")

        print(f"  {'-'*45}")
        print(f"  {'TOTAL':<10} {'':<10} ${total_max_loss:>18,.0f}")

        # Concentration warning
        if len(by_underlying) == 1:
            print("\n  ⚠️  100% concentration in single underlying!")
        elif len(by_underlying) <= 3:
            print(f"\n  ℹ️  Concentrated in {len(by_underlying)} underlyings")

        # Max loss vs net liq
        if total_max_loss > 0:
            loss_ratio = total_max_loss / net_liq * 100
            print(f"\n  Max Loss / Net Liq: {loss_ratio:.0f}%")
            if loss_ratio > 100:
                print("  ⚠️  Max loss exceeds account value!")

    # Summary warnings
    print("\n--- WARNINGS ---")
    warnings = []

    if margin_util > 60:
        warnings.append("High margin utilization")
    if len(by_underlying) == 1 and option_positions:
        warnings.append("Single-stock concentration")
    if net_liq < 5000:
        warnings.append("Small account - limited diversification")

    if warnings:
        for w in warnings:
            print(f"  ⚠️  {w}")
    else:
        print("  ✅ No major risk warnings")

    ib.disconnect()


def cmd_whatif(args):
    """What-if analysis for a potential new position"""
    ib = connect()

    print(f"{'='*60}")
    print(f"WHAT-IF ANALYSIS - {get_timestamp()}")
    print(f"{'='*60}")

    # Get current account state
    ib.sleep(2)
    account = DEFAULT_ACCOUNT
    acct_values = {v.tag: v.value for v in ib.accountValues() if v.account == account}

    net_liq = float(acct_values.get('NetLiquidation', 0))
    current_margin = float(acct_values.get('MaintMarginReq', 0))
    available = float(acct_values.get('AvailableFunds', 0))

    print("\nCurrent State:")
    print(f"  Net Liquidation: ${net_liq:,.2f}")
    print(f"  Current Margin:  ${current_margin:,.2f}")
    print(f"  Available Funds: ${available:,.2f}")

    # Estimate new position impact
    symbol = args.symbol.upper()
    qty = args.qty
    strike = args.strike

    # Get current stock price
    stock = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close

    print(f"\n{symbol} Current Price: ${spot:.2f}")
    print(f"Proposed: SELL {qty} PUT @ ${strike:.2f} strike")

    # Estimate margin requirement (rough: 25% of strike for cash-secured)
    est_margin = strike * 100 * qty * 0.25
    otm_pct = (spot - strike) / spot * 100 if spot > 0 else 0

    print("\n--- PROJECTED IMPACT ---")
    print(f"  Strike:            ${strike:.2f} ({otm_pct:.1f}% OTM)")
    print(f"  Est. Margin Req:   ${est_margin:,.0f}")
    print(f"  Max Loss if Assigned: ${strike * 100 * qty:,.0f}")

    new_margin = current_margin + est_margin
    new_margin_util = (new_margin / net_liq * 100) if net_liq > 0 else 0

    print("\n--- AFTER POSITION ---")
    print(f"  New Margin Req:    ${new_margin:,.0f}")
    print(f"  New Margin Util:   {new_margin_util:.1f}%")
    print(f"  Remaining Avail:   ${available - est_margin:,.0f}")

    if available < est_margin:
        print(f"\n  ❌ INSUFFICIENT FUNDS - need ${est_margin - available:,.0f} more")
    elif new_margin_util > 80:
        print(f"\n  ⚠️  WARNING: Would push margin utilization to {new_margin_util:.0f}%")
    else:
        print("\n  ✅ Position appears feasible")

    ib.disconnect()
