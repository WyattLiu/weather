#!/usr/bin/env python3
"""
Pull LNE (cash-settled NG options on NYMEX) call option chains for multiple
NG futures months via IBKR.

Connects to IBKR, gets NG futures prices for May-Oct 2026, then pulls
LNE call option chains at strikes -15% to +15% of each future's price.

Displays: strike, bid, ask, mid, IV, delta, OI, volume for each month.
"""

from ib_insync import IB, Future, FuturesOption
import sys

# IBKR connection settings
IBKR_HOST = '192.168.1.127'
IBKR_PORT = 20009
CLIENT_ID = 97

# NG futures months to scan: (label, YYYYMM, futures month code for display)
NG_MONTHS = [
    ('May 2026', '202605', 'NGK6'),
    ('Jun 2026', '202606', 'NGM6'),
    ('Jul 2026', '202607', 'NGN6'),
    ('Aug 2026', '202608', 'NGQ6'),
    ('Sep 2026', '202609', 'NGU6'),
    ('Oct 2026', '202610', 'NGV6'),
]


def get_futures_price(ib, fut_contract):
    """Get current price for a futures contract. Falls back through
    last -> close -> bid/ask midpoint."""
    ib.reqMktData(fut_contract, '', False, False)
    ib.sleep(3)
    t = ib.ticker(fut_contract)

    price = None
    if t.last and t.last > 0:
        price = t.last
    elif t.close and t.close > 0:
        price = t.close
    else:
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        if bid > 0 and ask > 0:
            price = (bid + ask) / 2
        elif bid > 0:
            price = bid
        elif ask > 0:
            price = ask

    return price


def get_option_chain(ib, fut_contract, spot):
    """Get LNE call option chain for a given NG futures contract.

    Uses reqSecDefOptParams to discover available expirations and strikes
    for this underlying, then filters to LNE trading class calls within
    -15% to +15% of spot.
    """
    # Get option parameters for this futures contract
    opt_params = ib.reqSecDefOptParams(
        fut_contract.symbol, 'NYMEX', 'FUT', fut_contract.conId)

    if not opt_params:
        opt_params = ib.reqSecDefOptParams(
            fut_contract.symbol, '', 'FUT', fut_contract.conId)

    if not opt_params:
        print(f"    No option params found for {fut_contract.localSymbol}")
        return []

    # Find LNE-specific params, or fall back to all params
    lne_params = [op for op in opt_params if op.tradingClass == 'LNE']
    params_to_use = lne_params if lne_params else opt_params

    # Collect all available expirations and strikes
    all_exps = set()
    all_strikes = set()
    for op in params_to_use:
        all_exps.update(op.expirations)
        all_strikes.update(op.strikes)

    # Filter strikes to -15% to +15% of spot
    low = spot * 0.85
    high = spot * 1.15
    strikes = sorted([s for s in all_strikes if low <= s <= high])

    if not strikes:
        print(f"    No strikes in range ${low:.2f} - ${high:.2f}")
        return []

    # Filter expirations: find the one(s) closest to the futures expiry month
    # For LNE, we want the expiration matching the futures month
    fut_ym = fut_contract.lastTradeDateOrContractMonth[:6]  # YYYYMM
    matching_exps = sorted([e for e in all_exps if e[:6] == fut_ym])

    if not matching_exps:
        # If no exact month match, find closest expiration
        all_sorted = sorted(all_exps)
        matching_exps = [e for e in all_sorted if e[:6] >= fut_ym][:1]

    if not matching_exps:
        matching_exps = sorted(all_exps)[-1:]

    if not matching_exps:
        print("    No matching expirations found")
        return []

    # Use the last (latest) expiration in the matching month
    target_exp = matching_exps[-1]

    print(f"    Using expiration: {target_exp}, {len(strikes)} strikes "
          f"(${strikes[0]:.2f} - ${strikes[-1]:.2f})")

    # Build option contracts with LNE trading class
    opts = [FuturesOption(
        symbol='NG',
        lastTradeDateOrContractMonth=target_exp,
        strike=s,
        right='C',
        exchange='NYMEX',
        tradingClass='LNE'
    ) for s in strikes]

    # Qualify contracts
    try:
        qualified = ib.qualifyContracts(*opts)
    except Exception as e:
        print(f"    Error qualifying LNE options: {e}")
        return []

    valid = [o for o in qualified if o.conId > 0]
    if not valid:
        print(f"    No valid LNE contracts qualified (tried {len(opts)} strikes)")
        # Try without tradingClass as fallback
        opts_fallback = [FuturesOption(
            symbol='NG',
            lastTradeDateOrContractMonth=target_exp,
            strike=s,
            right='C',
            exchange='NYMEX'
        ) for s in strikes]
        try:
            qualified = ib.qualifyContracts(*opts_fallback)
            valid = [o for o in qualified if o.conId > 0]
            if valid:
                print(f"    Fallback: qualified {len(valid)} contracts without LNE class")
        except Exception as e:
            print(f"    Fallback also failed: {e}")
            return []

    if not valid:
        return []

    print(f"    Qualified {len(valid)} contracts, requesting market data...")

    # Request market data for all valid contracts
    for opt in valid:
        ib.reqMktData(opt, '', False, False)

    # Wait for data to populate
    ib.sleep(4)

    # Collect results
    results = []
    for opt in valid:
        t = ib.ticker(opt)
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0

        # Skip if no market data at all
        mid = 0
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask

        # Greeks
        delta = None
        iv = None
        gamma = None
        theta = None
        if t.modelGreeks:
            delta = t.modelGreeks.delta
            iv = t.modelGreeks.impliedVol
            gamma = t.modelGreeks.gamma
            theta = t.modelGreeks.theta

        # Volume and OI
        volume = t.volume if t.volume and t.volume >= 0 else 0
        # OI not directly available from ticker; check contract details
        # callOpenInterest is sometimes in t.openInterest or via snapshot
        oi = 0
        if hasattr(t, 'openInterest') and t.openInterest and t.openInterest > 0:
            oi = t.openInterest

        otm_pct = (opt.strike - spot) / spot * 100

        results.append({
            'strike': opt.strike,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'iv': iv,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'oi': oi,
            'volume': volume,
            'otm_pct': otm_pct,
            'expiry': target_exp,
            'trading_class': opt.tradingClass or 'LNE',
        })

    # Cancel market data to free up slots
    for opt in valid:
        ib.cancelMktData(opt)

    return sorted(results, key=lambda x: x['strike'])


def print_chain(label, spot, results, expiry):
    """Print option chain in a clean table format."""
    print(f"\n{'='*90}")
    print(f"  {label}  |  Underlying: ${spot:.3f}  |  Exp: {expiry}  |  "
          f"Trading Class: LNE (Cash-Settled)")
    print(f"{'='*90}")

    if not results:
        print("  No option data available.")
        return

    # Header
    print(f"  {'Strike':>8} {'OTM%':>7} {'Bid':>8} {'Ask':>8} {'Mid':>8} "
          f"{'IV':>7} {'Delta':>7} {'Gamma':>7} {'Theta':>7} {'Vol':>6} {'OI':>6}")
    print(f"  {'-'*88}")

    for r in results:
        iv_s = f"{r['iv']*100:.1f}%" if r['iv'] else '   N/A'
        d_s = f"{r['delta']:.3f}" if r['delta'] is not None else '  N/A'
        g_s = f"{r['gamma']:.4f}" if r['gamma'] is not None else ' N/A'
        th_s = f"{r['theta']:.4f}" if r['theta'] is not None else ' N/A'
        vol_s = f"{int(r['volume']):>5}" if r['volume'] > 0 else '    -'
        oi_s = f"{int(r['oi']):>5}" if r['oi'] > 0 else '    -'

        # Highlight ATM strike
        marker = ' <-- ATM' if abs(r['otm_pct']) < 1.5 else ''

        print(f"  ${r['strike']:>7.2f} {r['otm_pct']:>+6.1f}% "
              f"${r['bid']:>7.3f} ${r['ask']:>7.3f} ${r['mid']:>7.3f} "
              f"{iv_s:>7} {d_s:>7} {g_s:>7} {th_s:>7} {vol_s:>6} {oi_s:>6}{marker}")

    # Count rows with actual data
    with_data = [r for r in results if r['bid'] > 0 or r['ask'] > 0]
    print(f"\n  {len(with_data)}/{len(results)} strikes with market data")


def main():
    print("=" * 90)
    print("  NG LNE Call Option Chain Scanner")
    print("  Months: May-Oct 2026  |  Strikes: +/-15% of spot")
    print("=" * 90)

    # Connect
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID, timeout=30)
        print(f"\nConnected to IBKR at {IBKR_HOST}:{IBKR_PORT} (clientId={CLIENT_ID})")
    except Exception as e:
        print(f"\nFailed to connect to IBKR: {e}")
        sys.exit(1)

    try:
        # Step 1: Get futures prices for all months
        print("\n--- Fetching NG Futures Prices ---")
        futures_data = []

        for label, ym, code in NG_MONTHS:
            fut = Future(symbol='NG', lastTradeDateOrContractMonth=ym,
                         exchange='NYMEX')
            try:
                qualified = ib.qualifyContracts(fut)
                if not qualified or fut.conId == 0:
                    print(f"  {code} ({label}): Could not qualify contract")
                    continue
            except Exception as e:
                print(f"  {code} ({label}): Qualification error: {e}")
                continue

            spot = get_futures_price(ib, fut)
            if spot and spot > 0:
                print(f"  {code} ({label}): ${spot:.3f}")
                futures_data.append((label, code, ym, fut, spot))
            else:
                print(f"  {code} ({label}): No price available")

        if not futures_data:
            print("\nNo futures prices available. Check if market is open.")
            ib.disconnect()
            sys.exit(1)

        # Step 2: Pull LNE call option chains for each month
        print("\n--- Fetching LNE Call Option Chains ---")

        for label, code, ym, fut, spot in futures_data:
            print(f"\n  Processing {code} ({label}), spot=${spot:.3f}...")
            results = get_option_chain(ib, fut, spot)
            expiry = results[0]['expiry'] if results else 'N/A'
            print_chain(f"{code} ({label}) LNE CALLS", spot, results, expiry)

        # Summary table
        print(f"\n{'='*90}")
        print("  SUMMARY: NG Futures Curve (May-Oct 2026)")
        print(f"{'='*90}")
        print(f"  {'Contract':>10} {'Month':>10} {'Price':>10}")
        print(f"  {'-'*32}")
        for label, code, ym, fut, spot in futures_data:
            print(f"  {code:>10} {label:>10} ${spot:>8.3f}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ib.disconnect()
        print("\nDisconnected from IBKR.")


if __name__ == '__main__':
    main()
