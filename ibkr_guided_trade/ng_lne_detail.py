#!/usr/bin/env python3
"""
Detailed LNE call option chains for NG Jul 2026 (NGN6) and Aug 2026 (NGQ6).

Connects to IBKR, gets current futures prices, then pulls LNE call chains
at strikes from $2.80 to $4.00 in $0.05 increments for each month.

Displays: Strike, Bid, Ask, Mid, Extrinsic Value, IV, Delta, Gamma, Theta, Vega,
          and Cost per contract in dollars.

Also prints a summary comparing ATM vs ~0.65 delta vs ~0.75 delta for both months.
"""

from ib_insync import IB, Future, FuturesOption
import sys

# IBKR connection settings
IBKR_HOST = '192.168.1.127'
IBKR_PORT = 20009
CLIENT_ID = 98

# Months to scan
NG_MONTHS = [
    ('Jul 2026', '202607', 'NGN6'),
    ('Aug 2026', '202608', 'NGQ6'),
]

# Strike range: $2.80 to $4.00 in $0.05 increments
STRIKE_LOW = 2.80
STRIKE_HIGH = 4.00
STRIKE_STEP = 0.05

# NG multiplier: 10,000 MMBtu per contract
NG_MULTIPLIER = 10000


def generate_strikes():
    """Generate strike list from STRIKE_LOW to STRIKE_HIGH in STRIKE_STEP increments."""
    strikes = []
    s = STRIKE_LOW
    while s <= STRIKE_HIGH + 0.001:
        strikes.append(round(s, 2))
        s += STRIKE_STEP
    return strikes


def get_futures_price(ib, fut_contract):
    """Get current price for a futures contract."""
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


def get_lne_chain(ib, fut_contract, spot, strikes):
    """Pull LNE call option chain for a given NG futures contract at specified strikes.

    Uses reqSecDefOptParams to discover the correct expiration for LNE options
    on this underlying, then qualifies and requests market data for each strike.
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

    # Find LNE-specific params
    lne_params = [op for op in opt_params if op.tradingClass == 'LNE']
    params_to_use = lne_params if lne_params else opt_params

    # Collect available expirations and strikes from IBKR
    all_exps = set()
    avail_strikes = set()
    for op in params_to_use:
        all_exps.update(op.expirations)
        avail_strikes.update(op.strikes)

    # Filter our desired strikes to only those available at IBKR
    valid_strikes = sorted([s for s in strikes if s in avail_strikes])

    if not valid_strikes:
        # Try rounding to check for floating point mismatches
        avail_sorted = sorted(avail_strikes)
        print(f"    Available strikes sample: {avail_sorted[:10]}...{avail_sorted[-10:]}")
        valid_strikes = sorted([s for s in strikes
                                if any(abs(s - a) < 0.001 for a in avail_strikes)])

    if not valid_strikes:
        print(f"    None of our requested strikes are available at IBKR")
        return []

    # Find the expiration matching the futures month
    fut_ym = fut_contract.lastTradeDateOrContractMonth[:6]
    matching_exps = sorted([e for e in all_exps if e[:6] == fut_ym])

    if not matching_exps:
        all_sorted = sorted(all_exps)
        matching_exps = [e for e in all_sorted if e[:6] >= fut_ym][:1]

    if not matching_exps:
        matching_exps = sorted(all_exps)[-1:]

    if not matching_exps:
        print(f"    No matching expirations found")
        return []

    # Use the last (latest) expiration in the matching month
    target_exp = matching_exps[-1]

    print(f"    Expiration: {target_exp}")
    print(f"    Strikes: {len(valid_strikes)} "
          f"(${valid_strikes[0]:.2f} - ${valid_strikes[-1]:.2f})")

    # Build option contracts with LNE trading class
    opts = [FuturesOption(
        symbol='NG',
        lastTradeDateOrContractMonth=target_exp,
        strike=s,
        right='C',
        exchange='NYMEX',
        tradingClass='LNE'
    ) for s in valid_strikes]

    # Qualify contracts
    try:
        qualified = ib.qualifyContracts(*opts)
    except Exception as e:
        print(f"    Error qualifying LNE options: {e}")
        return []

    valid = [o for o in qualified if o.conId > 0]
    if not valid:
        print(f"    No valid LNE contracts qualified (tried {len(opts)} strikes)")
        # Fallback: try without tradingClass
        opts_fb = [FuturesOption(
            symbol='NG',
            lastTradeDateOrContractMonth=target_exp,
            strike=s,
            right='C',
            exchange='NYMEX'
        ) for s in valid_strikes]
        try:
            qualified = ib.qualifyContracts(*opts_fb)
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

    # Wait for greeks to populate
    ib.sleep(3)

    # Collect results
    results = []
    for opt in valid:
        t = ib.ticker(opt)
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0

        mid = 0
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask

        # Intrinsic value = max(0, spot - strike) for calls
        intrinsic = max(0.0, spot - opt.strike)
        # Extrinsic = mid - intrinsic (time value)
        extrinsic = mid - intrinsic if mid > 0 else 0

        # Greeks from model
        delta = None
        gamma = None
        theta = None
        vega = None
        iv = None
        if t.modelGreeks:
            delta = t.modelGreeks.delta
            gamma = t.modelGreeks.gamma
            theta = t.modelGreeks.theta
            vega = t.modelGreeks.vega
            iv = t.modelGreeks.impliedVol

        # Cost per contract in dollars
        cost = mid * NG_MULTIPLIER if mid > 0 else 0

        results.append({
            'strike': opt.strike,
            'bid': bid,
            'ask': ask,
            'mid': mid,
            'intrinsic': intrinsic,
            'extrinsic': extrinsic,
            'iv': iv,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'cost': cost,
            'expiry': target_exp,
        })

    # Cancel market data to free up slots
    for opt in valid:
        ib.cancelMktData(opt)

    return sorted(results, key=lambda x: x['strike'])


def print_chain(label, spot, results, expiry):
    """Print option chain in a clean table format."""
    w = 130
    print(f"\n{'=' * w}")
    print(f"  {label}")
    print(f"  Underlying: ${spot:.3f}  |  Exp: {expiry}  |  "
          f"Trading Class: LNE (Cash-Settled)  |  Multiplier: {NG_MULTIPLIER:,}")
    print(f"{'=' * w}")

    if not results:
        print("  No option data available.")
        return

    # Header
    print(f"  {'Strike':>8} {'Bid':>8} {'Ask':>8} {'Mid':>8} "
          f"{'Extrin':>8} {'IV':>7} {'Delta':>7} {'Gamma':>7} "
          f"{'Theta':>7} {'Vega':>7} {'Cost/Ct':>10}")
    print(f"  {'-' * (w - 4)}")

    for r in results:
        iv_s = f"{r['iv']*100:.1f}%" if r['iv'] else '   N/A'
        d_s = f"{r['delta']:.4f}" if r['delta'] is not None else '  N/A'
        g_s = f"{r['gamma']:.4f}" if r['gamma'] is not None else '  N/A'
        th_s = f"{r['theta']:.4f}" if r['theta'] is not None else '  N/A'
        v_s = f"{r['vega']:.4f}" if r['vega'] is not None else '  N/A'
        ext_s = f"${r['extrinsic']:.4f}" if r['mid'] > 0 else '     N/A'
        cost_s = f"${r['cost']:,.0f}" if r['cost'] > 0 else '     N/A'

        # Mark ATM and ITM/OTM
        if abs(r['strike'] - spot) < 0.025:
            marker = ' <-- ATM'
        elif r['strike'] < spot:
            marker = ''  # ITM
        else:
            marker = ''

        print(f"  ${r['strike']:>7.2f} ${r['bid']:>7.4f} ${r['ask']:>7.4f} ${r['mid']:>7.4f} "
              f"{ext_s:>8} {iv_s:>7} {d_s:>7} {g_s:>7} "
              f"{th_s:>7} {v_s:>7} {cost_s:>10}{marker}")

    # Count rows with actual data
    with_data = [r for r in results if r['bid'] > 0 or r['ask'] > 0]
    print(f"\n  {len(with_data)}/{len(results)} strikes with market data")


def find_closest_delta(results, target_delta):
    """Find the result closest to a target delta value."""
    with_delta = [r for r in results if r['delta'] is not None and r['mid'] > 0]
    if not with_delta:
        return None
    return min(with_delta, key=lambda r: abs(r['delta'] - target_delta))


def print_summary(all_month_data):
    """Print comparison summary: ATM vs 0.65 delta vs 0.75 delta for all months."""
    w = 130
    print(f"\n{'=' * w}")
    print(f"  SUMMARY: ATM vs ITM (0.65 delta) vs Deep ITM (0.75 delta)")
    print(f"{'=' * w}")

    targets = [
        ('ATM (~0.50 delta)', 0.50),
        ('ITM (~0.65 delta)', 0.65),
        ('Deep ITM (~0.75 delta)', 0.75),
    ]

    for label, code, spot, results in all_month_data:
        print(f"\n  {code} ({label}) -- Underlying: ${spot:.3f}")
        print(f"  {'Category':<25} {'Strike':>8} {'Mid':>8} {'Extrin':>8} "
              f"{'Cost/Ct':>10} {'IV':>7} {'Delta':>7} {'Theta':>7} {'Vega':>7}")
        print(f"  {'-' * 100}")

        for cat_label, target_d in targets:
            r = find_closest_delta(results, target_d)
            if r:
                iv_s = f"{r['iv']*100:.1f}%" if r['iv'] else '  N/A'
                d_s = f"{r['delta']:.4f}" if r['delta'] is not None else ' N/A'
                th_s = f"{r['theta']:.4f}" if r['theta'] is not None else ' N/A'
                v_s = f"{r['vega']:.4f}" if r['vega'] is not None else ' N/A'
                cost_s = f"${r['cost']:,.0f}" if r['cost'] > 0 else '    N/A'
                ext_s = f"${r['extrinsic']:.4f}" if r['mid'] > 0 else '    N/A'

                print(f"  {cat_label:<25} ${r['strike']:>7.2f} ${r['mid']:>7.4f} "
                      f"{ext_s:>8} {cost_s:>10} {iv_s:>7} {d_s:>7} {th_s:>7} {v_s:>7}")

                # Also show cost breakdown
                intrinsic_cost = r['intrinsic'] * NG_MULTIPLIER
                extrinsic_cost = r['extrinsic'] * NG_MULTIPLIER
                print(f"  {'':25} {'Intrinsic:':>10} ${intrinsic_cost:>8,.0f}  "
                      f"{'Extrinsic:':>12} ${extrinsic_cost:>8,.0f}  "
                      f"{'Total:':>8} ${r['cost']:>8,.0f}")
            else:
                print(f"  {cat_label:<25}  -- no data --")

    # Cross-month comparison
    print(f"\n  {'=' * 60}")
    print(f"  Cross-Month Comparison (same delta target)")
    print(f"  {'=' * 60}")

    for cat_label, target_d in targets:
        print(f"\n  {cat_label}:")
        for label, code, spot, results in all_month_data:
            r = find_closest_delta(results, target_d)
            if r:
                cost_s = f"${r['cost']:,.0f}" if r['cost'] > 0 else 'N/A'
                ext_cost = r['extrinsic'] * NG_MULTIPLIER
                print(f"    {code}: ${r['strike']:.2f} strike, "
                      f"delta={r['delta']:.4f}, "
                      f"cost={cost_s}, "
                      f"extrinsic=${ext_cost:,.0f}, "
                      f"theta={r['theta']:.4f}" if r['theta'] else f"    {code}: no theta")


def main():
    print("=" * 80)
    print("  NG LNE Detailed Call Option Chains")
    print(f"  Months: Jul 2026 (NGN6), Aug 2026 (NGQ6)")
    print(f"  Strikes: ${STRIKE_LOW:.2f} - ${STRIKE_HIGH:.2f} "
          f"in ${STRIKE_STEP:.2f} increments")
    print(f"  Multiplier: {NG_MULTIPLIER:,} MMBtu/contract "
          f"($0.01 move = ${NG_MULTIPLIER * 0.01:,.0f}/contract)")
    print("=" * 80)

    strikes = generate_strikes()
    print(f"  Generated {len(strikes)} strikes")

    # Connect
    ib = IB()
    try:
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID, timeout=30)
        print(f"\n  Connected to IBKR at {IBKR_HOST}:{IBKR_PORT} (clientId={CLIENT_ID})")
    except Exception as e:
        print(f"\n  Failed to connect to IBKR: {e}")
        sys.exit(1)

    all_month_data = []

    try:
        # Step 1: Get futures prices
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

        # Step 2: Pull LNE call option chains
        print(f"\n--- Fetching LNE Call Option Chains ---")

        for label, code, ym, fut, spot in futures_data:
            print(f"\n  Processing {code} ({label}), spot=${spot:.3f}...")
            results = get_lne_chain(ib, fut, spot, strikes)
            expiry = results[0]['expiry'] if results else 'N/A'
            print_chain(f"{code} ({label}) LNE CALLS", spot, results, expiry)
            all_month_data.append((label, code, spot, results))

        # Step 3: Summary
        if all_month_data:
            print_summary(all_month_data)

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
