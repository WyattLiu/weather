"""
Module 1: Direct Trading Commands

Commands for viewing status, placing orders, and scanning options.
"""
# pyright: reportOptionalMemberAccess=false

from ib_insync import Stock, Option, LimitOrder, Contract, ComboLeg
from datetime import datetime, timedelta

from .common import connect, get_timestamp, DEFAULT_ACCOUNT, ET


def cmd_status(args):
    """Show all open orders and positions"""
    ib = connect()

    # Fetch ALL orders from server (not just this client's)
    ib.reqAllOpenOrders()
    ib.sleep(2)

    print(f"Time: {get_timestamp()}")
    print(f"Accounts: {ib.managedAccounts()}")

    print("\n=== OPEN ORDERS ===")
    trades = ib.openTrades()
    if trades:
        for t in trades:
            c = t.contract
            o = t.order
            s = t.orderStatus
            # Check if it's a combo/spread order
            if c.secType == 'BAG' and c.comboLegs:
                legs_desc = []
                for leg in c.comboLegs:
                    legs_desc.append(f"{leg.action}:{leg.conId}")
                print(f"  {c.symbol} SPREAD: {o.action} {o.totalQuantity:.0f} @ ${o.lmtPrice:.2f} - {s.status}")
                print(f"    Legs: {legs_desc}")
            else:
                symbol = c.localSymbol if hasattr(c, 'localSymbol') and c.localSymbol else c.symbol
                print(f"  {symbol}: {o.action} {o.totalQuantity:.0f} @ ${o.lmtPrice:.2f} - {s.status}")
    else:
        print("  No open orders")

    print("\n=== POSITIONS ===")
    positions = ib.positions()
    if positions:
        for p in positions:
            symbol = p.contract.localSymbol if p.contract.localSymbol else p.contract.symbol
            print(f"  {symbol}: {p.position:.0f} @ avg ${p.avgCost:.2f}")
    else:
        print("  No positions")

    ib.disconnect()


def cmd_cancel_all(args):
    """Cancel all open orders globally"""
    ib = connect()

    print("Cancelling ALL open orders...")
    ib.reqGlobalCancel()
    ib.sleep(3)

    # Verify
    ib.reqAllOpenOrders()
    ib.sleep(2)
    remaining = len(ib.openTrades())
    print(f"Remaining orders: {remaining}")

    ib.disconnect()


def cmd_modify_order(args):
    """Modify an existing order's price (works for both single-leg and spreads)"""
    ib = connect()

    ib.reqAllOpenOrders()
    ib.sleep(2)

    trades = ib.openTrades()
    if not trades:
        print("No open orders to modify")
        ib.disconnect()
        return

    # Find matching order - check both regular and spread orders
    target_trade = None
    for t in trades:
        c = t.contract
        # For spreads, match by underlying symbol
        if c.secType == 'BAG':
            if args.symbol.upper() == c.symbol.upper():
                # For spreads, compare absolute values since credits are stored as negative
                if args.old_price is None or abs(abs(t.order.lmtPrice) - args.old_price) < 0.01:
                    target_trade = t
                    break
        else:
            # Regular single-leg orders
            symbol = c.localSymbol if c.localSymbol else c.symbol
            if args.symbol.upper() in symbol.upper():
                if args.old_price is None or abs(t.order.lmtPrice - args.old_price) < 0.01:
                    target_trade = t
                    break

    if not target_trade:
        print(f"No matching order found for {args.symbol}")
        print("Open orders:")
        for t in trades:
            c = t.contract
            if c.secType == 'BAG':
                print(f"  {c.symbol} SPREAD: {t.order.action} {t.order.totalQuantity:.0f} @ ${t.order.lmtPrice:.2f}")
            else:
                symbol = c.localSymbol if c.localSymbol else c.symbol
                print(f"  {symbol}: {t.order.action} {t.order.totalQuantity:.0f} @ ${t.order.lmtPrice:.2f}")
        ib.disconnect()
        return

    # Modify the order
    old_price = target_trade.order.lmtPrice
    c = target_trade.contract

    # For spreads, new_price should be negative (credit)
    # User provides positive number, we negate it for credit spreads
    if c.secType == 'BAG' and args.new_price > 0:
        target_trade.order.lmtPrice = -args.new_price
    else:
        target_trade.order.lmtPrice = args.new_price

    ib.placeOrder(target_trade.contract, target_trade.order)
    ib.sleep(2)

    if c.secType == 'BAG':
        print(f"Modified: {c.symbol} SPREAD ${abs(old_price):.2f} → ${abs(target_trade.order.lmtPrice):.2f} credit")
    else:
        symbol = c.localSymbol or c.symbol
        print(f"Modified: {symbol} ${old_price:.2f} → ${args.new_price:.2f}")
    print(f"Status: {target_trade.orderStatus.status}")

    ib.disconnect()


def cmd_quote(args):
    """Get stock quote"""
    ib = connect()

    stock = Stock(args.symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    t = ib.ticker(stock)
    print(f"{args.symbol}: Last=${t.last:.2f}, Bid=${t.bid:.2f}, Ask=${t.ask:.2f}")

    ib.disconnect()


def cmd_opt_chain(args):
    """Show options chain for a symbol with recommendations"""
    ib = connect()

    stock = Stock(args.symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close
    if not spot or not (spot > 0):
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    print(f"{args.symbol} Spot: ${spot:.2f}")

    # Get option chains
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

    # Find expiration closest to target DTE
    target_date = datetime.now() + timedelta(days=args.dte)
    best_exp = None
    best_diff = 999

    for chain in chains:
        if chain.exchange == 'SMART':
            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d')
                diff = abs((exp_date - target_date).days)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp

    if not best_exp:
        print("No expirations found")
        ib.disconnect()
        return

    exp_date = datetime.strptime(best_exp, '%Y%m%d')
    actual_dte = (exp_date - datetime.now()).days
    print(f"Expiry: {best_exp} ({actual_dte} DTE)")

    # Get strikes around spot (OTM puts are below spot)
    all_strikes = set()
    for chain in chains:
        if chain.exchange == 'SMART':
            all_strikes.update(chain.strikes)

    strikes = sorted([s for s in all_strikes if spot * 0.80 <= s <= spot * 1.05])
    print(f"Analyzing {len(strikes)} strikes...")

    # Create put contracts
    puts = [Option(args.symbol, best_exp, strike, 'P', 'SMART') for strike in strikes]
    ib.qualifyContracts(*puts)

    for opt in puts:
        ib.reqMktData(opt, '', False, False)
    ib.sleep(3)

    print(f"\n{'='*80}")
    print(f"{args.symbol} PUT OPTIONS - Expiry: {best_exp} ({actual_dte} DTE)")
    print(f"{'='*80}")
    print(f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Spread':>8} {'Spread%':>8} {'Volume':>8} {'Delta':>8}")
    print("-" * 80)

    results = []
    for opt in puts:
        t = ib.ticker(opt)
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        spread = ask - bid if ask > 0 and bid > 0 else 0
        spread_pct = (spread / ((bid + ask) / 2) * 100) if (bid + ask) > 0 else 999
        delta = t.modelGreeks.delta if t.modelGreeks else None
        vol = t.volume if t.volume else 0

        if bid > 0:
            results.append({
                'strike': opt.strike,
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'volume': vol,
                'delta': delta,
                'expiry': best_exp
            })
            delta_str = f"{delta:.3f}" if delta else "N/A"
            print(f"{opt.strike:>8.1f} {bid:>8.2f} {ask:>8.2f} {spread:>8.2f} {spread_pct:>7.1f}% {vol:>8} {delta_str:>8}")

    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR SELLING PUTS:")
    print(f"{'='*80}")

    # Filter for good liquidity (tight spreads) and reasonable premium
    good_puts = [r for r in results if r['spread_pct'] < 20 and r['bid'] >= 0.05]
    good_puts.sort(key=lambda x: x['spread_pct'])

    for p in good_puts[:5]:
        otm_pct = (spot - p['strike']) / spot * 100
        premium_pct = p['bid'] / p['strike'] * 100
        delta_str = f"delta={p['delta']:.3f}" if p['delta'] else ""
        margin_est = p['strike'] * 100 * 0.25  # rough margin estimate
        print(f"  ${p['strike']:.1f} ({otm_pct:.1f}% OTM) - Bid ${p['bid']:.2f} ({premium_pct:.2f}% of strike) - Spread {p['spread_pct']:.1f}% {delta_str}")
        print(f"    Est margin: ~${margin_est:.0f} | Premium: ${p['bid']*100:.0f}/contract")

    if not good_puts:
        print("  No puts with good liquidity found. Try during market hours.")

    ib.disconnect()


def cmd_scan_puts(args):
    """Scan multiple expirations for the best put opportunities with comprehensive analysis"""
    ib = connect()

    stock = Stock(args.symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close
    if not spot or not (spot > 0):
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    print(f"{args.symbol} Spot: ${spot:.2f}")
    print(f"Scanning expirations from {args.min_dte} to {args.max_dte} DTE...\n")

    # Get option chains
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

    # Find all expirations in DTE range
    target_expirations = []
    for chain in chains:
        if chain.exchange == 'SMART':
            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d')
                dte = (exp_date - datetime.now()).days
                if args.min_dte <= dte <= args.max_dte:
                    target_expirations.append((exp, dte))

    target_expirations = sorted(set(target_expirations), key=lambda x: x[1])

    if not target_expirations:
        print("No expirations found in DTE range")
        ib.disconnect()
        return

    print(f"Found {len(target_expirations)} expirations: {[e[0] for e in target_expirations]}")

    # Get strikes around spot - wider range for more options
    all_strikes = set()
    for chain in chains:
        if chain.exchange == 'SMART':
            all_strikes.update(chain.strikes)

    strikes = sorted([s for s in all_strikes if spot * 0.75 <= s <= spot * 1.05])
    print(f"Analyzing {len(strikes)} strikes: {strikes}")

    all_results = []

    for exp, dte in target_expirations:
        print(f"\n{'='*80}")
        print(f"EXPIRY: {exp} ({dte} DTE)")
        print(f"{'='*80}")
        print(f"{'Strike':>7} {'OTM%':>6} {'Bid':>6} {'Ask':>6} {'Mid':>6} {'Sprd%':>6} {'Vol':>5} {'OI':>6} {'Delta':>7} {'IV':>6} {'Grade'}")
        print("-" * 80)

        puts = [Option(args.symbol, exp, strike, 'P', 'SMART') for strike in strikes]
        ib.qualifyContracts(*puts)

        for opt in puts:
            ib.reqMktData(opt, '', False, False)
        ib.sleep(3)

        for opt in puts:
            t = ib.ticker(opt)
            bid = t.bid if t.bid and t.bid > 0 else 0
            ask = t.ask if t.ask and t.ask > 0 else 0

            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pct = (spread / mid * 100) if mid > 0 else 999

                # Get greeks
                delta = t.modelGreeks.delta if t.modelGreeks else None
                iv = t.modelGreeks.impliedVol if t.modelGreeks else None

                # Volume and open interest
                vol = int(t.volume) if t.volume and t.volume > 0 else 0

                otm_pct = (spot - opt.strike) / spot * 100

                if otm_pct >= 0:  # OTM or ATM puts
                    # Calculate metrics
                    ann_return_bid = (bid / (opt.strike * 0.25)) * (365 / dte) * 100
                    ann_return_mid = (mid / (opt.strike * 0.25)) * (365 / dte) * 100

                    # Liquidity score: combination of spread, volume
                    liquidity_score = 0
                    if spread_pct < 10:
                        liquidity_score += 3
                    elif spread_pct < 15:
                        liquidity_score += 2
                    elif spread_pct < 25:
                        liquidity_score += 1

                    if vol >= 100:
                        liquidity_score += 3
                    elif vol >= 20:
                        liquidity_score += 2
                    elif vol >= 5:
                        liquidity_score += 1

                    # Grade based on liquidity
                    if liquidity_score >= 5:
                        grade = "A"
                    elif liquidity_score >= 3:
                        grade = "B"
                    elif liquidity_score >= 2:
                        grade = "C"
                    else:
                        grade = "D"

                    mid_fillable = spread_pct < 20

                    all_results.append({
                        'expiry': exp,
                        'dte': dte,
                        'strike': opt.strike,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'volume': vol,
                        'delta': delta,
                        'iv': iv,
                        'otm_pct': otm_pct,
                        'ann_return_bid': ann_return_bid,
                        'ann_return_mid': ann_return_mid,
                        'liquidity_score': liquidity_score,
                        'grade': grade,
                        'mid_fillable': mid_fillable
                    })

                    delta_str = f"{delta:.3f}" if delta else "N/A"
                    iv_str = f"{iv*100:.0f}%" if iv else "N/A"
                    print(f"${opt.strike:>6.1f} {otm_pct:>5.1f}% ${bid:>5.2f} ${ask:>5.2f} ${mid:>5.2f} {spread_pct:>5.0f}% {vol:>5} {'N/A':>6} {delta_str:>7} {iv_str:>6} [{grade}]")

            ib.cancelMktData(opt)

    # Summary
    print(f"\n{'='*80}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Best for conservative selling (at bid)
    print("\n📊 BEST AT BID (conservative, high fill probability):")
    print("-" * 60)
    good_bid = [r for r in all_results if r['spread_pct'] < 15 and r['otm_pct'] >= 8]
    good_bid.sort(key=lambda x: (-x['ann_return_bid'], x['spread_pct']))

    for i, opt in enumerate(good_bid[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}P ({opt['dte']}d, {opt['otm_pct']:.0f}% OTM)")
        print(f"     Bid ${opt['bid']:.2f} → ${opt['bid']*100:.0f} premium | ~{opt['ann_return_bid']:.0f}% ann | {delta_str}")

    # Mid-price opportunities
    print("\n🎯 MID-PRICE OPPORTUNITIES (place at mid, MM may fill):")
    print("-" * 60)
    mid_opps = [r for r in all_results if r['mid_fillable'] and r['otm_pct'] >= 5 and r['spread_pct'] < 25]
    mid_opps.sort(key=lambda x: (-x['ann_return_mid'], x['spread_pct']))

    for i, opt in enumerate(mid_opps[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        improvement = opt['mid'] - opt['bid']
        pct_improvement = (improvement / opt['bid'] * 100) if opt['bid'] > 0 else 0
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}P ({opt['dte']}d, {opt['otm_pct']:.0f}% OTM)")
        print(f"     Mid ${opt['mid']:.2f} (vs bid ${opt['bid']:.2f}, +${improvement:.2f}/+{pct_improvement:.0f}%) | ~{opt['ann_return_mid']:.0f}% ann | {delta_str}")

    # High volume/liquidity
    print("\n💧 HIGHEST LIQUIDITY (today's volume):")
    print("-" * 60)
    liquid = [r for r in all_results if r['volume'] > 0 and r['otm_pct'] >= 5]
    liquid.sort(key=lambda x: -x['volume'])

    for i, opt in enumerate(liquid[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}P - Vol: {opt['volume']} | Bid ${opt['bid']:.2f} | {delta_str}")

    # Wide spreads but good premium
    print("\n⏳ WIDE SPREADS - PLACE AT MID (patient orders):")
    print("-" * 60)
    wide_but_good = [r for r in all_results if 25 <= r['spread_pct'] <= 100 and r['otm_pct'] >= 8 and r['mid'] >= 0.10]
    wide_but_good.sort(key=lambda x: -x['ann_return_mid'])

    for i, opt in enumerate(wide_but_good[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}P ({opt['dte']}d, {opt['otm_pct']:.0f}% OTM)")
        print(f"     Bid ${opt['bid']:.2f} / Ask ${opt['ask']:.2f} → Try Mid ${opt['mid']:.2f} | {delta_str}")

    # Quick picks
    print(f"\n{'='*80}")
    print("QUICK PICKS:")
    print(f"{'='*80}")

    best = [r for r in all_results if r['grade'] in ['A', 'B'] and r['otm_pct'] >= 5]
    best.sort(key=lambda x: (-x['liquidity_score'], -x['ann_return_mid']))

    if best:
        top = best[0]
        print(f"  🏆 TOP PICK: {top['expiry']} ${top['strike']:.1f} Put")
        print(f"     {top['dte']} DTE, {top['otm_pct']:.1f}% OTM, Grade {top['grade']}")
        print(f"     Bid: ${top['bid']:.2f} | Mid: ${top['mid']:.2f} | Ask: ${top['ask']:.2f}")
        print(f"     Premium at bid: ${top['bid']*100:.0f} | at mid: ${top['mid']*100:.0f}")
        if top['delta']:
            print(f"     Delta: {top['delta']:.3f}")

    ib.disconnect()


def cmd_scan_calls(args):
    """Scan multiple expirations for covered call opportunities"""
    ib = connect()

    stock = Stock(args.symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close
    if not spot or not (spot > 0):
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    print(f"{args.symbol} Spot: ${spot:.2f}")
    print(f"Scanning expirations from {args.min_dte} to {args.max_dte} DTE...\n")

    # Get option chains
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

    # Find all expirations in DTE range
    target_expirations = []
    for chain in chains:
        if chain.exchange == 'SMART':
            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d')
                dte = (exp_date - datetime.now()).days
                if args.min_dte <= dte <= args.max_dte:
                    target_expirations.append((exp, dte))

    target_expirations = sorted(set(target_expirations), key=lambda x: x[1])

    if not target_expirations:
        print("No expirations found in DTE range")
        ib.disconnect()
        return

    print(f"Found {len(target_expirations)} expirations: {[e[0] for e in target_expirations]}")

    # Get strikes for covered calls - include ITM, ATM, and OTM
    all_strikes = set()
    for chain in chains:
        if chain.exchange == 'SMART':
            all_strikes.update(chain.strikes)

    strikes = sorted([s for s in all_strikes if spot * 0.70 <= s <= spot * 1.30])
    print(f"Analyzing {len(strikes)} strikes: {strikes}")

    all_results = []

    for exp, dte in target_expirations:
        print(f"\n{'='*80}")
        print(f"EXPIRY: {exp} ({dte} DTE)")
        print(f"{'='*80}")
        print(f"{'Strike':>7} {'OTM%':>6} {'Bid':>6} {'Ask':>6} {'Mid':>6} {'Sprd%':>6} {'Vol':>5} {'OI':>6} {'Delta':>7} {'IV':>6} {'Grade'}")
        print("-" * 80)

        calls = [Option(args.symbol, exp, strike, 'C', 'SMART') for strike in strikes]
        ib.qualifyContracts(*calls)

        for opt in calls:
            ib.reqMktData(opt, '', False, False)
        ib.sleep(3)

        for opt in calls:
            t = ib.ticker(opt)
            bid = t.bid if t.bid and t.bid > 0 else 0
            ask = t.ask if t.ask and t.ask > 0 else 0

            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread = ask - bid
                spread_pct = (spread / mid * 100) if mid > 0 else 999

                # Get greeks
                delta = t.modelGreeks.delta if t.modelGreeks else None
                iv = t.modelGreeks.impliedVol if t.modelGreeks else None

                # Volume and open interest
                vol = int(t.volume) if t.volume and t.volume > 0 else 0

                # OTM% for calls is (strike - spot) / spot (negative = ITM)
                otm_pct = (opt.strike - spot) / spot * 100

                if True:  # Show all strikes (ITM, ATM, OTM)
                    # Calculate metrics
                    ann_return_bid = (bid / spot) * (365 / dte) * 100
                    ann_return_mid = (mid / spot) * (365 / dte) * 100

                    # Liquidity score
                    liquidity_score = 0
                    if spread_pct < 10:
                        liquidity_score += 3
                    elif spread_pct < 15:
                        liquidity_score += 2
                    elif spread_pct < 25:
                        liquidity_score += 1

                    if vol >= 100:
                        liquidity_score += 3
                    elif vol >= 20:
                        liquidity_score += 2
                    elif vol >= 5:
                        liquidity_score += 1

                    # Grade based on spread
                    if spread_pct < 10:
                        grade = "A"
                    elif spread_pct < 20:
                        grade = "B"
                    elif spread_pct < 50:
                        grade = "C"
                    else:
                        grade = "D"

                    mid_fillable = spread_pct < 20

                    all_results.append({
                        'expiry': exp,
                        'dte': dte,
                        'strike': opt.strike,
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread': spread,
                        'spread_pct': spread_pct,
                        'volume': vol,
                        'delta': delta,
                        'iv': iv,
                        'otm_pct': otm_pct,
                        'ann_return_bid': ann_return_bid,
                        'ann_return_mid': ann_return_mid,
                        'liquidity_score': liquidity_score,
                        'grade': grade,
                        'mid_fillable': mid_fillable
                    })

                    delta_str = f"{delta:.3f}" if delta else "N/A"
                    iv_str = f"{iv*100:.0f}%" if iv else "N/A"
                    print(f"${opt.strike:>6.1f} {otm_pct:>5.1f}% ${bid:>5.2f} ${ask:>5.2f} ${mid:>5.2f} {spread_pct:>5.0f}% {vol:>5} {'N/A':>6} {delta_str:>7} {iv_str:>6} [{grade}]")

            ib.cancelMktData(opt)

    # Summary
    print(f"\n{'='*80}")
    print("COVERED CALL ANALYSIS SUMMARY")
    print(f"{'='*80}")

    # Best for conservative selling (at bid)
    print("\n📊 BEST AT BID (conservative, high fill probability):")
    print("-" * 60)
    good_bid = [r for r in all_results if r['spread_pct'] < 15 and r['otm_pct'] >= 3]
    good_bid.sort(key=lambda x: (-x['ann_return_bid'], x['spread_pct']))

    for i, opt in enumerate(good_bid[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        called_away = opt['strike']
        total_return = ((called_away - spot + opt['bid']) / spot) * 100
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}C ({opt['dte']}d, {opt['otm_pct']:.0f}% OTM)")
        print(f"     Bid ${opt['bid']:.2f} | If called: ${called_away:.2f} (+{total_return:.1f}% total) | {delta_str}")

    # Mid-price opportunities
    print("\n🎯 MID-PRICE OPPORTUNITIES (place at mid, MM may fill):")
    print("-" * 60)
    mid_opps = [r for r in all_results if r['mid_fillable'] and r['otm_pct'] >= 3 and r['spread_pct'] < 25]
    mid_opps.sort(key=lambda x: (-x['ann_return_mid'], x['spread_pct']))

    for i, opt in enumerate(mid_opps[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        improvement = opt['mid'] - opt['bid']
        pct_improvement = (improvement / opt['bid'] * 100) if opt['bid'] > 0 else 0
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}C ({opt['dte']}d, {opt['otm_pct']:.0f}% OTM)")
        print(f"     Mid ${opt['mid']:.2f} (vs bid ${opt['bid']:.2f}, +${improvement:.2f}/+{pct_improvement:.0f}%) | {delta_str}")

    # High volume/liquidity
    print("\n💧 HIGHEST LIQUIDITY (today's volume):")
    print("-" * 60)
    liquid = [r for r in all_results if r['volume'] > 0 and r['otm_pct'] >= 0]
    liquid.sort(key=lambda x: -x['volume'])

    for i, opt in enumerate(liquid[:5]):
        delta_str = f"δ={opt['delta']:.2f}" if opt['delta'] else ""
        print(f"  {i+1}. {opt['expiry']} ${opt['strike']:.1f}C - Vol: {opt['volume']} | Bid ${opt['bid']:.2f} | {delta_str}")

    # Quick picks
    print(f"\n{'='*80}")
    print("QUICK PICKS:")
    print(f"{'='*80}")

    best = [r for r in all_results if r['grade'] in ['A', 'B'] and r['otm_pct'] >= 3]
    best.sort(key=lambda x: (-x['liquidity_score'], -x['ann_return_mid']))

    if best:
        top = best[0]
        called_away = top['strike']
        total_return = ((called_away - spot + top['mid']) / spot) * 100
        print(f"  🏆 TOP PICK: {top['expiry']} ${top['strike']:.1f} Call")
        print(f"     {top['dte']} DTE, {top['otm_pct']:.1f}% OTM, Grade {top['grade']}")
        print(f"     Bid: ${top['bid']:.2f} | Mid: ${top['mid']:.2f} | Ask: ${top['ask']:.2f}")
        print(f"     Premium at bid: ${top['bid']*100:.0f} | at mid: ${top['mid']*100:.0f}")
        print(f"     If called away at ${called_away:.2f}: +{total_return:.1f}% total return")
        if top['delta']:
            print(f"     Delta: {top['delta']:.3f}")

    ib.disconnect()


def cmd_snapshot(args):
    """Full account P&L snapshot with positions and balances"""
    ib = connect()

    print(f"{'='*60}")
    print(f"IBKR ACCOUNT SNAPSHOT - {get_timestamp()}")
    print(f"{'='*60}")

    # Wait for data to stream in
    ib.sleep(3)

    # Get account values for each account
    for account in ib.managedAccounts():
        print(f"\n{'='*60}")
        print(f"ACCOUNT: {account}")
        print(f"{'='*60}")

        # Get account values
        acct_values = {v.tag: v.value for v in ib.accountValues() if v.account == account}

        print(f"\n--- BALANCES ---")
        print(f"  Net Liquidation:    ${float(acct_values.get('NetLiquidation', 0)):>12,.2f}")
        print(f"  Total Cash:         ${float(acct_values.get('TotalCashValue', 0)):>12,.2f}")
        print(f"  Available Funds:    ${float(acct_values.get('AvailableFunds', 0)):>12,.2f}")
        print(f"  Buying Power:       ${float(acct_values.get('BuyingPower', 0)):>12,.2f}")

        print(f"\n--- MARGIN ---")
        print(f"  Init Margin Req:    ${float(acct_values.get('InitMarginReq', 0)):>12,.2f}")
        print(f"  Maint Margin Req:   ${float(acct_values.get('MaintMarginReq', 0)):>12,.2f}")
        print(f"  Excess Liquidity:   ${float(acct_values.get('ExcessLiquidity', 0)):>12,.2f}")

        print(f"\n--- P&L ---")
        print(f"  Unrealized P&L:     ${float(acct_values.get('UnrealizedPnL', 0)):>12,.2f}")
        print(f"  Realized P&L:       ${float(acct_values.get('RealizedPnL', 0)):>12,.2f}")

        # Get portfolio items for this account (includes P&L)
        portfolio = [p for p in ib.portfolio() if p.account == account]

        if portfolio:
            print(f"\n--- POSITIONS (with P&L) ---")
            total_unrealized = 0
            for item in portfolio:
                symbol = item.contract.localSymbol or item.contract.symbol
                qty = item.position
                mkt_value = item.marketValue
                avg_cost = item.averageCost
                unrealized = item.unrealizedPNL
                realized = item.realizedPNL
                total_unrealized += unrealized if unrealized else 0

                print(f"\n  {symbol}")
                print(f"    Quantity:       {qty:>10.0f}")
                print(f"    Market Value:   ${mkt_value:>10.2f}")
                print(f"    Avg Cost:       ${avg_cost:>10.2f}")
                print(f"    Unrealized P&L: ${unrealized:>10.2f}" if unrealized else "    Unrealized P&L:        N/A")
                print(f"    Realized P&L:   ${realized:>10.2f}" if realized else "    Realized P&L:          N/A")

            print(f"\n  {'─'*29}")
            print(f"  Total Position P&L: ${total_unrealized:>10.2f}")
        else:
            # Fallback to positions() if portfolio() is empty
            positions = [p for p in ib.positions() if p.account == account]
            if positions:
                print(f"\n--- POSITIONS ---")
                for p in positions:
                    symbol = p.contract.localSymbol or p.contract.symbol
                    print(f"\n  {symbol}")
                    print(f"    Quantity:       {p.position:>10.0f}")
                    print(f"    Avg Cost:       ${p.avgCost:>10.2f}")

    ib.disconnect()


def cmd_sell_put(args):
    """Sell a put option"""
    ib = connect()

    # Check existing orders first
    ib.reqAllOpenOrders()
    ib.sleep(2)
    existing = [t for t in ib.openTrades()
                if t.contract.symbol == args.symbol and t.order.lmtPrice == args.price]

    if existing:
        print(f"WARNING: Already have order at ${args.price:.2f}")
        print("Use 'cancel-all' first if you want to replace orders")
        ib.disconnect()
        return

    opt = Option(args.symbol, args.expiry, args.strike, 'P', 'SMART')
    ib.qualifyContracts(opt)

    order = LimitOrder(
        action='SELL',
        totalQuantity=args.qty,
        lmtPrice=args.price,
        tif='GTC',
        account=DEFAULT_ACCOUNT
    )

    trade = ib.placeOrder(opt, order)
    ib.sleep(2)

    print(f"Placed: SELL {args.qty} {opt.localSymbol} @ ${args.price:.2f}")
    print(f"Status: {trade.orderStatus.status}")

    ib.disconnect()


def cmd_spread(args):
    """Place a vertical spread (bear call or bull put)

    Three modes:
        1. CREDIT (default): Sell short_strike, Buy long_strike → receive credit
           spread NVDA 20260320 P 190 185

        2. DEBIT (--open-debit): Buy short_strike, Sell long_strike → pay debit
           spread NVDA 20260227 P 190 185 --open-debit
           (Used for long spreads, e.g., bear put debit spread, bull call debit spread)

        3. CLOSE (--close): Auto-detects your position and reverses it
           spread NVDA 20260320 P 190 185 --close
           (Checks positions to determine if you're long or short the spread)
    """
    ib = connect()

    is_closing = getattr(args, 'close', False)
    is_open_debit = getattr(args, 'open_debit', False)

    if is_closing and is_open_debit:
        print("ERROR: Cannot use --close and --open-debit together")
        ib.disconnect()
        return

    # Create option contracts
    short_opt = Option(args.symbol, args.expiry, args.short_strike, args.right, 'SMART')
    long_opt = Option(args.symbol, args.expiry, args.long_strike, args.right, 'SMART')

    ib.qualifyContracts(short_opt, long_opt)

    # For --close, auto-detect position direction
    if is_closing:
        positions = ib.positions()
        short_pos = 0
        long_pos = 0
        for p in positions:
            if p.contract.conId == short_opt.conId:
                short_pos = p.position
            elif p.contract.conId == long_opt.conId:
                long_pos = p.position

        if short_pos == 0 and long_pos == 0:
            print(f"ERROR: No position found for {args.symbol} {args.expiry} {args.right} {args.short_strike}/{args.long_strike}")
            ib.disconnect()
            return

        # Determine direction: are we long or short the spread?
        # Credit spread = short the short_strike, long the long_strike (short_pos < 0, long_pos > 0)
        # Debit spread = long the short_strike, short the long_strike (short_pos > 0, long_pos < 0)
        if short_pos < 0 and long_pos > 0:
            close_direction = 'close_credit'  # We sold a credit spread, close by buying it back (debit)
            close_qty = min(abs(short_pos), abs(long_pos))
        elif short_pos > 0 and long_pos < 0:
            close_direction = 'close_debit'  # We bought a debit spread, close by selling it back (credit)
            close_qty = min(abs(short_pos), abs(long_pos))
        else:
            print(f"WARNING: Unexpected position shape: short_strike={short_pos}, long_strike={long_pos}")
            print(f"  Cannot auto-detect direction. Use explicit --open-debit or default credit mode.")
            ib.disconnect()
            return

        # Override qty if user didn't specify or specified less than position
        if args.qty == 1 and close_qty > 1:
            print(f"  Position size: {close_qty} contracts (use --qty to close partial)")
        args.qty = min(args.qty, close_qty) if args.qty > 1 else close_qty

    # Get current prices for reference
    ib.reqMktData(short_opt, '', False, False)
    ib.reqMktData(long_opt, '', False, False)
    ib.sleep(2)

    short_t = ib.ticker(short_opt)
    long_t = ib.ticker(long_opt)

    short_mid = (short_t.bid + short_t.ask) / 2 if short_t.bid and short_t.ask else 0
    long_mid = (long_t.bid + long_t.ask) / 2 if long_t.bid and long_t.ask else 0

    print(f"Spread: {args.symbol} {args.expiry} {args.right}")
    print(f"  Short ${args.short_strike}: Bid ${short_t.bid:.2f} / Ask ${short_t.ask:.2f}")
    print(f"  Long  ${args.long_strike}: Bid ${long_t.bid:.2f} / Ask ${long_t.ask:.2f}")

    # ──────────────────────────────────────────────────────────────
    # CLOSE MODE: auto-detected direction
    # ──────────────────────────────────────────────────────────────
    if is_closing:
        if close_direction == 'close_credit':
            # We're short a credit spread → close by BUYING it back (debit)
            # BUY short_strike, SELL long_strike
            natural_debit = short_t.ask - long_t.bid if short_t.ask and long_t.bid else 0
            mid_debit = short_mid - long_mid
            aggressive_debit = round(mid_debit - (natural_debit - mid_debit) * 0.5, 2)

            print(f"\n  === CLOSING CREDIT SPREAD (buying back) ===")
            print(f"  Position: short {args.short_strike} / long {args.long_strike} × {args.qty}")
            print(f"  Natural debit: ${natural_debit:.2f} (guaranteed fill)")
            print(f"  Mid debit:     ${mid_debit:.2f}")
            print(f"  Aggressive:    ${aggressive_debit:.2f} (lower = better)")

            debit_arg = getattr(args, 'debit', None)
            if debit_arg:
                target_debit = debit_arg
            elif getattr(args, 'aggressive', False):
                target_debit = aggressive_debit
            else:
                target_debit = round(natural_debit, 2)

            print(f"\n  Close at ${target_debit:.2f} debit")

            if args.dry_run:
                print(f"\n  [DRY RUN - Order not placed]")
                print(f"  To place: remove --dry-run flag")
                ib.disconnect()
                return

            print(f"\n  Placing close order at ${target_debit:.2f} debit...")

            combo = Contract()
            combo.symbol = args.symbol
            combo.secType = 'BAG'
            combo.currency = 'USD'
            combo.exchange = 'SMART'

            leg1 = ComboLeg()
            leg1.conId = short_opt.conId
            leg1.ratio = 1
            leg1.action = 'BUY'
            leg1.exchange = 'SMART'

            leg2 = ComboLeg()
            leg2.conId = long_opt.conId
            leg2.ratio = 1
            leg2.action = 'SELL'
            leg2.exchange = 'SMART'

            combo.comboLegs = [leg1, leg2]

            order = LimitOrder('BUY', args.qty, target_debit)
            order.account = DEFAULT_ACCOUNT
            order.tif = 'GTC'

        else:
            # We're long a debit spread → close by SELLING it back (credit)
            # SELL short_strike, BUY long_strike
            natural_credit = short_t.bid - long_t.ask if short_t.bid and long_t.ask else 0
            mid_credit = short_mid - long_mid
            aggressive_credit = round(mid_credit + (mid_credit - natural_credit) * 0.5, 2)

            print(f"\n  === CLOSING DEBIT SPREAD (selling back) ===")
            print(f"  Position: long {args.short_strike} / short {args.long_strike} × {args.qty}")
            print(f"  Natural credit: ${natural_credit:.2f} (guaranteed fill)")
            print(f"  Mid credit:     ${mid_credit:.2f}")
            print(f"  Aggressive:     ${aggressive_credit:.2f} (higher = better)")

            credit_arg = getattr(args, 'credit', None)
            if credit_arg:
                target_credit = credit_arg
            elif getattr(args, 'aggressive', False):
                target_credit = aggressive_credit
            else:
                target_credit = round(natural_credit, 2)

            print(f"\n  Close at ${target_credit:.2f} credit")

            if args.dry_run:
                print(f"\n  [DRY RUN - Order not placed]")
                print(f"  To place: remove --dry-run flag")
                ib.disconnect()
                return

            print(f"\n  Placing close order at ${target_credit:.2f} credit...")

            combo = Contract()
            combo.symbol = args.symbol
            combo.secType = 'BAG'
            combo.currency = 'USD'
            combo.exchange = 'SMART'

            leg1 = ComboLeg()
            leg1.conId = short_opt.conId
            leg1.ratio = 1
            leg1.action = 'SELL'
            leg1.exchange = 'SMART'

            leg2 = ComboLeg()
            leg2.conId = long_opt.conId
            leg2.ratio = 1
            leg2.action = 'BUY'
            leg2.exchange = 'SMART'

            combo.comboLegs = [leg1, leg2]

            order = LimitOrder('BUY', args.qty, -target_credit)
            order.account = DEFAULT_ACCOUNT
            order.tif = 'GTC'

        trade = ib.placeOrder(combo, order)
        ib.sleep(3)

        print(f"  Status: {trade.orderStatus.status}")

        ib.disconnect()
        return

    # ──────────────────────────────────────────────────────────────
    # OPEN DEBIT MODE: Buy short_strike, Sell long_strike → pay debit
    # ──────────────────────────────────────────────────────────────
    if is_open_debit:
        natural_debit = short_t.ask - long_t.bid if short_t.ask and long_t.bid else 0
        mid_debit = short_mid - long_mid
        aggressive_debit = round(mid_debit - (natural_debit - mid_debit) * 0.5, 2)

        print(f"\n  === OPENING DEBIT SPREAD ===")
        print(f"  Natural debit: ${natural_debit:.2f} (guaranteed fill)")
        print(f"  Mid debit:     ${mid_debit:.2f}")
        print(f"  Aggressive:    ${aggressive_debit:.2f} (lower = better)")

        debit_arg = getattr(args, 'debit', None)
        if debit_arg:
            target_debit = debit_arg
        elif getattr(args, 'aggressive', False):
            target_debit = aggressive_debit
        else:
            target_debit = round(mid_debit, 2)

        spread_type = "Bear Put" if args.right == 'P' else "Bull Call"
        width = abs(args.long_strike - args.short_strike)
        max_loss = target_debit * 100
        max_profit = (width * 100) - max_loss
        breakeven = args.short_strike - target_debit if args.right == 'P' else args.short_strike + target_debit

        print(f"\n{spread_type} Debit Spread:")
        print(f"  BUY  {args.qty} {args.symbol} ${args.short_strike} {args.right}")
        print(f"  SELL {args.qty} {args.symbol} ${args.long_strike} {args.right}")
        print()
        print(f"  Target Debit:   ${target_debit:.2f} (${max_loss:.0f})")
        print(f"  Max Profit:     ${max_profit:.0f}")
        print(f"  Breakeven:      ${breakeven:.2f}")
        print(f"  Risk/Reward:    {max_loss/max_profit:.1f}:1" if max_profit > 0 else "")

        if args.dry_run:
            print(f"\n  [DRY RUN - Order not placed]")
            print(f"  To place: remove --dry-run flag")
            ib.disconnect()
            return

        print(f"\n  Placing order at ${target_debit:.2f} debit...")

        combo = Contract()
        combo.symbol = args.symbol
        combo.secType = 'BAG'
        combo.currency = 'USD'
        combo.exchange = 'SMART'

        leg1 = ComboLeg()
        leg1.conId = short_opt.conId
        leg1.ratio = 1
        leg1.action = 'BUY'
        leg1.exchange = 'SMART'

        leg2 = ComboLeg()
        leg2.conId = long_opt.conId
        leg2.ratio = 1
        leg2.action = 'SELL'
        leg2.exchange = 'SMART'

        combo.comboLegs = [leg1, leg2]

        order = LimitOrder('BUY', args.qty, target_debit)
        order.account = DEFAULT_ACCOUNT
        order.tif = 'GTC'

        trade = ib.placeOrder(combo, order)
        ib.sleep(3)

        print(f"  Status: {trade.orderStatus.status}")

        ib.disconnect()
        return

    # ──────────────────────────────────────────────────────────────
    # OPEN CREDIT MODE (default): Sell short_strike, Buy long_strike → receive credit
    # ──────────────────────────────────────────────────────────────
    natural_credit = short_t.bid - long_t.ask if short_t.bid and long_t.ask else 0
    mid_credit = short_mid - long_mid

    print(f"  Natural credit: ${natural_credit:.2f}")
    print(f"  Mid credit:     ${mid_credit:.2f}")
    print()

    aggressive_credit = round(mid_credit + (mid_credit - natural_credit) * 0.5, 2)

    print(f"  Aggressive:     ${aggressive_credit:.2f}")

    if args.credit:
        target_credit = args.credit
    elif getattr(args, 'aggressive', False):
        target_credit = aggressive_credit
    elif args.use_mid:
        target_credit = round(mid_credit, 2)
    else:
        target_credit = round(mid_credit, 2)

    spread_type = "Bear Call" if args.right == 'C' else "Bull Put"
    width = abs(args.long_strike - args.short_strike)
    max_loss = (width * 100) - (target_credit * 100)
    max_profit = target_credit * 100
    breakeven = args.short_strike + target_credit if args.right == 'C' else args.short_strike - target_credit

    print(f"\n{spread_type} Spread Analysis:")
    print(f"  SELL {args.qty} {args.symbol} ${args.short_strike} {args.right}")
    print(f"  BUY  {args.qty} {args.symbol} ${args.long_strike} {args.right}")
    print()
    print(f"  Target Credit:  ${target_credit:.2f} (${max_profit:.0f})")
    print(f"  Max Loss:       ${max_loss:.0f}")
    print(f"  Breakeven:      ${breakeven:.2f}")
    print(f"  Risk/Reward:    {max_loss/max_profit:.1f}:1")

    if args.dry_run:
        print(f"\n  [DRY RUN - Order not placed]")
        print(f"  To place: remove --dry-run flag")
        ib.disconnect()
        return

    print(f"\n  Placing order at ${target_credit:.2f} credit...")

    combo = Contract()
    combo.symbol = args.symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'

    leg1 = ComboLeg()
    leg1.conId = short_opt.conId
    leg1.ratio = 1
    leg1.action = 'SELL'
    leg1.exchange = 'SMART'

    leg2 = ComboLeg()
    leg2.conId = long_opt.conId
    leg2.ratio = 1
    leg2.action = 'BUY'
    leg2.exchange = 'SMART'

    combo.comboLegs = [leg1, leg2]

    order = LimitOrder('BUY', args.qty, -target_credit)
    order.account = DEFAULT_ACCOUNT
    order.tif = 'GTC'

    trade = ib.placeOrder(combo, order)
    ib.sleep(3)

    print(f"  Status: {trade.orderStatus.status}")

    ib.disconnect()


def cmd_reverse_calendar(args):
    """
    Place a 4-leg reverse calendar spread (double diagonal calendar)

    Reverse Calendar: Buy short-dated, Sell long-dated options
    - BUY  short_expiry PUT  @ put_strike
    - SELL long_expiry  PUT  @ put_strike
    - BUY  short_expiry CALL @ call_strike
    - SELL long_expiry  CALL @ call_strike

    This profits from IV crush after earnings.

    Usage:
        rc INTC 20260123 20260130 53 55              # Open at mid price
        rc INTC 20260123 20260130 53 55 --credit 1.10  # Specify credit
        rc INTC 20260123 20260130 53 55 --dry-run   # Preview only
        rc INTC 20260123 20260130 53 55 --close     # Close position
        rc INTC 20260123 20260130 53 55 --close --debit 0.80  # Close at specific debit
    """
    is_closing = getattr(args, 'close', False)
    ib = connect()

    # Create the 4 option contracts
    short_put = Option(args.symbol, args.short_expiry, args.put_strike, 'P', 'SMART')
    long_put = Option(args.symbol, args.long_expiry, args.put_strike, 'P', 'SMART')
    short_call = Option(args.symbol, args.short_expiry, args.call_strike, 'C', 'SMART')
    long_call = Option(args.symbol, args.long_expiry, args.call_strike, 'C', 'SMART')

    ib.qualifyContracts(short_put, long_put, short_call, long_call)

    # Get current prices
    ib.reqMktData(short_put, '', False, False)
    ib.reqMktData(long_put, '', False, False)
    ib.reqMktData(short_call, '', False, False)
    ib.reqMktData(long_call, '', False, False)
    ib.sleep(2)

    sp_t = ib.ticker(short_put)
    lp_t = ib.ticker(long_put)
    sc_t = ib.ticker(short_call)
    lc_t = ib.ticker(long_call)

    # Calculate mid prices
    sp_mid = (sp_t.bid + sp_t.ask) / 2 if sp_t.bid and sp_t.ask else 0
    lp_mid = (lp_t.bid + lp_t.ask) / 2 if lp_t.bid and lp_t.ask else 0
    sc_mid = (sc_t.bid + sc_t.ask) / 2 if sc_t.bid and sc_t.ask else 0
    lc_mid = (lc_t.bid + lc_t.ask) / 2 if lc_t.bid and lc_t.ask else 0

    # Reverse calendar credit = (sell long-dated) - (buy short-dated)
    put_credit = lp_t.bid - sp_t.ask if lp_t.bid and sp_t.ask else 0
    call_credit = lc_t.bid - sc_t.ask if lc_t.bid and sc_t.ask else 0
    natural_credit = put_credit + call_credit

    put_mid_credit = lp_mid - sp_mid
    call_mid_credit = lc_mid - sc_mid
    mid_credit = put_mid_credit + call_mid_credit

    print(f"\n{'='*70}")
    print(f"REVERSE CALENDAR: {args.symbol}")
    print(f"{'='*70}")
    print(f"\nShort expiry: {args.short_expiry}  |  Long expiry: {args.long_expiry}")
    print(f"Put strike: ${args.put_strike}  |  Call strike: ${args.call_strike}")
    print()
    print(f"  LEG 1: BUY  {args.short_expiry} ${args.put_strike} Put")
    print(f"         Bid ${sp_t.bid:.2f} / Ask ${sp_t.ask:.2f} (mid ${sp_mid:.2f})")
    print(f"  LEG 2: SELL {args.long_expiry} ${args.put_strike} Put")
    print(f"         Bid ${lp_t.bid:.2f} / Ask ${lp_t.ask:.2f} (mid ${lp_mid:.2f})")
    print(f"  LEG 3: BUY  {args.short_expiry} ${args.call_strike} Call")
    print(f"         Bid ${sc_t.bid:.2f} / Ask ${sc_t.ask:.2f} (mid ${sc_mid:.2f})")
    print(f"  LEG 4: SELL {args.long_expiry} ${args.call_strike} Call")
    print(f"         Bid ${lc_t.bid:.2f} / Ask ${lc_t.ask:.2f} (mid ${lc_mid:.2f})")
    print()

    if is_closing:
        # CLOSING: Calculate debit to close (reverse of opening)
        # To close: SELL short-dated (we own), BUY long-dated (we're short)

        # Validate quote data - check for bad/stale quotes
        def validate_quote(ticker, name):
            warnings = []
            if ticker.bid is None or ticker.ask is None:
                warnings.append(f"{name}: Missing bid/ask")
            elif ticker.bid < 0:
                warnings.append(f"{name}: Negative bid (${ticker.bid}) - using 0")
            elif ticker.bid > ticker.ask:
                warnings.append(f"{name}: Crossed market (bid > ask)")
            elif ticker.ask - ticker.bid > 1.0:
                warnings.append(f"{name}: Wide spread (${ticker.ask - ticker.bid:.2f})")
            return warnings

        quote_warnings = []
        quote_warnings.extend(validate_quote(sp_t, "Short Put"))
        quote_warnings.extend(validate_quote(lp_t, "Long Put"))
        quote_warnings.extend(validate_quote(sc_t, "Short Call"))
        quote_warnings.extend(validate_quote(lc_t, "Long Call"))

        if quote_warnings:
            print(f"  ⚠️  QUOTE WARNINGS:")
            for w in quote_warnings:
                print(f"      {w}")
            print()

        # Use max(0, bid) to handle negative bids (bad data)
        sp_bid = max(0, sp_t.bid) if sp_t.bid else 0
        sc_bid = max(0, sc_t.bid) if sc_t.bid else 0

        # Natural debit (worst case - what market will definitely fill)
        put_natural = sp_bid - (lp_t.ask if lp_t.ask else 0)
        call_natural = sc_bid - (lc_t.ask if lc_t.ask else 0)
        natural_debit = -(put_natural + call_natural)

        # Mid debit (use validated mid prices)
        sp_mid_valid = (sp_bid + sp_t.ask) / 2 if sp_t.ask else sp_bid
        sc_mid_valid = (sc_bid + sc_t.ask) / 2 if sc_t.ask else sc_bid
        put_mid_debit = sp_mid_valid - lp_mid
        call_mid_debit = sc_mid_valid - lc_mid
        mid_debit = -(put_mid_debit + call_mid_debit)

        print(f"  === CLOSING POSITION ===")
        print(f"  Natural debit: ${natural_debit:.2f} (guaranteed fill)")
        print(f"  Mid debit:     ${mid_debit:.2f}")

        # Sanity check - natural should be <= mid
        if natural_debit > mid_debit + 0.10:
            print(f"  ⚠️  WARNING: Natural > Mid (bad quote data likely)")
            print(f"      Using natural as reference instead of mid")
            mid_debit = natural_debit + 0.30

        # Determine target debit
        debit_arg = getattr(args, 'debit', None)
        use_algo = getattr(args, 'algo', False)

        if debit_arg:
            target_debit = debit_arg
            print(f"\n  Target debit: ${target_debit:.2f} (user specified)")
        else:
            # Default: start at natural - $0.05 (aggressive)
            target_debit = max(0.01, round(natural_debit - 0.05, 2))
            print(f"\n  Starting debit: ${target_debit:.2f} (natural - $0.05)")
            print(f"  Will work up to: ${round(mid_debit + 0.10, 2):.2f} (mid + $0.10)")
            use_algo = True  # Always use algo unless specific debit given

        # Dry run check
        if args.dry_run:
            print(f"\n  [DRY RUN - Order not placed]")
            print(f"  To place: remove --dry-run flag")
            if use_algo:
                print(f"  Algo will raise $0.01 every 10s until filled")
            ib.disconnect()
            return

        # Create combo contract with REVERSED legs
        combo = Contract()
        combo.symbol = args.symbol
        combo.secType = 'BAG'
        combo.currency = 'USD'
        combo.exchange = 'SMART'

        # REVERSED: Leg 1: SELL short-dated put (close long)
        leg1 = ComboLeg()
        leg1.conId = short_put.conId
        leg1.ratio = 1
        leg1.action = 'SELL'
        leg1.exchange = 'SMART'

        # REVERSED: Leg 2: BUY long-dated put (close short)
        leg2 = ComboLeg()
        leg2.conId = long_put.conId
        leg2.ratio = 1
        leg2.action = 'BUY'
        leg2.exchange = 'SMART'

        # REVERSED: Leg 3: SELL short-dated call (close long)
        leg3 = ComboLeg()
        leg3.conId = short_call.conId
        leg3.ratio = 1
        leg3.action = 'SELL'
        leg3.exchange = 'SMART'

        # REVERSED: Leg 4: BUY long-dated call (close short)
        leg4 = ComboLeg()
        leg4.conId = long_call.conId
        leg4.ratio = 1
        leg4.action = 'BUY'
        leg4.exchange = 'SMART'

        combo.comboLegs = [leg1, leg2, leg3, leg4]

        if use_algo and not debit_arg:
            # ASCENDING PRICE ALGO: Start low, work up
            print(f"\n  Starting ascending price algo...")

            max_debit = round(mid_debit + 0.10, 2)
            current_debit = target_debit

            order = LimitOrder('BUY', args.qty, current_debit)
            order.account = DEFAULT_ACCOUNT
            order.tif = 'DAY'

            trade = ib.placeOrder(combo, order)
            ib.sleep(3)

            print(f"  Order placed at ${current_debit:.2f}")

            while current_debit < max_debit:
                ib.sleep(1)

                if trade.orderStatus.status == 'Filled':
                    print(f"  ✓ FILLED at ${current_debit:.2f}!")
                    break

                current_debit += 0.01
                current_debit = round(current_debit, 2)

                print(f"  Raising to ${current_debit:.2f}...")
                trade.order.lmtPrice = current_debit
                ib.placeOrder(combo, trade.order)
                ib.sleep(10)

                if trade.orderStatus.status == 'Filled':
                    print(f"  ✓ FILLED at ${current_debit:.2f}!")
                    break

            if trade.orderStatus.status != 'Filled':
                print(f"\n  Not filled at max ${max_debit:.2f}")
                print(f"  Order still working - check status")
        else:
            # Single order at specified debit
            print(f"\n  Placing CLOSE order at ${target_debit:.2f}...")

            order = LimitOrder('BUY', args.qty, target_debit)
            order.account = DEFAULT_ACCOUNT
            order.tif = 'DAY'

            trade = ib.placeOrder(combo, order)
            ib.sleep(3)

            print(f"  Order ID: {trade.order.orderId}")
            print(f"  Status: {trade.orderStatus.status}")

            if trade.orderStatus.status in ['Submitted', 'PreSubmitted']:
                print(f"\n  Close order submitted for ${target_debit:.2f} debit")
                print(f"  Monitor with: python ibkr_trading.py status")

        ib.disconnect()
        return

    # OPENING position
    print(f"  Put calendar credit:  ${put_mid_credit:.2f}")
    print(f"  Call calendar credit: ${call_mid_credit:.2f}")
    print(f"  ----------------------------")
    print(f"  Natural credit: ${natural_credit:.2f}")
    print(f"  Mid credit:     ${mid_credit:.2f}")

    # Determine target credit
    if args.credit:
        target_credit = args.credit
    else:
        target_credit = round(mid_credit, 2)

    print(f"\n  Target credit: ${target_credit:.2f} (${target_credit * 100:.0f} per combo)")

    # Dry run check
    if args.dry_run:
        print(f"\n  [DRY RUN - Order not placed]")
        print(f"  To place: remove --dry-run flag")
        ib.disconnect()
        return

    print(f"\n  Placing 4-leg combo order...")

    # Create combo contract
    combo = Contract()
    combo.symbol = args.symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'

    # Leg 1: BUY short-dated put
    leg1 = ComboLeg()
    leg1.conId = short_put.conId
    leg1.ratio = 1
    leg1.action = 'BUY'
    leg1.exchange = 'SMART'

    # Leg 2: SELL long-dated put
    leg2 = ComboLeg()
    leg2.conId = long_put.conId
    leg2.ratio = 1
    leg2.action = 'SELL'
    leg2.exchange = 'SMART'

    # Leg 3: BUY short-dated call
    leg3 = ComboLeg()
    leg3.conId = short_call.conId
    leg3.ratio = 1
    leg3.action = 'BUY'
    leg3.exchange = 'SMART'

    # Leg 4: SELL long-dated call
    leg4 = ComboLeg()
    leg4.conId = long_call.conId
    leg4.ratio = 1
    leg4.action = 'SELL'
    leg4.exchange = 'SMART'

    combo.comboLegs = [leg1, leg2, leg3, leg4]

    # For credit spreads, use BUY action with negative limit price
    order = LimitOrder('BUY', args.qty, -target_credit)
    order.account = DEFAULT_ACCOUNT
    order.tif = 'DAY'  # Day order for earnings trades

    trade = ib.placeOrder(combo, order)
    ib.sleep(3)

    print(f"  Order ID: {trade.order.orderId}")
    print(f"  Status: {trade.orderStatus.status}")

    if trade.orderStatus.status == 'Submitted':
        print(f"\n  Order submitted for ${target_credit:.2f} credit")
        print(f"  Monitor with: python ibkr_trading.py status")

    ib.disconnect()


def cmd_straddle(args):
    """Show straddle pricing near forward ATM for scaling into a long vol position.

    Finds the forward ATM strike (put-call parity), shows straddle mid/greeks
    at nearby strikes, and shows current straddle positions + net delta.
    """
    import math

    ib = connect()

    stock = Stock(args.symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close
    if not spot or not (spot > 0):
        bid = ticker.bid if ticker.bid and ticker.bid > 0 else 0
        ask = ticker.ask if ticker.ask and ticker.ask > 0 else 0
        spot = (bid + ask) / 2 if bid > 0 and ask > 0 else bid or ask
    print(f"{args.symbol} Spot: ${spot:.2f}")

    # Get option chains
    chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

    # Find expiration closest to target DTE
    target_date = datetime.now() + timedelta(days=args.dte)
    best_exp = None
    best_diff = 999

    for chain in chains:
        if chain.exchange == 'SMART':
            for exp in chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d')
                diff = abs((exp_date - target_date).days)
                if diff < best_diff:
                    best_diff = diff
                    best_exp = exp

    if not best_exp:
        print("No expirations found")
        ib.disconnect()
        return

    exp_date = datetime.strptime(best_exp, '%Y%m%d')
    actual_dte = (exp_date - datetime.now()).days
    print(f"Expiry: {best_exp} ({actual_dte} DTE)")

    # Get strikes near ATM (±5% for straddles)
    all_strikes = set()
    for chain in chains:
        if chain.exchange == 'SMART':
            all_strikes.update(chain.strikes)

    strikes = sorted([s for s in all_strikes if spot * 0.95 <= s <= spot * 1.05])
    if not strikes:
        print("No strikes found near ATM")
        ib.disconnect()
        return

    # Prefer whole-dollar strikes, thin to ~15
    whole = [s for s in strikes if abs(s - round(s)) < 0.01]
    if len(whole) >= 5:
        strikes = whole
    if len(strikes) > 15:
        fives = [s for s in strikes if abs(s % 5) < 0.01 or abs(s % 5 - 5) < 0.01]
        if len(fives) >= 5:
            strikes = fives

    print(f"Fetching {len(strikes)} strikes...")

    # Qualify calls and puts
    calls = [Option(args.symbol, best_exp, s, 'C', 'SMART') for s in strikes]
    puts = [Option(args.symbol, best_exp, s, 'P', 'SMART') for s in strikes]
    ib.qualifyContracts(*calls, *puts)

    for opt in calls + puts:
        ib.reqMktData(opt, '', False, False)
    ib.sleep(4)

    # Build chain data
    chain_data = {}
    for opt in calls:
        t = ib.ticker(opt)
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        greeks = t.modelGreeks
        chain_data.setdefault(opt.strike, {})['call'] = {
            'conId': opt.conId, 'bid': bid, 'ask': ask, 'mid': mid,
            'delta': greeks.delta if greeks else None,
            'gamma': greeks.gamma if greeks else None,
            'theta': greeks.theta if greeks else None,
            'vega': greeks.vega if greeks else None,
            'iv': greeks.impliedVol if greeks else None,
        }

    for opt in puts:
        t = ib.ticker(opt)
        bid = t.bid if t.bid and t.bid > 0 else 0
        ask = t.ask if t.ask and t.ask > 0 else 0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        greeks = t.modelGreeks
        chain_data.setdefault(opt.strike, {})['put'] = {
            'conId': opt.conId, 'bid': bid, 'ask': ask, 'mid': mid,
            'delta': greeks.delta if greeks else None,
            'gamma': greeks.gamma if greeks else None,
            'theta': greeks.theta if greeks else None,
            'vega': greeks.vega if greeks else None,
            'iv': greeks.impliedVol if greeks else None,
        }

    # Find forward ATM: strike where |call_mid - put_mid| is smallest
    best_fwd = None
    best_fwd_diff = float('inf')
    for strike, sides in chain_data.items():
        c = sides.get('call')
        p = sides.get('put')
        if not c or not p or c['mid'] <= 0 or p['mid'] <= 0:
            continue
        diff = abs(c['mid'] - p['mid'])
        if diff < best_fwd_diff:
            best_fwd_diff = diff
            fwd_price = strike + c['mid'] - p['mid']
            best_fwd = {
                'strike': strike, 'forward': fwd_price,
                'straddle_mid': c['mid'] + p['mid'],
            }

    # Current positions
    positions = ib.positions()
    straddle_positions = {}  # {strike: qty}
    orphan_legs = []
    for p in positions:
        c = p.contract
        if c.symbol != args.symbol or c.secType != 'OPT':
            continue
        if c.lastTradeDateOrContractMonth != best_exp:
            continue
        strike = c.strike
        qty = int(p.position)
        if qty <= 0:
            continue
        if strike not in straddle_positions:
            straddle_positions[strike] = {'C': 0, 'P': 0}
        straddle_positions[strike][c.right] += qty

    # Match straddles
    held_straddles = {}
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0
    for strike, legs in straddle_positions.items():
        n = min(legs['C'], legs['P'])
        if n > 0:
            held_straddles[strike] = n
            sides = chain_data.get(strike, {})
            c = sides.get('call', {})
            p = sides.get('put', {})
            cd = c.get('delta') or 0
            pd = p.get('delta') or 0
            cg = c.get('gamma') or 0
            pg = p.get('gamma') or 0
            ct = c.get('theta') or 0
            pt = p.get('theta') or 0
            cv = c.get('vega') or 0
            pv = p.get('vega') or 0
            mult = n * 100
            total_delta += (cd + pd) * mult
            total_gamma += (cg + pg) * mult
            total_theta += (ct + pt) * mult
            total_vega += (cv + pv) * mult
        # Orphan legs
        extra_c = legs['C'] - n
        extra_p = legs['P'] - n
        if extra_c > 0:
            orphan_legs.append(f"{extra_c}x ${strike}C")
        if extra_p > 0:
            orphan_legs.append(f"{extra_p}x ${strike}P")

    # Display
    print(f"\n{'=' * 80}")
    print(f"  {args.symbol} STRADDLE SCAN — {best_exp} ({actual_dte} DTE)")
    print(f"{'=' * 80}")

    if best_fwd:
        print(f"\n  Forward ATM:  ${best_fwd['strike']:.0f} (fwd price ${best_fwd['forward']:.2f})")
        print(f"  ATM Straddle: ${best_fwd['straddle_mid']:.2f} mid")

        # Breakeven range
        be_up = best_fwd['strike'] + best_fwd['straddle_mid']
        be_dn = best_fwd['strike'] - best_fwd['straddle_mid']
        be_pct = best_fwd['straddle_mid'] / spot * 100
        print(f"  Breakeven:    ${be_dn:.2f} — ${be_up:.2f} ({be_pct:.1f}% move)")

    # Straddle chain table
    print(f"\n  {'Strike':>7} {'C Bid':>7} {'C Ask':>7} {'P Bid':>7} {'P Ask':>7} "
          f"{'Strad':>7} {'IV':>6} {'Delta':>7} {'Gamma':>7} {'Theta':>7} {'Vega':>6}")
    print(f"  {'-' * 85}")

    for strike in sorted(chain_data.keys()):
        sides = chain_data[strike]
        c = sides.get('call', {})
        p = sides.get('put', {})
        if c.get('mid', 0) <= 0 or p.get('mid', 0) <= 0:
            continue

        strad_mid = c['mid'] + p['mid']
        avg_iv = ((c.get('iv') or 0) + (p.get('iv') or 0)) / 2
        strad_delta = (c.get('delta') or 0) + (p.get('delta') or 0)
        strad_gamma = (c.get('gamma') or 0) + (p.get('gamma') or 0)
        strad_theta = (c.get('theta') or 0) + (p.get('theta') or 0)
        strad_vega = (c.get('vega') or 0) + (p.get('vega') or 0)

        fwd_marker = ' <<' if best_fwd and strike == best_fwd['strike'] else ''
        held = held_straddles.get(strike, 0)
        held_marker = f' [{held}x]' if held > 0 else ''

        iv_str = f'{avg_iv * 100:.1f}%' if avg_iv > 0 else 'n/a'
        print(f"  ${strike:>6.0f} {c['bid']:>7.2f} {c['ask']:>7.2f} {p['bid']:>7.2f} {p['ask']:>7.2f} "
              f"{strad_mid:>7.2f} {iv_str:>6} {strad_delta:>+7.2f} {strad_gamma:>7.4f} "
              f"{strad_theta:>7.2f} {strad_vega:>6.2f}{fwd_marker}{held_marker}")

    # Current position summary
    if held_straddles:
        total_qty = sum(held_straddles.values())
        parts = [f'{qty}x ${k:.0f}' for k, qty in sorted(held_straddles.items())]
        print(f"\n  CURRENT POSITION: {' + '.join(parts)} = {total_qty} straddles")
        print(f"    Delta: {total_delta:+,.0f}  Gamma: {total_gamma:+,.1f}  "
              f"Theta: {total_theta:+,.1f}/day  Vega: {total_vega:+,.1f}")

        if best_fwd and total_delta != 0:
            # Suggest rebalancing
            fwd_sides = chain_data.get(best_fwd['strike'], {})
            fc = fwd_sides.get('call', {})
            fp = fwd_sides.get('put', {})
            per_strad_delta = ((fc.get('delta') or 0) + (fp.get('delta') or 0)) * 100
            if abs(per_strad_delta) > 0.01:
                rebal_straddles = -total_delta / per_strad_delta
                if abs(rebal_straddles) >= 0.5:
                    print(f"    To neutralize: buy ~{abs(rebal_straddles):.0f}x ${best_fwd['strike']:.0f} straddle "
                          f"(per-strad delta {per_strad_delta:+.0f})")
        if orphan_legs:
            print(f"    Orphan legs: {', '.join(orphan_legs)}")
    else:
        if best_fwd:
            # Brenner-Subrahmanyam: Straddle ≈ 0.798 × F × σ × √T
            fwd_sides = chain_data.get(best_fwd['strike'], {})
            fc = fwd_sides.get('call', {})
            fp = fwd_sides.get('put', {})
            avg_iv = ((fc.get('iv') or 0) + (fp.get('iv') or 0)) / 2
            if avg_iv > 0 and actual_dte > 0:
                fair = 0.798 * best_fwd['forward'] * avg_iv * math.sqrt(actual_dte / 365)
                print(f"\n  Fair value (B-S approx): ${fair:.2f} vs mid ${best_fwd['straddle_mid']:.2f}")

            print(f"\n  SCALING SUGGESTION:")
            per_gamma = ((fc.get('gamma') or 0) + (fp.get('gamma') or 0)) * 100
            per_vega = ((fc.get('vega') or 0) + (fp.get('vega') or 0)) * 100
            per_theta = ((fc.get('theta') or 0) + (fp.get('theta') or 0)) * 100
            cost_1 = best_fwd['straddle_mid'] * 100
            print(f"    1x ${best_fwd['strike']:.0f} straddle = ${cost_1:,.0f}")
            print(f"      Gamma: {per_gamma:+.1f}  Vega: {per_vega:+.1f}  Theta: {per_theta:+.1f}/day")
            for n in [5, 10, 20]:
                print(f"    {n}x = ${cost_1 * n:,.0f} cost | "
                      f"G {per_gamma * n:+.0f} V {per_vega * n:+.0f} Th {per_theta * n:+.0f}/day")

    print(f"\n  Buy command:")
    if best_fwd:
        strad_ask = 0
        fwd_sides = chain_data.get(best_fwd['strike'], {})
        fc = fwd_sides.get('call', {})
        fp = fwd_sides.get('put', {})
        strad_ask = fc.get('ask', 0) + fp.get('ask', 0)
        strad_mid = best_fwd['straddle_mid']
        print(f"    python ibkr_trading.py buy-straddle {args.symbol} {best_exp} "
              f"{best_fwd['strike']:.0f} {strad_mid:.2f}")
    print(f"{'=' * 80}")

    for opt in calls + puts:
        ib.cancelMktData(opt)

    ib.disconnect()


def cmd_buy_straddle(args):
    """Buy a straddle (1 call + 1 put at same strike) as a combo order."""
    ib = connect()

    call = Option(args.symbol, args.expiry, args.strike, 'C', 'SMART')
    put = Option(args.symbol, args.expiry, args.strike, 'P', 'SMART')
    ib.qualifyContracts(call, put)

    # Get current prices
    ib.reqMktData(call, '', False, False)
    ib.reqMktData(put, '', False, False)
    ib.sleep(2)

    ct = ib.ticker(call)
    pt = ib.ticker(put)

    c_bid = ct.bid if ct.bid and ct.bid > 0 else 0
    c_ask = ct.ask if ct.ask and ct.ask > 0 else 0
    p_bid = pt.bid if pt.bid and pt.bid > 0 else 0
    p_ask = pt.ask if pt.ask and pt.ask > 0 else 0
    c_mid = (c_bid + c_ask) / 2
    p_mid = (p_bid + p_ask) / 2

    natural = c_ask + p_ask
    mid = c_mid + p_mid

    exp_date = datetime.strptime(args.expiry, '%Y%m%d')
    actual_dte = (exp_date - datetime.now()).days

    print(f"\n  {args.symbol} ${args.strike:.0f} Straddle — {args.expiry} ({actual_dte} DTE)")
    print(f"  Call: ${c_bid:.2f} / ${c_ask:.2f} (mid ${c_mid:.2f})")
    print(f"  Put:  ${p_bid:.2f} / ${p_ask:.2f} (mid ${p_mid:.2f})")
    print(f"  Natural: ${natural:.2f}  Mid: ${mid:.2f}")

    # Greeks
    cg = ct.modelGreeks
    pg = pt.modelGreeks
    if cg and pg and cg.delta is not None and pg.delta is not None:
        strad_delta = (cg.delta + pg.delta) * args.qty * 100
        strad_gamma = ((cg.gamma or 0) + (pg.gamma or 0)) * args.qty * 100
        strad_theta = ((cg.theta or 0) + (pg.theta or 0)) * args.qty * 100
        strad_vega = ((cg.vega or 0) + (pg.vega or 0)) * args.qty * 100
        avg_iv = ((cg.impliedVol or 0) + (pg.impliedVol or 0)) / 2
        print(f"  IV: {avg_iv * 100:.1f}%")
        print(f"  Greeks ({args.qty}x): D {strad_delta:+.0f}  G {strad_gamma:+.1f}  "
              f"Th {strad_theta:+.1f}/day  V {strad_vega:+.1f}")

    target = args.price
    total_cost = target * args.qty * 100
    print(f"\n  Order: BUY {args.qty}x straddle @ ${target:.2f} debit")
    print(f"  Total cost: ${total_cost:,.0f}")

    if target > natural:
        print(f"  WARNING: Price ${target:.2f} > natural ${natural:.2f} (overpaying)")
    elif target < mid:
        print(f"  Below mid — may take time to fill")

    if args.dry_run:
        print(f"\n  [DRY RUN - Order not placed]")
        ib.disconnect()
        return

    # Place as combo order: BUY 1 call + BUY 1 put
    combo = Contract()
    combo.symbol = args.symbol
    combo.secType = 'BAG'
    combo.currency = 'USD'
    combo.exchange = 'SMART'

    leg1 = ComboLeg()
    leg1.conId = call.conId
    leg1.ratio = 1
    leg1.action = 'BUY'
    leg1.exchange = 'SMART'

    leg2 = ComboLeg()
    leg2.conId = put.conId
    leg2.ratio = 1
    leg2.action = 'BUY'
    leg2.exchange = 'SMART'

    combo.comboLegs = [leg1, leg2]

    order = LimitOrder('BUY', args.qty, target)
    order.account = DEFAULT_ACCOUNT
    order.tif = 'DAY'

    trade = ib.placeOrder(combo, order)
    ib.sleep(3)

    print(f"\n  Order ID: {trade.order.orderId}")
    print(f"  Status: {trade.orderStatus.status}")
    print(f"  Monitor: python ibkr_trading.py status")

    ib.disconnect()
