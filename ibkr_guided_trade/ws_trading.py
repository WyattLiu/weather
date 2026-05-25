#!/usr/bin/env python3
"""
Wealthsimple Trade CLI — Using GraphQL API

Thin CLI shim on top of the ``ws_sdk`` package. All GraphQL queries,
mutations, session helpers, and order primitives now live in
``ws_sdk/``; this file only contains the ``argparse`` plumbing plus the
``cmd_*`` handler functions that format the CLI output.

Existing scripts that do ``from ws_trading import ...`` keep working
because every legacy symbol is re-exported from ``ws_sdk``.

Usage:
    python ws_trading.py status                         # Account balances
    python ws_trading.py positions                      # Current positions
    python ws_trading.py orders                         # Recent activities
    python ws_trading.py open-orders                    # Open orders (margin account, filtered)
    python ws_trading.py refresh                        # Re-export cookies from browser

Stock Trading:
    python ws_trading.py buy UNG 100 12.50              # Buy 100 UNG at $12.50
    python ws_trading.py sell UNG 100 13.00             # Sell 100 UNG at $13.00
    python ws_trading.py cancel <order-id>              # Cancel an order
    python ws_trading.py modify <order-id> 12.75        # Modify order price

Options:
    python ws_trading.py opt-expiry UNG                 # List option expiry dates
    python ws_trading.py opt-chain UNG 2026-01-17 PUT   # Show put options
    python ws_trading.py buy-opt <sec-id> 1 0.50        # Buy 1 contract at $0.50
    python ws_trading.py sell-opt <sec-id> 1 0.75       # Sell 1 contract at $0.75
    python ws_trading.py order-status <order-id>        # Canonical order status (NEW)

Straddle Scanning:
    python ws_trading.py straddle-scan SPY              # Scan ~25 DTE forward ATM
    python ws_trading.py straddle-scan SPY --dte 30     # Target 30 DTE
    python ws_trading.py straddle-scan SPY --expiry 2026-04-03

Multi-leg:
    python ws_trading.py straddle SPY 2026-02-20 692 18.50          # Buy straddle at $18.50 debit
    python ws_trading.py straddle SPY 2026-02-20 692 18.50 --qty 2  # 2 contracts
    python ws_trading.py straddle SPY 2026-02-20 692 18.50 --close  # Close straddle
    python ws_trading.py straddle SPY 2026-02-20 692 18.50 --dry-run
    python ws_trading.py multileg-status order-batch-00YGqBb0frTx
"""
from __future__ import annotations

import argparse
from datetime import datetime

# ---- Legacy backward-compat surface (used by ws_ung_strangle, spy_scalp, ...) --
# Every name that used to live in this file is re-exported from ws_sdk so
# existing consumers keep working unchanged.
from ws_sdk import (
    # session / auth
    get_session,
    load_config,
    save_config,
    load_cookies,
    save_cookies,
    extract_oauth_data,
    is_token_expired,
    refresh_access_token,
    update_cookies_with_new_token,
    extract_access_token,
    extract_identity_from_cookies,
    extract_accounts_from_cookies,
    # graphql transport
    graphql_query,
    # GraphQL operation strings
    QUERY_FETCH_FINANCIALS,
    QUERY_FETCH_POSITIONS,
    QUERY_FETCH_ACTIVITIES,
    QUERY_FETCH_SECURITY,
    QUERY_SECURITY_SEARCH,
    QUERY_OPTION_EXPIRATION_DATES,
    QUERY_OPTION_CHAIN,
    QUERY_MULTILEG_ORDER,
    QUERY_EXTENDED_ORDER,
    QUERY_ALL_ACCOUNTS,
    MUTATION_ORDER_CREATE,
    MUTATION_ORDER_CANCEL,
    MUTATION_ORDER_MODIFY,
    MUTATION_ORDER_EXECUTION_CREATE,
    MUTATION_PREFLIGHT_CHECK,
    # order primitives
    generate_order_id,
    place_order,
    place_multileg_order,
    cancel_order,
    modify_order,
    preflight_multileg,
    fetch_multileg_order,
    fetch_extended_order,
    wait_for_order,
    wait_for_multileg_order,
    # accounts / margin discovery
    DEFAULT_ACCOUNT_ID,
    fetch_all_accounts,
    get_margin_account_id,
    # security catalog
    KNOWN_SECURITIES,
    # new SDK surface
    WSClient,
    OrderStatus,
    OrderSide,
    AccountType,
    OrderTimeout,
    OrderNotFound,
    OrderRejected,
)

# Re-export CONFIG_DIR and friends for the small number of callers that
# reach directly into ws_trading for these paths.
from ws_sdk.auth import CONFIG_DIR, COOKIES_FILE, CONFIG_FILE, GRAPHQL_URL  # noqa: F401


# ==================== COMMANDS ====================

def cmd_status(args):
    """Show account status and balances"""
    session = get_session()
    config = load_config()
    cookies = load_cookies()

    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)
    if not identity_id:
        print("No identity_id found. Please re-export cookies.")
        return

    print("=" * 60)
    print(f"WEALTHSIMPLE ACCOUNT STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Use USD for consistency
    data = graphql_query(session, "FetchIdentityCurrentFinancials", QUERY_FETCH_FINANCIALS, {
        "identityId": identity_id,
        "currency": "USD"
    })

    if not data:
        return

    identity = data.get('identity', {})
    financials = identity.get('financials', {})
    current = financials.get('current', {})

    nlv = current.get('netLiquidationValueV2', {})
    deposits = current.get('netDeposits', {})
    returns = current.get('simpleReturns', {})

    print(f"\n--- COMBINED PORTFOLIO (USD) ---")

    if nlv:
        print(f"  Net Liquidation:  ${float(nlv.get('amount', 0)):,.2f}")

    if deposits:
        print(f"  Net Deposits:     ${float(deposits.get('amount', 0)):,.2f}")

    if returns:
        ret_amt = returns.get('amount', {})
        ret_rate = returns.get('rate', 0)
        if ret_amt:
            print(f"  Returns (P&L):    ${float(ret_amt.get('amount', 0)):+,.2f} ({float(ret_rate)*100:+.2f}%)")


def cmd_positions(args):
    """Show current positions"""
    session = get_session()
    config = load_config()
    cookies = load_cookies()

    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)
    if not identity_id:
        print("No identity_id found. Please re-export cookies.")
        return

    print("=" * 60)
    print("POSITIONS")
    print("=" * 60)

    data = graphql_query(session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, {
        "identityId": identity_id,
        "currency": "CAD",
        "first": 50,
        "aggregated": True,
        "currencyOverride": "MARKET",
        "sort": "TODAY_GAIN",
        "includeSecurity": True,
        "includeAccountData": True,
        "includeOneDayReturnsBaseline": True
    })

    if not data:
        return

    identity = data.get('identity', {})
    financials = identity.get('financials', {})
    current = financials.get('current', {})
    positions_data = current.get('positions', {})
    positions = positions_data.get('edges', [])

    if not positions:
        print("\nNo positions found")
        return

    print(f"\n{'Symbol':<24} {'Qty':>8} {'Avg Cost':>10} {'Mkt Value':>12} {'P&L':>12} {'P&L%':>8}")
    print("-" * 80)

    total_value = 0
    total_pnl = 0
    total_cost = 0

    for edge in positions:
        pos = edge.get('node', {})
        security = pos.get('security', {})

        # Get symbol from security
        stock = security.get('stock', {})
        symbol = stock.get('symbol', 'N/A')

        # For options, show strike/expiry
        option_details = security.get('optionDetails', {})
        if option_details:
            opt_type = option_details.get('optionType', '')[:1]  # C or P
            strike = option_details.get('strikePrice', 0)
            expiry = option_details.get('expiryDate', '')[5:10]  # MM-DD format
            underlying = option_details.get('underlyingSecurity', {}).get('stock', {}).get('symbol', '')
            symbol = f"{underlying} {expiry} ${strike}{opt_type}"

        # quantity is already negative for short positions - don't double-negate
        qty = float(pos.get('quantity', 0))

        # Use marketAveragePrice (USD) directly instead of calculating from book value
        avg_price = pos.get('marketAveragePrice', pos.get('averagePrice', {}))
        avg_cost = float(avg_price.get('amount', 0))

        market = pos.get('totalValue', {})
        market_value = float(market.get('amount', 0))

        book = pos.get('marketBookValue', pos.get('bookValue', {}))
        book_value = float(book.get('amount', 0))

        unrealized = pos.get('marketUnrealizedReturns', pos.get('unrealizedReturns', {}))
        pnl = float(unrealized.get('amount', 0)) if unrealized else (market_value - book_value)
        pnl_pct = (pnl / abs(book_value) * 100) if book_value != 0 else 0

        total_value += market_value
        total_pnl += pnl
        total_cost += abs(book_value)

        print(f"{symbol:<24} {qty:>8.0f} ${avg_cost:>8.2f} ${market_value:>10.2f} ${pnl:>+10.2f} {pnl_pct:>+7.1f}%")

    print("-" * 80)
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost != 0 else 0
    print(f"{'TOTAL':<24} {'':<8} {'':<10} ${total_value:>10.2f} ${total_pnl:>+10.2f} {total_pnl_pct:>+7.1f}%")


def cmd_orders(args):
    """Show recent orders/activities"""
    session = get_session()

    print("=" * 60)
    print("RECENT ACTIVITIES")
    print("=" * 60)

    data = graphql_query(session, "FetchActivityFeedItems", QUERY_FETCH_ACTIVITIES, {
        "first": 30,
        "orderBy": "OCCURRED_AT_DESC"
    })

    if not data:
        return

    activities = data.get('activityFeedItems', {}).get('edges', [])

    if not activities:
        print("\nNo activities found")
        return

    print(f"\n{'Date':<12} {'Type':<12} {'Symbol':<10} {'Status':<12} {'Qty':>8} {'Amount':>12}")
    print("-" * 72)

    for edge in activities:
        act = edge.get('node', {})

        occurred = act.get('occurredAt', '')[:10] if act.get('occurredAt') else ''
        act_type = act.get('subType') or act.get('type') or 'N/A'
        if len(act_type) > 12:
            act_type = act_type[:10] + '..'
        status = act.get('unifiedStatus') or act.get('status') or ''

        symbol = act.get('assetSymbol', '') or ''
        qty = act.get('assetQuantity', '') or ''

        amount = act.get('amount', 0)
        amount_str = f"${float(amount):,.2f}" if amount else ''
        sign = act.get('amountSign', '')
        if sign == 'negative':
            amount_str = f"-{amount_str}"

        print(f"{occurred:<12} {act_type:<12} {symbol:<10} {status:<12} {str(qty):>8} {amount_str:>12}")


def cmd_quote(args):
    """Get quote for a symbol - requires security ID"""
    session = get_session()

    # For now, we need the security ID. In a full implementation, we'd search first.
    print(f"\nNote: Quote lookup requires a security ID.")
    print(f"Use positions to see security IDs for your holdings.")
    print(f"\nTo look up {args.symbol}, you would need to search for its security ID first.")


def cmd_refresh(args):
    """Instructions to refresh cookies"""
    print("=" * 60)
    print("REFRESH COOKIES")
    print("=" * 60)
    print("""
Your access token expires every 30 minutes.
To refresh:

1. Log into https://my.wealthsimple.com in your browser
2. Open DevTools (F12) > Console
3. Run this command:

let cookies = {}; document.cookie.split(';').forEach(c => { let [k,v] = c.trim().split('='); if(k) cookies[k]=v; }); console.log(JSON.stringify(cookies, null, 2));

4. Copy the JSON output
5. Save to: ~/.ws_trade/cookies.json

Then run your command again.
""")


def cmd_buy(args):
    """Place a limit buy order"""
    session = get_session()

    symbol = args.symbol.upper()
    quantity = args.quantity
    limit_price = args.price

    # Look up security ID
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known symbols: {', '.join(KNOWN_SECURITIES.keys())}")
        print("Add security ID to KNOWN_SECURITIES in ws_trading.py")
        return

    # Confirmation
    total_cost = quantity * limit_price
    print("=" * 60)
    print("ORDER CONFIRMATION")
    print("=" * 60)
    print(f"  Action:      BUY")
    print(f"  Symbol:      {symbol}")
    print(f"  Quantity:    {quantity}")
    print(f"  Limit Price: ${limit_price:.2f}")
    print(f"  Total Cost:  ${total_cost:,.2f} (approx)")
    print(f"  Time:        DAY order")
    print("=" * 60)

    confirm = input("\nConfirm order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Order cancelled.")
        return

    print("\nPlacing order...")
    result = place_order(session, "BUY_QUANTITY", security_id, quantity, limit_price)

    order_result = result.get('result', {})
    order_data = order_result.get('soOrdersCreateOrder', {})
    errors = order_data.get('errors', [])

    if errors:
        print("Order FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        order_info = order_data.get('order', {})
        print(f"Order SUBMITTED!")
        print(f"  Order ID: {result.get('order_id')}")
        print(f"  Server ID: {order_info.get('orderId')}")
        print(f"  Created: {order_info.get('createdAt')}")


def cmd_sell(args):
    """Place a limit sell order"""
    session = get_session()

    symbol = args.symbol.upper()
    quantity = args.quantity
    limit_price = args.price

    # Look up security ID
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known symbols: {', '.join(KNOWN_SECURITIES.keys())}")
        return

    # Confirmation
    total_value = quantity * limit_price
    print("=" * 60)
    print("ORDER CONFIRMATION")
    print("=" * 60)
    print(f"  Action:      SELL")
    print(f"  Symbol:      {symbol}")
    print(f"  Quantity:    {quantity}")
    print(f"  Limit Price: ${limit_price:.2f}")
    print(f"  Total Value: ${total_value:,.2f} (approx)")
    print(f"  Time:        DAY order")
    print("=" * 60)

    confirm = input("\nConfirm order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Order cancelled.")
        return

    print("\nPlacing order...")
    result = place_order(session, "SELL_QUANTITY", security_id, quantity, limit_price)

    order_result = result.get('result', {})
    order_data = order_result.get('soOrdersCreateOrder', {})
    errors = order_data.get('errors', [])

    if errors:
        print("Order FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        order_info = order_data.get('order', {})
        print(f"Order SUBMITTED!")
        print(f"  Order ID: {result.get('order_id')}")
        print(f"  Server ID: {order_info.get('orderId')}")
        print(f"  Created: {order_info.get('createdAt')}")


def cmd_cancel(args):
    """Cancel an open order"""
    session = get_session()

    order_id = args.order_id

    print(f"Cancelling order: {order_id}")

    confirm = input("Confirm cancel? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Cancel aborted.")
        return

    result = cancel_order(session, order_id)

    cancel_data = result.get('orderServiceCancelOrder', {})
    errors = cancel_data.get('errors', [])

    if errors:
        print("Cancel FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        print(f"Order cancelled: {cancel_data.get('externalId')}")


def cmd_modify(args):
    """Modify an order's limit price"""
    session = get_session()

    order_id = args.order_id
    new_price = args.price

    print(f"Modifying order: {order_id}")
    print(f"New limit price: ${new_price:.2f}")

    confirm = input("Confirm modification? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Modification aborted.")
        return

    result = modify_order(session, order_id, new_price)

    modify_data = result.get('soOrdersModifyOrder', {})
    errors = modify_data.get('errors', [])

    if errors:
        print("Modify FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        print("Order modified successfully!")


def cmd_open_orders(args):
    """Show open/pending orders with order IDs for cancellation.

    Uses the SDK's margin-only filter so crypto activities and recurring
    orders from other accounts never pollute the output. Each order
    surfaced from the activity feed is then verified against
    ``fetch_extended_order`` so the status/qty/price shown here matches
    the truth source on WS's side — not the laggy activity feed.
    """
    session = get_session()

    print("=" * 100)
    print("OPEN ORDERS")
    print("=" * 100)

    ws = WSClient(session=session)

    # Still pull the raw activity-feed nodes so we can render price / symbol
    # in the legacy format that spy_scalp.py and ws_ung_strangle.py parse.
    data = graphql_query(session, "FetchActivityFeedItems", QUERY_FETCH_ACTIVITIES, {
        "first": 50,
        "orderBy": "OCCURRED_AT_DESC",
        "condition": {
            "accountIds": [ws.account_id],
            "types": ["DIY_BUY", "DIY_SELL", "OPTIONS_BUY", "OPTIONS_SELL", "OPTIONS_MULTILEG"],
            "unifiedStatuses": ["PENDING", "SUBMITTED", "WORKING"],
        },
    })

    if not data:
        print("\nNo open orders")
        return

    activities = data.get('activityFeedItems', {}).get('edges', [])

    # Exclude recurring/managed orders even if they slipped past the server filter.
    EXCLUDED_SUBTYPES = {'RECURRING_ORDER_UPCOMING', 'RECURRING_ORDER', 'AUTO_INVEST'}

    open_orders = []
    for edge in activities:
        act = edge.get('node', {})
        sub = (act.get('subType') or '').upper()
        if sub in EXCLUDED_SUBTYPES:
            continue
        status = act.get('unifiedStatus') or act.get('status') or ''
        if status.upper() not in ('PENDING', 'SUBMITTED', 'WORKING'):
            continue

        # Verify against the canonical extended-order query. If it's
        # already terminal (fill / cancel / reject) drop it — this is
        # where the "still see 4 orders" bug was coming from.
        ext_id = act.get('externalCanonicalId') or act.get('canonicalId') or ''
        if ext_id:
            verified = fetch_extended_order(session, ext_id)
            if verified is not None and verified.is_terminal:
                continue

        open_orders.append(act)

    if not open_orders:
        print("\nNo open orders")
        return

    print(f"\n{'Symbol':<8} {'Type':<6} {'Qty':>8} {'Price':>10} {'Order ID':<50}")
    print("-" * 100)

    for act in open_orders:
        symbol = act.get('assetSymbol', '') or ''
        order_type = act.get('type', '')
        if 'BUY' in order_type.upper():
            side = 'BUY'
        elif 'SELL' in order_type.upper():
            side = 'SELL'
        else:
            side = order_type[:6]

        qty = act.get('assetQuantity', '') or ''
        amount = act.get('amount', 0)
        qty_float = float(qty) if qty else 0
        price = float(amount) / qty_float if qty_float else 0

        order_id = act.get('externalCanonicalId', '') or act.get('canonicalId', '')

        # For options, show strike/expiry
        strike = act.get('strikePrice')
        expiry = act.get('expiryDate')
        contract_type = act.get('contractType')
        if strike and expiry:
            symbol = f"{symbol} {expiry[:10]} ${strike}{contract_type[0] if contract_type else ''}"

        print(f"{symbol:<8} {side:<6} {qty_float:>8.0f} ${price:>9.2f} {order_id:<50}")

    print("-" * 100)
    print(f"\nTotal: {len(open_orders)} open order(s)")
    print("\nTo cancel: python ws_trading.py cancel <order-id>")


def cmd_order_status(args):
    """Show canonical status for a single order by external ID.

    This is the new-in-SDK command: instead of grepping the activity
    feed for ``unifiedStatus == 'COMPLETED'`` (which is laggy and
    sometimes wrong), it calls ``soOrdersExtendedOrder`` directly and
    prints the authoritative fill state.
    """
    session = get_session()
    order_id = args.order_id

    print(f"Fetching order: {order_id}")
    order = fetch_extended_order(session, order_id)
    if order is None:
        print("Order not found (either too new or never existed).")
        return

    print("=" * 60)
    print(f"ORDER {order.external_id}")
    print("=" * 60)
    print(f"  Status:       {order.status.value}")
    print(f"  Side:         {order.side.value if order.side else '-'}")
    print(f"  Open/Close:   {order.open_close.value if order.open_close else '-'}")
    print(f"  Security:     {order.security_id}")
    print(f"  Submitted:    {order.submitted_quantity}")
    print(f"  Filled:       {order.filled_quantity}")
    if order.average_filled_price is not None:
        print(f"  Avg fill:     ${order.average_filled_price}")
    if order.limit_price is not None:
        print(f"  Limit price:  ${order.limit_price}")
    print(f"  TIF:          {order.time_in_force or '-'}")
    print(f"  Submitted at: {order.submitted_at or '-'}")
    print(f"  Last filled:  {order.last_filled_at or '-'}")
    if order.rejection_cause:
        print(f"  Rejection:    {order.rejection_cause} ({order.rejection_code})")
    print(f"  Account:      {order.canonical_account_id or '-'}")
    print("-" * 60)
    if order.is_filled:
        print("  → FILLED")
    elif order.status == OrderStatus.CANCELLED:
        print("  → CANCELLED")
    elif order.status == OrderStatus.REJECTED:
        print("  → REJECTED")
    elif order.is_partially_filled:
        print(f"  → PARTIAL (remaining: {order.remaining_quantity})")
    else:
        print("  → WORKING")


def cmd_opt_expiry(args):
    """List option expiration dates for a symbol"""
    session = get_session()

    symbol = args.symbol.upper()
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known symbols: {', '.join(KNOWN_SECURITIES.keys())}")
        return

    # Get dates for next 3 years
    from datetime import timedelta
    today = datetime.now().strftime('%Y-%m-%d')
    max_date = (datetime.now() + timedelta(days=365*3)).strftime('%Y-%m-%d')

    print(f"Fetching option expiration dates for {symbol}...")

    data = graphql_query(session, "FetchOptionExpirationDates", QUERY_OPTION_EXPIRATION_DATES, {
        "securityId": security_id,
        "minDate": today,
        "maxDate": max_date
    })

    if not data:
        return

    security = data.get('security', {})
    exp_data = security.get('optionExpirationDates', {})
    dates = exp_data.get('expirationDates', [])

    if not dates:
        print("No expiration dates found")
        return

    print(f"\n{'='*40}")
    print(f"OPTION EXPIRATION DATES - {symbol}")
    print(f"{'='*40}")
    print(f"\nFound {len(dates)} expiration dates:\n")

    for i, date in enumerate(dates[:20]):  # Show first 20
        print(f"  {i+1:2}. {date}")

    if len(dates) > 20:
        print(f"  ... and {len(dates) - 20} more")


def cmd_opt_chain(args):
    """Show option chain for a symbol/expiry/type"""
    session = get_session()

    symbol = args.symbol.upper()
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known symbols: {', '.join(KNOWN_SECURITIES.keys())}")
        return

    expiry_date = args.expiry
    option_type = args.type.upper()  # CALL or PUT

    if option_type not in ('CALL', 'PUT'):
        print("Option type must be CALL or PUT")
        return

    print(f"Fetching {option_type} options for {symbol} expiring {expiry_date}...")

    data = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
        "id": security_id,
        "expiryDate": expiry_date,
        "optionType": option_type,
        "realTimeQuote": True,
        "includeGreeks": True
    })

    if not data:
        return

    security = data.get('security', {})
    chain = security.get('optionChain', {})
    edges = chain.get('edges', [])

    if not edges:
        print("No options found for this expiry/type")
        return

    print(f"\n{'='*100}")
    print(f"{symbol} {option_type}S - Expiry: {expiry_date}")
    print(f"{'='*100}")
    print(f"\n{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Last':>8} {'OI':>8} {'IV':>8} {'Delta':>8} {'Theta':>8} {'Security ID':<42}")
    print("-" * 100)

    for edge in edges:
        node = edge.get('node', {})
        sec_id = node.get('id', '')
        quote = node.get('quoteV2', {})
        details = node.get('optionDetails', {})

        strike = float(details.get('strikePrice', 0))
        bid = float(quote.get('bid', 0))
        ask = float(quote.get('ask', 0))
        last = float(quote.get('last', 0))
        oi = int(quote.get('openInterest', 0))

        greeks = details.get('greekSymbols', {}) or {}
        iv = float(greeks.get('impliedVolatility', 0)) * 100 if greeks else 0
        delta = float(greeks.get('delta', 0)) if greeks else 0
        theta = float(greeks.get('theta', 0)) if greeks else 0

        itm = quote.get('inTheMoney', False)
        itm_mark = '*' if itm else ' '

        print(f"{strike:>7.2f}{itm_mark} {bid:>8.2f} {ask:>8.2f} {last:>8.2f} {oi:>8} {iv:>7.1f}% {delta:>+8.3f} {theta:>+8.4f} {sec_id:<42}")

    print("-" * 100)
    print("* = In The Money")
    print("\nTo trade an option, use the Security ID with buy-opt/sell-opt commands")


def cmd_buy_opt(args):
    """Place a limit buy order for an option"""
    session = get_session()

    security_id = args.security_id
    quantity = args.quantity
    limit_price = args.price
    open_close = args.open_close.upper() if args.open_close else "OPEN"

    if not security_id.startswith('sec-o-'):
        print(f"Invalid option security ID: {security_id}")
        print("Option IDs should start with 'sec-o-'")
        return

    if open_close not in ('OPEN', 'CLOSE'):
        print("open_close must be OPEN or CLOSE")
        return

    # Confirmation
    total_cost = quantity * limit_price * 100  # Options are per 100 shares
    print("=" * 60)
    print("OPTION ORDER CONFIRMATION")
    print("=" * 60)
    print(f"  Action:      BUY TO {open_close}")
    print(f"  Security ID: {security_id}")
    print(f"  Quantity:    {quantity} contract(s)")
    print(f"  Limit Price: ${limit_price:.2f} per share")
    print(f"  Total Cost:  ${total_cost:,.2f} (approx)")
    print(f"  Time:        DAY order")
    print("=" * 60)

    confirm = input("\nConfirm order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Order cancelled.")
        return

    print("\nPlacing order...")
    result = place_order(session, "BUY_QUANTITY", security_id, quantity, limit_price, open_close=open_close)

    order_result = result.get('result', {})
    order_data = order_result.get('soOrdersCreateOrder', {})
    errors = order_data.get('errors', [])

    if errors:
        print("Order FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        order_info = order_data.get('order', {})
        print(f"Order SUBMITTED!")
        print(f"  Order ID: {result.get('order_id')}")
        print(f"  Server ID: {order_info.get('orderId')}")
        print(f"  Created: {order_info.get('createdAt')}")


def cmd_sell_opt(args):
    """Place a limit sell order for an option"""
    session = get_session()

    security_id = args.security_id
    quantity = args.quantity
    limit_price = args.price
    open_close = args.open_close.upper() if args.open_close else "CLOSE"

    if not security_id.startswith('sec-o-'):
        print(f"Invalid option security ID: {security_id}")
        print("Option IDs should start with 'sec-o-'")
        return

    if open_close not in ('OPEN', 'CLOSE'):
        print("open_close must be OPEN or CLOSE")
        return

    # Confirmation
    total_value = quantity * limit_price * 100  # Options are per 100 shares
    print("=" * 60)
    print("OPTION ORDER CONFIRMATION")
    print("=" * 60)
    print(f"  Action:      SELL TO {open_close}")
    print(f"  Security ID: {security_id}")
    print(f"  Quantity:    {quantity} contract(s)")
    print(f"  Limit Price: ${limit_price:.2f} per share")
    print(f"  Total Value: ${total_value:,.2f} (approx)")
    print(f"  Time:        DAY order")
    print("=" * 60)

    confirm = input("\nConfirm order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Order cancelled.")
        return

    print("\nPlacing order...")
    result = place_order(session, "SELL_QUANTITY", security_id, quantity, limit_price, open_close=open_close)

    order_result = result.get('result', {})
    order_data = order_result.get('soOrdersCreateOrder', {})
    errors = order_data.get('errors', [])

    if errors:
        print("Order FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        order_info = order_data.get('order', {})
        print(f"Order SUBMITTED!")
        print(f"  Order ID: {result.get('order_id')}")
        print(f"  Server ID: {order_info.get('orderId')}")
        print(f"  Created: {order_info.get('createdAt')}")


def cmd_search(args):
    """Search for a security and show its ID"""
    session = get_session()

    query = args.query.upper()

    print(f"Searching for: {query}")
    print("-" * 60)

    result = graphql_query(session, "FetchSecuritySearchResult", QUERY_SECURITY_SEARCH, {
        "query": query,
        "securityGroupIds": None
    })

    search_results = result.get('securitySearch', {}).get('results', [])

    if not search_results:
        print("No results found")
        return

    for sec in search_results[:10]:  # Show top 10
        stock = sec.get('stock', {})
        symbol = stock.get('symbol', 'N/A')
        name = stock.get('name', 'N/A')
        exchange = stock.get('primaryExchange', '')
        sec_id = sec.get('id', '')
        sec_type = sec.get('securityType', '')
        options_ok = sec.get('optionsEligible', False)

        print(f"\n{symbol} - {name}")
        print(f"  ID:       {sec_id}")
        print(f"  Exchange: {exchange}")
        print(f"  Type:     {sec_type}")
        print(f"  Options:  {'Yes' if options_ok else 'No'}")

    print("\n" + "-" * 60)
    print("To add a security, update KNOWN_SECURITIES dict in ws_trading.py")


def cmd_straddle_scan(args):
    """Scan for straddle opportunities: forward ATM, pricing, greeks, delta-neutral scaling."""
    import math
    from datetime import timedelta

    session = get_session()
    symbol = args.symbol.upper()
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}. Known: {', '.join(KNOWN_SECURITIES.keys())}")
        return

    # Find expiry — scan multiple candidates for OI/vol unless --expiry given
    today = datetime.now().strftime('%Y-%m-%d')
    max_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')

    exp_data = graphql_query(session, "FetchOptionExpirationDates", QUERY_OPTION_EXPIRATION_DATES, {
        "securityId": security_id, "minDate": today, "maxDate": max_date
    })
    all_dates = exp_data.get('security', {}).get('optionExpirationDates', {}).get('expirationDates', []) if exp_data else []
    if not all_dates:
        print("No expiration dates found")
        return

    if getattr(args, 'expiry', None):
        best_exp = args.expiry
        actual_dte = (datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days
        print(f"{symbol} straddle scan — expiry {best_exp} ({actual_dte} DTE)")
    else:
        # Scan candidate expiries (20-60 DTE) for OI, volume, spread quality
        min_dte = max(15, args.dte - 10)
        max_dte = args.dte + 35
        candidates = []
        for d in all_dates:
            dte = (datetime.strptime(d, '%Y-%m-%d') - datetime.now()).days
            if min_dte <= dte <= max_dte:
                candidates.append((d, dte))

        if not candidates:
            # Fallback: closest to target
            target_date = datetime.now() + timedelta(days=args.dte)
            best_exp = min(all_dates, key=lambda d: abs((datetime.strptime(d, '%Y-%m-%d') - target_date).days))
            actual_dte = (datetime.strptime(best_exp, '%Y-%m-%d') - datetime.now()).days
            print(f"{symbol} straddle scan — expiry {best_exp} ({actual_dte} DTE)")
        else:
            # Quick-scan each candidate: fetch ATM call+put, check OI/vol/spread
            print(f"Scanning {len(candidates)} expiries ({min_dte}-{max_dte} DTE) for best liquidity...")
            expiry_scores = []

            for exp_date, dte in candidates:
                atm_oi = 0
                atm_vol = 0
                strad_mid = 0
                atm_iv = 0

                for opt_type in ('CALL', 'PUT'):
                    cdata = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
                        "id": security_id, "expiryDate": exp_date, "optionType": opt_type,
                        "realTimeQuote": True, "includeGreeks": True, "first": 80,
                    })
                    if not cdata:
                        continue

                    # Find nearest-ATM strike (|delta| closest to 0.50)
                    best_atm = None
                    best_atm_diff = float('inf')
                    for edge in cdata.get('security', {}).get('optionChain', {}).get('edges', []):
                        node = edge.get('node', {})
                        details = node.get('optionDetails', {})
                        greeks = details.get('greekSymbols', {}) or {}
                        delta = abs(float(greeks.get('delta', 0) or 0))
                        diff = abs(delta - 0.50)
                        if diff < best_atm_diff:
                            best_atm_diff = diff
                            quote = node.get('quoteV2', {})
                            best_atm = {
                                'oi': int(quote.get('openInterest', 0) or 0),
                                'vol': int(quote.get('volume', 0) or quote.get('vol', 0) or 0),
                                'bid': float(quote.get('bid', 0) or 0),
                                'ask': float(quote.get('ask', 0) or 0),
                                'iv': float(greeks.get('impliedVolatility', 0) or 0),
                            }

                    if best_atm:
                        atm_oi += best_atm['oi']
                        atm_vol += best_atm['vol']
                        mid = (best_atm['bid'] + best_atm['ask']) / 2
                        strad_mid += mid
                        if best_atm['iv'] > 0:
                            atm_iv = (atm_iv + best_atm['iv']) / 2 if atm_iv > 0 else best_atm['iv']

                if strad_mid > 0:
                    # Compute combined spread %
                    # (We summed mids; approximate spread from individual spreads)
                    pass

                # Score: heavily weight OI, then vol, prefer monthlies (3rd Friday)
                exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                is_monthly = exp_dt.weekday() == 4 and 15 <= exp_dt.day <= 21  # 3rd Friday
                is_eom = exp_dt.month != (exp_dt + timedelta(days=3)).month  # end of month

                score = atm_oi * 1.0 + atm_vol * 5.0
                if is_monthly or is_eom:
                    score *= 1.5  # Prefer monthlies — usually deeper liquidity

                expiry_scores.append({
                    'date': exp_date, 'dte': dte, 'oi': atm_oi, 'vol': atm_vol,
                    'strad_mid': strad_mid, 'iv': atm_iv, 'score': score,
                    'monthly': is_monthly or is_eom,
                })

            # Display comparison
            expiry_scores.sort(key=lambda x: -x['score'])
            print(f"\n  {'Expiry':>12} {'DTE':>5} {'ATM OI':>8} {'ATM Vol':>8} {'Strad Mid':>10} {'IV':>6} {'Type':>7} {'Score':>8}")
            print(f"  {'-' * 72}")
            for i, e in enumerate(expiry_scores):
                iv_str = f"{e['iv'] * 100:.1f}%" if e['iv'] > 0 else '  n/a'
                typ = 'MONTH' if e['monthly'] else 'weekly'
                marker = ' << BEST' if i == 0 else ''
                print(f"  {e['date']:>12} {e['dte']:>5} {e['oi']:>8,} {e['vol']:>8,} "
                      f"${e['strad_mid']:>8.2f} {iv_str:>6} {typ:>7} {e['score']:>8,.0f}{marker}")

            best_exp = expiry_scores[0]['date']
            actual_dte = expiry_scores[0]['dte']
            print(f"\n  Selected: {best_exp} ({actual_dte} DTE)")
            print(f"  Override with: --expiry YYYY-MM-DD")
    print(f"\n{symbol} straddle scan — expiry {best_exp} ({actual_dte} DTE)")

    # Fetch both chains with greeks
    chain = {}  # {strike: {call: {...}, put: {...}}}
    for opt_type in ('CALL', 'PUT'):
        cdata = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
            "id": security_id, "expiryDate": best_exp, "optionType": opt_type,
            "realTimeQuote": True, "includeGreeks": True
        })
        if not cdata:
            print(f"Failed to fetch {opt_type} chain")
            return

        for edge in cdata.get('security', {}).get('optionChain', {}).get('edges', []):
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            quote = node.get('quoteV2', {})
            greeks = details.get('greekSymbols', {}) or {}
            strike = float(details.get('strikePrice', 0))
            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
            spot = float(quote.get('underlyingSpot', 0) or 0)

            entry = {
                'sec_id': node.get('id', ''),
                'bid': bid, 'ask': ask, 'mid': mid,
                'delta': float(greeks.get('delta', 0) or 0),
                'gamma': float(greeks.get('gamma', 0) or 0),
                'theta': float(greeks.get('theta', 0) or 0),
                'vega': float(greeks.get('vega', 0) or 0),
                'iv': float(greeks.get('impliedVolatility', 0) or 0),
                'oi': int(quote.get('openInterest', 0) or 0),
                'spot': spot,
                'market_status': quote.get('marketStatus', ''),
            }
            chain.setdefault(strike, {})[opt_type.lower()] = entry

    # Get spot from chain
    spot = 0
    for sides in chain.values():
        for side in sides.values():
            if side.get('spot', 0) > 0:
                spot = side['spot']
                break
        if spot > 0:
            break

    if not spot:
        # Fallback to quote
        qdata = graphql_query(session, "FetchSecurityQuoteV2", """
query FetchSecurityQuoteV2($id: ID!) { security(id: $id) { quoteV2 { price } } }
""", {"id": security_id})
        spot = float(qdata.get('security', {}).get('quoteV2', {}).get('price', 0) or 0) if qdata else 0

    if spot <= 0:
        print("Could not get spot price")
        return

    print(f"Spot: ${spot:.2f}")

    # Filter to strikes near ATM (±5%)
    near_strikes = sorted([s for s in chain.keys()
                           if spot * 0.95 <= s <= spot * 1.05
                           and 'call' in chain[s] and 'put' in chain[s]])

    if not near_strikes:
        print("No strikes with both call and put near ATM")
        return

    # Find forward ATM
    best_fwd = None
    best_diff = float('inf')
    for strike in near_strikes:
        c = chain[strike].get('call', {})
        p = chain[strike].get('put', {})
        if c.get('mid', 0) <= 0 or p.get('mid', 0) <= 0:
            continue
        diff = abs(c['mid'] - p['mid'])
        if diff < best_diff:
            best_diff = diff
            fwd_price = strike + c['mid'] - p['mid']
            best_fwd = {
                'strike': strike, 'forward': fwd_price,
                'straddle_mid': c['mid'] + p['mid'],
                'straddle_bid': c['bid'] + p['bid'],
                'straddle_ask': c['ask'] + p['ask'],
            }

    # Fetch current positions
    config = load_config()
    cookies = load_cookies()
    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)
    held_straddles = {}  # {strike: qty}
    orphan_legs = []
    if identity_id:
        pos_data = graphql_query(session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, {
            "identityId": identity_id, "currency": "CAD", "first": 50,
            "aggregated": True, "currencyOverride": "MARKET",
            "sort": "TODAY_GAIN", "includeSecurity": True,
            "includeAccountData": True, "includeOneDayReturnsBaseline": True,
        })
        pos_calls = {}  # {strike: qty}
        pos_puts = {}
        for edge in (pos_data or {}).get('identity', {}).get('financials', {}).get('current', {}).get('positions', {}).get('edges', []):
            pos = edge.get('node', {})
            sec = pos.get('security', {})
            opt = sec.get('optionDetails', {})
            if not opt:
                continue
            und = opt.get('underlyingSecurity', {}).get('stock', {}).get('symbol', '')
            if und != symbol:
                continue
            exp = opt.get('expiryDate', '')
            if exp != best_exp:
                continue
            k = float(opt.get('strikePrice', 0))
            qty = int(float(pos.get('quantity', 0)))
            if qty <= 0:
                continue
            opt_type = opt.get('optionType', '')
            if opt_type == 'CALL':
                pos_calls[k] = pos_calls.get(k, 0) + qty
            elif opt_type == 'PUT':
                pos_puts[k] = pos_puts.get(k, 0) + qty

        for k in sorted(set(pos_calls.keys()) | set(pos_puts.keys())):
            nc = pos_calls.get(k, 0)
            np_ = pos_puts.get(k, 0)
            n = min(nc, np_)
            if n > 0:
                held_straddles[k] = n
            if nc > n:
                orphan_legs.append(f"{nc - n}x ${k:.0f}C")
            if np_ > n:
                orphan_legs.append(f"{np_ - n}x ${k:.0f}P")

    # Compute position greeks
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0
    total_value = 0
    for strike, qty in held_straddles.items():
        sides = chain.get(strike, {})
        c = sides.get('call', {})
        p = sides.get('put', {})
        mult = qty * 100
        total_delta += (c.get('delta', 0) + p.get('delta', 0)) * mult
        total_gamma += (c.get('gamma', 0) + p.get('gamma', 0)) * mult
        total_theta += (c.get('theta', 0) + p.get('theta', 0)) * mult
        total_vega += (c.get('vega', 0) + p.get('vega', 0)) * mult
        total_value += (c.get('mid', 0) + p.get('mid', 0)) * mult

    # Display
    mkt = chain.get(near_strikes[0], {}).get('call', {}).get('market_status', '')
    mkt_str = ' [AFTER HOURS]' if mkt and mkt != 'OPEN' else ''
    print(f"\n{'=' * 80}")
    print(f"  {symbol} STRADDLE SCAN — {best_exp} ({actual_dte} DTE){mkt_str}")
    print(f"{'=' * 80}")

    if best_fwd:
        print(f"\n  Forward ATM:  ${best_fwd['strike']:.0f} (fwd ${best_fwd['forward']:.2f})")
        be_up = best_fwd['strike'] + best_fwd['straddle_mid']
        be_dn = best_fwd['strike'] - best_fwd['straddle_mid']
        be_pct = best_fwd['straddle_mid'] / spot * 100
        print(f"  Straddle:     bid ${best_fwd['straddle_bid']:.2f} / mid ${best_fwd['straddle_mid']:.2f} / ask ${best_fwd['straddle_ask']:.2f}")
        print(f"  Breakeven:    ${be_dn:.2f} — ${be_up:.2f} ({be_pct:.1f}% move)")
        # Fair value
        fwd_sides = chain.get(best_fwd['strike'], {})
        fc = fwd_sides.get('call', {})
        fp = fwd_sides.get('put', {})
        avg_iv = (fc.get('iv', 0) + fp.get('iv', 0)) / 2
        if avg_iv > 0 and actual_dte > 0:
            fair = 0.798 * best_fwd['forward'] * avg_iv * math.sqrt(actual_dte / 365)
            print(f"  Fair (B-S):   ${fair:.2f}")
        if avg_iv > 0:
            print(f"  ATM IV:       {avg_iv * 100:.1f}%")

    # Chain table
    print(f"\n  {'Strike':>7} {'C Bid':>7} {'C Ask':>7} {'P Bid':>7} {'P Ask':>7} "
          f"{'Strad':>7} {'IV':>6} {'Delta':>7} {'Gamma':>7} {'Theta':>7} {'Vega':>6} {'OI':>6}")
    print(f"  {'-' * 92}")

    for strike in near_strikes:
        sides = chain[strike]
        c = sides.get('call', {})
        p = sides.get('put', {})
        if c.get('mid', 0) <= 0 or p.get('mid', 0) <= 0:
            continue

        strad_mid = c['mid'] + p['mid']
        avg_iv = (c.get('iv', 0) + p.get('iv', 0)) / 2
        strad_delta = c.get('delta', 0) + p.get('delta', 0)
        strad_gamma = c.get('gamma', 0) + p.get('gamma', 0)
        strad_theta = c.get('theta', 0) + p.get('theta', 0)
        strad_vega = c.get('vega', 0) + p.get('vega', 0)
        total_oi = c.get('oi', 0) + p.get('oi', 0)

        markers = ''
        if best_fwd and strike == best_fwd['strike']:
            markers += ' <<'
        held = held_straddles.get(strike, 0)
        if held > 0:
            markers += f' [{held}x]'

        iv_str = f'{avg_iv * 100:.1f}%' if avg_iv > 0 else '  n/a'
        print(f"  ${strike:>6.0f} {c['bid']:>7.2f} {c['ask']:>7.2f} {p['bid']:>7.2f} {p['ask']:>7.2f} "
              f"{strad_mid:>7.2f} {iv_str:>6} {strad_delta:>+7.2f} {strad_gamma:>7.4f} "
              f"{strad_theta:>7.2f} {strad_vega:>6.2f} {total_oi:>6}{markers}")

    # Position summary
    if held_straddles:
        total_qty = sum(held_straddles.values())
        parts = [f'{qty}x ${k:.0f}' for k, qty in sorted(held_straddles.items())]
        print(f"\n  CURRENT: {' + '.join(parts)} = {total_qty} straddles")
        print(f"    Mkt Value: ${total_value:,.0f}  Delta: {total_delta:+,.0f}  "
              f"Gamma: {total_gamma:+,.1f}  Theta: {total_theta:+,.1f}/day  Vega: {total_vega:+,.1f}")
        if orphan_legs:
            print(f"    Orphan legs: {', '.join(orphan_legs)}")

        # Delta neutralization recommendation
        if best_fwd and abs(total_delta) > 5:
            fwd_sides = chain.get(best_fwd['strike'], {})
            fc = fwd_sides.get('call', {})
            fp = fwd_sides.get('put', {})
            per_strad_delta = (fc.get('delta', 0) + fp.get('delta', 0)) * 100
            if abs(per_strad_delta) > 0.01:
                n_needed = -total_delta / per_strad_delta
                if n_needed > 0.5:
                    strad_bid = fc.get('bid', 0) + fp.get('bid', 0)
                    strad_mid = fc.get('mid', 0) + fp.get('mid', 0)
                    strad_ask = fc.get('ask', 0) + fp.get('ask', 0)
                    # Price approach: start near bid, work toward mid
                    start_price = round(strad_bid + 0.01, 2)
                    mid_price = round(strad_mid, 2)
                    n_round = max(1, round(n_needed))

                    print(f"\n  SCALING PLAN (delta neutralize):")
                    print(f"    Buy {n_round}x ${best_fwd['strike']:.0f} straddle (per-strad delta {per_strad_delta:+.0f})")
                    print(f"    New delta: {total_delta + per_strad_delta * n_round:+.0f}")
                    print(f"    Price approach: start ${start_price:.2f} → creep to ${mid_price:.2f}")
                    print(f"    Cost: ${strad_mid * n_round * 100:,.0f} at mid")
                    print(f"\n    Command:")
                    print(f"    python ws_trading.py straddle {symbol} {best_exp} "
                          f"{best_fwd['strike']:.0f} {start_price:.2f} --qty {n_round}")
                elif n_needed < -0.5:
                    print(f"\n  Delta is {total_delta:+.0f} — consider closing {abs(round(n_needed))} straddle(s) to neutralize")
            else:
                print(f"\n  Position is near delta-neutral ({total_delta:+.0f})")
    else:
        # No position — suggest initial entry
        if best_fwd:
            fwd_sides = chain.get(best_fwd['strike'], {})
            fc = fwd_sides.get('call', {})
            fp = fwd_sides.get('put', {})
            per_gamma = (fc.get('gamma', 0) + fp.get('gamma', 0)) * 100
            per_vega = (fc.get('vega', 0) + fp.get('vega', 0)) * 100
            per_theta = (fc.get('theta', 0) + fp.get('theta', 0)) * 100
            strad_bid = fc.get('bid', 0) + fp.get('bid', 0)
            strad_mid = fc.get('mid', 0) + fp.get('mid', 0)
            strad_ask = fc.get('ask', 0) + fp.get('ask', 0)
            start_price = round(strad_bid + 0.01, 2)

            print(f"\n  NO POSITION — Initial entry suggestion:")
            print(f"    Strike: ${best_fwd['strike']:.0f} (forward ATM)")
            print(f"    1x straddle = ${strad_mid * 100:,.0f} at mid")
            print(f"      Gamma: {per_gamma:+.1f}/strad  Vega: {per_vega:+.1f}/strad  Theta: {per_theta:+.1f}/day/strad")
            print(f"    Pricing approach:")
            print(f"      Start:   ${start_price:.2f} (bid + $0.01)")
            print(f"      Mid:     ${strad_mid:.2f}")
            print(f"      Natural: ${strad_ask:.2f} (guaranteed fill)")
            for n in [1, 5, 10]:
                print(f"    {n}x = ${strad_mid * n * 100:,.0f} | "
                      f"G {per_gamma * n:+.0f}  V {per_vega * n:+.0f}  Th {per_theta * n:+.0f}/day")
            print(f"\n    Command (start near bid):")
            print(f"    python ws_trading.py straddle {symbol} {best_exp} "
                  f"{best_fwd['strike']:.0f} {start_price:.2f}")

    print(f"{'=' * 80}")


def cmd_straddle(args):
    """Place a multi-leg straddle order (buy put + buy call at same strike)"""
    session = get_session()

    symbol = args.symbol.upper()
    security_id = KNOWN_SECURITIES.get(symbol)
    if not security_id:
        print(f"Unknown symbol: {symbol}")
        print(f"Known symbols: {', '.join(KNOWN_SECURITIES.keys())}")
        return

    expiry_date = args.expiry
    strike = args.strike
    limit_price = args.price  # net debit per contract
    qty = args.qty
    open_close = "CLOSE" if args.close else "OPEN"
    dry_run = args.dry_run

    # Fetch both put and call chains to find security IDs
    print(f"Looking up {symbol} {expiry_date} ${strike} straddle...")

    put_sec_id = None
    call_sec_id = None
    put_quote = {}
    call_quote = {}

    for opt_type in ('PUT', 'CALL'):
        data = graphql_query(session, "FetchOptionChain", QUERY_OPTION_CHAIN, {
            "id": security_id,
            "expiryDate": expiry_date,
            "optionType": opt_type,
            "realTimeQuote": True,
            "includeGreeks": False
        })

        if not data:
            print(f"Failed to fetch {opt_type} chain")
            return

        edges = data.get('security', {}).get('optionChain', {}).get('edges', [])
        for edge in edges:
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            node_strike = float(details.get('strikePrice', 0))
            if abs(node_strike - strike) < 0.01:
                quote = node.get('quoteV2', {})
                if opt_type == 'PUT':
                    put_sec_id = node.get('id')
                    put_quote = quote
                else:
                    call_sec_id = node.get('id')
                    call_quote = quote
                break

    if not put_sec_id or not call_sec_id:
        print(f"Could not find both put and call at strike ${strike}")
        if not put_sec_id:
            print("  Missing: PUT")
        if not call_sec_id:
            print("  Missing: CALL")
        return

    put_bid = float(put_quote.get('bid', 0))
    put_ask = float(put_quote.get('ask', 0))
    put_mid = (put_bid + put_ask) / 2
    call_bid = float(call_quote.get('bid', 0))
    call_ask = float(call_quote.get('ask', 0))
    call_mid = (call_bid + call_ask) / 2

    natural_debit = put_ask + call_ask
    mid_debit = put_mid + call_mid

    # If closing, limit price should be negative (credit)
    effective_price = -limit_price if args.close else limit_price

    print("=" * 60)
    print(f"{'CLOSE' if args.close else 'OPEN'} STRADDLE ORDER")
    print("=" * 60)
    print(f"  Symbol:      {symbol}")
    print(f"  Expiry:      {expiry_date}")
    print(f"  Strike:      ${strike:.2f}")
    print(f"  Contracts:   {qty}")
    print(f"  Put:         {put_sec_id}")
    print(f"    Bid/Ask:   ${put_bid:.2f} / ${put_ask:.2f} (mid: ${put_mid:.2f})")
    print(f"  Call:        {call_sec_id}")
    print(f"    Bid/Ask:   ${call_bid:.2f} / ${call_ask:.2f} (mid: ${call_mid:.2f})")
    print(f"  ---")
    print(f"  Mid debit:   ${mid_debit:.2f}")
    print(f"  Natural:     ${natural_debit:.2f}")
    print(f"  Your limit:  ${limit_price:.2f} {'(credit)' if args.close else '(debit)'}")
    print(f"  Total cost:  ${limit_price * qty * 100:,.2f}")
    print(f"  Time:        DAY order")
    print("=" * 60)

    if dry_run:
        print("\n[DRY RUN] Order not placed.")
        return

    if args.close:
        order_type_put = "SELL_QUANTITY"
        order_type_call = "SELL_QUANTITY"
    else:
        order_type_put = "BUY_QUANTITY"
        order_type_call = "BUY_QUANTITY"

    legs = [
        {"securityId": put_sec_id, "orderType": order_type_put, "openClose": open_close},
        {"securityId": call_sec_id, "orderType": order_type_call, "openClose": open_close},
    ]

    confirm = input("\nConfirm order? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("Order cancelled.")
        return

    print("\nPlacing multi-leg order...")
    result = place_multileg_order(session, legs, effective_price, quantity_multiplier=qty)

    order_result = result.get('result', {})
    exec_data = order_result.get('soOrdersCreateOrderExecution', {})
    errors = exec_data.get('errors', [])

    if errors:
        print("Order FAILED:")
        for err in errors:
            print(f"  - {err.get('code')}: {err.get('message')}")
    else:
        orders = exec_data.get('orders', [])
        print(f"Straddle order SUBMITTED!")
        print(f"  External ID: {result.get('order_id')}")
        for o in orders:
            print(f"  Leg: {o.get('orderId')} (created: {o.get('createdAt')})")


def cmd_multileg_status(args):
    """Show details of a multi-leg order"""
    session = get_session()
    batch_id = args.batch_id

    print(f"Fetching multi-leg order: {batch_id}")
    data = fetch_multileg_order(session, batch_id)

    if not data:
        print("No data returned")
        return

    order = data.get('soOrdersMultilegOrder')
    if not order:
        print("Order not found")
        return

    print("=" * 60)
    print("MULTI-LEG ORDER DETAILS")
    print("=" * 60)
    print(f"  Batch ID:    {order.get('orderBatchId')}")
    print(f"  Strategy:    {order.get('optionStrategy')}")
    print(f"  Status:      {order.get('status')}")
    print(f"  Limit Price: {order.get('limitPrice')}")
    print(f"  Time Force:  {order.get('timeInForce')}")
    print(f"  Total Fee:   {order.get('totalFee')}")
    print(f"  Submitted:   {order.get('submittedAtUtc')}")
    print(f"  Updated:     {order.get('updatedAtUtc')}")

    legs = order.get('legs', [])
    if legs:
        print(f"\n  LEGS ({len(legs)}):")
        for i, leg in enumerate(legs):
            print(f"    Leg {i+1}:")
            print(f"      Symbol:     {leg.get('symbol')}")
            print(f"      Security:   {leg.get('securityId')}")
            print(f"      Side:       {leg.get('side')}")
            print(f"      Open/Close: {leg.get('openClose')}")
            print(f"      Status:     {leg.get('status')}")
            print(f"      Qty:        {leg.get('submittedQuantity')}")
            fill = leg.get('averageFillPrice', {})
            if fill:
                print(f"      Fill Price: ${float(fill.get('amount', 0)):.2f}")
            print(f"      Fill Qty:   {leg.get('filledQuantity')}")


def main():
    parser = argparse.ArgumentParser(description='Wealthsimple Trade CLI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # status
    subparsers.add_parser('status', help='Show account status and balances')

    # positions
    subparsers.add_parser('positions', help='Show current positions')

    # orders
    subparsers.add_parser('orders', help='Show recent orders/activities')

    # open-orders
    subparsers.add_parser('open-orders', help='Show open orders with IDs for cancellation')

    # quote
    p = subparsers.add_parser('quote', help='Get quote for a symbol')
    p.add_argument('symbol', help='Stock symbol')

    # buy
    p = subparsers.add_parser('buy', help='Place a limit buy order')
    p.add_argument('symbol', help='Stock symbol (e.g., UNG)')
    p.add_argument('quantity', type=int, help='Number of shares')
    p.add_argument('price', type=float, help='Limit price')

    # sell
    p = subparsers.add_parser('sell', help='Place a limit sell order')
    p.add_argument('symbol', help='Stock symbol (e.g., UNG)')
    p.add_argument('quantity', type=int, help='Number of shares')
    p.add_argument('price', type=float, help='Limit price')

    # cancel
    p = subparsers.add_parser('cancel', help='Cancel an open order')
    p.add_argument('order_id', help='Order ID (e.g., order-xxxx-xxxx-xxxx)')

    # modify
    p = subparsers.add_parser('modify', help='Modify an order limit price')
    p.add_argument('order_id', help='Order ID (e.g., order-xxxx-xxxx-xxxx)')
    p.add_argument('price', type=float, help='New limit price')

    # refresh
    subparsers.add_parser('refresh', help='Instructions to refresh cookies')

    # opt-expiry
    p = subparsers.add_parser('opt-expiry', help='List option expiration dates')
    p.add_argument('symbol', help='Underlying stock symbol (e.g., UNG)')

    # opt-chain
    p = subparsers.add_parser('opt-chain', help='Show option chain')
    p.add_argument('symbol', help='Underlying stock symbol (e.g., UNG)')
    p.add_argument('expiry', help='Expiration date (YYYY-MM-DD)')
    p.add_argument('type', help='Option type (CALL or PUT)')

    # buy-opt
    p = subparsers.add_parser('buy-opt', help='Place a limit buy order for an option')
    p.add_argument('security_id', help='Option security ID (sec-o-xxx)')
    p.add_argument('quantity', type=int, help='Number of contracts')
    p.add_argument('price', type=float, help='Limit price per share')
    p.add_argument('--open-close', dest='open_close', default='OPEN', help='OPEN or CLOSE (default: OPEN)')

    # sell-opt
    p = subparsers.add_parser('sell-opt', help='Place a limit sell order for an option')
    p.add_argument('security_id', help='Option security ID (sec-o-xxx)')
    p.add_argument('quantity', type=int, help='Number of contracts')
    p.add_argument('price', type=float, help='Limit price per share')
    p.add_argument('--open-close', dest='open_close', default='CLOSE', help='OPEN or CLOSE (default: CLOSE)')

    # search
    p = subparsers.add_parser('search', help='Search for a security and show its ID')
    p.add_argument('query', help='Search query (symbol or company name)')

    # straddle-scan
    p = subparsers.add_parser('straddle-scan', help='Scan straddle pricing near forward ATM')
    p.add_argument('symbol', help='Underlying stock symbol (e.g., SPY)')
    p.add_argument('--dte', type=int, default=25, help='Target DTE (default: 25)')
    p.add_argument('--expiry', default=None, help='Exact expiry date YYYY-MM-DD (overrides --dte)')

    # straddle
    p = subparsers.add_parser('straddle', help='Place a multi-leg straddle order')
    p.add_argument('symbol', help='Underlying stock symbol (e.g., SPY)')
    p.add_argument('expiry', help='Expiration date (YYYY-MM-DD)')
    p.add_argument('strike', type=float, help='Strike price')
    p.add_argument('price', type=float, help='Net debit limit per contract')
    p.add_argument('--qty', type=int, default=1, help='Number of contracts (default: 1)')
    p.add_argument('--close', action='store_true', help='Close existing straddle (sell both legs)')
    p.add_argument('--dry-run', action='store_true', help='Show details without placing order')

    # multileg-status
    p = subparsers.add_parser('multileg-status', help='Show multi-leg order details')
    p.add_argument('batch_id', help='Order batch ID (e.g., order-batch-00YGqBb0frTx)')

    # order-status (NEW — uses FetchSoOrdersExtendedOrder)
    p = subparsers.add_parser('order-status', help='Show canonical single-order status by external ID')
    p.add_argument('order_id', help='External order ID (e.g., order-xxxx-xxxx)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'status': cmd_status,
        'positions': cmd_positions,
        'orders': cmd_orders,
        'open-orders': cmd_open_orders,
        'quote': cmd_quote,
        'buy': cmd_buy,
        'sell': cmd_sell,
        'cancel': cmd_cancel,
        'modify': cmd_modify,
        'refresh': cmd_refresh,
        'opt-expiry': cmd_opt_expiry,
        'opt-chain': cmd_opt_chain,
        'buy-opt': cmd_buy_opt,
        'sell-opt': cmd_sell_opt,
        'search': cmd_search,
        'straddle-scan': cmd_straddle_scan,
        'straddle': cmd_straddle,
        'multileg-status': cmd_multileg_status,
        'order-status': cmd_order_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
