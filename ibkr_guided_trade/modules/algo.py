"""
Module 3: Algorithmic Trading - Oscillation Strategy

Buy low / sell high with parallel orders and SQLite persistence.
"""
# pyright: reportOptionalMemberAccess=false
# pyright: reportOperatorIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false

import sqlite3
import json
from datetime import datetime
from ib_insync import Stock, LimitOrder

from .common import connect, get_timestamp, DEFAULT_ACCOUNT, DB_PATH


def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            strategy_type TEXT DEFAULT 'oscillation',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_id INTEGER NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            limit_price REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            ibkr_order_id INTEGER,
            ibkr_perm_id INTEGER,
            filled_qty INTEGER DEFAULT 0,
            filled_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (strategy_id) REFERENCES strategies(id)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            filled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (order_id) REFERENCES orders(id)
        )
    ''')

    conn.commit()
    conn.close()


def get_db():
    """Get database connection"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_algo_list(args):
    """List all strategies"""
    conn = get_db()
    c = conn.cursor()

    strategies = c.execute('''
        SELECT s.*,
               (SELECT COUNT(*) FROM orders WHERE strategy_id = s.id AND status = 'submitted') as pending_orders,
               (SELECT COUNT(*) FROM orders WHERE strategy_id = s.id AND status = 'filled') as filled_orders
        FROM strategies s
        ORDER BY s.status = 'active' DESC, s.updated_at DESC
    ''').fetchall()

    print(f"{'='*70}")
    print(f"ALGO STRATEGIES - {get_timestamp()}")
    print(f"{'='*70}")

    if not strategies:
        print("\nNo strategies found. Use 'algo add SYMBOL' to create one.")
        conn.close()
        return

    print(f"\n{'ID':>4} {'Symbol':<8} {'Type':<12} {'Status':<10} {'Pending':>8} {'Filled':>8}")
    print("-" * 70)

    for s in strategies:
        config = json.loads(s['config']) if s['config'] else {}
        print(f"{s['id']:>4} {s['symbol']:<8} {s['strategy_type']:<12} {s['status']:<10} {s['pending_orders']:>8} {s['filled_orders']:>8}")

        # Show config summary
        if config:
            buy_pct = config.get('buy_pct', 0)
            sell_pct = config.get('sell_pct', 0)
            sell_profit = config.get('sell_profit', 0)
            qty = config.get('qty', 0)
            max_shares = config.get('max_shares', 0)
            print(f"     └─ Buy: {buy_pct}% below | Sell: {sell_pct}% above avg (or +${sell_profit}) | Qty: {qty} | Max: {max_shares}")

    conn.close()


def cmd_algo_add(args):
    """Add a new oscillation strategy"""
    symbol = args.symbol.upper()

    # Get current price
    ib = connect()
    stock = Stock(symbol, 'SMART', 'USD')
    ib.qualifyContracts(stock)
    ib.reqMktData(stock)
    ib.sleep(2)

    ticker = ib.ticker(stock)
    spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close

    if not spot or spot <= 0:
        print(f"Could not get price for {symbol}")
        ib.disconnect()
        return

    print(f"{'='*60}")
    print(f"CREATE OSCILLATION STRATEGY - {symbol}")
    print(f"{'='*60}")
    print(f"\nCurrent Price: ${spot:.2f}")

    # Parse arguments or use defaults
    buy_pct = args.buy_pct
    qty = args.qty
    max_shares = args.max_shares
    sell_pct = args.sell_pct
    sell_profit = args.sell_profit

    # Calculate buy price
    buy_price = round(spot * (1 - buy_pct / 100), 2)

    print(f"\n--- BUY CONFIGURATION ---")
    print(f"  Buy {buy_pct}% below current: ${buy_price:.2f}")
    print(f"  Quantity per order: {qty} shares")
    print(f"  Max accumulation: {max_shares} shares")

    print(f"\n--- SELL CONFIGURATION ---")
    if sell_pct > 0:
        print(f"  Sell when {sell_pct}% above average cost")
    if sell_profit > 0:
        print(f"  OR when +${sell_profit:.2f} profit per share")

    # Create strategy config
    config = {
        'buy_pct': buy_pct,
        'qty': qty,
        'max_shares': max_shares,
        'sell_pct': sell_pct,
        'sell_profit': sell_profit,
        'initial_price': spot
    }

    # Save to database
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        INSERT INTO strategies (symbol, strategy_type, config)
        VALUES (?, 'oscillation', ?)
    ''', (symbol, json.dumps(config)))
    strategy_id = c.lastrowid
    conn.commit()
    conn.close()

    print(f"\n✅ Strategy #{strategy_id} created for {symbol}")
    print(f"   Run 'algo run' to submit initial orders.")

    ib.disconnect()


def cmd_algo_show(args):
    """Show details of a specific strategy"""
    conn = get_db()
    c = conn.cursor()

    strategy = c.execute('SELECT * FROM strategies WHERE id = ?', (args.id,)).fetchone()

    if not strategy:
        print(f"Strategy #{args.id} not found")
        conn.close()
        return

    config = json.loads(strategy['config']) if strategy['config'] else {}

    print(f"{'='*60}")
    print(f"STRATEGY #{strategy['id']} - {strategy['symbol']}")
    print(f"{'='*60}")
    print(f"\nType: {strategy['strategy_type']}")
    print(f"Status: {strategy['status']}")
    print(f"Created: {strategy['created_at']}")

    print(f"\n--- CONFIG ---")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Show orders
    orders = c.execute('''
        SELECT * FROM orders WHERE strategy_id = ?
        ORDER BY created_at DESC LIMIT 20
    ''', (args.id,)).fetchall()

    if orders:
        print(f"\n--- ORDERS (last 20) ---")
        print(f"{'ID':>6} {'Side':<6} {'Qty':>6} {'Price':>10} {'Status':<12} {'Filled':>8}")
        print("-" * 55)
        for o in orders:
            filled = f"{o['filled_qty']}/{o['quantity']}" if o['filled_qty'] else "-"
            print(f"{o['id']:>6} {o['side']:<6} {o['quantity']:>6} ${o['limit_price']:>9.2f} {o['status']:<12} {filled:>8}")

    conn.close()


def cmd_algo_pause(args):
    """Pause a strategy"""
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE strategies SET status = "paused", updated_at = ? WHERE id = ?',
              (datetime.now(), args.id))
    if c.rowcount:
        print(f"Strategy #{args.id} paused")
    else:
        print(f"Strategy #{args.id} not found")
    conn.commit()
    conn.close()


def cmd_algo_resume(args):
    """Resume a paused strategy"""
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE strategies SET status = "active", updated_at = ? WHERE id = ?',
              (datetime.now(), args.id))
    if c.rowcount:
        print(f"Strategy #{args.id} resumed")
    else:
        print(f"Strategy #{args.id} not found")
    conn.commit()
    conn.close()


def cmd_algo_cancel(args):
    """Cancel a strategy and all its pending orders"""
    conn = get_db()
    c = conn.cursor()

    strategy = c.execute('SELECT * FROM strategies WHERE id = ?', (args.id,)).fetchone()
    if not strategy:
        print(f"Strategy #{args.id} not found")
        conn.close()
        return

    # Cancel pending orders in IBKR
    pending_orders = c.execute('''
        SELECT * FROM orders WHERE strategy_id = ? AND status = 'submitted'
    ''', (args.id,)).fetchall()

    if pending_orders:
        ib = connect()
        ib.reqAllOpenOrders()
        ib.sleep(2)

        for order in pending_orders:
            if order['ibkr_order_id']:
                # Find and cancel the order
                for trade in ib.openTrades():
                    if trade.order.orderId == order['ibkr_order_id']:
                        ib.cancelOrder(trade.order)
                        print(f"  Cancelled IBKR order {order['ibkr_order_id']}")

        ib.disconnect()

    # Update database
    c.execute('UPDATE orders SET status = "cancelled", updated_at = ? WHERE strategy_id = ? AND status IN ("pending", "submitted")',
              (datetime.now(), args.id))
    c.execute('UPDATE strategies SET status = "cancelled", updated_at = ? WHERE id = ?',
              (datetime.now(), args.id))

    conn.commit()
    conn.close()
    print(f"Strategy #{args.id} cancelled")


def cmd_algo_sync(args):
    """Sync order statuses with IBKR"""
    conn = get_db()
    c = conn.cursor()

    # Get all submitted orders
    submitted = c.execute('''
        SELECT o.*, s.symbol FROM orders o
        JOIN strategies s ON o.strategy_id = s.id
        WHERE o.status = 'submitted'
    ''').fetchall()

    if not submitted:
        print("No submitted orders to sync")
        conn.close()
        return

    print(f"Syncing {len(submitted)} orders with IBKR...")

    ib = connect()
    ib.reqAllOpenOrders()
    ib.sleep(2)

    # Get all open trades
    open_trades = {t.order.orderId: t for t in ib.openTrades()}

    # Get executions/fills
    executions = ib.executions()

    for order in submitted:
        order_id = order['ibkr_order_id']

        if order_id in open_trades:
            # Order still open
            trade = open_trades[order_id]
            filled = trade.orderStatus.filled
            if filled > order['filled_qty']:
                c.execute('''
                    UPDATE orders SET filled_qty = ?, updated_at = ? WHERE id = ?
                ''', (filled, datetime.now(), order['id']))
                print(f"  Order {order['id']}: filled {filled}/{order['quantity']}")
        else:
            # Order not in open trades - check if filled or cancelled
            # Check fills by looking at positions
            c.execute('''
                UPDATE orders SET status = 'filled', filled_qty = quantity, updated_at = ?
                WHERE id = ? AND filled_qty < quantity
            ''', (datetime.now(), order['id']))
            if c.rowcount:
                print(f"  Order {order['id']}: marked as filled")

    conn.commit()
    conn.close()
    ib.disconnect()
    print("Sync complete")


def cmd_algo_run(args):
    """Execute pending orders for all active strategies"""
    conn = get_db()
    c = conn.cursor()

    # Get active strategies
    strategies = c.execute('''
        SELECT * FROM strategies WHERE status = 'active'
    ''').fetchall()

    if not strategies:
        print("No active strategies")
        conn.close()
        return

    print(f"{'='*60}")
    print(f"ALGO RUN - {get_timestamp()}")
    print(f"{'='*60}")

    ib = connect()
    ib.sleep(2)

    # Get current positions
    positions = {p.contract.symbol: p.position for p in ib.positions()
                 if p.contract.secType == 'STK'}

    for strategy in strategies:
        symbol = strategy['symbol']
        config = json.loads(strategy['config']) if strategy['config'] else {}
        strategy_id = strategy['id']

        print(f"\n--- {symbol} (Strategy #{strategy_id}) ---")

        # Get current price
        stock = Stock(symbol, 'SMART', 'USD')
        ib.qualifyContracts(stock)
        ib.reqMktData(stock)
        ib.sleep(2)

        ticker = ib.ticker(stock)
        spot = ticker.last if ticker.last and ticker.last > 0 else ticker.close

        if not spot or spot <= 0:
            print(f"  Could not get price for {symbol}, skipping")
            continue

        print(f"  Current price: ${spot:.2f}")

        # Get current position
        current_pos = positions.get(symbol, 0)
        print(f"  Current position: {current_pos} shares")

        # Get strategy config
        buy_pct = config.get('buy_pct', 2)
        qty = config.get('qty', 10)
        max_shares = config.get('max_shares', 100)
        sell_pct = config.get('sell_pct', 3)
        sell_profit = config.get('sell_profit', 0)

        # Check existing submitted orders
        submitted_buys = c.execute('''
            SELECT SUM(quantity) FROM orders
            WHERE strategy_id = ? AND side = 'BUY' AND status = 'submitted'
        ''', (strategy_id,)).fetchone()[0] or 0

        submitted_sells = c.execute('''
            SELECT SUM(quantity) FROM orders
            WHERE strategy_id = ? AND side = 'SELL' AND status = 'submitted'
        ''', (strategy_id,)).fetchone()[0] or 0

        pending_accumulation = current_pos + submitted_buys
        print(f"  Pending accumulation: {pending_accumulation} (pos: {current_pos}, buy orders: {submitted_buys})")

        # BUY LOGIC: If below max accumulation, place buy order
        if pending_accumulation < max_shares:
            buy_price = round(spot * (1 - buy_pct / 100), 2)

            # Check if we already have a buy order at this price
            existing_buy = c.execute('''
                SELECT * FROM orders
                WHERE strategy_id = ? AND side = 'BUY' AND status = 'submitted'
                AND ABS(limit_price - ?) < 0.01
            ''', (strategy_id, buy_price)).fetchone()

            if not existing_buy:
                # Place new buy order
                order = LimitOrder(
                    action='BUY',
                    totalQuantity=qty,
                    lmtPrice=buy_price,
                    tif='GTC',
                    account=DEFAULT_ACCOUNT
                )

                trade = ib.placeOrder(stock, order)
                ib.sleep(1)

                # Record in database
                c.execute('''
                    INSERT INTO orders (strategy_id, side, quantity, limit_price, status, ibkr_order_id)
                    VALUES (?, 'BUY', ?, ?, 'submitted', ?)
                ''', (strategy_id, qty, buy_price, trade.order.orderId))

                print(f"  ✅ Placed BUY {qty} @ ${buy_price:.2f} (Order ID: {trade.order.orderId})")
            else:
                print(f"  Buy order already exists at ${buy_price:.2f}")
        else:
            print(f"  Max accumulation reached ({max_shares}), no new buy orders")

        # SELL LOGIC: If holding shares, check sell trigger
        if current_pos > 0:
            # Get average cost from positions
            for p in ib.positions():
                if p.contract.symbol == symbol and p.contract.secType == 'STK':
                    avg_cost = p.avgCost
                    break
            else:
                avg_cost = spot  # Fallback

            # Calculate sell trigger price
            sell_trigger = None
            if sell_pct > 0:
                sell_trigger = avg_cost * (1 + sell_pct / 100)
            if sell_profit > 0:
                profit_trigger = avg_cost + sell_profit
                if sell_trigger is None or profit_trigger < sell_trigger:
                    sell_trigger = profit_trigger

            print(f"  Avg cost: ${avg_cost:.2f}, Sell trigger: ${sell_trigger:.2f}")

            if spot >= sell_trigger:
                # Price is above sell trigger - place sell order
                sell_qty = min(qty, current_pos - submitted_sells)
                if sell_qty > 0:
                    order = LimitOrder(
                        action='SELL',
                        totalQuantity=sell_qty,
                        lmtPrice=round(sell_trigger, 2),
                        tif='GTC',
                        account=DEFAULT_ACCOUNT
                    )

                    trade = ib.placeOrder(stock, order)
                    ib.sleep(1)

                    c.execute('''
                        INSERT INTO orders (strategy_id, side, quantity, limit_price, status, ibkr_order_id)
                        VALUES (?, 'SELL', ?, ?, 'submitted', ?)
                    ''', (strategy_id, sell_qty, sell_trigger, trade.order.orderId))

                    print(f"  ✅ Placed SELL {sell_qty} @ ${sell_trigger:.2f} (Order ID: {trade.order.orderId})")
            else:
                print(f"  Price ${spot:.2f} below sell trigger ${sell_trigger:.2f}, waiting...")

        ib.cancelMktData(stock)

    conn.commit()
    conn.close()
    ib.disconnect()

    print(f"\n{'='*60}")
    print("Run complete")
