#!/usr/bin/env python3
"""
UNG Smart Grid Bot — Cost Basis Reducer

Places buy orders below current price, sell orders above avg cost.
Scaled buying: buys more shares the further price dips below avg cost.
Never sells below avg cost — only reduces basis on profitable fills.

Usage:
    python ws_ung_grid.py              # Live trading
    python ws_ung_grid.py --dry-run    # Simulate without placing orders
    python ws_ung_grid.py --cancel     # Cancel all open UNG orders and exit
    python ws_ung_grid.py --status     # Show position + active grid, exit
"""

import argparse
import time
import math
import random
from datetime import datetime, timedelta

from ws_trading import (
    get_session, graphql_query, place_order, cancel_order,
    load_config, load_cookies, extract_identity_from_cookies,
    extract_oauth_data, is_token_expired, refresh_access_token,
    update_cookies_with_new_token,
    QUERY_FETCH_POSITIONS, QUERY_FETCH_ACTIVITIES,
    KNOWN_SECURITIES, DEFAULT_ACCOUNT_ID,
)

# ── Config ─────────────────────────────────────────────────────────────────────

UNG_SEC_ID       = KNOWN_SECURITIES['UNG']
GRID_SPACING     = 0.10   # $ between grid levels
QTY_BASE         = 100    # base shares per grid level
MAX_SHARES       = 5000   # hard cap — never buy above this total
BUY_LEVELS       = 5      # how many active buy orders to maintain
SELL_LEVELS      = 5      # how many active sell orders to maintain
POLL_INTERVAL    = 30     # seconds between fill-detection polls
RECENTER_MINS    = 30     # recenter grid around current price every N mins
MIN_SELL_PROFIT  = 0.02   # min profit per share above avg cost to place sell

# Scaled buy qty: buy more aggressively the further below avg cost
# (pct_below_avg_cost → qty_multiplier)
SCALE_TABLE = [
    (5.0, 2.5),
    (3.0, 2.0),
    (1.0, 1.5),
    (0.0, 1.0),
]


def ts():
    return datetime.now().strftime('%H:%M:%S')


def log(msg):
    print(f'[{ts()}] {msg}')


# ── WS helpers ─────────────────────────────────────────────────────────────────

def ensure_session(session_holder):
    """Refresh token if needed, return valid session."""
    cookies = load_cookies()
    oauth_data = extract_oauth_data(cookies)
    if is_token_expired(oauth_data):
        log('Token expired — refreshing...')
        device_id = cookies.get('wssdi', 'cli-device-001')
        new_token = refresh_access_token(oauth_data, device_id)
        if new_token:
            update_cookies_with_new_token(new_token)
            log('Token refreshed.')
        else:
            log('ERROR: token refresh failed!')
            return None
    return get_session()


def get_position(session):
    """Return (shares, avg_cost_usd) for UNG from WS."""
    config = load_config()
    cookies = load_cookies()
    identity_id = config.get('identity_id') or extract_identity_from_cookies(cookies)
    data = graphql_query(session, 'FetchIdentityPositions', QUERY_FETCH_POSITIONS, {
        'identityId': identity_id, 'currency': 'CAD', 'first': 50,
        'aggregated': True, 'currencyOverride': 'MARKET',
        'sort': 'TODAY_GAIN', 'includeSecurity': True,
        'includeAccountData': True, 'includeOneDayReturnsBaseline': True,
    })
    edges = (data or {}).get('identity', {}).get('financials', {}) \
                        .get('current', {}).get('positions', {}).get('edges', [])
    for edge in edges:
        node = edge.get('node', {})
        sec = node.get('security', {}) or {}
        if sec.get('id') == UNG_SEC_ID:
            qty = int(float(node.get('quantity', 0)))
            avg = node.get('marketAveragePrice', node.get('averagePrice', {}))
            avg_cost = float(avg.get('amount', 0)) if avg else 0
            return qty, avg_cost
    return 0, 0.0


def get_quote(session):
    """Return current UNG bid/ask from WS."""
    QUERY = """
    query FetchUNGQuote($id: ID!) {
      security(id: $id) {
        quoteV2 { bid ask price quotedAsOf }
      }
    }"""
    data = graphql_query(session, 'FetchUNGQuote', QUERY, {'id': UNG_SEC_ID})
    q = (data or {}).get('security', {}).get('quoteV2', {})
    bid = float(q.get('bid', 0) or 0)
    ask = float(q.get('ask', 0) or 0)
    price = float(q.get('price', 0) or 0)
    mid = (bid + ask) / 2 if bid and ask else price
    return bid, ask, mid


def get_open_order_ids(session):
    """Return set of our order external IDs that are still pending."""
    data = graphql_query(session, 'FetchActivityFeedItems', QUERY_FETCH_ACTIVITIES, {
        'first': 100, 'orderBy': 'OCCURRED_AT_DESC',
    })
    pending = set()
    for edge in (data or {}).get('activityFeedItems', {}).get('edges', []):
        act = edge.get('node', {})
        status = (act.get('unifiedStatus') or act.get('status') or '').upper()
        symbol = act.get('assetSymbol', '')
        if symbol == 'UNG' and status in ('PENDING', 'SUBMITTED'):
            oid = act.get('externalCanonicalId') or act.get('canonicalId') or ''
            if oid:
                pending.add(oid)
    return pending


def cancel_all_ung_orders(session):
    """Cancel all open UNG orders. Returns count cancelled."""
    pending_ids = get_open_order_ids(session)
    cancelled = 0
    for oid in pending_ids:
        result = cancel_order(session, oid)
        errs = (result or {}).get('orderServiceCancelOrder', {}).get('errors', [])
        if not errs:
            cancelled += 1
            log(f'  Cancelled {oid}')
        time.sleep(0.3)
    return cancelled


# ── Grid logic ─────────────────────────────────────────────────────────────────

def scaled_qty(price, avg_cost, current_shares):
    """Buy qty scaled by distance below avg cost, capped by remaining capacity."""
    remaining = MAX_SHARES - current_shares
    if remaining <= 0:
        return 0
    pct_below = max(0.0, (avg_cost - price) / avg_cost * 100) if avg_cost > 0 else 0.0
    mult = 1.0
    for threshold, m in SCALE_TABLE:
        if pct_below >= threshold:
            mult = m
            break
    qty = int(QTY_BASE * mult)
    return min(qty, remaining)


def compute_grid(current_price, avg_cost, current_shares):
    """
    Return (buy_levels, sell_levels) as sorted lists of prices.
    buy_levels:  prices below current_price (descending = closest first)
    sell_levels: prices above avg_cost + MIN_SELL_PROFIT (ascending)
    """
    buy_prices = []
    for i in range(1, BUY_LEVELS + 1):
        p = round(current_price - i * GRID_SPACING, 2)
        if p > 0 and (MAX_SHARES - current_shares) > 0:
            buy_prices.append(p)

    sell_prices = []
    sell_floor = round(avg_cost + MIN_SELL_PROFIT + GRID_SPACING, 2) if avg_cost > 0 else round(current_price + GRID_SPACING, 2)
    for i in range(SELL_LEVELS):
        p = round(sell_floor + i * GRID_SPACING, 2)
        sell_prices.append(p)

    return buy_prices, sell_prices


# ── Main bot ───────────────────────────────────────────────────────────────────

class UNGGridBot:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.session = None

        # Position state (synced from WS)
        self.shares = 0
        self.avg_cost = 0.0

        # Active orders: order_id -> {side, price, qty}
        self.active = {}

        self.last_recenter = datetime.min
        self.total_buys = 0
        self.total_sells = 0
        self.realized_pnl = 0.0

    def refresh_session(self):
        self.session = ensure_session(self.session)

    def sync_position(self):
        shares, avg_cost = get_position(self.session)
        if shares != self.shares or abs(avg_cost - self.avg_cost) > 0.001:
            log(f'Position sync: {shares} shares @ ${avg_cost:.3f} avg  (was {self.shares} @ ${self.avg_cost:.3f})')
        self.shares = shares
        self.avg_cost = avg_cost

    def committed_shares(self):
        """Shares held + qty in all pending buy orders (worst-case if all fill)."""
        pending_buys = sum(o['qty'] for o in self.active.values() if o['side'] == 'BUY')
        return self.shares + pending_buys

    def place(self, side, price, qty):
        """Place a limit order. Returns order_id or None."""
        prefix = '[DRY] ' if self.dry_run else ''
        log(f'{prefix}{side} {qty} UNG @ ${price:.2f}')
        if self.dry_run:
            fake_id = f'dry-{side[:1]}-{price:.2f}-{int(time.time())}'
            self.active[fake_id] = {'side': side, 'price': price, 'qty': qty}
            return fake_id
        order_type = 'BUY_QUANTITY' if side == 'BUY' else 'SELL_QUANTITY'
        result = place_order(self.session, order_type, UNG_SEC_ID, qty, price)
        errs = result.get('result', {}).get('soOrdersCreateOrder', {}).get('errors', [])
        if errs:
            log(f'  ORDER ERROR: {errs[0].get("message")}')
            return None
        oid = result.get('order_id')
        self.active[oid] = {'side': side, 'price': price, 'qty': qty}
        time.sleep(0.3)
        return oid

    def cancel_order(self, oid):
        if self.dry_run:
            self.active.pop(oid, None)
            return
        result = cancel_order(self.session, oid)
        errs = (result or {}).get('orderServiceCancelOrder', {}).get('errors', [])
        if not errs:
            self.active.pop(oid, None)
        time.sleep(0.2)

    def detect_fills(self):
        """
        Detect fills using position delta as ground truth.

        1. Prune orders from self.active that are no longer WS-pending.
        2. Reconcile with actual position change: only treat pruned orders as
           fills if the position actually moved in that direction.
        Returns list of filled order dicts.
        """
        if self.dry_run:
            return []

        pending_ids = get_open_order_ids(self.session)
        prev_shares = self.shares

        # Prune orders no longer in WS pending (fill OR cancel)
        disappeared = {}
        for oid, info in list(self.active.items()):
            if oid not in pending_ids and not oid.startswith('dry-'):
                disappeared[oid] = info
                del self.active[oid]

        if not disappeared:
            return []

        # Get actual position from WS to cross-check
        ws_shares, _ = get_position(self.session)
        delta = ws_shares - prev_shares   # >0 = net buy, <0 = net sell

        if delta == 0:
            # Position unchanged → disappeared orders were likely cancelled, not filled
            if disappeared:
                log(f'  {len(disappeared)} orders disappeared but position unchanged — treating as cancelled')
            return []

        # Position changed — infer fills from disappeared orders that match the delta direction
        buy_gone  = [o for o in disappeared.values() if o['side'] == 'BUY']
        sell_gone = [o for o in disappeared.values() if o['side'] == 'SELL']

        filled = []
        if delta > 0 and buy_gone:
            # Net position increased — buys filled
            for info in buy_gone:
                filled.append(info)
        if delta < 0 and sell_gone:
            # Net position decreased — sells filled
            for info in sell_gone:
                filled.append(info)

        if len(filled) < len(disappeared):
            not_filled = len(disappeared) - len(filled)
            log(f'  {not_filled} orders treated as cancelled (position delta={delta:+d})')

        return filled

    def handle_fill(self, fill):
        """Process a fill: update P&L, place counter-order."""
        side  = fill['side']
        price = fill['price']
        qty   = fill['qty']

        if side == 'BUY':
            self.total_buys += qty
            # Update avg cost
            total_cost = self.shares * self.avg_cost + qty * price
            self.shares += qty
            self.avg_cost = total_cost / self.shares if self.shares else 0
            log(f'  FILL BUY  {qty}@${price:.2f} → avg_cost=${self.avg_cost:.3f} ({self.shares} shares)')

            # Place counter sell at fill + spacing, only if above avg_cost
            sell_price = round(price + GRID_SPACING, 2)
            if sell_price > self.avg_cost + MIN_SELL_PROFIT:
                self.place('SELL', sell_price, qty)
            else:
                # Sell at avg_cost + min profit instead
                sell_price = round(self.avg_cost + MIN_SELL_PROFIT + 0.01, 2)
                log(f'  Adjusting sell to ${sell_price:.2f} (above avg cost)')
                self.place('SELL', sell_price, qty)

        else:  # SELL
            self.total_sells += qty
            profit = (price - self.avg_cost) * qty
            self.realized_pnl += profit
            self.shares -= qty
            if self.shares <= 0:
                self.shares = 0
                self.avg_cost = 0
            log(f'  FILL SELL {qty}@${price:.2f} → profit=${profit:.2f} | realized=${self.realized_pnl:.2f}')

            # Place counter buy at fill - spacing, only if under cap
            buy_price = round(price - GRID_SPACING, 2)
            qty_next = scaled_qty(buy_price, self.avg_cost, self.committed_shares())
            if qty_next > 0:
                self.place('BUY', buy_price, qty_next)
            else:
                log(f'  Counter-buy skipped — committed {self.committed_shares()} >= cap {MAX_SHARES}')

    def place_trim_sells(self, current_price):
        """
        When holding more than MAX_SHARES, place a tight ladder of sells
        on the excess shares starting at current_price+$0.01.
        These deliberately ignore the avg_cost floor — we're trimming excess
        the bot never should have bought.
        """
        excess = self.shares - MAX_SHARES
        if excess <= 0:
            return
        n_orders = max(1, excess // QTY_BASE)
        log(f'  TRIM: {excess} excess shares → placing {n_orders} trim sells starting ${current_price + 0.01:.2f}')
        for i in range(n_orders):
            p = round(current_price + 0.01 + i * 0.02, 2)
            self.place('SELL', p, QTY_BASE)

    def place_initial_grid(self, current_price):
        """Place initial buy + sell orders around current price."""
        log(f'Placing initial grid | price=${current_price:.2f} avg_cost=${self.avg_cost:.3f}')
        committed = self.committed_shares()
        buy_prices, sell_prices = compute_grid(current_price, self.avg_cost, committed)

        # If over cap: trim sells take priority, no normal grid sells/buys
        if self.shares > MAX_SHARES:
            self.place_trim_sells(current_price)
        else:
            for p in buy_prices:
                qty = scaled_qty(p, self.avg_cost, self.committed_shares())
                if qty > 0:
                    self.place('BUY', p, qty)

            for p in sell_prices:
                if self.shares > 0:
                    self.place('SELL', p, QTY_BASE)

        self.last_recenter = datetime.now()

    def recenter(self, current_price):
        """Cancel stale orders far from price, fill in missing levels."""
        log(f'Recentering grid @ ${current_price:.2f}')

        # Cancel orders > (BUY_LEVELS+1) grid levels away on buy side,
        # or sell orders now below avg_cost floor
        stale = []
        for oid, info in list(self.active.items()):
            if info['side'] == 'BUY' and info['price'] < current_price - (BUY_LEVELS + 1) * GRID_SPACING:
                stale.append(oid)
            elif info['side'] == 'SELL' and info['price'] < self.avg_cost + MIN_SELL_PROFIT:
                stale.append(oid)

        for oid in stale:
            log(f'  Cancelling stale {self.active[oid]["side"]} @ ${self.active[oid]["price"]:.2f}')
            self.cancel_order(oid)

        # Count active buys/sells
        active_buys  = [o for o in self.active.values() if o['side'] == 'BUY']
        active_sells = [o for o in self.active.values() if o['side'] == 'SELL']
        active_buy_prices  = {o['price'] for o in active_buys}
        active_sell_prices = {o['price'] for o in active_sells}

        committed = self.committed_shares()
        buy_prices, sell_prices = compute_grid(current_price, self.avg_cost, committed)

        # Fill missing buy levels (only if under cap)
        for p in buy_prices:
            if p not in active_buy_prices:
                qty = scaled_qty(p, self.avg_cost, self.committed_shares())
                if qty > 0:
                    self.place('BUY', p, qty)

        # Fill missing sell levels
        for p in sell_prices:
            if p not in active_sell_prices and self.shares > 0:
                self.place('SELL', p, QTY_BASE)

        self.last_recenter = datetime.now()

    def print_status(self, current_price):
        buys  = sorted([o for o in self.active.values() if o['side'] == 'BUY'],  key=lambda x: -x['price'])
        sells = sorted([o for o in self.active.values() if o['side'] == 'SELL'], key=lambda x:  x['price'])
        unreal = (current_price - self.avg_cost) * self.shares
        committed = self.committed_shares()
        print(f'\n  ── Grid Status @ {ts()} ──────────────────────────')
        print(f'  Position: {self.shares} shares @ ${self.avg_cost:.3f} avg  |  price ${current_price:.2f}')
        print(f'  Unrealized: ${unreal:.2f}  |  Realized P&L: ${self.realized_pnl:.2f}  |  Net: ${unreal + self.realized_pnl:.2f}')
        print(f'  Committed: {committed} shares ({self.shares} held + {committed - self.shares} in buy orders)  cap={MAX_SHARES}')
        sell_str = '  '.join(f'${o["price"]:.2f}×{o["qty"]}' for o in sells)
        buy_str  = '  '.join(f'${o["price"]:.2f}×{o["qty"]}' for o in buys)
        print(f'  SELL orders: {sell_str or "(none)"}')
        print(f'  ──── ${current_price:.2f} ────')
        print(f'  BUY  orders: {buy_str or "(none)"}')
        print()

    def run(self):
        log('=' * 60)
        log(f'UNG SMART GRID BOT  {"[DRY RUN]" if self.dry_run else "[LIVE]"}')
        log(f'Grid: ${GRID_SPACING:.2f} spacing | {BUY_LEVELS} buy + {SELL_LEVELS} sell levels')
        log(f'Base qty: {QTY_BASE} shares | Max position: {MAX_SHARES}')
        log('=' * 60)

        self.refresh_session()
        self.sync_position()

        _, _, mid = get_quote(self.session)
        log(f'UNG mid: ${mid:.3f}')

        # Cancel any existing UNG orders before starting fresh
        log('Cancelling existing UNG orders...')
        cancelled = cancel_all_ung_orders(self.session)
        log(f'Cancelled {cancelled} orders.')

        self.place_initial_grid(mid)
        self.print_status(mid)

        while True:
            try:
                time.sleep(POLL_INTERVAL + random.uniform(-3, 3))

                # Token refresh
                self.refresh_session()

                # Check fills
                fills = self.detect_fills()
                for fill in fills:
                    self.handle_fill(fill)

                # Always sync position from WS every cycle to prevent drift
                self.sync_position()

                # Get fresh price
                _, _, mid = get_quote(self.session)

                # If over cap and no active sell orders → re-place trim sells immediately
                active_sells = [o for o in self.active.values() if o['side'] == 'SELL']
                if self.shares > MAX_SHARES and not active_sells:
                    log(f'Over cap ({self.shares}/{MAX_SHARES}) with no sells — re-placing trim sells')
                    self.place_trim_sells(mid)
                    self.last_recenter = datetime.now()

                # Recenter periodically (normal grid maintenance)
                elif (datetime.now() - self.last_recenter).total_seconds() / 60 >= RECENTER_MINS:
                    self.recenter(mid)

                self.print_status(mid)

            except KeyboardInterrupt:
                log('Stopping...')
                break
            except Exception as e:
                log(f'ERROR: {e}')
                time.sleep(10)

        log(f'Final: {self.shares} shares @ ${self.avg_cost:.3f} avg | realized P&L ${self.realized_pnl:.2f}')


def main():
    parser = argparse.ArgumentParser(description='UNG Smart Grid Bot')
    parser.add_argument('--dry-run', action='store_true', help='Simulate only')
    parser.add_argument('--cancel',  action='store_true', help='Cancel all UNG orders and exit')
    parser.add_argument('--status',  action='store_true', help='Show position and exit')
    args = parser.parse_args()

    session = get_session()

    if args.cancel:
        log('Cancelling all UNG orders...')
        n = cancel_all_ung_orders(session)
        log(f'Done — cancelled {n} orders.')
        return

    if args.status:
        shares, avg_cost = get_position(session)
        _, _, mid = get_quote(session)
        unreal = (mid - avg_cost) * shares if avg_cost else 0
        print(f'UNG: {shares} shares @ ${avg_cost:.3f} avg | price ${mid:.3f} | P&L ${unreal:.2f}')
        return

    bot = UNGGridBot(dry_run=args.dry_run)
    bot.run()


if __name__ == '__main__':
    main()
