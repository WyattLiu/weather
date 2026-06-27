#!/usr/bin/env python3
"""
UNG Accumulator Bot for Wealthsimple

Strategy:
- Entry: RSI oversold (< 30) + low volatility (price range < 1% over 10 periods)
- Buy 100 shares at ask, max 1000 total
- Exit: When price rallies (> 3% from avg cost), sell 100 shares per minute
- Auto-refresh OAuth token as needed

Usage:
    python ws_ung_accumulator.py [--dry-run]
"""

import argparse
import time
import random
import requests
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

# Import from ws_trading
from ws_trading import (
    get_session, graphql_query, load_cookies, extract_oauth_data,
    is_token_expired, refresh_access_token, update_cookies_with_new_token,
    place_order, KNOWN_SECURITIES, QUERY_FETCH_POSITIONS
)

# Configuration
UNG_SECURITY_ID = KNOWN_SECURITIES['UNG']
MAX_SHARES = 3500
BUY_QTY = 10
SELL_QTY = 100
MAX_BUY_PRICE = 11.45  # Won't buy above this price

# RSI parameters
RSI_PERIOD = 30
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

# Volatility parameters (for consolidation detection)
VOLATILITY_WINDOW = 10  # Number of price samples
MAX_VOLATILITY_PCT = 1.0  # Max price range % for "consolidation"

# Exit parameters
RALLY_THRESHOLD_PCT = 3.0  # Sell when up 3% from avg cost
SELL_INTERVAL_SEC = 60  # Sell 100 shares per minute when rallying

# Polling interval
POLL_INTERVAL_SEC = 60  # Check price every 60 seconds

# Yahoo Finance settings for historical data
YAHOO_SYMBOL = "UNG"


def fetch_yahoo_history_extended(symbol: str, period: int = 120) -> list:
    """
    Fetch historical 1-minute bars from Yahoo Finance INCLUDING extended hours.
    Uses includePrePost=true to get pre-market and post-market data.

    Args:
        symbol: Stock symbol (e.g., "UNG")
        period: Number of recent bars to return

    Returns:
        List of (timestamp, close_price) tuples, most recent last
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": "1m",
        "range": "1d",
        "includePrePost": "true"  # KEY: This enables extended hours data!
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        result = data.get("chart", {}).get("result", [])
        if not result:
            return []

        chart = result[0]
        timestamps = chart.get("timestamp", [])
        indicators = chart.get("indicators", {})
        quotes = indicators.get("quote", [{}])[0]
        closes = quotes.get("close", [])

        if not timestamps or not closes:
            return []

        # Pair timestamps with closes, filter out None values
        bars = []
        for ts, close in zip(timestamps, closes):
            if close is not None:
                bars.append((ts, close))

        # Return most recent N bars
        return bars[-period:] if len(bars) > period else bars

    except Exception as e:
        print(f"Yahoo Finance fetch error: {e}")
        return []


# Quote query (from HAR)
QUERY_QUOTE = """
query FetchSecurityQuoteV2($id: ID!, $currency: Currency = null) {
  security(id: $id) {
    id
    quoteV2(currency: $currency) {
      ...SecurityQuoteV2
      __typename
    }
    __typename
  }
}

fragment StreamedSecurityQuoteV2 on UnifiedQuote {
  __typename
  securityId
  ask
  bid
  currency
  price
  sessionPrice
  quotedAsOf
  ... on EquityQuote {
    marketStatus
    askSize
    bidSize
    close
    high
    last
    lastSize
    low
    open
    mid
    volume: vol
    referenceClose
    __typename
  }
  ... on OptionQuote {
    marketStatus
    askSize
    bidSize
    close
    high
    last
    lastSize
    low
    open
    mid
    volume: vol
    breakEven
    inTheMoney
    liquidityStatus
    openInterest
    underlyingSpot
    __typename
  }
}

fragment SecurityQuoteV2 on UnifiedQuote {
  ...StreamedSecurityQuoteV2
  previousBaseline
  __typename
}
"""


class UNGAccumulator:
    def __init__(self, dry_run=False, end_time=None):
        self.dry_run = dry_run
        self.end_time = end_time or (datetime.now() + timedelta(hours=12))

        self.session = None
        self.prices = deque(maxlen=max(RSI_PERIOD + 1, VOLATILITY_WINDOW))
        self.gains = deque(maxlen=RSI_PERIOD)
        self.losses = deque(maxlen=RSI_PERIOD)

        self.shares_bought = 0
        self.total_cost = 0.0
        self.avg_cost = 0.0

        self.last_buy_time = None
        self.last_sell_time = None

        self.log_file = Path("ung_accumulator.log")

    def log(self, msg):
        """Log message with timestamp"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def jitter_sleep(self, base_sec=1.0, variance=0.5):
        """Sleep with random jitter to avoid detection"""
        sleep_time = base_sec + random.uniform(-variance, variance)
        time.sleep(max(0.5, sleep_time))

    def ensure_session(self):
        """Ensure we have a valid session, refresh token if needed"""
        cookies = load_cookies()
        oauth_data = extract_oauth_data(cookies)

        if is_token_expired(oauth_data):
            self.log("Token expired, refreshing...")
            self.jitter_sleep(1.0, 0.5)  # Jitter before refresh
            device_id = cookies.get('wssdi', 'cli-device-001')
            new_token = refresh_access_token(oauth_data, device_id)
            if new_token:
                update_cookies_with_new_token(new_token)
                self.log("Token refreshed successfully")
            else:
                self.log("ERROR: Failed to refresh token!")
                return False

        self.session = get_session()
        return True

    def get_quote(self):
        """Get current UNG quote"""
        if not self.ensure_session():
            return None

        self.jitter_sleep(1.0, 0.5)  # Jitter before quote request

        data = graphql_query(self.session, "FetchSecurityQuoteV2", QUERY_QUOTE, {
            "id": UNG_SECURITY_ID,
            "currency": None
        })

        if not data:
            return None

        security = data.get('security', {})
        quote = security.get('quoteV2', {})

        return {
            'bid': float(quote.get('bid', 0)),
            'ask': float(quote.get('ask', 0)),
            'last': float(quote.get('last', 0)),
            'price': float(quote.get('price', 0)),
            'market_status': quote.get('marketStatus', 'UNKNOWN'),
            'quoted_as_of': quote.get('quotedAsOf', ''),
        }

    def get_current_position(self):
        """Get current UNG position from account"""
        if not self.ensure_session():
            return 0, 0.0

        self.jitter_sleep(1.0, 0.5)  # Jitter before position request

        cookies = load_cookies()
        oauth_data = extract_oauth_data(cookies)
        identity_id = oauth_data.get('identity_canonical_id')

        if not identity_id:
            return 0, 0.0

        data = graphql_query(self.session, "FetchIdentityPositions", QUERY_FETCH_POSITIONS, {
            "identityId": identity_id,
            "currency": "CAD",
            "first": 50,
            "aggregated": True,
            "currencyOverride": "MARKET",
            "includeSecurity": True,
            "includeAccountData": False,
            "includeOneDayReturnsBaseline": False
        })

        if not data:
            return 0, 0.0

        positions = data.get('identity', {}).get('financials', {}).get('current', {}).get('positions', {}).get('edges', [])

        for edge in positions:
            pos = edge.get('node', {})
            sec = pos.get('security', {})
            if sec.get('id') == UNG_SECURITY_ID:
                qty = float(pos.get('quantity', 0))
                # Only consider long stock positions, not options
                if qty > 0:
                    # Use marketAveragePrice (USD) instead of averagePrice (CAD)
                    avg_price = pos.get('marketAveragePrice', pos.get('averagePrice', {}))
                    avg = float(avg_price.get('amount', 0))
                    return int(qty), avg

        return 0, 0.0

    def calculate_rsi(self):
        """Calculate RSI from recent price changes"""
        if len(self.prices) < RSI_PERIOD + 1:
            return 50  # Neutral if not enough data

        # Calculate gains and losses
        prices_list = list(self.prices)
        gains = []
        losses = []

        for i in range(1, len(prices_list)):
            change = prices_list[i] - prices_list[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        # Use last RSI_PERIOD values
        recent_gains = gains[-RSI_PERIOD:]
        recent_losses = losses[-RSI_PERIOD:]

        avg_gain = sum(recent_gains) / RSI_PERIOD
        avg_loss = sum(recent_losses) / RSI_PERIOD

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_volatility(self):
        """Calculate price volatility (range as % of mean)"""
        if len(self.prices) < VOLATILITY_WINDOW:
            return float('inf')  # High volatility if not enough data

        recent = list(self.prices)[-VOLATILITY_WINDOW:]
        high = max(recent)
        low = min(recent)
        mean = sum(recent) / len(recent)

        if mean == 0:
            return float('inf')

        volatility = ((high - low) / mean) * 100
        return volatility

    def should_buy(self, quote):
        """Check if we should buy based on RSI and volatility"""
        if self.shares_bought >= MAX_SHARES:
            return False, "Max shares reached"

        # Need enough price data
        if len(self.prices) < RSI_PERIOD + 1:
            return False, f"Collecting data ({len(self.prices)}/{RSI_PERIOD + 1})"

        rsi = self.calculate_rsi()
        volatility = self.calculate_volatility()

        self.log(f"RSI: {rsi:.1f}, Volatility: {volatility:.2f}%")

        # Check price limit
        bid = quote['bid']
        if bid > MAX_BUY_PRICE:
            return False, f"Price ${bid:.2f} > ${MAX_BUY_PRICE} limit"

        if rsi > RSI_OVERSOLD:
            return False, f"RSI {rsi:.1f} > {RSI_OVERSOLD} (not oversold)"

        if volatility > MAX_VOLATILITY_PCT:
            return False, f"Volatility {volatility:.2f}% > {MAX_VOLATILITY_PCT}% (not consolidated)"

        return True, f"RSI {rsi:.1f} oversold + volatility {volatility:.2f}% low"

    def should_sell(self, current_price):
        """Check if we should sell based on RSI overbought AND above water"""
        if self.shares_bought == 0:
            return False, "No shares to sell"

        if self.avg_cost == 0:
            return False, "Unknown avg cost"

        # Must be above water (profitable)
        gain_pct = ((current_price - self.avg_cost) / self.avg_cost) * 100
        if gain_pct <= 0:
            return False, f"Underwater ({gain_pct:.2f}%) - not selling"

        # Need enough price data for RSI
        if len(self.prices) < RSI_PERIOD + 1:
            return False, f"Collecting data for RSI ({len(self.prices)}/{RSI_PERIOD + 1})"

        rsi = self.calculate_rsi()

        # RSI must be overbought
        if rsi < RSI_OVERBOUGHT:
            return False, f"RSI {rsi:.1f} < {RSI_OVERBOUGHT} (not overbought), gain {gain_pct:.2f}%"

        return True, f"RSI {rsi:.1f} overbought + up {gain_pct:.2f}% from ${self.avg_cost:.2f}"

    def execute_buy(self, bid_price):
        """Execute a buy order at bid (limit order for better fill)"""
        self.log(f"{'[DRY RUN] ' if self.dry_run else ''}BUY {BUY_QTY} UNG @ ${bid_price:.2f}")

        if self.dry_run:
            # Simulate buy
            self.shares_bought += BUY_QTY
            self.total_cost += BUY_QTY * bid_price
            self.avg_cost = self.total_cost / self.shares_bought
            self.last_buy_time = datetime.now()
            return True

        try:
            result = place_order(
                self.session,
                "BUY_QUANTITY",
                UNG_SECURITY_ID,
                BUY_QTY,
                bid_price
            )

            order_data = result.get('result', {}).get('soOrdersCreateOrder', {})
            errors = order_data.get('errors', [])

            if errors:
                self.log(f"BUY FAILED: {errors[0].get('message')}")
                return False

            order_id = result.get('order_id')
            self.log(f"BUY ORDER SUBMITTED: {order_id}")

            # Update tracking (order may still be pending)
            self.shares_bought += BUY_QTY
            self.total_cost += BUY_QTY * bid_price
            self.avg_cost = self.total_cost / self.shares_bought
            self.last_buy_time = datetime.now()

            return True

        except Exception as e:
            self.log(f"BUY ERROR: {e}")
            return False

    def execute_sell(self, bid_price):
        """Execute a sell order"""
        sell_qty = min(SELL_QTY, self.shares_bought)

        self.log(f"{'[DRY RUN] ' if self.dry_run else ''}SELL {sell_qty} UNG @ ${bid_price:.2f}")

        if self.dry_run:
            # Simulate sell
            self.shares_bought -= sell_qty
            self.total_cost = self.shares_bought * self.avg_cost
            self.last_sell_time = datetime.now()

            if self.shares_bought == 0:
                self.avg_cost = 0
                self.total_cost = 0

            return True

        try:
            result = place_order(
                self.session,
                "SELL_QUANTITY",
                UNG_SECURITY_ID,
                sell_qty,
                bid_price
            )

            order_data = result.get('result', {}).get('soOrdersCreateOrder', {})
            errors = order_data.get('errors', [])

            if errors:
                self.log(f"SELL FAILED: {errors[0].get('message')}")
                return False

            order_id = result.get('order_id')
            self.log(f"SELL ORDER SUBMITTED: {order_id}")

            # Update tracking
            self.shares_bought -= sell_qty
            self.total_cost = self.shares_bought * self.avg_cost
            self.last_sell_time = datetime.now()

            if self.shares_bought == 0:
                self.avg_cost = 0
                self.total_cost = 0

            return True

        except Exception as e:
            self.log(f"SELL ERROR: {e}")
            return False

    def run(self):
        """Main loop"""
        self.log("=" * 60)
        self.log("UNG ACCUMULATOR BOT STARTING")
        self.log(f"End time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Dry run: {self.dry_run}")
        self.log(f"Max shares: {MAX_SHARES}")
        self.log(f"RSI oversold threshold: {RSI_OVERSOLD}")
        self.log(f"Max volatility for entry: {MAX_VOLATILITY_PCT}%")
        self.log(f"Rally exit threshold: {RALLY_THRESHOLD_PCT}%")
        self.log("=" * 60)

        # Get initial position
        current_qty, current_avg = self.get_current_position()
        if current_qty > 0:
            self.log(f"Existing position: {current_qty} shares @ ${current_avg:.2f} avg")
            self.shares_bought = current_qty
            self.avg_cost = current_avg
            self.total_cost = current_qty * current_avg

        # Bootstrap RSI with Yahoo Finance extended hours data
        self.log("Fetching extended hours history from Yahoo Finance...")
        history = fetch_yahoo_history_extended(YAHOO_SYMBOL, period=RSI_PERIOD + 10)

        if history:
            for ts, price in history:
                self.prices.append(price)
            dt_first = datetime.fromtimestamp(history[0][0])
            dt_last = datetime.fromtimestamp(history[-1][0])
            self.log(f"Loaded {len(history)} bars from {dt_first.strftime('%H:%M')} to {dt_last.strftime('%H:%M')}")
            if len(self.prices) >= RSI_PERIOD + 1:
                rsi = self.calculate_rsi()
                self.log(f"Initial RSI: {rsi:.1f}")
        else:
            self.log(f"No Yahoo history, will build RSI from live polling (need {RSI_PERIOD + 1} samples)...")

        while datetime.now() < self.end_time:
            try:
                # Get quote
                quote = self.get_quote()
                if not quote:
                    self.log("Failed to get quote, retrying...")
                    time.sleep(10)
                    continue

                bid = quote['bid']
                ask = quote['ask']
                status = quote['market_status']

                # Use mid-price for calculations - more reliable than 'price' field
                # which can have stale/indicative values in pre-market
                mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else quote['last'] or quote['price']

                # Add to price history (use mid-price, not the unreliable 'price' field)
                if mid_price > 0:
                    self.prices.append(mid_price)

                self.log(f"UNG: ${mid_price:.2f} (bid: ${bid:.2f}, ask: ${ask:.2f}) [{status}] | Held: {self.shares_bought} shares")

                # Check sell conditions (RSI overbought + above water)
                should_sell, sell_reason = self.should_sell(mid_price)

                if should_sell:
                    # Check if enough time since last sell
                    if self.last_sell_time:
                        elapsed = (datetime.now() - self.last_sell_time).total_seconds()
                        if elapsed < SELL_INTERVAL_SEC:
                            self.log(f"SELL SIGNAL: {sell_reason} - waiting {SELL_INTERVAL_SEC - elapsed:.0f}s")
                        else:
                            self.log(f"SELL SIGNAL: {sell_reason}")
                            self.execute_sell(bid)
                    else:
                        self.log(f"SELL SIGNAL: {sell_reason}")
                        self.execute_sell(bid)
                else:
                    # Check for buy opportunity
                    should_buy, reason = self.should_buy(quote)
                    if should_buy:
                        self.log(f"BUY SIGNAL: {reason}")
                        self.execute_buy(bid)
                    else:
                        self.log(f"No action: {reason}")

                # Sleep with jitter
                sleep_time = POLL_INTERVAL_SEC + random.uniform(-5, 5)
                time.sleep(max(10, sleep_time))

            except KeyboardInterrupt:
                self.log("Interrupted by user")
                break
            except Exception as e:
                self.log(f"ERROR: {e}")
                time.sleep(30)

        self.log("=" * 60)
        self.log("BOT STOPPED")
        self.log(f"Final position: {self.shares_bought} shares @ ${self.avg_cost:.2f} avg")
        self.log("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='UNG Accumulator Bot')
    parser.add_argument('--dry-run', action='store_true', help='Simulate trades without executing')
    parser.add_argument('--hours', type=float, default=12, help='Run for N hours (default: 12)')
    parser.add_argument('--until', type=str, help='Run until specific time HH:MM (e.g., 09:29)')
    args = parser.parse_args()

    now = datetime.now()

    # If --until is specified, use that as end time
    if args.until:
        hour, minute = map(int, args.until.split(':'))
        end_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # If that time already passed today, it means tomorrow
        if end_time <= now:
            end_time += timedelta(days=1)
        print(f"Will run until: {end_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        end_time = now + timedelta(hours=args.hours)

        # Calculate hours until 9 AM tomorrow
        tomorrow_9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
        if now.hour >= 9:
            tomorrow_9am += timedelta(days=1)

        hours_until_9am = (tomorrow_9am - now).total_seconds() / 3600

        print(f"Hours until 9 AM tomorrow: {hours_until_9am:.1f}")

        # Use the earlier of --hours or 9 AM tomorrow
        if args.hours > hours_until_9am:
            end_time = tomorrow_9am
            print(f"Will run until 9 AM tomorrow: {end_time}")

    bot = UNGAccumulator(dry_run=args.dry_run, end_time=end_time)
    bot.run()


if __name__ == '__main__':
    main()
