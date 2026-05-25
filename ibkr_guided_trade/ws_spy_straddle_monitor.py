#!/usr/bin/env python3
"""
SPY Straddle Monitor — tracks forward ATM, realized vs implied vol, theta burn.

Auto-detects positions from Wealthsimple. Supports multi-strike portfolios.
Shows roll analysis with credit ranges when forward ATM drifts.

Usage:
    python ws_spy_straddle_monitor.py --once                    # Single snapshot
    python ws_spy_straddle_monitor.py                           # Poll every 30s
    python ws_spy_straddle_monitor.py --active                  # Active mode (3s poll)
    python ws_spy_straddle_monitor.py --expiry 2026-03-27       # Override expiry
    python ws_spy_straddle_monitor.py --cost 30.60              # Per-straddle cost basis
"""

import argparse
import csv
import math
import time
import urllib.request
import json
from datetime import datetime, timezone
from pathlib import Path

import ws_trading as ws

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / 'spy_data'
PRICE_CSV = DATA_DIR / 'spy_price_history.csv'
YAHOO_CHART_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1m&range={range}'

# ─── Price history persistence ────────────────────────────────────────────────


def load_price_history() -> list[tuple[float, float]]:
    """Load [(unix_ts, price), ...] from CSV."""
    if not PRICE_CSV.exists():
        return []
    rows = []
    with open(PRICE_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    rows.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
    return rows


def save_price_point(ts: float, price: float):
    """Append a single price point to CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PRICE_CSV.exists()
    with open(PRICE_CSV, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['timestamp', 'price'])
        writer.writerow([f'{ts:.0f}', f'{price:.2f}'])


def save_price_history(history: list[tuple[float, float]]):
    """Overwrite CSV with full history (used after bootstrap merge)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRICE_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'price'])
        for ts, price in history:
            writer.writerow([f'{ts:.0f}', f'{price:.2f}'])


def fetch_ibkr_spy_history(days: int = 5) -> list[tuple[float, float]]:
    """Bootstrap 1-min SPY bars from IBKR. Returns [(ts, close), ...]."""
    try:
        from modules.common import connect
        from ib_insync import Stock
        import calendar
    except ImportError:
        print('  [WARN] ib_insync not available, skipping IBKR bootstrap')
        return []

    try:
        ib = connect(client_id=99)  # dedicated read-only client ID
    except Exception as e:
        print(f'  [WARN] IBKR connect failed: {e}')
        return []

    try:
        contract = Stock('SPY', 'ARCA', 'USD')
        ib.qualifyContracts(contract)

        duration = f'{days} D'
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=True,
        )

        points = []
        for bar in bars:
            ts = calendar.timegm(bar.date.timetuple())
            points.append((float(ts), float(bar.close)))

        print(f'  IBKR: fetched {len(points)} 1-min bars ({days}d)')
        return points

    except Exception as e:
        print(f'  [WARN] IBKR historical data failed: {e}')
        return []
    finally:
        ib.disconnect()


def fetch_yahoo_spy_history(days: int = 5) -> list[tuple[float, float]]:
    """Bootstrap 1-min SPY bars from Yahoo Finance. Returns [(ts, close), ...]."""
    rng = f'{days}d'
    url = YAHOO_CHART_URL.format(range=rng)
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0',
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f'  [WARN] Yahoo fetch failed: {e}')
        return []

    result = data.get('chart', {}).get('result', [])
    if not result:
        return []

    timestamps = result[0].get('timestamp', [])
    closes = result[0].get('indicators', {}).get('quote', [{}])[0].get('close', [])

    points = []
    for ts, c in zip(timestamps, closes):
        if ts is not None and c is not None:
            points.append((float(ts), float(c)))
    return points


# ─── Realized vol ─────────────────────────────────────────────────────────────


def compute_realized_vol(history: list[tuple[float, float]], window_seconds: int) -> float | None:
    """Annualized RV from log returns over a time window.

    Uses actual elapsed trading time for annualization:
    - 6.5 hours per trading day, 252 trading days per year
    - Skips overnight/weekend gaps (>30 min between ticks)
    """
    if len(history) < 10:
        return None

    cutoff = history[-1][0] - window_seconds
    window = [(ts, p) for ts, p in history if ts >= cutoff]

    if len(window) < 10:
        return None

    # Log returns, skipping overnight gaps
    log_rets = []
    for i in range(1, len(window)):
        if window[i - 1][1] > 0 and window[i][1] > 0:
            dt = window[i][0] - window[i - 1][0]
            if dt < 1800:
                log_rets.append(math.log(window[i][1] / window[i - 1][1]))

    if len(log_rets) < 5:
        return None

    mean = sum(log_rets) / len(log_rets)
    var = sum((r - mean) ** 2 for r in log_rets) / (len(log_rets) - 1)

    # Average interval for annualization
    total_elapsed = 0
    count = 0
    for i in range(1, len(window)):
        dt = window[i][0] - window[i - 1][0]
        if dt < 1800:
            total_elapsed += dt
            count += 1
    avg_interval = total_elapsed / count if count > 0 else 60

    trading_secs_per_year = 252 * 6.5 * 3600
    intervals_per_year = trading_secs_per_year / avg_interval

    return math.sqrt(var * intervals_per_year)


# ─── Position auto-detection ─────────────────────────────────────────────────


def fetch_positions(session) -> list[dict]:
    """Fetch current SPY option positions from WS. Returns [{strike, qty, expiry, type}, ...]."""
    config = ws.load_config()
    cookies = ws.load_cookies()
    identity_id = config.get('identity_id') or ws.extract_identity_from_cookies(cookies)
    if not identity_id:
        return []

    data = ws.graphql_query(session, 'FetchIdentityPositions', ws.QUERY_FETCH_POSITIONS, {
        'identityId': identity_id, 'currency': 'CAD', 'first': 50,
        'aggregated': True, 'currencyOverride': 'MARKET',
        'sort': 'TODAY_GAIN', 'includeSecurity': True,
        'includeAccountData': True, 'includeOneDayReturnsBaseline': True,
    })

    positions = []
    for edge in data.get('identity', {}).get('financials', {}).get('current', {}).get('positions', {}).get('edges', []):
        pos = edge.get('node', {})
        sec = pos.get('security', {})
        opt = sec.get('optionDetails', {})
        if not opt:
            continue
        und = opt.get('underlyingSecurity', {}).get('stock', {}).get('symbol', '')
        if und != 'SPY':
            continue
        positions.append({
            'strike': float(opt.get('strikePrice', 0)),
            'qty': int(float(pos.get('quantity', 0))),
            'expiry': opt.get('expiryDate', ''),
            'type': opt.get('optionType', ''),
            'market_value': float(pos.get('totalValue', {}).get('amount', 0)),
            'book_value': float(pos.get('marketBookValue', pos.get('bookValue', {})).get('amount', 0)),
        })

    return positions


def group_straddle_positions(positions: list[dict], expiry: str) -> dict[float, int]:
    """Group positions into straddles by strike. Returns {strike: qty}.

    A straddle = min(call_qty, put_qty) at same strike/expiry.
    """
    calls = {}
    puts = {}
    for p in positions:
        if p['expiry'] != expiry:
            continue
        if p['type'] == 'CALL':
            calls[p['strike']] = calls.get(p['strike'], 0) + p['qty']
        elif p['type'] == 'PUT':
            puts[p['strike']] = puts.get(p['strike'], 0) + p['qty']

    straddles = {}
    all_strikes = set(calls.keys()) | set(puts.keys())
    for k in sorted(all_strikes):
        c = calls.get(k, 0)
        p = puts.get(k, 0)
        n = min(c, p)
        if n > 0:
            straddles[k] = n

    return straddles


# ─── Option chain fetching ────────────────────────────────────────────────────


def fetch_option_chain_both_sides(session, expiry: str) -> dict:
    """Fetch call + put chains with greeks. Returns {strike: {call: {...}, put: {...}}}."""
    spy_id = ws.KNOWN_SECURITIES['SPY']
    chain = {}

    for opt_type in ('CALL', 'PUT'):
        data = ws.graphql_query(session, 'FetchOptionChain', ws.QUERY_OPTION_CHAIN, {
            'id': spy_id,
            'expiryDate': expiry,
            'optionType': opt_type,
            'realTimeQuote': True,
            'includeGreeks': True,
        })
        if not data:
            continue

        edges = data.get('security', {}).get('optionChain', {}).get('edges', [])
        for edge in edges:
            node = edge.get('node', {})
            details = node.get('optionDetails', {})
            quote = node.get('quoteV2', {})
            greeks = details.get('greekSymbols', {}) or {}

            strike = float(details.get('strikePrice', 0))
            bid = float(quote.get('bid', 0) or 0)
            ask = float(quote.get('ask', 0) or 0)
            mid = (bid + ask) / 2 if (bid and ask) else 0
            market_status = quote.get('marketStatus', '')

            entry = {
                'sec_id': node.get('id', ''),
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'last': float(quote.get('last', 0) or 0),
                'oi': int(quote.get('openInterest', 0) or 0),
                'iv': float(greeks.get('impliedVolatility', 0) or 0),
                'delta': float(greeks.get('delta', 0) or 0),
                'gamma': float(greeks.get('gamma', 0) or 0),
                'theta': float(greeks.get('theta', 0) or 0),
                'vega': float(greeks.get('vega', 0) or 0),
                'spot': float(quote.get('underlyingSpot', 0) or 0),
                'market_status': market_status,
            }

            if strike not in chain:
                chain[strike] = {}
            chain[strike][opt_type.lower()] = entry

    return chain


def find_forward_atm(chain_data: dict) -> dict | None:
    """Find strike where |call_mid - put_mid| is smallest (forward ATM).

    Forward price = K + C - P  (put-call parity).
    """
    best = None
    best_diff = float('inf')

    for strike, sides in chain_data.items():
        call = sides.get('call')
        put = sides.get('put')
        if not call or not put:
            continue
        if call['mid'] <= 0 or put['mid'] <= 0:
            continue

        diff = abs(call['mid'] - put['mid'])
        if diff < best_diff:
            best_diff = diff
            straddle_mid = call['mid'] + put['mid']
            avg_iv = (call['iv'] + put['iv']) / 2
            # Forward price from put-call parity: F = K + C - P
            fwd_price = strike + call['mid'] - put['mid']
            best = {
                'strike': strike,
                'straddle_mid': straddle_mid,
                'call_mid': call['mid'],
                'put_mid': put['mid'],
                'forward_price': fwd_price,
                'iv': avg_iv,
                'call_iv': call['iv'],
                'put_iv': put['iv'],
                'delta': call['delta'] + put['delta'],
                'gamma': call['gamma'] + put['gamma'],
                'theta': call['theta'] + put['theta'],
                'vega': call['vega'] + put['vega'],
                'spot': call['spot'] or put['spot'],
                'market_status': call.get('market_status', ''),
            }

    return best


def get_position_greeks(chain_data: dict, strike: float, qty: int) -> dict | None:
    """Get greeks for position at given strike * qty * 100 multiplier."""
    sides = chain_data.get(strike)
    if not sides:
        return None
    call = sides.get('call')
    put = sides.get('put')
    if not call or not put:
        return None

    mult = qty * 100
    straddle_mid = call['mid'] + put['mid']
    avg_iv = (call['iv'] + put['iv']) / 2

    return {
        'straddle_mid': straddle_mid,
        'market_value': straddle_mid * mult,
        'iv': avg_iv,
        'delta': (call['delta'] + put['delta']) * mult,
        'gamma': (call['gamma'] + put['gamma']) * mult,
        'theta': (call['theta'] + put['theta']) * mult,
        'vega': (call['vega'] + put['vega']) * mult,
        'call_bid': call['bid'],
        'call_ask': call['ask'],
        'call_mid': call['mid'],
        'put_bid': put['bid'],
        'put_ask': put['ask'],
        'put_mid': put['mid'],
        'per_straddle_delta': call['delta'] + put['delta'],
        'per_straddle_gamma': call['gamma'] + put['gamma'],
        'per_straddle_theta': call['theta'] + put['theta'],
    }


def compute_roll_credit(chain_data: dict, from_strike: float, to_strike: float) -> dict | None:
    """Compute roll credit range from one strike to another."""
    src = chain_data.get(from_strike)
    dst = chain_data.get(to_strike)
    if not src or not dst:
        return None
    sc = src.get('call')
    sp = src.get('put')
    dc = dst.get('call')
    dp = dst.get('put')
    if not all([sc, sp, dc, dp]):
        return None

    sell_bid = sc['bid'] + sp['bid']
    sell_ask = sc['ask'] + sp['ask']
    sell_mid = sc['mid'] + sp['mid']
    buy_bid = dc['bid'] + dp['bid']
    buy_ask = dc['ask'] + dp['ask']
    buy_mid = dc['mid'] + dp['mid']

    return {
        'max_credit': round(sell_ask - buy_bid, 2),
        'mid_credit': round(sell_mid - buy_mid, 2),
        'min_credit': round(sell_bid - buy_ask, 2),
        'sell_mid': round(sell_mid, 2),
        'buy_mid': round(buy_mid, 2),
    }


# ─── DTE helpers ──────────────────────────────────────────────────────────────


def compute_dte(expiry: str) -> float:
    """Calendar DTE including fractional day."""
    exp_dt = datetime.strptime(expiry, '%Y-%m-%d').replace(
        hour=16, minute=0, tzinfo=timezone.utc
    )
    now = datetime.now(timezone.utc)
    return max((exp_dt - now).total_seconds() / 86400, 0)


def dte_warning(dte: float) -> str:
    if dte <= 7:
        return ' !! URGENT'
    elif dte <= 14:
        return ' ! WARN'
    return ''


# ─── Display ──────────────────────────────────────────────────────────────────


def compact_line(spot, dte, fwd_strike, iv, rv_iv_spread, delta, theta_day, rv_signal, market_status):
    """Print single compact status line."""
    ts = datetime.now().strftime('%H:%M:%S')
    iv_pct = iv * 100 if iv else 0
    spread_str = f'{rv_iv_spread * 100:+.1f}%' if rv_iv_spread is not None else 'n/a'
    signal = rv_signal or ''
    warn = dte_warning(dte)
    mkt = ' [AH]' if market_status and market_status != 'OPEN' else ''

    print(
        f'[{ts}]{mkt} SPY ${spot:.2f} | DTE {dte:.1f}{warn} | '
        f'Fwd ${fwd_strike:.0f} | IV {iv_pct:.1f}% | '
        f'RV-IV {spread_str} {signal} | '
        f'D {delta:+.0f} | Th ${theta_day:+,.0f}/day',
        flush=True,
    )


def full_summary(spot, dte, expiry, fwd, straddles, chain, rv_1d, rv_5d, iv,
                  price_history, cost, market_status):
    """Print detailed summary block with multi-strike support."""
    warn = dte_warning(dte)
    mkt = ' [AFTER HOURS]' if market_status and market_status != 'OPEN' else ''
    print('\n' + '=' * 72)
    print(f'  SPY STRADDLE MONITOR — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}{mkt}')
    print('=' * 72)

    # Market
    print(f'\n  MARKET')
    print(f'    SPY Spot:       ${spot:.2f}')
    print(f'    DTE:            {dte:.1f} days{warn}')
    print(f'    Expiry:         {expiry}')

    # Forward ATM
    if fwd:
        print(f'\n  FORWARD ATM')
        print(f'    Forward Strike: ${fwd["strike"]:.0f}')
        print(f'    Forward Price:  ${fwd["forward_price"]:.2f}')
        print(f'    Straddle Mid:   ${fwd["straddle_mid"]:.2f}')

    # Per-strike positions + aggregated totals
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0
    total_value = 0
    total_qty = 0

    for strike in sorted(straddles.keys()):
        qty = straddles[strike]
        total_qty += qty
        pos = get_position_greeks(chain, strike, qty)
        if not pos:
            print(f'\n  POSITION ({qty}x ${strike:.0f} straddle) — no chain data')
            continue

        total_delta += pos['delta']
        total_gamma += pos['gamma']
        total_theta += pos['theta']
        total_vega += pos['vega']
        total_value += pos['market_value']

        # Drift from forward
        drift = fwd['strike'] - strike if fwd else 0
        drift_str = f'  (drift {drift:+.0f})' if drift != 0 else ''
        roll_flag = '  << ROLL >>' if abs(drift) >= 2 else ''

        print(f'\n  POSITION ({qty}x ${strike:.0f} straddle){drift_str}{roll_flag}')
        print(f'    Call:     ${pos["call_bid"]:.2f} / ${pos["call_ask"]:.2f} (mid ${pos["call_mid"]:.2f})')
        print(f'    Put:      ${pos["put_bid"]:.2f} / ${pos["put_ask"]:.2f} (mid ${pos["put_mid"]:.2f})')
        print(f'    Straddle: ${pos["straddle_mid"]:.2f}  |  Mkt Val: ${pos["market_value"]:,.2f}')
        print(f'    D {pos["delta"]:+,.0f}  G {pos["gamma"]:+,.1f}  Th {pos["theta"]:+,.1f}  V {pos["vega"]:+,.1f}')

        # Roll analysis if drifted
        if fwd and abs(drift) >= 1:
            roll = compute_roll_credit(chain, strike, fwd['strike'])
            if roll:
                print(f'    Roll -> ${fwd["strike"]:.0f}: credit ${roll["mid_credit"]:.2f} '
                      f'(range ${roll["min_credit"]:.2f}–${roll["max_credit"]:.2f})')
                print(f'      x{qty}: ${roll["mid_credit"] * qty * 100:,.0f} mid credit')

    # Aggregated totals
    if len(straddles) > 1:
        print(f'\n  TOTAL ({total_qty}x straddles across {len(straddles)} strikes)')
    else:
        print(f'\n  TOTAL ({total_qty}x straddles)')
    print(f'    Mkt Val:  ${total_value:,.2f}')
    if cost:
        total_cost = cost * total_qty * 100
        pnl = total_value - total_cost
        pnl_pct = (pnl / total_cost) * 100 if total_cost else 0
        print(f'    Cost:     ${total_cost:,.2f} (${cost:.2f}/share)')
        print(f'    P&L:      ${pnl:+,.2f} ({pnl_pct:+.1f}%)')
    print(f'    Delta:    {total_delta:+,.0f}')
    print(f'    Gamma:    {total_gamma:+,.1f}')
    print(f'    Theta:    {total_theta:+,.1f}')
    print(f'    Vega:     {total_vega:+,.1f}')

    # Volatility
    iv_pct = iv * 100 if iv else 0
    rv_1d_pct = rv_1d * 100 if rv_1d else None
    rv_5d_pct = rv_5d * 100 if rv_5d else None
    print(f'\n  VOLATILITY')
    print(f'    ATM IV:       {iv_pct:.1f}%')
    if rv_1d_pct is not None:
        spread_1d = rv_1d_pct - iv_pct
        sig_1d = '[G>T]' if spread_1d > 0 else '[T>G]'
        print(f'    RV (1-day):   {rv_1d_pct:.1f}%  (RV-IV: {spread_1d:+.1f}% {sig_1d})')
    else:
        print(f'    RV (1-day):   n/a (need more data)')
    if rv_5d_pct is not None:
        spread_5d = rv_5d_pct - iv_pct
        sig_5d = '[G>T]' if spread_5d > 0 else '[T>G]'
        print(f'    RV (5-day):   {rv_5d_pct:.1f}%  (RV-IV: {spread_5d:+.1f}% {sig_5d})')
    else:
        print(f'    RV (5-day):   n/a (need more data)')
    print(f'    Prices in DB: {len(price_history)}')

    # Theta burn (aggregated)
    if total_theta != 0:
        daily_theta = total_theta
        weekly_theta = daily_theta * 5
        to_expiry_theta = daily_theta * dte

        # Breakeven: use weighted average gamma/theta across all strikes
        total_per_gamma = 0
        total_per_theta = 0
        for strike, qty in straddles.items():
            pos = get_position_greeks(chain, strike, 1)
            if pos:
                total_per_gamma += pos['per_straddle_gamma'] * qty
                total_per_theta += pos['per_straddle_theta'] * qty
        avg_gamma = total_per_gamma / total_qty if total_qty else 0
        avg_theta = total_per_theta / total_qty if total_qty else 0
        if avg_gamma > 0 and avg_theta < 0:
            breakeven_move = math.sqrt(2 * abs(avg_theta) / (avg_gamma * 100))
        else:
            breakeven_move = 0

        print(f'\n  THETA BURN')
        print(f'    Daily:          ${daily_theta:+,.0f}')
        print(f'    Weekly:         ${weekly_theta:+,.0f}')
        print(f'    To Expiry:      ${to_expiry_theta:+,.0f}')
        if breakeven_move > 0 and spot > 0:
            print(f'    Breakeven Move: ${breakeven_move:.2f} SPY ({breakeven_move / spot * 100:.2f}%)')

    print('=' * 72 + '\n')


# ─── Main loop ────────────────────────────────────────────────────────────────


def run_monitor(args):
    expiry = args.expiry
    cost = args.cost
    poll = 3 if args.active else args.poll
    once = args.once

    # Bootstrap price history
    price_history = load_price_history()
    if len(price_history) < 100:
        print('Bootstrapping price history (trying IBKR, then Yahoo)...')
        bootstrap = fetch_ibkr_spy_history(5)
        if not bootstrap:
            print('  Falling back to Yahoo Finance...')
            bootstrap = fetch_yahoo_spy_history(5)
        if bootstrap:
            existing_ts = {ts for ts, _ in price_history}
            for ts, p in bootstrap:
                if ts not in existing_ts:
                    price_history.append((ts, p))
            price_history.sort()
            save_price_history(price_history)
            print(f'  Total: {len(price_history)} price points')
        else:
            print('  Bootstrap failed, will build history from live data')
    else:
        print(f'Loaded {len(price_history)} price points from history')

    # Connect
    print('Connecting to Wealthsimple...')
    session = ws.get_session()
    last_token_refresh = time.time()
    print('Session ready.')

    # Auto-detect positions
    print('Detecting positions...')
    ws_positions = fetch_positions(session)
    straddles = group_straddle_positions(ws_positions, expiry)

    if straddles:
        parts = [f'{qty}x ${k:.0f}' for k, qty in sorted(straddles.items())]
        total = sum(straddles.values())
        print(f'  Found: {" + ".join(parts)} = {total} straddles')
    else:
        print(f'  No SPY straddle positions found for {expiry}')
        if not once:
            print('  Will monitor forward ATM only')
        straddles = {}

    # Allow CLI override
    if args.strike and args.qty and not straddles:
        straddles = {args.strike: args.qty}
        print(f'  Using CLI override: {args.qty}x ${args.strike:.0f}')

    print(f'Poll: {poll}s\n')

    cycle = 0
    last_fwd_strike = None

    while True:
        try:
            cycle += 1

            # Proactive token refresh every 10 min
            if time.time() - last_token_refresh > 600:
                session = ws.get_session()
                last_token_refresh = time.time()

            # Fetch chain
            chain = fetch_option_chain_both_sides(session, expiry)
            if not chain:
                print(f'[{datetime.now().strftime("%H:%M:%S")}] Failed to fetch chain, refreshing session...')
                session = ws.get_session()
                last_token_refresh = time.time()
                if once:
                    return
                time.sleep(poll)
                continue

            # Forward ATM
            fwd = find_forward_atm(chain)
            spot = fwd['spot'] if fwd else 0
            market_status = fwd.get('market_status', '') if fwd else ''

            # SPY spot fallback
            if spot == 0:
                spy_quote = ws.graphql_query(session, 'FetchSecurityQuoteV2', ws.QUERY_FETCH_SECURITY, {
                    'id': ws.KNOWN_SECURITIES['SPY'],
                })
                spot = float(spy_quote.get('security', {}).get('quoteV2', {}).get('price', 0) or 0)

            # Record price
            now_ts = time.time()
            if spot > 0:
                price_history.append((now_ts, spot))
                save_price_point(now_ts, spot)

            # DTE
            dte = compute_dte(expiry)

            # IV from forward ATM
            iv = fwd['iv'] if fwd else 0

            # Realized vol
            rv_1d = compute_realized_vol(price_history, 1 * 24 * 3600)
            rv_5d = compute_realized_vol(price_history, 5 * 24 * 3600)

            # RV-IV spread
            rv_iv_spread = (rv_1d - iv) if rv_1d is not None and iv else None
            rv_signal = ''
            if rv_iv_spread is not None:
                rv_signal = '[G>T]' if rv_iv_spread > 0 else '[T>G]'

            # Aggregate delta/theta for compact line
            total_delta = 0
            total_theta = 0
            for strike, qty in straddles.items():
                pos = get_position_greeks(chain, strike, qty)
                if pos:
                    total_delta += pos['delta']
                    total_theta += pos['theta']

            # Detect forward ATM shift
            fwd_shifted = False
            fwd_strike = fwd['strike'] if fwd else 0
            if last_fwd_strike is not None and fwd_strike != last_fwd_strike:
                fwd_shifted = True
            last_fwd_strike = fwd_strike

            # Re-detect positions every 50 cycles (~25 min) in case rolls filled
            if cycle % 50 == 0:
                ws_positions = fetch_positions(session)
                new_straddles = group_straddle_positions(ws_positions, expiry)
                if new_straddles != straddles:
                    straddles = new_straddles
                    parts = [f'{qty}x ${k:.0f}' for k, qty in sorted(straddles.items())]
                    print(f'  [POSITION UPDATE] {" + ".join(parts)}')

            # Full summary: first cycle, every 20 cycles, on fwd shift, or --once
            show_full = (cycle == 1) or (cycle % 20 == 0) or fwd_shifted or once

            if show_full:
                full_summary(
                    spot, dte, expiry, fwd, straddles, chain, rv_1d, rv_5d, iv,
                    price_history, cost, market_status,
                )
            else:
                compact_line(
                    spot, dte, fwd_strike, iv, rv_iv_spread,
                    total_delta, total_theta, rv_signal, market_status,
                )

            if once:
                return

            time.sleep(poll)

        except KeyboardInterrupt:
            print('\nStopping monitor.')
            return
        except Exception as e:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] Error: {e}')
            # Try refreshing session on error
            try:
                session = ws.get_session()
                last_token_refresh = time.time()
            except Exception:
                pass
            if once:
                return
            time.sleep(poll)


def main():
    parser = argparse.ArgumentParser(description='SPY Straddle Monitor')
    parser.add_argument('--poll', type=int, default=30, help='Poll interval seconds (default 30)')
    parser.add_argument('--active', action='store_true', help='Active mode (3s poll)')
    parser.add_argument('--once', action='store_true', help='Single snapshot then exit')
    parser.add_argument('--strike', type=float, default=None, help='Override position strike')
    parser.add_argument('--expiry', default='2026-03-31', help='Expiry date (default 2026-03-31)')
    parser.add_argument('--qty', type=int, default=None, help='Override straddle count')
    parser.add_argument('--cost', type=float, default=None, help='Per-straddle cost basis (for P&L)')

    args = parser.parse_args()
    run_monitor(args)


if __name__ == '__main__':
    main()
