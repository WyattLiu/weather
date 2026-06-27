"""
Module 4: Market Scanner

3-phase scan to find credit spread opportunities (bull puts + bear calls):
  Phase 1: Screen stocks for sharp drops OR rallies, declining RV, adequate liquidity
  Phase 2: IV screen — ATM put IV at target DTE, compute IV-RV edge
  Phase 3: Deep scan — full put/call chain, OI walls, both spread types, composite scoring
  Portfolio: Markowitz mean-variance optimization across all spread candidates
"""

import json
import math
from datetime import datetime, date
from pathlib import Path

from ib_insync import Stock, Option

from .common import connect

SCANNER_CLIENT_ID = 94

DATA_DIR = Path(__file__).parent.parent / 'scan_data'

DEFAULT_FILTERS = {
    'min_drop': 10,
    'min_price': 10,
    'min_vol': 1_000_000,
    'min_edge': 5,
    'target_dte': 35,
    'top': 10,
}

# S&P 500 constituents (503 tickers, from Wikipedia Feb 2026)
SP500_TICKERS = [
    'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM',
    'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM',
    'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP',
    'AMT', 'AMZN', 'ANET', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APO', 'APP',
    'APTV', 'ARE', 'ARES', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP',
    'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B', 'BG',
    'BIIB', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK.B', 'BRO',
    'BSX', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE',
    'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW',
    'CHTR', 'CI', 'CIEN', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI',
    'CMS', 'CNC', 'CNP', 'COF', 'COIN', 'COO', 'COP', 'COR', 'COST', 'CPAY',
    'CPB', 'CPRT', 'CPT', 'CRH', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX',
    'CTAS', 'CTRA', 'CTSH', 'CTVA', 'CVNA', 'CVS', 'CVX', 'D', 'DAL', 'DASH',
    'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DG', 'DGX', 'DHI', 'DHR', 'DIS',
    'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
    'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL',
    'ELV', 'EME', 'EMR', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES',
    'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXE', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FICO', 'FIS',
    'FISV', 'FITB', 'FIX', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD',
    'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
    'HBAN', 'HCA', 'HD', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HOOD', 'HPE',
    'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBKR', 'IBM',
    'ICE', 'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IQV',
    'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBHT', 'JBL', 'JCI',
    'JKHY', 'JNJ', 'JPM', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KKR', 'KLAC',
    'KMB', 'KMI', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX',
    'LII', 'LIN', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS',
    'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK',
    'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MKC', 'MLM', 'MMM', 'MNST',
    'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRSH', 'MS', 'MSCI',
    'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NDSN', 'NEE',
    'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS',
    'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC',
    'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PAYC', 'PAYX', 'PCAR', 'PCG',
    'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD',
    'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU',
    'PSA', 'PSKY', 'PSX', 'PTC', 'PWR', 'PYPL', 'Q', 'QCOM', 'RCL', 'REG',
    'REGN', 'RF', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
    'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA',
    'SNDK', 'SNPS', 'SO', 'SOLV', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT',
    'STX', 'STZ', 'SW', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP',
    'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TGT', 'TJX', 'TKO', 'TMO',
    'TMUS', 'TPL', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN',
    'TT', 'TTD', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS',
    'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO',
    'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT',
    'WBD', 'WDAY', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB',
    'WSM', 'WST', 'WTW', 'WY', 'WYNN', 'XEL', 'XOM', 'XYL', 'XYZ', 'YUM',
    'ZBH', 'ZBRA', 'ZTS',
]

# Curated high-IV / post-earnings drop watchlist
QUICK_LIST = [
    'GOOGL', 'AMZN', 'INTC', 'PYPL', 'PINS', 'UBER', 'F', 'NKE', 'SBUX',
    'PFE', 'MRNA', 'BA', 'DIS', 'NFLX', 'META', 'TSLA', 'AMD', 'MU',
    'SNAP', 'COIN', 'HOOD', 'ABNB', 'DKNG', 'RBLX', 'SHOP', 'SQ',
    'CVS', 'WBD', 'PARA', 'EL', 'NEM', 'FCX', 'ALB', 'FSLR',
    'SMCI', 'PLTR', 'CRWD', 'PANW', 'SNOW', 'DDOG', 'NET',
    'UNG', 'MARA', 'RIOT', 'MSTR', 'GM', 'LUV', 'AAL', 'DAL',
]


# ============ Helpers ============

def scanner_connect():
    """Connect with scanner-specific client ID."""
    return connect(client_id=SCANNER_CLIENT_ID)


def compute_rv(closes, window):
    """Compute annualized realized volatility from close prices."""
    if len(closes) < window + 1:
        return None
    recent = closes[-(window + 1):]
    log_returns = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    if not log_returns:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance * 252) * 100


def compute_support_resistance(bars, current):
    """Find support/resistance levels using adaptive bucket touch counting.

    Uses lows for support, highs for resistance. Bucket size scales with price:
    <$20: $0.50, $20-100: $1, $100-500: $2, $500+: $5.
    """
    if not bars:
        return [], []

    if current < 20:
        bucket_size = 0.5
    elif current < 100:
        bucket_size = 1
    elif current < 500:
        bucket_size = 2
    else:
        bucket_size = 5

    # Count low touches for support, high touches for resistance
    low_touches = {}
    high_touches = {}
    for b in bars:
        low_bucket = round(b.low / bucket_size) * bucket_size
        high_bucket = round(b.high / bucket_size) * bucket_size
        low_touches[low_bucket] = low_touches.get(low_bucket, 0) + 1
        high_touches[high_bucket] = high_touches.get(high_bucket, 0) + 1

    # Also count close prices near lows (confirms support held)
    for b in bars:
        close_bucket = round(b.close / bucket_size) * bucket_size
        if b.close <= b.open:  # down day that held — stronger support signal
            low_touches[close_bucket] = low_touches.get(close_bucket, 0) + 0.5

    support = sorted(
        [(p, int(c)) for p, c in low_touches.items() if p < current and c >= 3],
        key=lambda x: x[1], reverse=True
    )[:5]
    resistance = sorted(
        [(p, int(c)) for p, c in high_touches.items() if p > current and c >= 3],
        key=lambda x: x[1], reverse=True
    )[:5]
    return support, resistance


def _thin_strikes(strikes, max_count):
    """Thin a strike list to max_count, preferring whole-dollar then round numbers."""
    if len(strikes) <= max_count:
        return strikes
    # Prefer whole-dollar strikes (filters out $0.50 and $2.50 increments)
    whole = [s for s in strikes if abs(s - round(s)) < 0.01]
    if len(whole) >= 5:
        strikes = whole
    if len(strikes) <= max_count:
        return strikes
    # Still too many — prefer $5 multiples
    fives = [s for s in strikes if abs(s % 5) < 0.01 or abs(s % 5 - 5) < 0.01]
    if len(fives) >= 5:
        strikes = fives
    if len(strikes) <= max_count:
        return strikes
    # Last resort: uniform step
    step = max(1, len(strikes) // max_count)
    return strikes[::step]


def near_support(strike, support_levels, tolerance_pct=2.0):
    """Check if a strike is within tolerance% of a support level. Returns (bool, level, touches)."""
    for level, touches in support_levels:
        if abs(strike - level) / level * 100 <= tolerance_pct:
            return True, level, touches
    return False, None, None


def near_resistance(strike, resistance_levels, tolerance_pct=2.0):
    """Check if a strike is within tolerance% of a resistance level. Returns (bool, level, touches)."""
    for level, touches in resistance_levels:
        if abs(strike - level) / level * 100 <= tolerance_pct:
            return True, level, touches
    return False, None, None


def detect_sharp_drop(bars):
    """True if >50% of total 60d drop occurred in any 10-day window."""
    if len(bars) < 60:
        return False
    recent = bars[-60:]
    closes = [b.close for b in recent]
    high_60d = max(b.high for b in recent)
    total_drop = high_60d - closes[-1]
    if total_drop <= 0:
        return False

    # Check each 10-day window
    for i in range(len(closes) - 10):
        window_high = max(closes[i:i + 10])
        window_low = min(closes[i:i + 10])
        window_drop = window_high - window_low
        if window_drop > total_drop * 0.5:
            return True
    return False


def detect_sharp_rally(bars):
    """True if >50% of total 60d rally occurred in any 10-day window."""
    if len(bars) < 60:
        return False
    recent = bars[-60:]
    closes = [b.close for b in recent]
    low_60d = min(b.low for b in recent)
    total_rally = closes[-1] - low_60d
    if total_rally <= 0:
        return False

    for i in range(len(closes) - 10):
        window_low = min(closes[i:i + 10])
        window_high = max(closes[i:i + 10])
        window_rally = window_high - window_low
        if window_rally > total_rally * 0.5:
            return True
    return False


def oi_grade(oi):
    """Grade open interest: A=5000+, B=1000+, C=200+, D=below."""
    if oi >= 5000:
        return 'A'
    elif oi >= 1000:
        return 'B'
    elif oi >= 200:
        return 'C'
    return 'D'


def print_progress(msg):
    print(f'\r  {msg}' + ' ' * 20, end='', flush=True)


# ============ Caching ============

def save_cache(data, phase):
    """Save scan results to scan_data/phase{N}_YYYYMMDD.json."""
    DATA_DIR.mkdir(exist_ok=True)
    filepath = DATA_DIR / f'phase{phase}_{date.today().strftime("%Y%m%d")}.json'
    with open(filepath, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'data': data}, f, indent=2)
    print(f'  Cached → {filepath}')


def load_cache(phase, max_age_hours=4):
    """Load cached phase data if fresh enough. Returns (data, age_str) or (None, None)."""
    filepath = DATA_DIR / f'phase{phase}_{date.today().strftime("%Y%m%d")}.json'
    if not filepath.exists():
        return None, None
    try:
        with open(filepath) as f:
            cached = json.load(f)
        ts = datetime.fromisoformat(cached['timestamp'])
        age_sec = (datetime.now() - ts).total_seconds()
        if age_sec > max_age_hours * 3600:
            return None, None
        age_str = f'{age_sec / 60:.0f}m ago'
        return cached['data'], age_str
    except Exception:
        return None, None


# ============ Phase 1: Historical Screen ============

def phase1_screen(ib, tickers, filters, progress_cb=None):
    """Phase 1: Pull 90d daily bars, compute vol/drop/support metrics.

    Returns dict of {ticker: {...metrics...}} for candidates passing filters.
    """
    min_drop = filters.get('min_drop', DEFAULT_FILTERS['min_drop'])
    min_price = filters.get('min_price', DEFAULT_FILTERS['min_price'])
    min_vol = filters.get('min_vol', DEFAULT_FILTERS['min_vol'])

    results = {}
    total = len(tickers)
    batch_size = 45

    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        if progress_cb:
            progress_cb(f'Phase 1: Batch {batch_num}/{total_batches} ({batch[0]}-{batch[-1]})')

        stocks = []
        valid_symbols = []
        for sym in batch:
            ibkr_sym = sym.replace('.', ' ')
            stock = Stock(ibkr_sym, 'SMART', 'USD')
            stocks.append(stock)
            valid_symbols.append(sym)

        try:
            ib.qualifyContracts(*stocks)
        except Exception as e:
            print(f'\n  Qualify error batch {batch_num}: {e}')
            continue

        for i, stock in enumerate(stocks):
            sym = valid_symbols[i]
            if not stock.conId:
                continue

            try:
                bars = ib.reqHistoricalData(
                    stock, endDateTime='', durationStr='6 M',
                    barSizeSetting='1 day', whatToShow='TRADES',
                    useRTH=True, formatDate=1, timeout=10,
                )
                if not bars or len(bars) < 61:
                    continue

                closes = [b.close for b in bars]
                spot = closes[-1]

                # Price filter
                if spot < min_price:
                    continue

                # Volume filter (avg 20d)
                volumes = [b.volume for b in bars[-20:]]
                avg_vol = sum(volumes) / len(volumes) if volumes else 0
                if avg_vol < min_vol:
                    continue

                # Realized vols
                rv10 = compute_rv(closes, 10)
                rv20 = compute_rv(closes, 20)
                rv30 = compute_rv(closes, 30)
                rv60 = compute_rv(closes, 60)

                if not rv10 or not rv30:
                    continue

                # Drop from 60d high
                recent_60 = bars[-60:]
                high_60d = max(b.high for b in recent_60)
                drop_pct = (spot - high_60d) / high_60d * 100  # negative = drop
                days_since_high = next(
                    (len(recent_60) - 1 - i for i, b in enumerate(recent_60) if b.high == high_60d),
                    0
                )

                # Rise from 60d low
                low_60d = min(b.low for b in recent_60)
                rise_pct = (spot - low_60d) / low_60d * 100 if low_60d > 0 else 0
                days_since_low = next(
                    (len(recent_60) - 1 - i for i, b in enumerate(recent_60) if b.low == low_60d),
                    0
                )

                # Direction filter: either drop or rally qualifies
                has_drop = abs(drop_pct) >= min_drop
                has_rally = rise_pct >= min_drop
                if not has_drop and not has_rally:
                    continue

                # RV declining filter: RV10 < RV30
                if rv10 >= rv30:
                    continue

                # Determine direction
                if has_drop and has_rally:
                    direction = 'both'
                elif has_drop:
                    direction = 'put'
                else:
                    direction = 'call'

                # Bounce from 20d low
                low_20d = min(b.low for b in bars[-20:])
                bounce_pct = (spot - low_20d) / low_20d * 100

                # Pullback from 20d high
                high_20d = max(b.high for b in bars[-20:])
                pullback_pct = (high_20d - spot) / high_20d * 100

                # Sharp move detection
                is_sharp = detect_sharp_drop(bars)
                is_sharp_rally = detect_sharp_rally(bars)

                # SMA
                sma20 = sum(closes[-20:]) / 20
                sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None

                # Support/resistance
                support, resistance = compute_support_resistance(bars[-90:], spot)

                results[sym] = {
                    'spot': spot,
                    'rv10': rv10,
                    'rv20': rv20,
                    'rv30': rv30,
                    'rv60': rv60,
                    'high_60d': high_60d,
                    'drop_pct': drop_pct,
                    'days_since_high': days_since_high,
                    'low_60d': low_60d,
                    'rise_pct': rise_pct,
                    'days_since_low': days_since_low,
                    'low_20d': low_20d,
                    'bounce_pct': bounce_pct,
                    'high_20d': high_20d,
                    'pullback_pct': pullback_pct,
                    'avg_vol': avg_vol,
                    'is_sharp': is_sharp,
                    'is_sharp_rally': is_sharp_rally,
                    'direction': direction,
                    'sma20': sma20,
                    'sma50': sma50,
                    'support': support,
                    'resistance': resistance,
                }

            except Exception:
                pass

            if i % 5 == 4:
                ib.sleep(1)

        ib.sleep(1)

    return results


# ============ Phase 2: IV Screen ============

def _ranked_expiries(smart_chain, today, target_dte):
    """Return expiries ranked by preference: monthlies first, then by DTE closeness."""
    candidates = []
    for exp in smart_chain.expirations:
        exp_date = datetime.strptime(exp, '%Y%m%d').date()
        dte = (exp_date - today).days
        if 20 <= dte <= 55:
            first = exp_date.replace(day=1)
            days_to_fri = (4 - first.weekday()) % 7
            third_friday = first.replace(day=1 + days_to_fri + 14)
            is_monthly = (exp_date == third_friday)
            candidates.append((exp, dte, is_monthly))

    # Sort: monthlies first, then by distance from target DTE
    candidates.sort(key=lambda x: (not x[2], abs(x[1] - target_dte)))
    return [(exp, dte) for exp, dte, _ in candidates]


def phase2_iv_screen(ib, candidates, target_dte, progress_cb=None):
    """Phase 2: Get ATM put IV, compute IV-RV edge.

    Batch-qualifies 20 nearby strikes to catch $5 increments.
    Falls back to next expiry if first one yields no IV.
    """
    today = date.today()
    results = {}
    symbols = list(candidates.keys())
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(f'Phase 2: {idx + 1}/{total} — {sym}')

        info = candidates[sym]
        spot = info['spot']
        ibkr_sym = sym.replace('.', ' ')
        stock = Stock(ibkr_sym, 'SMART', 'USD')

        try:
            ib.qualifyContracts(stock)
            if not stock.conId:
                continue

            chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
            smart_chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not smart_chain:
                continue

            expiries = _ranked_expiries(smart_chain, today, target_dte)
            if not expiries:
                continue

            all_strikes = sorted(smart_chain.strikes)
            strikes_by_dist = sorted(all_strikes, key=lambda s: abs(s - spot))[:20]

            # Try up to 3 expiries until we get IV
            atm_iv = None
            atm_strike = None
            used_exp = None
            actual_dte = None

            for exp, dte in expiries[:3]:
                put_candidates = [Option(ibkr_sym, exp, s, 'P', 'SMART') for s in strikes_by_dist]
                ib.qualifyContracts(*put_candidates)

                qualified = [(p, abs(p.strike - spot)) for p in put_candidates if p.conId]
                if not qualified:
                    continue
                qualified.sort(key=lambda x: x[1])
                atm_put = qualified[0][0]

                t = ib.reqMktData(atm_put, genericTickList='106')
                ib.sleep(4)

                if t.modelGreeks and t.modelGreeks.impliedVol:
                    atm_iv = t.modelGreeks.impliedVol * 100
                    atm_strike = atm_put.strike
                    used_exp = exp
                    actual_dte = dte

                ib.cancelMktData(atm_put)

                if atm_iv is not None:
                    break

            if atm_iv is None:
                continue

            rv10 = info.get('rv10', 0) or 0
            rv30 = info.get('rv30', 0) or 0
            edge_rv10 = atm_iv - rv10
            edge_rv30 = atm_iv - rv30

            results[sym] = {
                **info,
                'expiry': used_exp,
                'dte': actual_dte,
                'atm_strike': atm_strike,
                'atm_iv': atm_iv,
                'edge_rv10': edge_rv10,
                'edge_rv30': edge_rv30,
            }

        except Exception:
            pass

        if idx % 3 == 2:
            ib.sleep(1)

    return results


# ============ Phase 3: Deep Scan ============

def phase3_deep_scan(ib, candidates, target_dte, progress_cb=None):
    """Phase 3: Full put+call chain, OI walls, spread analysis, composite scoring."""
    results = {}
    symbols = list(candidates.keys())
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(f'Phase 3: {idx + 1}/{total} — {sym} deep scan')

        info = candidates[sym]
        spot = info['spot']
        expiry = info['expiry']
        dte = info['dte']
        ibkr_sym = sym.replace('.', ' ')
        stock = Stock(ibkr_sym, 'SMART', 'USD')

        try:
            ib.qualifyContracts(stock)
            chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
            smart_chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not smart_chain:
                continue

            all_strikes = sorted(smart_chain.strikes)

            # === Put chain: 3%-20% OTM ===
            put_strikes = [s for s in all_strikes if spot * 0.80 <= s <= spot * 0.97]
            put_strikes = _thin_strikes(put_strikes, 20)

            if not put_strikes:
                continue

            puts = [Option(ibkr_sym, expiry, s, 'P', 'SMART') for s in put_strikes]
            ib.qualifyContracts(*puts)
            puts = [p for p in puts if p.conId]

            if not puts:
                continue

            # Stream put data (OI needs streaming, not snapshot)
            put_tickers = {}
            for p in puts:
                t = ib.reqMktData(p, genericTickList='101,106')
                put_tickers[p.strike] = t

            # === Call chain: ATM to 20% OTM (OI walls + bear call spreads) ===
            call_strikes = [s for s in all_strikes if spot * 0.99 <= s <= spot * 1.20]
            call_strikes = _thin_strikes(call_strikes, 20)

            calls = [Option(ibkr_sym, expiry, s, 'C', 'SMART') for s in call_strikes]
            ib.qualifyContracts(*calls)
            calls = [c for c in calls if c.conId]

            call_tickers = {}
            for c in calls:
                t = ib.reqMktData(c, genericTickList='101,106')
                call_tickers[c.strike] = t

            ib.sleep(6)

            # Collect put data
            put_data = {}
            for strike, t in put_tickers.items():
                bid = t.bid if t.bid and t.bid > 0 else 0
                ask = t.ask if t.ask and t.ask > 0 else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                oi = 0
                if hasattr(t, 'putOpenInterest') and t.putOpenInterest and t.putOpenInterest == t.putOpenInterest:
                    oi = int(t.putOpenInterest)

                iv = delta = gamma = vega = theta = None
                if t.modelGreeks:
                    iv = t.modelGreeks.impliedVol * 100 if t.modelGreeks.impliedVol else None
                    delta = t.modelGreeks.delta
                    gamma = t.modelGreeks.gamma
                    vega = t.modelGreeks.vega
                    theta = t.modelGreeks.theta

                if bid > 0:
                    put_data[strike] = {
                        'bid': bid, 'ask': ask, 'mid': mid,
                        'oi': oi, 'oi_grade': oi_grade(oi),
                        'iv': iv, 'delta': delta, 'gamma': gamma,
                        'vega': vega, 'theta': theta,
                        'otm_pct': (spot - strike) / spot * 100,
                    }

            # Collect call data (full: bid/ask/mid/OI/greeks for bear call spreads)
            call_data = {}
            for strike, t in call_tickers.items():
                bid = t.bid if t.bid and t.bid > 0 else 0
                ask = t.ask if t.ask and t.ask > 0 else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                oi = 0
                if hasattr(t, 'callOpenInterest') and t.callOpenInterest and t.callOpenInterest == t.callOpenInterest:
                    oi = int(t.callOpenInterest)

                iv = delta = gamma = vega = theta = None
                if t.modelGreeks:
                    iv = t.modelGreeks.impliedVol * 100 if t.modelGreeks.impliedVol else None
                    delta = t.modelGreeks.delta
                    gamma = t.modelGreeks.gamma
                    vega = t.modelGreeks.vega
                    theta = t.modelGreeks.theta

                call_data[strike] = {
                    'bid': bid, 'ask': ask, 'mid': mid,
                    'oi': oi, 'oi_grade': oi_grade(oi),
                    'iv': iv, 'delta': delta, 'gamma': gamma,
                    'vega': vega, 'theta': theta,
                    'otm_pct': (strike - spot) / spot * 100,
                }

            # Cancel all market data
            for t in put_tickers.values():
                ib.cancelMktData(t.contract)
            for t in call_tickers.values():
                ib.cancelMktData(t.contract)

            # === Compute bull put spreads ===
            sorted_strikes = sorted(put_data.keys(), reverse=True)
            spreads = []

            for i, short_k in enumerate(sorted_strikes):
                for long_k in sorted_strikes[i + 1:]:
                    width = short_k - long_k
                    if width < 1 or width > 15:
                        continue

                    short_d = put_data[short_k]
                    long_d = put_data[long_k]

                    mid_credit = short_d['mid'] - long_d['mid']
                    nat_credit = short_d['bid'] - long_d['ask']
                    if mid_credit <= 0:
                        continue

                    max_risk = width - mid_credit
                    ror = (mid_credit / max_risk) * 100 if max_risk > 0 else 0
                    ann_ror = ror * (365 / dte) if dte > 0 else 0
                    min_oi = min(short_d['oi'], long_d['oi'])

                    # Composite score
                    edge = max(info.get('edge_rv10', 0) or 0,
                               info.get('edge_rv30', 0) or 0)
                    rv10 = info.get('rv10', 0) or 0
                    rv30 = info.get('rv30', 0) or 0
                    rv_decline = max(0, (rv30 - rv10) / rv30 * 100) if rv30 > 0 else 0
                    otm = short_d['otm_pct']
                    drop = abs(info.get('drop_pct', 0))
                    bounce = info.get('bounce_pct', 0) or 0

                    # Support bonus: short strike near historical support
                    support_levels = info.get('support', [])
                    at_sup, _, sup_touches = near_support(short_k, support_levels)
                    sup_bonus = min((sup_touches or 0) / 5, 2.0) * 10 if at_sup else 0

                    score = (
                        min(edge / 20, 2.0) * 20 +          # edge: 20%
                        min(rv_decline / 30, 2.0) * 15 +    # rv decline: 15%
                        min(otm / 12, 2.0) * 15 +           # otm distance: 15%
                        min(min_oi / 3000, 2.0) * 15 +      # oi: 15%
                        min(drop / 20, 2.0) * 10 +          # drop magnitude: 10%
                        min(bounce / 5, 2.0) * 10 +         # bounce: 10%
                        sup_bonus                            # support: up to 15%
                    )

                    # Spread greeks (per contract, *100 for notional)
                    spread_delta = (short_d.get('delta') or 0) - (long_d.get('delta') or 0)
                    spread_gamma = (short_d.get('gamma') or 0) - (long_d.get('gamma') or 0)
                    spread_vega = (short_d.get('vega') or 0) - (long_d.get('vega') or 0)
                    spread_theta = (short_d.get('theta') or 0) - (long_d.get('theta') or 0)

                    spreads.append({
                        'short_strike': short_k,
                        'long_strike': long_k,
                        'width': width,
                        'mid_credit': mid_credit,
                        'nat_credit': nat_credit,
                        'max_risk': max_risk,
                        'ror': ror,
                        'ann_ror': ann_ror,
                        'short_otm_pct': otm,
                        'short_oi': short_d['oi'],
                        'long_oi': long_d['oi'],
                        'min_oi': min_oi,
                        'short_delta': short_d['delta'],
                        'short_iv': short_d['iv'],
                        'spread_delta': spread_delta,
                        'spread_gamma': spread_gamma,
                        'spread_vega': spread_vega,
                        'spread_theta': spread_theta,
                        'score': score,
                    })

            spreads.sort(key=lambda s: s['score'], reverse=True)

            # === Compute bear call spreads ===
            # Short lower-strike call, long higher-strike call
            call_strikes_sorted = sorted([k for k in call_data.keys() if call_data[k].get('bid', 0) > 0
                                          and call_data[k].get('otm_pct', 0) > 2])
            bear_call_spreads = []

            for i, short_k in enumerate(call_strikes_sorted):
                for long_k in call_strikes_sorted[i + 1:]:
                    width = long_k - short_k
                    if width < 1 or width > 15:
                        continue

                    short_d = call_data[short_k]
                    long_d = call_data[long_k]

                    mid_credit = short_d['mid'] - long_d['mid']
                    nat_credit = short_d['bid'] - long_d['ask']
                    if mid_credit <= 0:
                        continue

                    max_risk = width - mid_credit
                    ror = (mid_credit / max_risk) * 100 if max_risk > 0 else 0
                    ann_ror = ror * (365 / dte) if dte > 0 else 0
                    min_oi = min(short_d['oi'], long_d['oi'])

                    # Bear call composite score
                    edge = max(info.get('edge_rv10', 0) or 0,
                               info.get('edge_rv30', 0) or 0)
                    rv10 = info.get('rv10', 0) or 0
                    rv30 = info.get('rv30', 0) or 0
                    rv_decline = max(0, (rv30 - rv10) / rv30 * 100) if rv30 > 0 else 0
                    otm = short_d['otm_pct']
                    rise = info.get('rise_pct', 0) or 0
                    pullback = info.get('pullback_pct', 0) or 0

                    # Resistance bonus: short strike near historical resistance
                    resistance_levels = info.get('resistance', [])
                    at_res, _, res_touches = near_resistance(short_k, resistance_levels)
                    res_bonus = min((res_touches or 0) / 5, 2.0) * 10 if at_res else 0

                    bc_score = (
                        min(edge / 20, 2.0) * 20 +          # edge: 20%
                        min(rv_decline / 30, 2.0) * 15 +    # rv decline: 15%
                        min(otm / 12, 2.0) * 15 +           # otm distance: 15%
                        min(min_oi / 3000, 2.0) * 15 +      # oi: 15%
                        min(rise / 20, 2.0) * 10 +          # rally magnitude: 10%
                        min(pullback / 5, 2.0) * 10 +       # pullback: 10%
                        res_bonus                            # resistance: up to 15%
                    )

                    # Spread greeks (short - long, per contract)
                    bc_spread_delta = (short_d.get('delta') or 0) - (long_d.get('delta') or 0)
                    bc_spread_gamma = (short_d.get('gamma') or 0) - (long_d.get('gamma') or 0)
                    bc_spread_vega = (short_d.get('vega') or 0) - (long_d.get('vega') or 0)
                    bc_spread_theta = (short_d.get('theta') or 0) - (long_d.get('theta') or 0)

                    bear_call_spreads.append({
                        'short_strike': short_k,
                        'long_strike': long_k,
                        'width': width,
                        'mid_credit': mid_credit,
                        'nat_credit': nat_credit,
                        'max_risk': max_risk,
                        'ror': ror,
                        'ann_ror': ann_ror,
                        'short_otm_pct': otm,
                        'short_oi': short_d['oi'],
                        'long_oi': long_d['oi'],
                        'min_oi': min_oi,
                        'short_delta': short_d['delta'],
                        'short_iv': short_d['iv'],
                        'spread_delta': bc_spread_delta,
                        'spread_gamma': bc_spread_gamma,
                        'spread_vega': bc_spread_vega,
                        'spread_theta': bc_spread_theta,
                        'score': bc_score,
                        'type': 'bear_call',
                    })

            bear_call_spreads.sort(key=lambda s: s['score'], reverse=True)

            # Put OI walls: strikes with highest put OI (support)
            put_oi_walls = sorted(put_data.items(), key=lambda x: x[1]['oi'], reverse=True)[:3]

            # Call OI walls: strikes with highest call OI (resistance)
            call_oi_walls = sorted(call_data.items(), key=lambda x: x[1]['oi'], reverse=True)[:3]

            # Best spread across both types
            best_put = spreads[0] if spreads else None
            best_call = bear_call_spreads[0] if bear_call_spreads else None
            if best_put and best_call:
                best_overall_score = max(best_put['score'], best_call['score'])
            elif best_put:
                best_overall_score = best_put['score']
            elif best_call:
                best_overall_score = best_call['score']
            else:
                best_overall_score = 0

            results[sym] = {
                **info,
                'put_data': put_data,
                'call_data': call_data,
                'put_oi_walls': put_oi_walls,
                'call_oi_walls': call_oi_walls,
                'spreads': spreads[:5],
                'bear_call_spreads': bear_call_spreads[:5],
                'best_spread': best_put,
                'best_bear_call': best_call,
                'composite_score': best_overall_score,
            }

        except Exception as e:
            print(f'\n  Error scanning {sym}: {e}')

        ib.sleep(1)

    return results


# ============ Output ============

def print_phase1_table(results):
    """Print Phase 1 results sorted by drop magnitude."""
    sorted_r = sorted(results.items(), key=lambda x: x[1].get('drop_pct', 0))

    print(f'\n{"#":>3} {"Ticker":>7} {"Spot":>8} {"Drop%":>7} {"Rise%":>7} {"Dir":>5} {"RV10":>6} {"RV30":>6} '
          f'{"Decline":>8} {"Sharp":>6} {"AvgVol":>8}')
    print('-' * 95)

    for i, (sym, d) in enumerate(sorted_r):
        rv10 = d.get('rv10', 0) or 0
        rv30 = d.get('rv30', 0) or 0
        decline = ((rv30 - rv10) / rv30 * 100) if rv30 > 0 else 0
        direction = d.get('direction', 'put')
        dir_str = {'put': 'PUT', 'call': 'CALL', 'both': 'BOTH'}.get(direction, '?')
        sharp = 'YES' if d.get('is_sharp') or d.get('is_sharp_rally') else ''
        vol_m = d.get('avg_vol', 0) / 1e6

        print(f'{i+1:>3} {sym:>7} ${d["spot"]:>7.2f} {d["drop_pct"]:>6.1f}% {d.get("rise_pct", 0):>5.1f}% '
              f'{dir_str:>5} {rv10:>5.1f}% {rv30:>5.1f}% {decline:>6.1f}%  {sharp:>5} {vol_m:>6.1f}M')

    n_puts = sum(1 for _, d in results.items() if d.get('direction') in ('put', 'both'))
    n_calls = sum(1 for _, d in results.items() if d.get('direction') in ('call', 'both'))
    print(f'\n  {len(results)} stocks passed Phase 1 ({n_puts} put candidates, {n_calls} call candidates)')


def print_phase2_table(results):
    """Print Phase 2 results sorted by edge."""
    sorted_r = sorted(results.items(), key=lambda x: x[1].get('edge_rv30', 0), reverse=True)

    print(f'\n{"#":>3} {"Ticker":>7} {"Spot":>8} {"RV10":>6} {"RV30":>6} {"ATM IV":>7} '
          f'{"Edge10":>7} {"Edge30":>7} {"Drop%":>7} {"DTE":>4} {"Verdict":>10}')
    print('-' * 90)

    for i, (sym, d) in enumerate(sorted_r):
        rv10 = d.get('rv10', 0) or 0
        rv30 = d.get('rv30', 0) or 0
        edge10 = d.get('edge_rv10', 0)
        edge30 = d.get('edge_rv30', 0)

        if edge30 > 15:
            verdict = 'STRONG'
        elif edge30 > 5:
            verdict = 'good'
        elif edge30 > 0:
            verdict = 'slight'
        else:
            verdict = 'no edge'

        print(f'{i+1:>3} {sym:>7} ${d["spot"]:>7.2f} {rv10:>5.1f}% {rv30:>5.1f}% {d["atm_iv"]:>5.1f}%  '
              f'{edge10:>+5.1f}% {edge30:>+5.1f}% {d["drop_pct"]:>6.1f}% {d["dte"]:>4} {verdict:>10}')

    print(f'\n  {len(results)} stocks passed Phase 2 IV screen')


def print_phase3_report(sym, data):
    """Print detailed Phase 3 report for a single stock."""
    spot = data['spot']
    print(f'\n{"=" * 80}')
    print(f'  {sym} @ ${spot:.2f} | IV {data.get("atm_iv", 0):.1f}% | '
          f'RV10 {data.get("rv10", 0) or 0:.1f}% | RV30 {data.get("rv30", 0) or 0:.1f}% | '
          f'Edge {data.get("edge_rv30", 0):+.1f}%')
    print(f'  Drop {data.get("drop_pct", 0):.1f}% from ${data.get("high_60d", 0):.2f} ({data.get("days_since_high", 0)}d ago) | '
          f'Bounce {data.get("bounce_pct", 0):.1f}% | '
          f'{"SHARP" if data.get("is_sharp") else "gradual"} drop | '
          f'{data.get("expiry", "?")} ({data.get("dte", 0)}d)')
    print(f'{"=" * 80}')

    support_levels = data.get('support', [])
    resistance_levels = data.get('resistance', [])

    # Support / Resistance levels
    if support_levels or resistance_levels:
        print('\n  KEY LEVELS:')
        if support_levels:
            sup_str = '  '.join(f'${p:.1f}({c}x)' for p, c in support_levels[:4])
            print(f'    Support:    {sup_str}')
        if resistance_levels:
            res_str = '  '.join(f'${p:.1f}({c}x)' for p, c in resistance_levels[:4])
            print(f'    Resistance: {res_str}')

    # Put chain — annotate strikes near support
    put_data = data.get('put_data', {})
    if put_data:
        print('\n  PUT CHAIN:')
        print(f'  {"Strike":>8} {"OTM%":>6} {"Bid":>7} {"Ask":>7} {"Mid":>7} {"OI":>7} {"Grd":>4} {"IV":>7} {"Delta":>7}')
        print(f'  {"-" * 62}')
        for strike in sorted(put_data.keys(), reverse=True):
            d = put_data[strike]
            iv_str = f'{d["iv"]:.1f}%' if d.get('iv') else 'N/A'
            delta_str = f'{d["delta"]:.3f}' if d.get('delta') else 'N/A'
            at_sup, _, _ = near_support(strike, support_levels)
            sup_tag = ' <<SUP' if at_sup else ''
            print(f'  ${strike:>7.1f} {d["otm_pct"]:>5.1f}% ${d["bid"]:>5.2f} ${d["ask"]:>5.2f} '
                  f'${d["mid"]:>5.2f} {d["oi"]:>7} {d["oi_grade"]:>4} {iv_str:>7} {delta_str:>7}{sup_tag}')

    # Call chain — annotate strikes near resistance
    call_data = data.get('call_data', {})
    otm_calls = {k: v for k, v in call_data.items() if v.get('bid', 0) > 0 and v.get('otm_pct', 0) > 1}
    if otm_calls:
        print('\n  CALL CHAIN (OTM):')
        print(f'  {"Strike":>8} {"OTM%":>6} {"Bid":>7} {"Ask":>7} {"Mid":>7} {"OI":>7} {"Grd":>4} {"IV":>7} {"Delta":>7}')
        print(f'  {"-" * 62}')
        for strike in sorted(otm_calls.keys()):
            d = otm_calls[strike]
            iv_str = f'{d["iv"]:.1f}%' if d.get('iv') else 'N/A'
            delta_str = f'{d["delta"]:.3f}' if d.get('delta') else 'N/A'
            at_res, _, _ = near_resistance(strike, resistance_levels)
            res_tag = ' <<RES' if at_res else ''
            print(f'  ${strike:>7.1f} {d["otm_pct"]:>5.1f}% ${d["bid"]:>5.2f} ${d["ask"]:>5.2f} '
                  f'${d["mid"]:>5.2f} {d["oi"]:>7} {d["oi_grade"]:>4} {iv_str:>7} {delta_str:>7}{res_tag}')

    # OI walls
    call_walls = data.get('call_oi_walls', [])
    put_walls = data.get('put_oi_walls', [])
    if call_walls or put_walls:
        print('\n  OI WALLS:')
        if put_walls:
            for strike, d in put_walls:
                if d['oi'] > 0:
                    print(f'    PUT  ${strike:.1f} — OI {d["oi"]:,} ({d["oi_grade"]})  [support]')
        if call_walls:
            for strike, d in call_walls:
                if d['oi'] > 0:
                    print(f'    CALL ${strike:.1f} — OI {d["oi"]:,} ({d["oi_grade"]})  [resistance]')

    # Top bull put spreads — flag support-anchored trades
    spreads = data.get('spreads', [])
    if spreads:
        print('\n  BULL PUT SPREADS:')
        print(f'  {"Spread":>13} {"OTM%":>6} {"MidCr":>7} {"NatCr":>7} {"Width":>6} {"Risk":>7} '
              f'{"RoR":>6} {"AnnRoR":>8} {"OI(S/L)":>12} {"Score":>6}')
        print(f'  {"-" * 88}')
        for s in spreads[:3]:
            spread_str = f'{s["short_strike"]:.0f}/{s["long_strike"]:.0f}P'
            short_at_sup, sup_lvl, _ = near_support(s['short_strike'], support_levels)
            sup_note = f'  short@sup ${sup_lvl:.0f}' if short_at_sup else ''
            print(f'  {spread_str:>13} {s["short_otm_pct"]:>5.1f}% ${s["mid_credit"]:>5.2f} ${s["nat_credit"]:>5.2f} '
                  f'${s["width"]:>4.0f} ${s["max_risk"]:>5.2f} {s["ror"]:>5.1f}% {s["ann_ror"]:>6.0f}%/y '
                  f'{s["short_oi"]:>5}/{s["long_oi"]:<5} {s["score"]:>5.1f}{sup_note}')

    # Top bear call spreads — flag resistance-anchored trades
    bc_spreads = data.get('bear_call_spreads', [])
    if bc_spreads:
        print('\n  BEAR CALL SPREADS:')
        print(f'  {"Spread":>13} {"OTM%":>6} {"MidCr":>7} {"NatCr":>7} {"Width":>6} {"Risk":>7} '
              f'{"RoR":>6} {"AnnRoR":>8} {"OI(S/L)":>12} {"Score":>6}')
        print(f'  {"-" * 88}')
        for s in bc_spreads[:3]:
            spread_str = f'{s["short_strike"]:.0f}/{s["long_strike"]:.0f}C'
            short_at_res, res_lvl, _ = near_resistance(s['short_strike'], resistance_levels)
            res_note = f'  short@res ${res_lvl:.0f}' if short_at_res else ''
            print(f'  {spread_str:>13} {s["short_otm_pct"]:>5.1f}% ${s["mid_credit"]:>5.2f} ${s["nat_credit"]:>5.2f} '
                  f'${s["width"]:>4.0f} ${s["max_risk"]:>5.2f} {s["ror"]:>5.1f}% {s["ann_ror"]:>6.0f}%/y '
                  f'{s["short_oi"]:>5}/{s["long_oi"]:<5} {s["score"]:>5.1f}{res_note}')


def print_final_rankings(results):
    """Print final rankings sorted by composite score."""
    ranked = sorted(results.items(), key=lambda x: x[1].get('composite_score', 0), reverse=True)

    print(f'\n{"=" * 100}')
    print('FINAL RANKINGS — by composite score')
    print(f'{"=" * 100}')
    print(f'{"#":>3} {"Symbol":>7} {"Spot":>8} {"Drop%":>7} {"Rise%":>7} {"IV":>6} {"Edge":>6} '
          f'{"Best Put":>12} {"Best Call":>12} {"Score":>6}')
    print('-' * 90)

    for i, (sym, d) in enumerate(ranked):
        bp = d.get('best_spread')
        bc = d.get('best_bear_call')
        put_str = f'{bp["short_strike"]:.0f}/{bp["long_strike"]:.0f}P' if bp else '-'
        call_str = f'{bc["short_strike"]:.0f}/{bc["long_strike"]:.0f}C' if bc else '-'
        score = d.get('composite_score', 0)

        print(f'{i+1:>3} {sym:>7} ${d["spot"]:>7.2f} {d.get("drop_pct", 0):>6.1f}% '
              f'{d.get("rise_pct", 0):>5.1f}% '
              f'{d.get("atm_iv", 0):>5.1f}% {d.get("edge_rv30", 0):>+5.1f}% '
              f'{put_str:>12} {call_str:>12} {score:>5.1f}')


# ============ Portfolio Optimization ============

def _run_portfolio_opt(ib, phase3_results, total_capital):
    """Run Markowitz portfolio optimization on scan results."""
    from .portfolio import run_portfolio_optimization, print_allocation

    print(f'\n{"=" * 100}')
    print('PORTFOLIO OPTIMIZATION — Efficient Frontier')
    print(f'{"=" * 100}')

    result = run_portfolio_optimization(
        phase3_results,
        ib=ib,
        total_capital=total_capital,
    )
    if result:
        print_allocation(result)


# ============ CLI ============

def cmd_scan(args):
    """Main scan entry point."""
    filters = {
        'min_drop': getattr(args, 'min_drop', DEFAULT_FILTERS['min_drop']),
        'min_price': DEFAULT_FILTERS['min_price'],
        'min_vol': DEFAULT_FILTERS['min_vol'],
        'min_edge': getattr(args, 'min_edge', DEFAULT_FILTERS['min_edge']),
    }
    target_dte = getattr(args, 'dte', DEFAULT_FILTERS['target_dte'])
    top_n = getattr(args, 'top', DEFAULT_FILTERS['top'])
    deep_only = getattr(args, 'deep', None)
    phase1_only = getattr(args, 'phase1', False)
    use_cached = getattr(args, 'cached', False)
    use_quick = getattr(args, 'quick', False)
    run_portfolio = getattr(args, 'portfolio', False)
    total_capital = getattr(args, 'capital', 10000)

    print('=' * 100)
    print('MARKET SCANNER — Credit Spread Opportunities (Bull Puts + Bear Calls)')
    print('=' * 100)

    ib = scanner_connect()

    try:
        # === --deep MODE: skip phase 1, go straight to deep dive ===
        if deep_only:
            tickers = [t.strip().upper() for t in deep_only.split(',')]
            print(f'Deep dive mode: {", ".join(tickers)}')

            # Build minimal candidates — need spot + RV + expiry
            candidates = {}
            for sym in tickers:
                ibkr_sym = sym.replace('.', ' ')
                stock = Stock(ibkr_sym, 'SMART', 'USD')
                try:
                    ib.qualifyContracts(stock)
                    if not stock.conId:
                        print(f'  {sym}: could not qualify')
                        continue

                    bars = ib.reqHistoricalData(
                        stock, endDateTime='', durationStr='6 M',
                        barSizeSetting='1 day', whatToShow='TRADES',
                        useRTH=True, formatDate=1, timeout=10,
                    )
                    if not bars or len(bars) < 31:
                        print(f'  {sym}: insufficient data')
                        continue

                    closes = [b.close for b in bars]
                    spot = closes[-1]
                    rv10 = compute_rv(closes, 10)
                    rv20 = compute_rv(closes, 20)
                    rv30 = compute_rv(closes, 30)
                    rv60 = compute_rv(closes, 60)

                    recent_60 = bars[-60:] if len(bars) >= 60 else bars
                    high_60d = max(b.high for b in recent_60)
                    drop_pct = (spot - high_60d) / high_60d * 100
                    days_since_high = next(
                        (len(recent_60) - 1 - i for i, b in enumerate(recent_60) if b.high == high_60d), 0
                    )
                    low_60d = min(b.low for b in recent_60)
                    rise_pct = (spot - low_60d) / low_60d * 100 if low_60d > 0 else 0
                    days_since_low = next(
                        (len(recent_60) - 1 - i for i, b in enumerate(recent_60) if b.low == low_60d), 0
                    )
                    low_20d = min(b.low for b in bars[-20:])
                    bounce_pct = (spot - low_20d) / low_20d * 100
                    high_20d = max(b.high for b in bars[-20:])
                    pullback_pct = (high_20d - spot) / high_20d * 100
                    is_sharp = detect_sharp_drop(bars)
                    is_sharp_rally = detect_sharp_rally(bars)
                    support, resistance = compute_support_resistance(
                        bars[-90:] if len(bars) >= 90 else bars, spot)

                    # Get IV — use ranked expiries + batch qualify
                    today = date.today()
                    chain_params = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)
                    smart_chain = next((c for c in chain_params if c.exchange == 'SMART'), None)
                    if not smart_chain:
                        print(f'  {sym}: no SMART chain')
                        continue

                    expiries = _ranked_expiries(smart_chain, today, target_dte)
                    if not expiries:
                        print(f'  {sym}: no suitable expiry')
                        continue

                    all_strikes = sorted(smart_chain.strikes)
                    strikes_by_dist = sorted(all_strikes, key=lambda s: abs(s - spot))[:20]

                    atm_iv = None
                    atm_strike = None
                    best_exp = None
                    actual_dte = None
                    for exp, dte in expiries[:3]:
                        put_candidates = [Option(ibkr_sym, exp, s, 'P', 'SMART') for s in strikes_by_dist]
                        ib.qualifyContracts(*put_candidates)
                        qualified = [(p, abs(p.strike - spot)) for p in put_candidates if p.conId]
                        if not qualified:
                            continue
                        qualified.sort(key=lambda x: x[1])
                        atm_put = qualified[0][0]
                        t = ib.reqMktData(atm_put, genericTickList='106')
                        ib.sleep(4)
                        if t.modelGreeks and t.modelGreeks.impliedVol:
                            atm_iv = t.modelGreeks.impliedVol * 100
                            atm_strike = atm_put.strike
                            best_exp = exp
                            actual_dte = dte
                        ib.cancelMktData(atm_put)
                        if atm_iv is not None:
                            break

                    if not best_exp:
                        print(f'  {sym}: no IV data')
                        continue

                    candidates[sym] = {
                        'spot': spot,
                        'rv10': rv10, 'rv20': rv20, 'rv30': rv30, 'rv60': rv60,
                        'high_60d': high_60d, 'drop_pct': drop_pct,
                        'days_since_high': days_since_high,
                        'low_60d': low_60d, 'rise_pct': rise_pct,
                        'days_since_low': days_since_low,
                        'low_20d': low_20d, 'bounce_pct': bounce_pct,
                        'high_20d': high_20d, 'pullback_pct': pullback_pct,
                        'is_sharp': is_sharp, 'is_sharp_rally': is_sharp_rally,
                        'direction': 'both',
                        'support': support, 'resistance': resistance,
                        'avg_vol': sum(b.volume for b in bars[-20:]) / 20,
                        'expiry': best_exp, 'dte': actual_dte,
                        'atm_strike': atm_strike,
                        'atm_iv': atm_iv or 0,
                        'edge_rv10': (atm_iv or 0) - (rv10 or 0),
                        'edge_rv30': (atm_iv or 0) - (rv30 or 0),
                    }
                    print(f'  {sym}: ${spot:.2f} | IV {atm_iv or 0:.1f}% | RV30 {rv30 or 0:.1f}% | '
                          f'Edge {(atm_iv or 0) - (rv30 or 0):+.1f}% | {best_exp} ({actual_dte}d)')

                except Exception as e:
                    print(f'  {sym}: error — {e}')

            if not candidates:
                print('\nNo valid candidates.')
                return

            print(f'\n{"=" * 100}')
            print(f'PHASE 3: Deep scan {len(candidates)} stocks')
            print(f'{"=" * 100}')

            phase3 = phase3_deep_scan(ib, candidates, target_dte, progress_cb=print_progress)
            print()

            for sym in tickers:
                if sym in phase3:
                    print_phase3_report(sym, phase3[sym])

            if len(phase3) > 1:
                print_final_rankings(phase3)

            if run_portfolio and len(phase3) >= 2:
                _run_portfolio_opt(ib, phase3, total_capital)

            print('\nNOTE: Analysis only — no orders placed')
            return

        # === Normal multi-phase flow ===
        tickers = QUICK_LIST if use_quick else SP500_TICKERS
        mode = 'Quick' if use_quick else 'Full S&P 500'
        print(f'{mode}: {len(tickers)} stocks | min drop {filters["min_drop"]}% | '
              f'min edge {filters["min_edge"]}% | DTE ~{target_dte}')

        # Phase 1
        print(f'\n{"=" * 100}')
        print('PHASE 1: Historical volatility screen')
        print(f'{"=" * 100}')

        phase1 = None
        if use_cached:
            phase1, age = load_cache(1)
            if phase1:
                print(f'  Loaded from cache ({age}): {len(phase1)} stocks')
            else:
                print('  No valid cache, running phase 1...')

        if phase1 is None:
            phase1 = phase1_screen(ib, tickers, filters, progress_cb=print_progress)
            print(f'\n  Phase 1 complete: {len(phase1)} stocks passed filters')
            save_cache(phase1, 1)

        print_phase1_table(phase1)

        if phase1_only:
            print('\nPhase 1 only — done.')
            return

        # Phase 2
        print(f'\n{"=" * 100}')
        print(f'PHASE 2: IV screen at ~{target_dte} DTE')
        print(f'{"=" * 100}')

        phase2 = phase2_iv_screen(ib, phase1, target_dte, progress_cb=print_progress)
        print(f'\n  Phase 2 data: {len(phase2)} stocks got IV')

        # Filter by min edge — use max(edge_rv10, edge_rv30) so stocks with
        # declining RV (RV30 inflated by crash) still pass if edge_rv10 is positive
        min_edge = filters['min_edge']
        phase2_filtered = {
            sym: d for sym, d in phase2.items()
            if max(d.get('edge_rv10', -999), d.get('edge_rv30', -999)) >= min_edge
        }

        print_phase2_table(phase2_filtered)
        save_cache(phase2_filtered, 2)

        if not phase2_filtered:
            print('\nNo candidates passed edge filter. Try --min-edge 0')
            return

        # Phase 3 — rank by best available edge
        ranked = sorted(phase2_filtered.items(),
                        key=lambda x: max(x[1].get('edge_rv10', 0), x[1].get('edge_rv30', 0)),
                        reverse=True)
        top_for_deep = dict(ranked[:top_n])

        print(f'\n{"=" * 100}')
        print(f'PHASE 3: Deep scan top {len(top_for_deep)} stocks')
        print(f'{"=" * 100}')

        phase3 = phase3_deep_scan(ib, top_for_deep, target_dte, progress_cb=print_progress)
        print()

        for sym, d in sorted(phase3.items(), key=lambda x: x[1].get('composite_score', 0), reverse=True):
            print_phase3_report(sym, d)

        print_final_rankings(phase3)

        if run_portfolio and len(phase3) >= 2:
            _run_portfolio_opt(ib, phase3, total_capital)

        print('\nNOTE: Analysis only — no orders placed')

    finally:
        ib.disconnect()
