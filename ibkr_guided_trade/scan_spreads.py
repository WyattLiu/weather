#!/usr/bin/env python3
"""
S&P 500 Bull Put Spread Scanner

3-phase scan:
  Phase 1: Quick screen all 503 stocks — price, 30d realized vol, recent drop
  Phase 2: IV screen on candidates — ATM put IV at target DTE, compute IV-RV edge
  Phase 3: Deep scan on top picks — full put chain with OI, bid/ask, spread analysis

Usage:
    python scan_spreads.py                     # Full scan (~15 min)
    python scan_spreads.py --top 20            # Show top 20 results
    python scan_spreads.py --quick             # Scan curated high-IV list only
    python scan_spreads.py --ticker GOOGL      # Deep scan single ticker
    python scan_spreads.py --min-iv 25         # Min IV filter
    python scan_spreads.py --min-edge 3        # Min IV-RV edge filter
    python scan_spreads.py --dte 35            # Target DTE (default 35)
    python scan_spreads.py --save              # Save results to CSV
    python scan_spreads.py --skip-phase1       # Skip to phase 2 (use cached phase 1)
"""

import argparse
import csv
import json
import math
from datetime import datetime, date
from pathlib import Path

from ib_insync import IB, Stock, Option
from modules.common import IBKR_HOST, IBKR_PORT

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

DATA_DIR = Path(__file__).parent / 'scan_data'
CACHE_FILE = DATA_DIR / 'scan_phase1_cache.json'


def connect_ib(client_id=95):
    ib = IB()
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=client_id, timeout=30)
    return ib


def compute_realized_vol(bars, window_days=30):
    """Compute annualized realized volatility from daily bars using log returns."""
    if len(bars) < window_days + 1:
        return None
    recent = bars[-(window_days + 1):]
    closes = [b.close for b in recent]
    log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    if not log_returns:
        return None
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance * 252) * 100  # annualized, as percentage


def compute_recent_drop(bars, lookback_days=20):
    """Compute max drop from recent high over lookback period."""
    if len(bars) < lookback_days:
        return None
    recent = bars[-lookback_days:]
    highs = [b.high for b in recent]
    peak = max(highs)
    current = recent[-1].close
    return (current - peak) / peak * 100  # negative = drop


def phase1_screen(ib, tickers, progress_cb=None):
    """Phase 1: Quick screen — get price, 30d RV, recent drop for all stocks.

    Returns dict of {ticker: {spot, rv30, rv10, drop_20d, drop_5d}} for candidates.
    """
    results = {}
    total = len(tickers)

    # Process in batches to respect IBKR limits
    batch_size = 45  # stay under 50 simultaneous requests

    for batch_start in range(0, total, batch_size):
        batch = tickers[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        if progress_cb:
            progress_cb(f'Phase 1: Batch {batch_num}/{total_batches} ({batch[0]}-{batch[-1]})')

        # Create stock contracts
        stocks = []
        valid_symbols = []
        for sym in batch:
            # IBKR doesn't like dots in symbols for some
            ibkr_sym = sym.replace('.', ' ')
            stock = Stock(ibkr_sym, 'SMART', 'USD')
            stocks.append(stock)
            valid_symbols.append(sym)

        try:
            ib.qualifyContracts(*stocks)
        except Exception as e:
            print(f'  Qualify error batch {batch_num}: {e}')
            continue

        # Request historical data for each stock (1 year daily bars for RV)
        for i, stock in enumerate(stocks):
            sym = valid_symbols[i]
            if not stock.conId:
                continue

            try:
                bars = ib.reqHistoricalData(
                    stock,
                    endDateTime='',
                    durationStr='1 Y',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1,
                    timeout=10,
                )

                if not bars or len(bars) < 31:
                    continue

                spot = bars[-1].close
                rv30 = compute_realized_vol(bars, 30)
                rv10 = compute_realized_vol(bars, 10)
                drop_20d = compute_recent_drop(bars, 20)
                drop_5d = compute_recent_drop(bars, 5)

                results[sym] = {
                    'spot': spot,
                    'rv30': rv30,
                    'rv10': rv10,
                    'drop_20d': drop_20d,
                    'drop_5d': drop_5d,
                }

            except Exception as e:
                # Skip stocks that error out
                pass

            # Rate limit: IBKR allows ~60 historical data requests per 10 min
            # Sleep briefly between requests
            if i % 5 == 4:
                ib.sleep(1)

        ib.sleep(1)

    return results


def phase2_iv_screen(ib, candidates, target_dte=35, progress_cb=None):
    """Phase 2: Get ATM put IV for each candidate at target DTE.

    Computes IV-RV edge. Returns enriched candidates dict.
    """
    today = date.today()
    results = {}
    symbols = list(candidates.keys())
    total = len(symbols)

    for idx, sym in enumerate(symbols):
        if progress_cb and idx % 10 == 0:
            progress_cb(f'Phase 2: {idx}/{total} — {sym}')

        info = candidates[sym]
        spot = info['spot']

        ibkr_sym = sym.replace('.', ' ')
        stock = Stock(ibkr_sym, 'SMART', 'USD')

        try:
            ib.qualifyContracts(stock)
            if not stock.conId:
                continue

            # Get option chain params
            chains = ib.reqSecDefOptParams(stock.symbol, '', stock.secType, stock.conId)

            smart_chain = None
            for c in chains:
                if c.exchange == 'SMART':
                    smart_chain = c
                    break

            if not smart_chain:
                continue

            # Find expiry closest to target DTE
            best_exp = None
            best_dte_diff = 999
            for exp in smart_chain.expirations:
                exp_date = datetime.strptime(exp, '%Y%m%d').date()
                dte = (exp_date - today).days
                if 20 <= dte <= 50:  # reasonable range around target
                    diff = abs(dte - target_dte)
                    if diff < best_dte_diff:
                        best_dte_diff = diff
                        best_exp = exp

            if not best_exp:
                continue

            exp_date = datetime.strptime(best_exp, '%Y%m%d').date()
            actual_dte = (exp_date - today).days

            # Find ATM strike
            all_strikes = sorted(smart_chain.strikes)
            atm_strike = min(all_strikes, key=lambda s: abs(s - spot))

            # Also get a ~10% OTM strike for spread analysis
            otm_target = spot * 0.90
            otm_strike = min(all_strikes, key=lambda s: abs(s - otm_target))

            # Request ATM and OTM put data
            atm_put = Option(ibkr_sym, best_exp, atm_strike, 'P', 'SMART')
            otm_put = Option(ibkr_sym, best_exp, otm_strike, 'P', 'SMART')

            ib.qualifyContracts(atm_put, otm_put)

            tickers = ib.reqTickers(atm_put, otm_put)
            ib.sleep(2)

            atm_t = tickers[0]
            otm_t = tickers[1]

            # Extract IV from ATM put
            atm_iv = None
            atm_delta = None
            if atm_t.modelGreeks:
                atm_iv = atm_t.modelGreeks.impliedVol
                atm_delta = atm_t.modelGreeks.delta

            if atm_iv is None:
                continue

            atm_iv_pct = atm_iv * 100
            rv30 = info.get('rv30', 0) or 0
            iv_rv_edge = atm_iv_pct - rv30  # positive = IV overpriced = selling edge

            # OTM put data
            otm_bid = otm_t.bid if otm_t.bid and otm_t.bid > 0 else 0
            otm_ask = otm_t.ask if otm_t.ask and otm_t.ask > 0 else 0
            otm_iv = None
            otm_delta = None
            if otm_t.modelGreeks:
                otm_iv = otm_t.modelGreeks.impliedVol
                otm_delta = otm_t.modelGreeks.delta

            results[sym] = {
                **info,
                'expiry': best_exp,
                'dte': actual_dte,
                'atm_strike': atm_strike,
                'atm_iv': atm_iv_pct,
                'atm_delta': atm_delta,
                'otm_strike': otm_strike,
                'otm_bid': otm_bid,
                'otm_ask': otm_ask,
                'otm_iv': otm_iv * 100 if otm_iv else None,
                'otm_delta': otm_delta,
                'iv_rv_edge': iv_rv_edge,
            }

        except Exception:
            pass

        # Rate limit
        if idx % 3 == 2:
            ib.sleep(1)

    return results


def phase3_deep_scan(ib, candidates, target_dte=35, progress_cb=None):
    """Phase 3: Deep scan top candidates — full put chain with OI, bid/ask, spreads."""
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

            smart_chain = None
            for c in chains:
                if c.exchange == 'SMART':
                    smart_chain = c
                    break

            if not smart_chain:
                continue

            # Get puts from 5% OTM to 15% OTM
            all_strikes = sorted(smart_chain.strikes)
            target_strikes = [s for s in all_strikes if spot * 0.83 <= s <= spot * 0.97]

            if len(target_strikes) > 12:
                # Sample evenly
                step = max(1, len(target_strikes) // 12)
                target_strikes = target_strikes[::step]

            if not target_strikes:
                continue

            # Create and qualify put contracts — filter out invalid (half-dollar etc.)
            puts = [Option(ibkr_sym, expiry, s, 'P', 'SMART') for s in target_strikes]
            ib.qualifyContracts(*puts)
            puts = [p for p in puts if p.conId]  # only keep qualified contracts

            if not puts:
                continue

            # Stream market data with OI
            put_tickers = {}
            for p in puts:
                t = ib.reqMktData(p, genericTickList='101,106')
                put_tickers[p.strike] = t

            ib.sleep(5)

            # Collect data
            put_data = {}
            for strike, t in put_tickers.items():
                bid = t.bid if t.bid and t.bid > 0 else 0
                ask = t.ask if t.ask and t.ask > 0 else 0
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                vol = int(t.volume) if t.volume and t.volume == t.volume and t.volume >= 0 else 0
                oi = 0
                if hasattr(t, 'putOpenInterest') and t.putOpenInterest and t.putOpenInterest == t.putOpenInterest:
                    oi = int(t.putOpenInterest)

                iv = delta = None
                if t.modelGreeks:
                    iv = t.modelGreeks.impliedVol * 100 if t.modelGreeks.impliedVol else None
                    delta = t.modelGreeks.delta if t.modelGreeks.delta else None

                if bid > 0:
                    put_data[strike] = {
                        'bid': bid, 'ask': ask, 'mid': mid,
                        'volume': vol, 'oi': oi,
                        'iv': iv, 'delta': delta,
                        'otm_pct': (spot - strike) / spot * 100,
                    }

            # Cancel market data
            for t in put_tickers.values():
                ib.cancelMktData(t.contract)

            # Compute best spreads
            sorted_strikes = sorted(put_data.keys(), reverse=True)
            spreads = []

            for i, short_k in enumerate(sorted_strikes):
                for long_k in sorted_strikes[i + 1:]:
                    width = short_k - long_k
                    if width < 2.5 or width > 15:
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
                    spread_width = (short_d['ask'] - short_d['bid']) + (long_d['ask'] - long_d['bid'])

                    spreads.append({
                        'short_strike': short_k,
                        'long_strike': long_k,
                        'width': width,
                        'mid_credit': mid_credit,
                        'nat_credit': nat_credit,
                        'max_risk': max_risk,
                        'ror': ror,
                        'ann_ror': ann_ror,
                        'short_otm_pct': short_d['otm_pct'],
                        'short_oi': short_d['oi'],
                        'long_oi': long_d['oi'],
                        'min_oi': min_oi,
                        'short_delta': short_d['delta'],
                        'short_iv': short_d['iv'],
                        'spread_width': spread_width,
                    })

            # Rank spreads: balance RoR, OTM distance, liquidity
            for s in spreads:
                otm_score = min(s['short_otm_pct'] / 10, 1.5) * 20  # prefer 8-15% OTM
                ror_score = min(s['ror'] / 30, 1.5) * 20
                oi_score = min(s['min_oi'] / 2000, 1.5) * 15
                edge_score = max(0, info.get('iv_rv_edge', 0)) * 2
                s['score'] = otm_score + ror_score + oi_score + edge_score

            spreads.sort(key=lambda s: s['score'], reverse=True)

            results[sym] = {
                **info,
                'put_data': put_data,
                'spreads': spreads[:5],  # top 5 spreads
                'best_spread': spreads[0] if spreads else None,
            }

        except Exception as e:
            print(f'  Error scanning {sym}: {e}')

        ib.sleep(1)

    return results


def print_progress(msg):
    print(f'\r  {msg}' + ' ' * 20, end='', flush=True)


def save_cache(data, filepath):
    """Save phase 1 results to JSON cache."""
    DATA_DIR.mkdir(exist_ok=True)
    # Convert to serializable format
    serializable = {}
    for k, v in data.items():
        serializable[k] = {key: val for key, val in v.items()}
    with open(filepath, 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'data': serializable}, f, indent=2)


def load_cache(filepath, max_age_hours=4):
    """Load phase 1 cache if fresh enough."""
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            cached = json.load(f)
        ts = datetime.fromisoformat(cached['timestamp'])
        age = (datetime.now() - ts).total_seconds() / 3600
        if age > max_age_hours:
            return None
        return cached['data']
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='S&P 500 Bull Put Spread Scanner')
    parser.add_argument('--quick', action='store_true', help='Scan curated watchlist only (~50 stocks)')
    parser.add_argument('--ticker', type=str, help='Deep scan single ticker')
    parser.add_argument('--top', type=int, default=15, help='Show top N results (default 15)')
    parser.add_argument('--min-iv', type=float, default=20, help='Min ATM IV %% (default 20)')
    parser.add_argument('--min-edge', type=float, default=0, help='Min IV-RV edge %% (default 0)')
    parser.add_argument('--dte', type=int, default=35, help='Target DTE (default 35)')
    parser.add_argument('--save', action='store_true', help='Save results to CSV')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip phase 1, use cache')
    parser.add_argument('--phase1-only', action='store_true', help='Run phase 1 only')
    parser.add_argument('--deep', type=int, default=10, help='Number of stocks for phase 3 deep scan (default 10)')
    args = parser.parse_args()

    print('=' * 100)
    print('S&P 500 BULL PUT SPREAD SCANNER')
    print('=' * 100)

    ib = connect_ib()

    try:
        # Determine ticker list
        if args.ticker:
            tickers = [args.ticker.upper()]
            print(f'Single ticker mode: {args.ticker.upper()}')
        elif args.quick:
            # Filter quick list to only S&P 500 members + extras
            tickers = QUICK_LIST
            print(f'Quick mode: {len(tickers)} stocks')
        else:
            tickers = SP500_TICKERS
            print(f'Full scan mode: {len(tickers)} stocks')

        # ============ PHASE 1: Quick Screen ============
        print(f'\n{"=" * 100}')
        print('PHASE 1: Historical volatility screen')
        print(f'{"=" * 100}')

        phase1 = None
        if args.skip_phase1:
            phase1 = load_cache(CACHE_FILE)
            if phase1:
                print(f'  Loaded cache: {len(phase1)} stocks')
            else:
                print('  No valid cache found, running phase 1...')

        if phase1 is None:
            phase1 = phase1_screen(ib, tickers, progress_cb=print_progress)
            print(f'\n  Phase 1 complete: {len(phase1)} stocks with data')
            save_cache(phase1, CACHE_FILE)
            print(f'  Cache saved to {CACHE_FILE}')

        if args.phase1_only:
            # Print phase 1 results sorted by RV
            sorted_p1 = sorted(phase1.items(), key=lambda x: x[1].get('rv30', 0) or 0, reverse=True)
            print(f'\n{"Ticker":>8} {"Spot":>8} {"RV30":>7} {"RV10":>7} {"Drop20":>8} {"Drop5":>8}')
            print('-' * 60)
            for sym, d in sorted_p1[:50]:
                rv30 = d.get('rv30', 0) or 0
                rv10 = d.get('rv10', 0) or 0
                drop20 = d.get('drop_20d', 0) or 0
                drop5 = d.get('drop_5d', 0) or 0
                print(f'{sym:>8} ${d["spot"]:>7.2f} {rv30:>6.1f}% {rv10:>6.1f}% {drop20:>7.1f}% {drop5:>7.1f}%')
            ib.disconnect()
            return

        # Filter phase 1: must have RV data, price > $5
        phase1_filtered = {
            sym: d for sym, d in phase1.items()
            if d.get('rv30') and d['spot'] > 5
        }
        print(f'  After filtering: {len(phase1_filtered)} stocks')

        # ============ PHASE 2: IV Screen ============
        print(f'\n{"=" * 100}')
        print(f'PHASE 2: IV screen at ~{args.dte} DTE')
        print(f'{"=" * 100}')

        phase2 = phase2_iv_screen(ib, phase1_filtered, target_dte=args.dte, progress_cb=print_progress)
        print(f'\n  Phase 2 complete: {len(phase2)} stocks with IV data')

        # Filter: min IV, min edge
        phase2_filtered = {
            sym: d for sym, d in phase2.items()
            if d.get('atm_iv', 0) >= args.min_iv
            and d.get('iv_rv_edge', -999) >= args.min_edge
        }

        # Sort by IV-RV edge (selling edge)
        ranked = sorted(phase2_filtered.items(), key=lambda x: x[1].get('iv_rv_edge', 0), reverse=True)

        print(f'\n  Filtered: {len(ranked)} stocks with IV >= {args.min_iv}% and edge >= {args.min_edge}%')

        # Print phase 2 results
        print(f'\n{"=" * 100}')
        print(f'PHASE 2 RESULTS — Ranked by IV-RV Edge (selling edge)')
        print(f'{"=" * 100}')
        print(f'{"#":>3} {"Ticker":>8} {"Spot":>8} {"RV30":>7} {"RV10":>7} {"ATM IV":>8} {"Edge":>7} {"Drop20":>8} {"DTE":>4} {"Expiry":>10}')
        print('-' * 90)

        for i, (sym, d) in enumerate(ranked[:args.top * 2]):
            rv30 = d.get('rv30', 0) or 0
            rv10 = d.get('rv10', 0) or 0
            atm_iv = d.get('atm_iv', 0)
            edge = d.get('iv_rv_edge', 0)
            drop20 = d.get('drop_20d', 0) or 0
            edge_flag = '+' if edge > 0 else ''

            print(f'{i+1:>3} {sym:>8} ${d["spot"]:>7.2f} {rv30:>6.1f}% {rv10:>6.1f}% {atm_iv:>6.1f}% {edge_flag}{edge:>5.1f}% {drop20:>7.1f}% {d["dte"]:>4} {d["expiry"]:>10}')

        # ============ PHASE 3: Deep Scan ============
        top_for_deep = dict(ranked[:args.deep])

        if not top_for_deep:
            print('\nNo candidates passed filters. Try lowering --min-iv or --min-edge.')
            ib.disconnect()
            return

        print(f'\n{"=" * 100}')
        print(f'PHASE 3: Deep scan top {len(top_for_deep)} stocks')
        print(f'{"=" * 100}')

        phase3 = phase3_deep_scan(ib, top_for_deep, target_dte=args.dte, progress_cb=print_progress)
        print()

        # Final results
        print(f'\n{"=" * 100}')
        print(f'FINAL RESULTS — Best Bull Put Spreads')
        print(f'{"=" * 100}')

        # Collect all best spreads with stock context
        all_spreads = []
        for sym, d in phase3.items():
            if d.get('best_spread'):
                bs = d['best_spread']
                all_spreads.append({
                    'symbol': sym,
                    'spot': d['spot'],
                    'atm_iv': d.get('atm_iv', 0),
                    'rv30': d.get('rv30', 0) or 0,
                    'iv_rv_edge': d.get('iv_rv_edge', 0),
                    'expiry': d['expiry'],
                    'dte': d['dte'],
                    **bs,
                })

        # Sort by composite score
        all_spreads.sort(key=lambda s: s.get('score', 0), reverse=True)

        print(f'\n{"#":>3} {"Symbol":>7} {"Spread":>13} {"Spot":>8} {"OTM%":>6} {"MidCr":>7} {"NatCr":>7} {"Risk":>7} {"RoR":>6} {"OI(S/L)":>13} {"IV":>6} {"RV30":>6} {"Edge":>6} {"DTE":>4}')
        print('-' * 120)

        for i, s in enumerate(all_spreads[:args.top]):
            spread_str = f'{s["short_strike"]:.0f}/{s["long_strike"]:.0f}P'
            edge_flag = '+' if s['iv_rv_edge'] > 0 else ''

            print(f'{i+1:>3} {s["symbol"]:>7} {spread_str:>13} ${s["spot"]:>7.2f} {s["short_otm_pct"]:>5.1f}% ${s["mid_credit"]:>5.2f} ${s["nat_credit"]:>5.2f} ${s["max_risk"]:>5.2f} {s["ror"]:>5.1f}% {s["short_oi"]:>6}/{s["long_oi"]:<6} {s.get("short_iv",0) or 0:>5.1f}% {s["rv30"]:>5.1f}% {edge_flag}{s["iv_rv_edge"]:>4.1f}% {s["dte"]:>4}')

        # Print detailed view for top 5
        print(f'\n{"=" * 100}')
        print('DETAILED VIEW — Top 5')
        print(f'{"=" * 100}')

        for i, (sym, d) in enumerate(list(phase3.items())[:5]):
            if not d.get('spreads'):
                continue

            print(f'\n--- {sym} @ ${d["spot"]:.2f} | IV {d.get("atm_iv",0):.1f}% | RV30 {d.get("rv30",0) or 0:.1f}% | Edge {d.get("iv_rv_edge",0):+.1f}% | {d["expiry"]} ({d["dte"]}d) ---')
            print(f'  {"Spread":>13} {"OTM%":>6} {"MidCr":>7} {"NatCr":>7} {"Width":>6} {"Risk":>7} {"RoR":>6} {"AnnRoR":>8} {"OI(S/L)":>13} {"Delta":>7}')

            for s in d['spreads']:
                spread_str = f'{s["short_strike"]:.0f}/{s["long_strike"]:.0f}P'
                delta_str = f'{s["short_delta"]:.3f}' if s.get('short_delta') else 'N/A'
                print(f'  {spread_str:>13} {s["short_otm_pct"]:>5.1f}% ${s["mid_credit"]:>5.2f} ${s["nat_credit"]:>5.2f} ${s["width"]:>4.0f} ${s["max_risk"]:>5.2f} {s["ror"]:>5.1f}% {s["ann_ror"]:>6.0f}%/y {s["short_oi"]:>6}/{s["long_oi"]:<6} {delta_str:>7}')

        # Save to CSV if requested
        if args.save and all_spreads:
            DATA_DIR.mkdir(exist_ok=True)
            csv_file = DATA_DIR / f'scan_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'symbol', 'spot', 'expiry', 'dte', 'atm_iv', 'rv30', 'iv_rv_edge',
                    'short_strike', 'long_strike', 'width', 'short_otm_pct',
                    'mid_credit', 'nat_credit', 'max_risk', 'ror', 'ann_ror',
                    'short_oi', 'long_oi', 'short_iv', 'short_delta', 'score',
                ])
                writer.writeheader()
                for s in all_spreads:
                    writer.writerow({k: s.get(k, '') for k in writer.fieldnames})
            print(f'\nResults saved to {csv_file}')

        print(f'\nNOTE: Analysis only — no orders placed')

    finally:
        ib.disconnect()


if __name__ == '__main__':
    main()
