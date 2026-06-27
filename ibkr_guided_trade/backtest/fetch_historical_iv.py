"""Fetch UNG historical 30-day IV from IBKR via reqHistoricalData.

IBKR exposes the underlying's IV30 series directly (similar to VIX for SPY)
via `whatToShow='OPTION_IMPLIED_VOLATILITY'`. This replaces our calibrated
`rv * 1.12` proxy with actual market IV — improves option pricing accuracy
especially for tail-strike pricing where realized-vol approximation breaks down.

Usage:
  cd /home/wyatt/weather/ibkr_guided_trade
  ../venv/bin/python backtest/fetch_historical_iv.py

Writes: backtest/cache/ung_historical_iv.csv (date, iv30)
The bulk runner can join this onto master_dataset.csv to override the proxy.

Notes:
- IBKR caps history at the contract's age; UNG IV30 should be available
  back to ~2011 (UNG inception).
- Requires TWS/IB Gateway running with API enabled on localhost:7497/4001.
- WhatToShow='OPTION_IMPLIED_VOLATILITY' is the underlying's 30-day IV
  computed from a constant-maturity strike-weighted basket — same family
  as VIX but for the underlying.
"""
import os
import sys
import pandas as pd
from datetime import datetime

try:
    from ib_insync import IB, Stock
except ImportError:
    print("ib_insync not installed. pip install ib_insync")
    sys.exit(1)

# Use the canonical IBKR endpoint from modules.common (192.168.1.127:20009)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from modules.common import IBKR_HOST, IBKR_PORT
except ImportError:
    IBKR_HOST, IBKR_PORT = '192.168.1.127', 20009

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_iv30(symbol='UNG', years=6, host=None, port=None, client_id=42):
    if host is None: host = IBKR_HOST
    if port is None: port = IBKR_PORT
    """Pull underlying IV30 series from IBKR."""
    ib = IB()
    print(f"Connecting to TWS at {host}:{port}...")
    try:
        ib.connect(host, port, clientId=client_id, timeout=15)
    except Exception as e:
        print(f"Failed to connect: {e}")
        print("Make sure TWS/IB Gateway is running with API enabled.")
        return None

    contract = Stock(symbol, 'ARCA', 'USD')
    ib.qualifyContracts(contract)

    print(f"Fetching {years}y of IV30 for {symbol}...")
    # IBKR caps single request at ~365 days for 1-day bars; loop in 1y chunks
    all_bars = []
    end = ''  # most recent
    for chunk in range(years):
        try:
            bars = ib.reqHistoricalData(
                contract, endDateTime=end, durationStr='365 D',
                barSizeSetting='1 day', whatToShow='OPTION_IMPLIED_VOLATILITY',
                useRTH=True, formatDate=1, timeout=30,
            )
            if not bars:
                print(f"  Chunk {chunk+1}: no data returned, stopping")
                break
            all_bars.extend(bars)
            print(f"  Chunk {chunk+1}: {len(bars)} bars ({bars[0].date} → {bars[-1].date})")
            # Move endDateTime back by 1y for next chunk
            earliest = bars[0].date
            if isinstance(earliest, str):
                earliest = datetime.strptime(earliest, '%Y%m%d').date()
            end = earliest.strftime('%Y%m%d %H:%M:%S')
        except Exception as e:
            print(f"  Chunk {chunk+1} failed: {e}")
            break

    ib.disconnect()

    if not all_bars:
        print("No data fetched")
        return None

    # Deduplicate by date (IBKR may overlap chunks)
    rows = {}
    for b in all_bars:
        d = b.date if hasattr(b, 'date') else None
        if d is not None:
            rows[d] = b.close
    df = pd.DataFrame.from_dict(rows, orient='index', columns=['iv30'])
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    out_path = os.path.join(CACHE_DIR, f'{symbol.lower()}_historical_iv.csv')
    df.to_csv(out_path)
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"IV30 range: {df['iv30'].min():.3f} → {df['iv30'].max():.3f}")
    print(f"Median IV30: {df['iv30'].median():.3f}")
    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='UNG')
    parser.add_argument('--years', type=int, default=6)
    parser.add_argument('--host', default=IBKR_HOST)
    parser.add_argument('--port', type=int, default=IBKR_PORT)
    parser.add_argument('--client-id', type=int, default=42)
    args = parser.parse_args()
    fetch_iv30(args.symbol, args.years, host=args.host, port=args.port, client_id=args.client_id)
