#!/bin/bash
# Daily HH spot refresh — pulls latest EIA Henry Hub daily spot prices
# and rebuilds master_dataset. Standalone (no need to run full pipeline).
#
# Cron: 30 17 * * 1-5  (5:30pm ET weekdays, after EIA daily publishes)
set -u
cd "$(dirname "$0")/.."
VENV=venv/bin/python

# Force-refresh EIA HH spot xls (remove cache)
rm -f backtest/cache/eia_hh_spot_daily.xls

# Rerun pipeline (will only re-download EIA monthly/weekly stale files;
# HH spot will refresh because we just deleted it)
$VENV backtest/historical_data_pipeline.py --years 5 2>&1 | grep -E "hh_spot|HH|basis|hh_basis" | head

# Quick sanity check
$VENV -c "
import pandas as pd
df = pd.read_csv('backtest/cache/master_dataset.csv', index_col=0, parse_dates=True)
hh = df['eia_hh_spot_daily'].dropna()
ng = df['NG'].dropna()
basis = (hh - ng).dropna()
latest = basis.dropna().tail(1)
print(f'Latest HH spot: \${hh.iloc[-1]:.2f}  NG futures: \${ng.iloc[-1]:.2f}  basis: \${(hh.iloc[-1]-ng.iloc[-1]):+.2f}')
if (hh.iloc[-1] - ng.iloc[-1]) > 0.40:
    print('🌩 BACKWARDATION STORM ACTIVE — defensive mode triggered')
else:
    print('✓ Basis normal — no defensive signal')
"
