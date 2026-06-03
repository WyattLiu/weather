#!/bin/bash
# Daily UNG IV30 maintenance — appends today's IV from IBKR.
# Call from cron. Idempotent: re-fetches last 30 days to fill any gaps.
#
# Cron suggestion:
#   30 16 * * 1-5  cd /home/wyatt/weather/ibkr_guided_trade && bash backtest/maintain_iv.sh
# (4:30pm ET on weekdays — after market close)

set -e
cd "$(dirname "$0")/.."
VENV=../venv/bin/python

# Pull latest 30d (cheap; idempotent on dates)
$VENV backtest/fetch_historical_iv.py --years 1 --client-id 43 2>&1 | tail -10

# Merge into master_dataset.csv on next pipeline run
$VENV backtest/historical_data_pipeline.py 2>&1 | tail -5

echo "[$(date +%F\ %T)] IV maintenance complete"
