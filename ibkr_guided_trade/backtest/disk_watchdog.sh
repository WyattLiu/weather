#!/bin/bash
# Auto-stop the ETF minute backfill if free disk drops below the floor (protects PG + data,
# since PG is on this same /dev/sda2). Checks every 2 min while a backfill is running.
FLOOR_KB=62914560   # 60 GB
LOG=/home/wyatt/weather/ibkr_guided_trade/backtest/log/etf_minute_chain.log
while pgrep -f backfill_etf_intraday >/dev/null 2>&1; do
  avail=$(df --output=avail / | tail -1 | tr -d ' ')
  if [ "${avail:-0}" -lt "$FLOOR_KB" ]; then
    echo "=== WATCHDOG $(date): free ${avail}KB < 60G floor — KILLING backfill to protect disk ===" >> "$LOG"
    pkill -9 -f backfill_etf_intraday; pkill -9 -f etf_minute_chain
    break
  fi
  sleep 120
done
