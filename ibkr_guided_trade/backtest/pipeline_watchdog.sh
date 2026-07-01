#!/bin/bash
# 15-minute data-pipeline watchdog. Runs pipeline_health.py, writes a machine-readable status
# (cache/pipeline_health_status.json) the dashboard can surface, and logs an ALERT block on RED or
# on a fresh transition into WARN. This is the monitoring layer the pipeline was missing — a stale
# feed / missed refresh / look-ahead regression is now caught within 15 min instead of by accident.
#
# Install:  crontab entry ->  */15 * * * * /home/wyatt/weather/ibkr_guided_trade/backtest/pipeline_watchdog.sh
set -u
ROOT=/home/wyatt/weather/ibkr_guided_trade
LOG="$ROOT/backtest/log/pipeline_watchdog.log"
STATE="$ROOT/backtest/cache/.watchdog_last_verdict"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT" || exit 1

OUT=$("$ROOT/venv/bin/python" backtest/pipeline_health.py --quiet --status 2>&1)
RC=$?
TS=$(date '+%Y-%m-%d %H:%M:%S')
case $RC in
  0) V=GREEN ;;
  1) V=WARN ;;
  2) V=RED ;;
  *) V=ERROR ;;
esac
PREV=$(cat "$STATE" 2>/dev/null || echo NONE)
echo "$V" > "$STATE"
echo "$TS $V" >> "$LOG"   # heartbeat every run

# Alert on RED/ERROR (every run while broken) or on a NEW transition into WARN.
if [ "$V" = "RED" ] || [ "$V" = "ERROR" ] || { [ "$V" = "WARN" ] && [ "$PREV" != "WARN" ] && [ "$PREV" != "RED" ]; }; then
  {
    echo "$TS ==== PIPELINE ALERT [$V] (was $PREV) ===="
    echo "$OUT" | sed 's/^/    /'
    echo ""
  } >> "$LOG"
fi

# keep the log from growing unbounded (tail to last 2000 lines)
tail -n 2000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
exit 0
