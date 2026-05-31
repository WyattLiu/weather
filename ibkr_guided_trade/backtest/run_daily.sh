#!/bin/bash
# Daily backtest refresh — runs data pipeline + replay engine + commits results
# Suitable for cron:
#   30 19 * * 1-5 /home/wyatt/ibkr_guided_trade/backtest/run_daily.sh
# (7:30pm weekdays = after market close + data refresh)

set -e

cd /home/wyatt/ibkr_guided_trade

LOG=/tmp/backtest_daily.log
echo "=== Backtest refresh $(date) ===" > $LOG

# 1. Refresh historical data
./venv/bin/python backtest/historical_data_pipeline.py --years 5 >> $LOG 2>&1

# 2. Run replay engine on all strategies
./venv/bin/python backtest/replay_engine.py --compare >> $LOG 2>&1

# 3. Sync to weather repo (mirror)
rsync -av --exclude='.git' --exclude='venv' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='*.har' --exclude='cookies*' \
    --exclude='*.png' --exclude='*.log' --exclude='.claude' \
    /home/wyatt/ibkr_guided_trade/backtest/ \
    /home/wyatt/weather/ibkr_guided_trade/backtest/ >> $LOG 2>&1

# 4. Git commit + push
cd /home/wyatt/weather
git add ibkr_guided_trade/backtest/ >> $LOG 2>&1 || true
if git diff --cached --quiet; then
    echo "No backtest changes to commit" >> $LOG
else
    git commit -m "Daily backtest refresh ($(date +%Y-%m-%d))

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>" >> $LOG 2>&1
    git push origin master >> $LOG 2>&1 || true
fi

echo "=== Done $(date) ===" >> $LOG
