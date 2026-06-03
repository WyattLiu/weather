#!/bin/bash
# Continuous backtest cycle — runs replay + ablation + bug watch,
# commits results if anything changed, reports anomalies.
#
# Run via the Claude Code cron (or system cron) every few hours.

set -u
cd "$(dirname "$0")/.."

VENV=venv/bin/python
LOG_DIR=backtest/results/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/cycle_$TS.log"

echo "=== Backtest cycle $TS ===" | tee "$LOG"

# 1) Run main backtest
echo "--- replay_engine ---" | tee -a "$LOG"
$VENV backtest/replay_engine.py >> "$LOG" 2>&1
REPLAY_EXIT=$?

# 2) Ablation on top strategies
for strat in elevator_close_surprise regime_aware_surprise regime_aware_roll_up; do
    echo "--- ablation:$strat ---" | tee -a "$LOG"
    $VENV backtest/ablation.py --strategy "$strat" >> "$LOG" 2>&1
done

# 3) Bug watch (exit code 1 = errors found)
echo "--- bug_watch ---" | tee -a "$LOG"
$VENV backtest/bug_watch.py 2>&1 | tee -a "$LOG"
BUG_EXIT=${PIPESTATUS[0]}

# 4) If results changed, commit
if [ -n "$(git status --porcelain backtest/results/)" ]; then
    echo "--- committing results ---" | tee -a "$LOG"
    git add backtest/results/
    git commit -m "Backtest cycle $TS (replay_exit=$REPLAY_EXIT bug_exit=$BUG_EXIT)" \
        >> "$LOG" 2>&1
fi

# 4.5) Strategy lifecycle: retire bottom-Sharpe strategies if active set
# exceeds target + threshold. Marks (not deletes) — strategies stay in code.
echo "--- retire_stale ---" | tee -a "$LOG"
$VENV backtest/retire_stale.py 2>&1 | tee -a "$LOG"
if [ -n "$(git status --porcelain backtest/strategy_lifecycle.json)" ]; then
    git add backtest/strategy_lifecycle.json
    git commit -m "Retire stale strategies (cycle $TS)" >> "$LOG" 2>&1
fi

# 5) Tail summary for the operator (also stays in log)
echo "--- summary ---"
$VENV -c "
import json
with open('backtest/results/summary.json') as f: s = json.load(f)
for k, v in sorted(s.items()):
    if isinstance(v, dict) and 'return_pct' in v:
        print(f\"  {k:<28} {v['return_pct']:>+7.1f}%  Sharpe {v.get('sharpe',0):+.2f}\")
"

exit $BUG_EXIT
