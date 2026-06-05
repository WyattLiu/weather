#!/bin/bash
# Cron-suitable: monitor account + execute best play (PAPER by default).
#
# Replaces backtest/run_cycle.sh in the Claude Code cron. Backtest cycles
# are no longer needed for daily optimization (engine state stable).
#
# To enable LIVE submission, set both:
#   export KERNEL_LIVE=1
# and pass --live flag below.
#
# Conservative defaults:
#  - Submits ONLY passive tier of the limit ladder (best premium for us)
#  - Lock file prevents double-execution if cron overlaps
#  - All decisions appended to live/log/trading_actions.jsonl

set -u
cd "$(dirname "$0")/.."

VENV=venv/bin/python
LOG_DIR=live/log
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$LOG_DIR/run_$TS.log"

echo "=== Kernel monitor + execute $TS ===" | tee "$RUN_LOG"

# 1) Ensure dashboard is up (it provides /api/state which the executor reads)
if ! curl -sf --max-time 3 http://127.0.0.1:10001/ -o /dev/null; then
    echo "WARN: dashboard not responding on :10001 — starting it" | tee -a "$RUN_LOG"
    nohup $VENV kernel_dashboard.py > /tmp/kdash.log 2>&1 &
    disown
    for i in $(seq 1 30); do
        if curl -sf --max-time 1 http://127.0.0.1:10001/ -o /dev/null 2>/dev/null; then
            break
        fi
        sleep 1
    done
fi

# 2) Trigger dashboard refresh so verdict reflects current account
curl -s --max-time 15 http://127.0.0.1:10001/api/refresh > /dev/null 2>&1 || true
sleep 2

# 3) Mode selection (auto = will submit up to daily cap if KERNEL_LIVE=1)
MODE="${KERNEL_MODE:-paper}"          # paper | review | auto
DAILY_MAX="${KERNEL_DAILY_MAX:-4}"    # cap orders/24h in auto mode
if [ "${KERNEL_LIVE:-0}" = "1" ] && [ "$MODE" = "auto" ]; then
    echo "🔴 AUTO+LIVE — will submit up to $DAILY_MAX orders/24h" | tee -a "$RUN_LOG"
elif [ "$MODE" = "review" ]; then
    echo "👁  review mode — interactive confirm (non-tty = decline)" | tee -a "$RUN_LOG"
elif [ "$MODE" = "auto" ]; then
    echo "🟡 auto mode but KERNEL_LIVE!=1 → orders blocked by env_guard" | tee -a "$RUN_LOG"
else
    echo "📝 paper mode — planned actions logged only" | tee -a "$RUN_LOG"
fi

# 3a) Escalation sweep — cancel stale unfilled orders (live-aware)
echo "--- escalation sweep ---" | tee -a "$RUN_LOG"
ESC_LIVE_FLAG=""
[ "${KERNEL_LIVE:-0}" = "1" ] && ESC_LIVE_FLAG="--live"
$VENV live/cancel_escalate.py $ESC_LIVE_FLAG 2>&1 | tee -a "$RUN_LOG" || true

# 3b) Submit new orders for today's best play
echo "--- execute kernel plan ($MODE) ---" | tee -a "$RUN_LOG"
$VENV live/execute_kernel_plan.py --mode "$MODE" --daily-max "$DAILY_MAX" 2>&1 | tee -a "$RUN_LOG"

# 4) Daily digest on every 24th cycle (~every 12hr if cycle every 30min)
HOUR=$(date +%H)
if [ "$HOUR" = "12" ] || [ "$HOUR" = "00" ]; then
    echo "--- daily digest ---" | tee -a "$RUN_LOG"
    $VENV live/daily_digest.py --hours 24 2>&1 | tee -a "$RUN_LOG"
fi

# 5) Always show last 3 log entries
echo "--- recent trading actions ---" | tee -a "$RUN_LOG"
$VENV live/trading_log.py 3 2>&1 | head -40 | tee -a "$RUN_LOG"

exit 0
