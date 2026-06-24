#!/bin/bash
# Auto-backup: commit any working changes and push to origin/master. Run every 15 min via cron.
# Push goes direct (repo root = /home/wyatt/weather; ibkr_guided_trade is a tracked subdir).
set -o pipefail
cd /home/wyatt/weather || exit 1
export GIT_SSH_COMMAND='ssh -o BatchMode=yes'   # never prompt; fail fast if key unavailable

ts() { date '+%Y-%m-%d %H:%M:%S'; }

git add -A
if ! git diff --cached --quiet; then
    git commit -q -m "auto-backup $(ts)" || { echo "$(ts) commit failed"; exit 1; }
    echo "$(ts) committed working changes"
fi

# push only if there are unpushed commits (avoids no-op network calls / noise)
if [ -n "$(git log origin/master..HEAD --oneline 2>/dev/null)" ]; then
    if git push origin master 2>&1; then
        echo "$(ts) pushed to origin/master"
    else
        echo "$(ts) push FAILED"
        exit 1
    fi
fi
