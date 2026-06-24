#!/bin/bash
# Auto-backup: commit CODE/doc changes and push to origin/master. Run every 15 min via cron.
# CODE ONLY by design — never sweeps in regenerable data/cache (xls, zip, csv, most json) so the
# repo doesn't bloat. Data is handled by the dedicated fetch crons. Push goes direct (repo root =
# /home/wyatt/weather; ibkr_guided_trade is a tracked subdir).
set -o pipefail
cd /home/wyatt/weather || exit 1
export GIT_SSH_COMMAND='ssh -o BatchMode=yes'   # never prompt; fail fast if key unavailable

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Stage each code/doc pattern INDEPENDENTLY — a single non-matching pathspec (e.g. no *.service this
# run) must NOT abort the whole `git add`, which would silently back up nothing.
for pat in '*.py' '*.md' '*.sh' '*.html' '*.css' '*.txt' '*.toml' '*.cfg' '*.ini' '.gitignore' '*.service' '*.sql'; do
    git add -- "$pat" 2>/dev/null
done

if ! git diff --cached --quiet; then
    git commit -q -m "auto-backup $(ts)" || { echo "$(ts) commit failed"; exit 1; }
    echo "$(ts) committed code changes"
fi

# push only if there are unpushed commits
if [ -n "$(git log origin/master..HEAD --oneline 2>/dev/null)" ]; then
    git push origin master 2>&1 && echo "$(ts) pushed" || { echo "$(ts) push FAILED"; exit 1; }
fi
