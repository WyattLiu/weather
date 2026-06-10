#!/bin/bash
# Daily refresh for DBA × weather signals.
# Cron: 0 18 * * 1-5  (6pm ET, after USDM Thursday update lands)
set -u
cd "$(dirname "$0")/../.."

echo "=== fetch ONI + DSCI + ETF prices ==="
venv/bin/python research/dba/fetch_data.py 2>&1 | tail -10

echo
echo "=== fetch CPC ENSO plume forecast ==="
venv/bin/python research/dba/fetch_enso_plume.py 2>&1 | tail -15

echo
echo "=== rebuild composite state ==="
venv/bin/python research/dba/composite_edge.py 2>&1 | tail -10
