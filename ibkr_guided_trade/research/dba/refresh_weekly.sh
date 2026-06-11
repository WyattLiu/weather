#!/bin/bash
# Weekly DBA fundamentals refresh — COT (Fri 3:30pm ET release),
# FAO FPI (monthly, ~1st Thursday), USDA PSD (monthly WASDE-aligned).
# Rebuilds the fundamentals panel + composite state so the live tilt
# (score-graduated confluence) uses fresh data.
# Cron: 0 10 * * 6  (Saturday 10:00, after Friday's COT post)
set -u
cd "$(dirname "$0")/../.."

echo "=== $(date -Is) weekly fundamentals refresh ==="
venv/bin/python research/dba/fundamentals_fetch.py 2>&1 | tail -5

echo "--- rebuild fundamentals panel ---"
venv/bin/python research/dba/fundamentals_scan.py 2>&1 | tail -3

echo "--- rebuild factor panel (oni/dxy/ng inputs) ---"
venv/bin/python research/dba/factor_scan.py 2>&1 | tail -2

echo "--- rebuild composite state (live tilt) ---"
venv/bin/python research/dba/composite_edge.py 2>&1 | tail -6
