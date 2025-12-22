#!/bin/bash
# Auto-fetch GFS forecast data
# Runs 4 times daily to catch each GFS cycle (00z, 06z, 12z, 18z)
# GFS data available ~5 hours after cycle time

SCRIPT_DIR="/home/wyatt/weather"
LOG_FILE="${SCRIPT_DIR}/gfs_fetch.log"
VENV_PATH="${SCRIPT_DIR}/venv/bin/activate"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "Starting GFS auto-fetch"
log "=========================================="

# Activate virtual environment
source "$VENV_PATH"

# Fetch latest GFS data
log "Fetching GFS forecast..."
cd "$SCRIPT_DIR"
python3 fetch_gfs.py >> "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    log "GFS fetch completed successfully"

    # Run GFS HDD analysis
    log "Running GFS HDD analysis..."
    python3 gfs_hdd_analysis.py >> "$LOG_FILE" 2>&1

    # Update combined ECMWF+GFS vs NG chart
    log "Updating combined HDD vs NG chart..."
    python3 hdd_ng_comparison.py >> "$LOG_FILE" 2>&1

    if [ $? -eq 0 ]; then
        log "Charts updated successfully"
    else
        log "WARNING: Chart update failed"
    fi
else
    log "ERROR: GFS fetch failed"
fi

# Cleanup old GFS files (keep last 7 days)
log "Cleaning up old GFS files..."
find "$SCRIPT_DIR" -name "gfs_*.grib2" -mtime +7 -delete 2>/dev/null

log "GFS auto-fetch finished"
log ""
