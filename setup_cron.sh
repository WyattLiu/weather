#!/bin/bash
# Setup cron jobs for automatic weather data fetching
# - ECMWF: 4-day forecast (00z, 12z)
# - GFS: 16-day forecast (00z, 06z, 12z, 18z)

echo "Setting up cron jobs for weather forecast fetching..."

# Create cron entries
CRON_ENTRIES="
# ============================================
# ECMWF Forecast Fetching (4-day, twice daily)
# ============================================
# 00z run - available ~4:30 UTC
30 4 * * * /home/wyatt/weather/auto_fetch.sh >> /home/wyatt/weather/cron.log 2>&1

# 12z run - available ~16:30 UTC
30 16 * * * /home/wyatt/weather/auto_fetch.sh >> /home/wyatt/weather/cron.log 2>&1

# ============================================
# GFS Forecast Fetching (16-day, 4x daily)
# ============================================
# 00z run - available ~5:00 UTC
0 5 * * * /home/wyatt/weather/auto_fetch_gfs.sh >> /home/wyatt/weather/cron.log 2>&1

# 06z run - available ~11:00 UTC
0 11 * * * /home/wyatt/weather/auto_fetch_gfs.sh >> /home/wyatt/weather/cron.log 2>&1

# 12z run - available ~17:00 UTC
0 17 * * * /home/wyatt/weather/auto_fetch_gfs.sh >> /home/wyatt/weather/cron.log 2>&1

# 18z run - available ~23:00 UTC
0 23 * * * /home/wyatt/weather/auto_fetch_gfs.sh >> /home/wyatt/weather/cron.log 2>&1
"

echo ""
echo "New cron schedule:"
echo "$CRON_ENTRIES"
echo ""

read -p "Install these cron jobs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Remove old entries and add new ones
    crontab -l 2>/dev/null | grep -v "auto_fetch" | grep -v "ECMWF" | grep -v "GFS" > /tmp/crontab.tmp
    echo "$CRON_ENTRIES" >> /tmp/crontab.tmp
    crontab /tmp/crontab.tmp
    rm /tmp/crontab.tmp

    echo ""
    echo "Cron jobs installed successfully!"
    echo ""
    echo "Schedule Summary:"
    echo "  ECMWF (4-day):  04:30 UTC, 16:30 UTC"
    echo "  GFS (16-day):   05:00, 11:00, 17:00, 23:00 UTC"
    echo ""
    echo "Logs: /home/wyatt/weather/cron.log"
else
    echo "Cancelled. No changes made."
fi

echo ""
echo "Current crontab:"
crontab -l 2>/dev/null || echo "(empty)"

echo ""
echo "To view logs:"
echo "  tail -f /home/wyatt/weather/cron.log"
echo ""
echo "To edit cron jobs manually:"
echo "  crontab -e"
