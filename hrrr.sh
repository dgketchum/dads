#!/usr/bin/env bash
set -euo pipefail

nohup uv run python -m grid.download_hrrr_archive --dest /mnt/mco_nas1/shared/hrrr_hourly --workers 20 --start-date 2014-11-15 --end-date 2026-03-04 > download_hrrr.log 2>&1 &

echo "PID: $!"
echo "Log: tail -f download_hrrr.log"
