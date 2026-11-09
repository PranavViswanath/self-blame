#!/bin/bash
# Kill PID $1 if MemAvailable drops below $2 GB (default 16). Logs every 30 s to $3.
PID=$1; MIN=${2:-16}; LOG=${3:-logs/watchdog.log}
while kill -0 "$PID" 2>/dev/null; do
  AV=$(awk '/MemAvailable/ {printf "%.1f", $2/1e6}' /proc/meminfo)
  echo "$(date +%H:%M:%S) pid=$PID MemAvailable=${AV}GB" >> "$LOG"
  if awk "BEGIN{exit !($AV < $MIN)}"; then
    echo "$(date +%H:%M:%S) KILL pid=$PID MemAvailable=${AV}GB < ${MIN}GB" >> "$LOG"
    kill -9 "$PID"; exit 1
  fi
  sleep 30
done
echo "$(date +%H:%M:%S) pid=$PID exited" >> "$LOG"
