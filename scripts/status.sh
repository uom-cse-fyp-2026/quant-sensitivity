#!/usr/bin/env bash
# Show whether a background job started with scripts/run_bg.sh is still running,
# and the last lines of its log.
#
#   scripts/status.sh <name>
set -euo pipefail
cd "$(dirname "$0")/.."

name=${1:?usage: scripts/status.sh <name>}
pidfile="logs/${name}.pid"
log=$(cat "logs/${name}.latest" 2>/dev/null || true)

if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "'$name' is running (pid $(cat "$pidfile"))"
else
    echo "'$name' is not running"
fi
if [ -n "$log" ] && [ -f "$log" ]; then
    echo "--- last lines of $log ---"
    tail -n "${LINES_TO_SHOW:-15}" "$log"
fi
