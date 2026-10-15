#!/usr/bin/env bash
# Run a command in the background so it keeps going after the SSH session closes.
#
#   scripts/run_bg.sh <name> <command...>
#   scripts/run_bg.sh pilot uv run python -m src.run_experiment --gate 0
#
# Output goes to logs/<name>-<UTC time>.log, ending with the command's exit status.
# The PID is saved in logs/<name>.pid. Everything stays inside the repo; nothing global.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 2 ]; then
    echo "usage: scripts/run_bg.sh <name> <command...>" >&2
    exit 2
fi
name=$1
shift

mkdir -p logs
pidfile="logs/${name}.pid"
if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "'$name' is already running (pid $(cat "$pidfile")); see scripts/status.sh $name" >&2
    exit 1
fi

log="logs/${name}-$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$log" > "logs/${name}.latest"

# setsid detaches from the terminal's session so closing SSH cannot signal the job.
detach=$(command -v setsid || true)
nohup $detach bash -c '
    echo "started $(date -u +%FT%TZ) on $(hostname): $*"
    "$@"
    status=$?
    echo "finished $(date -u +%FT%TZ), exit status $status"
    exit $status
' _ "$@" > "$log" 2>&1 < /dev/null &
echo $! > "$pidfile"

echo "started '$name' (pid $!)"
echo "  log:    tail -f $log"
echo "  status: scripts/status.sh $name"
echo "  stop:   kill \$(cat $pidfile)"
