# Short commands for the experiment. Run `just` to list them.
# Runs happen inside a tmux session, so you can detach with Ctrl+B then D, close SSH,
# and come back later with `just attach`.

set shell := ["bash", "-cu"]
set windows-shell := ["bash", "-cu"]

session := "qs"
config := "configs/experiment.yaml"

# List the commands
default:
    @just --list --unsorted

# Get the latest code and update the environment
pull:
    git pull --ff-only
    uv sync --locked

# Run the unit tests
test:
    uv run pytest

# Run a gate (default 0 = pilot) in tmux; Ctrl+B then D to detach, `just attach` to return
run gate="0":
    #!/usr/bin/env bash
    set -euo pipefail
    cmd='uv run python -m src.run_experiment --config {{config}} --gate {{gate}}; rc=$?; echo; echo "[finished: exit status $rc] press Enter to close"; read'
    if [ -n "${TMUX:-}" ]; then
        echo "already inside tmux; running here"
        exec bash -c "$cmd"
    fi
    if ! command -v tmux >/dev/null; then
        echo "tmux is not installed; run without it: uv run python -m src.run_experiment --gate {{gate}}" >&2
        exit 1
    fi
    if tmux has-session -t {{session}} 2>/dev/null; then
        echo "a run is already going; see it with: just attach" >&2
        exit 1
    fi
    tmux new-session -s {{session}} "$cmd"

# Pull, update, test, then run a gate (default 0 = pilot)
pull-run gate="0": pull test
    just run {{gate}}

# Bring the running job back to the foreground (detach again: Ctrl+B then D)
attach:
    tmux attach -t {{session}}

alias fg := attach

# Is a run going? Shows the end of the newest run log
status:
    #!/usr/bin/env bash
    if tmux has-session -t {{session}} 2>/dev/null; then
        echo "running (just attach to watch)"
    else
        echo "nothing running"
    fi
    log=$(ls -t artifacts/*/run.log 2>/dev/null | head -1)
    [ -n "$log" ] && { echo "--- $log ---"; tail -n 12 "$log"; }
    true

# Follow the newest run log live (Ctrl+C stops watching, not the run)
log:
    tail -f "$(ls -t artifacts/*/run.log | head -1)"

# Stop a running job
stop:
    tmux kill-session -t {{session}}

# Commit and push the newest finished run (or one you name: just push artifacts/<run_id>)
push run="":
    scripts/push_results.sh {{run}}

# List the runs on this machine, newest first
runs:
    @ls -td artifacts/*/ 2>/dev/null | grep -v calibration || echo "no runs yet"
