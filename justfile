# Short commands for the experiment. Run `just` to list them.
# `just run` works inside a tmux session, so you can detach with Ctrl+B then D, close SSH,
# and come back later with `just attach`.

set shell := ["bash", "-cu"]
set windows-shell := ["bash", "-cu"]

session := "qs"

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

# Run the experiment and its analysis in tmux; Ctrl+B then D to detach, `just attach` to return
run:
    #!/usr/bin/env bash
    set -euo pipefail
    cmd='uv run python -m src.run_experiment; rc=$?; echo; echo "[finished: exit status $rc] press Enter to close"; read'
    if [ -n "${TMUX:-}" ]; then
        exec bash -c "$cmd"
    fi
    if ! command -v tmux >/dev/null; then
        echo "tmux is not installed; run without it: uv run python -m src.run_experiment" >&2
        exit 1
    fi
    if tmux has-session -t {{session}} 2>/dev/null; then
        echo "a run is already going; see it with: just attach" >&2
        exit 1
    fi
    tmux new-session -s {{session}} "$cmd"

# Pull, update, test, then run
pull-run: pull test run

# Bring the running job back to the foreground (detach again: Ctrl+B then D)
attach:
    tmux attach -t {{session}}

alias fg := attach

# Is a run going? Shows the end of the newest run log
status:
    #!/usr/bin/env bash
    if tmux has-session -t {{session}} 2>/dev/null; then echo "running (just attach to watch)"; else echo "nothing running"; fi
    log=$(ls -t artifacts/*/run.log 2>/dev/null | head -1)
    [ -n "$log" ] && { echo "--- $log ---"; tail -n 12 "$log"; }
    true

# Stop a running job
stop:
    tmux kill-session -t {{session}}

# Redo the tables and figures for a run (default: newest), e.g. after changing the analysis
analyze run="":
    uv run python -m src.analysis --run "{{ if run == "" { `ls -td artifacts/*/ | head -1` } else { run } }}"

# Commit and push the newest finished run (or one you name: just push artifacts/<run_id>)
push run="":
    scripts/push_results.sh {{run}}

# List the runs on this machine, newest first
runs:
    @ls -td artifacts/*/ 2>/dev/null || echo "no runs yet"
