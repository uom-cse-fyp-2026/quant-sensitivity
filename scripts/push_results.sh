#!/usr/bin/env bash
# Commit a finished run folder to git and push it, compressing raw CSVs first.
# Run outputs are gitignored by default, so the folder is added with -f.
#
#   scripts/push_results.sh                       # newest finished run under artifacts/
#   scripts/push_results.sh artifacts/<run_id>    # a specific run
set -euo pipefail
cd "$(dirname "$0")/.."

run=${1:-$(ls -td artifacts/*/ 2>/dev/null | grep -v '/calibration/' | head -1)}
run=${run%/}
if [ -z "$run" ] || [ ! -f "$run/manifest.json" ]; then
    echo "no finished run found (a run folder gets manifest.json when it finishes): '$run'" >&2
    exit 1
fi

find "$run/raw" -name '*.csv' -exec gzip -f {} + 2>/dev/null || true
git add -f "$run"
if git diff --cached --quiet; then
    echo "$run is already committed"
else
    if [ "$(basename "$run")" = calibration ]; then
        sha=$(grep -o '"static_scale_checksum": "[0-9a-f]*' "$run/manifest.json" | cut -d'"' -f4)
        git commit -q -m "Gate 1: freeze static calibration scales (sha256 ${sha:0:12})"
    else
        git commit -q -m "Add results from VM: $(basename "$run")"
    fi
fi
# Pick up code pushed from the laptop meanwhile; results only add new files, so this is clean.
git pull --rebase --quiet
git push --quiet
echo "pushed $run"
