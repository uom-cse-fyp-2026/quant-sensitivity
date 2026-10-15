#!/usr/bin/env bash
# Commit a finished run folder to git and push it, compressing raw CSVs first.
# Run outputs are gitignored by default, so the folder is added with -f.
#
#   scripts/push_results.sh                       # newest run under artifacts/
#   scripts/push_results.sh artifacts/<run_id>    # a specific run
set -euo pipefail
cd "$(dirname "$0")/.."

run=${1:-$(ls -td artifacts/*/ 2>/dev/null | grep -v '/calibration/' | head -1)}
run=${run%/}
if [ -z "$run" ] || [ ! -f "$run/manifest.json" ]; then
    echo "no finished run found (expected a folder with manifest.json): '$run'" >&2
    exit 1
fi

find "$run/raw" -name '*.csv' -exec gzip -f {} + 2>/dev/null || true
git add -f "$run"
git commit -m "Add results from VM: $(basename "$run")"
git push
echo "pushed $run"
