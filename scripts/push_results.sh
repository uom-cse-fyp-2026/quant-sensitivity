#!/usr/bin/env bash
# Commit a finished run folder to git and push it, compressing the raw CSV first.
# Run outputs are gitignored, so the folder is added with -f.
#
#   scripts/push_results.sh                       # newest finished run under artifacts/
#   scripts/push_results.sh artifacts/<run_id>    # a specific run
set -euo pipefail
cd "$(dirname "$0")/.."

run=${1:-$(ls -td artifacts/*/ 2>/dev/null | head -1)}
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
    git commit -q -m "Add results from VM: $(basename "$run")"
fi
# Pick up code pushed from the laptop meanwhile; results only add new files, so this is clean.
git pull --rebase --quiet
git push --quiet
echo "pushed $run"
