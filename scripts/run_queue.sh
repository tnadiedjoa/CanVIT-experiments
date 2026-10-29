#!/usr/bin/env bash
# Runs scripts/ade20k_rollouts.py once per line of a queue file (its arguments, after the result's name),
# one after another, skipping lines whose result already exists.
#
#   scripts/run_queue.sh queues/reproduction.txt
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/tmp/canvit-nadiedjoa-24/hf}
export ADE20K_ROOT=${ADE20K_ROOT:-/tmp/canvit-nadiedjoa-24/data/ade20k/ADEChallengeData2016}
mkdir -p results logs

while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [[ -e "results/$name.json" ]]; then
        echo "skip $name"
        continue
    fi
    echo "$(date +%H:%M) start $name"
    # shellcheck disable=SC2086
    if .venv/bin/python -P scripts/ade20k_rollouts.py $args --output "results/$name.json" > "logs/$name.log" 2>&1; then
        echo "$(date +%H:%M) done  $name: $(grep -o 'mIoU by glimpse.*' "logs/$name.log" | cut -c 1-80)"
    else
        echo "$(date +%H:%M) FAIL  $name (logs/$name.log)"
    fi
done < "$1"
