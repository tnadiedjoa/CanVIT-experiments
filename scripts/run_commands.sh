#!/usr/bin/env bash
# Runs a queue file of Python commands one after another: each line is an output path, then the arguments of
# .venv/bin/python (a script or -m module and its options), which get `--output <path>`. Lines whose output already
# exists are skipped; each command logs to logs/<output name>.log.
#
#   scripts/run_commands.sh queues/cityscapes.txt
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/tmp/canvit-nadiedjoa-24/hf}
# `hf auth login` stores the token under the default HF_HOME, not the one above (DINOv3 is gated).
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export ADE20K_ROOT=${ADE20K_ROOT:-/tmp/canvit-nadiedjoa-24/data/ade20k/ADEChallengeData2016}
export CITYSCAPES_ROOT=${CITYSCAPES_ROOT:-/tmp/canvit-nadiedjoa-24/data/cityscapes}
mkdir -p logs

while read -r output args; do
    [[ -z "$output" || "$output" == \#* ]] && continue
    if [[ -e "$output" ]]; then
        echo "skip $output"
        continue
    fi
    name=$(basename "${output%.*}")
    echo "$(date +%H:%M) start $name"
    # shellcheck disable=SC2086
    if .venv/bin/python -P $args --output "$output" > "logs/$name.log" 2>&1; then
        echo "$(date +%H:%M) done  $name: $(grep -oE '(mIoU: |Saved the probe).*' "logs/$name.log" | tail -1 | cut -c 1-100)"
    else
        echo "$(date +%H:%M) FAIL  $name (logs/$name.log)"
    fi
done < "$1"
