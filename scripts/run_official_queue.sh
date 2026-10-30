#!/usr/bin/env bash
# Runs `python -m canvit_pytorch.evaluate` once per line of a queue file (the result's name, then the subcommand
# and its arguments), one after another, skipping lines whose result already exists.
#
#   scripts/run_official_queue.sh queues/dinov3_baselines.txt
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/tmp/canvit-nadiedjoa-24/hf}
# `hf auth login` stores the token under the default HF_HOME, not the one above (DINOv3 is gated).
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}
export ADE20K_ROOT=${ADE20K_ROOT:-/tmp/canvit-nadiedjoa-24/data/ade20k/ADEChallengeData2016}
mkdir -p results/official logs

while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if [[ -e "results/official/$name.pt" ]]; then
        echo "skip $name"
        continue
    fi
    echo "$(date +%H:%M) start $name"
    # shellcheck disable=SC2086
    if .venv/bin/python -m canvit_pytorch.evaluate $args --output "results/official/$name.pt" > "logs/$name.log" 2>&1; then
        echo "$(date +%H:%M) done  $name"
    else
        echo "$(date +%H:%M) FAIL  $name (logs/$name.log)"
    fi
done < "$1"
