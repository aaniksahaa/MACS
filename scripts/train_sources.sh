#!/usr/bin/env bash
# Train one source model per domain of a task (100 epochs, 500 for whole-mouse-brain); finished models are skipped.
# Usage: bash scripts/train_sources.sh <membrane|mitochondria> [arch=unet] [extra train_source.py options]
set -euo pipefail
cd "$(dirname "$0")/.."
TASK="${1:?usage: train_sources.sh <membrane|mitochondria> [arch] [options]}"
ARCH="${2:-unet}"
shift $(( $# >= 2 ? 2 : 1 ))
for domain in $(python -c "from common import domains; print(' '.join(domains.list_domains('$TASK')))"); do
    out="models/${TASK}_domains/${ARCH}/${domain}"
    if [[ -f "$out/model_best.pth" && -f "$out/model_final.pth" ]]; then
        echo "skip $domain ($out exists)"
        continue
    fi
    epochs=100
    [[ "$domain" == "whole-mouse-brain" ]] && epochs=500
    python train_source.py --task "$TASK" --domain "$domain" --model "$ARCH" --epochs "$epochs" "$@"
done
