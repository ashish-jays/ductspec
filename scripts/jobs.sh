#!/usr/bin/env bash
# Launch the training runs on Hugging Face Jobs.
#
# Jobs default to a 30 minute timeout and ephemeral storage, so both are set
# explicitly: a run that is killed at 30 minutes leaves nothing behind, and an
# adapter written to the container's filesystem is gone when it exits.
set -euo pipefail

ARM="${1:?usage: jobs.sh <raw|curated> [--smoke]}"
SMOKE="${2:-}"
FLAVOR="${DUCTSPEC_FLAVOR:-l4x1}"
REPO="${DUCTSPEC_HF_OWNER:-ashishr10}/ductspec-$ARM"

EXTRA=""
if [ "$SMOKE" = "--smoke" ]; then
  EXTRA="--max-steps 2 --max-samples 16"
  REPO="$REPO-smoke"
fi

hf jobs run \
  --flavor "$FLAVOR" \
  --timeout 3600 \
  --secrets HF_TOKEN \
  --env DUCTSPEC_GPU="$FLAVOR" \
  --env DUCTSPEC_GPU_USD="0.80" \
  hf.co/spaces/ductspec/trainer \
  python scripts/train.py "$ARM" --output /data/adapters/"$ARM" $EXTRA

echo "adapter will be pushed to: https://huggingface.co/$REPO"
