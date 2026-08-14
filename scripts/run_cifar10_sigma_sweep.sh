#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUT_ROOT AUTOENCODER_CHECKPOINT" >&2
  exit 2
fi

OUT_ROOT="$1"
AE_CHECKPOINT="$2"
[[ -f "$AE_CHECKPOINT" ]] || { echo "Checkpoint not found: $AE_CHECKPOINT" >&2; exit 2; }

python src/srul_sigma_knob_sweep.py \
  --dataset cifar10 \
  --data-root data \
  --out-dir "$OUT_ROOT/cifar10_sigma" \
  --num-classes 10 \
  --image-size 32 \
  --train-samples 0 \
  --test-samples 10000 \
  --seed 0 \
  --base-channels 96 \
  --latent-channels 32 \
  --ae-checkpoint "$AE_CHECKPOINT" \
  --sigma-values 0.05 0.15 0.30 \
  --prior-epochs 80 \
  --prior-width 256 \
  --prior-depth 6 \
  --batch-size 128 \
  --metric-batch-size 128 \
  --guidance-scale 2.0 \
  --sample-steps 100 \
  --metric-samples 10000 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --checkpoint-every 5 \
  --amp \
  --resume
