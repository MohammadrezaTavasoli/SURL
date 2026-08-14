#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 OUT_ROOT AUTOENCODER_CHECKPOINT" >&2
  exit 2
fi

OUT_ROOT="$1"
AE_CHECKPOINT="$2"
[[ -f "$AE_CHECKPOINT" ]] || { echo "Checkpoint not found: $AE_CHECKPOINT" >&2; exit 2; }

python src/srul_cifar_conditional_spatial_experiment.py \
  --out-dir "$OUT_ROOT/cifar10_conditional" \
  --ae-checkpoint "$AE_CHECKPOINT" \
  --skip-ae-training \
  --dataset cifar10 \
  --train-samples 0 \
  --test-samples 10000 \
  --seed 0 \
  --base-channels 96 \
  --latent-channels 32 \
  --prior-epochs 120 \
  --prior-lr 2e-4 \
  --prior-width 256 \
  --prior-depth 6 \
  --time-dim 128 \
  --time-sampling logit_normal \
  --methods cond_rfm \
  --label-drop-prob 0.10 \
  --ema-decay 0.999 \
  --guidance-scales 1.0 1.5 2.0 \
  --sample-steps 100 \
  --metric-samples 10000 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --batch-size 128 \
  --metric-batch-size 128 \
  --num-workers 2 \
  --checkpoint-every 5 \
  --amp \
  --resume
