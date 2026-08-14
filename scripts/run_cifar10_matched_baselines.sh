#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-runs/final_comparison}"
: "${SRUL_AE_CHECKPOINT:?Set SRUL_AE_CHECKPOINT to autoencoder_final.pt}"
: "${SRUL_PRIOR_CHECKPOINT:?Set SRUL_PRIOR_CHECKPOINT to cond_rfm_final.pt}"
[[ -f "$SRUL_AE_CHECKPOINT" ]] || { echo "Checkpoint not found: $SRUL_AE_CHECKPOINT" >&2; exit 2; }
[[ -f "$SRUL_PRIOR_CHECKPOINT" ]] || { echo "Checkpoint not found: $SRUL_PRIOR_CHECKPOINT" >&2; exit 2; }

python src/srul_baseline_comparison.py \
  --dataset cifar10 \
  --data-root ./data \
  --out-dir "$OUT_ROOT/CIFAR10" \
  --num-classes 10 \
  --image-size 32 \
  --train-samples 0 \
  --test-samples 10000 \
  --seed 0 \
  --methods srul_rfm euclidean_fm ldm \
  --srul-ae-checkpoint "$SRUL_AE_CHECKPOINT" \
  --srul-prior-checkpoint "$SRUL_PRIOR_CHECKPOINT" \
  --base-channels 96 \
  --latent-channels 32 \
  --ae-epochs 60 \
  --prior-epochs 120 \
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
