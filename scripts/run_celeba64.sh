#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${1:-runs}"
DATA_ROOT="${2:-data/celeba64}"
python src/srul_celeba64_conditional_experiment.py \
  --out-dir "$OUT_ROOT/celeba64" \
  --data-root "$DATA_ROOT" \
  --dataset celeba64 \
  --train-samples 30000 \
  --test-samples 5000 \
  --seed 0 \
  --image-size 64 \
  --num-classes 2 \
  --ae-epochs 40 \
  --prior-epochs 60 \
  --base-channels 64 \
  --latent-channels 32 \
  --prior-width 256 \
  --prior-depth 6 \
  --time-dim 128 \
  --batch-size 64 \
  --metric-batch-size 64 \
  --num-workers 2 \
  --sigma-enc 0.15 \
  --time-sampling logit_normal \
  --methods cond_rfm \
  --label-drop-prob 0.10 \
  --ema-decay 0.999 \
  --guidance-scales 1.0 2.0 \
  --sample-steps 100 \
  --metric-samples 5000 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --checkpoint-every 5 \
  --compute-lpips \
  --amp \
  --resume
