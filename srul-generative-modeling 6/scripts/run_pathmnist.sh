#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${1:-runs}"
mkdir -p data
python src/srul_medmnist_conditional_experiment.py \
  --out-dir "$OUT_ROOT/pathmnist" \
  --dataset pathmnist \
  --data-root data \
  --train-samples 0 \
  --test-samples 7180 \
  --seed 0 \
  --ae-epochs 50 \
  --prior-epochs 80 \
  --base-channels 96 \
  --latent-channels 32 \
  --prior-width 256 \
  --prior-depth 6 \
  --time-dim 128 \
  --batch-size 128 \
  --metric-batch-size 128 \
  --num-workers 2 \
  --sigma-enc 0.15 \
  --time-sampling logit_normal \
  --methods cond_rfm \
  --label-drop-prob 0.10 \
  --ema-decay 0.999 \
  --guidance-scales 1.0 2.0 \
  --sample-steps 100 \
  --metric-samples 7180 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --checkpoint-every 5 \
  --compute-lpips \
  --amp \
  --resume
