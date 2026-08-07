#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT="${1:-runs}"
python src/srul_cifar_spatial_tokens_experiment.py \
  --out-dir "$OUT_ROOT/cifar10_geometry" \
  --dataset cifar10 \
  --train-samples 0 \
  --test-samples 10000 \
  --seed 0 \
  --ae-epochs 60 \
  --prior-epochs 120 \
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
  --lambda-clean 1.0 \
  --lambda-noisy 0.5 \
  --lambda-edge 0.15 \
  --lambda-latent 0.05 \
  --lambda-lpips 0.10 \
  --methods chord rfm \
  --sample-steps 100 \
  --metric-samples 10000 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --pr-chunk-size 256 \
  --pr-nearest-k 5 \
  --checkpoint-every 5 \
  --compute-lpips \
  --amp \
  --resume
