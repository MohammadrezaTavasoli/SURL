#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT=${1:?Usage: $0 OUT_ROOT BASE_AUTOENCODER_CHECKPOINT}
AE_CKPT=${2:?Usage: $0 OUT_ROOT BASE_AUTOENCODER_CHECKPOINT}
python src/srul_sphere_style_uniformization.py \
  --dataset cifar10 --data-root data \
  --out-dir "$OUT_ROOT/CIFAR10_spherical_coverage" \
  --seed 0 --train-samples 0 --test-samples 10000 \
  --init-ae-checkpoint "$AE_CKPT" \
  --ae-epochs 30 --ae-lr 5e-5 \
  --alpha-max-deg 80 --alpha-mix-low-deg 80 --alpha-mix-high-deg 85 \
  --alpha-mix-prob 0.10 --small-scale-max 0.50 \
  --lambda-pix-recon-l1 1.0 --lambda-pix-recon-lpips 1.0 \
  --lambda-pix-cons-l1 0.5 --lambda-pix-cons-lpips 0.5 \
  --lambda-lat-cons 0.10 --lambda-edge 0.10 --lambda-clean-anchor 0.25 \
  --sigma-enc 0.05 --prior-epochs 80 --prior-lr 2e-4 \
  --prior-width 256 --prior-depth 6 --time-dim 128 \
  --time-sampling logit_normal --label-drop-prob 0.10 --ema-decay 0.999 \
  --guidance-scales 1.0 2.0 --sample-steps 100 \
  --batch-size 128 --metric-batch-size 128 --num-workers 2 \
  --metric-samples 10000 --pr-samples 5000 --recon-metric-samples 5000 \
  --uniformity-samples 10000 --checkpoint-every 5 --compute-lpips --amp --resume
