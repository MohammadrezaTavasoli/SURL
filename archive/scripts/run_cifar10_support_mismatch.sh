#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT=${1:-runs/final_comparison}
python src/srul_projected_rfm_mismatch.py \
  --dataset cifar10 \
  --data-root ./data \
  --out-dir "$OUT_ROOT/CIFAR10_projected_rfm" \
  --seed 0 \
  --num-classes 10 \
  --train-samples 0 \
  --test-samples 10000 \
  --euclidean-ae-checkpoint "${EUCLIDEAN_AE_CHECKPOINT:?Set EUCLIDEAN_AE_CHECKPOINT}" \
  --previous-metrics-csv "${MATCHED_METRICS_CSV:?Set MATCHED_METRICS_CSV}" \
  --prior-epochs 120 \
  --prior-lr 2e-4 \
  --prior-width 256 \
  --prior-depth 6 \
  --time-dim 128 \
  --time-sampling logit_normal \
  --batch-size 128 \
  --metric-batch-size 128 \
  --num-workers 2 \
  --label-drop-prob 0.10 \
  --ema-decay 0.999 \
  --guidance-scale 2.0 \
  --sample-steps 100 \
  --radius-mode class_token_mean \
  --metric-samples 10000 \
  --pr-samples 5000 \
  --reconstruction-metric-samples 5000 \
  --checkpoint-every 5 \
  --amp --resume
