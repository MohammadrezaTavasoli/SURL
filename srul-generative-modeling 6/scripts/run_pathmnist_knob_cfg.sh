#!/usr/bin/env bash
set -euo pipefail
OUT_ROOT=${1:?Usage: $0 OUT_ROOT AUTOENCODER_CHECKPOINT}
AE_CKPT=${2:?Usage: $0 OUT_ROOT AUTOENCODER_CHECKPOINT}
python src/srul_cross_dataset_knob_cfg_sweep.py \
  --dataset pathmnist --data-root data \
  --out-dir "$OUT_ROOT/PathMNIST_knob_cfg" \
  --seed 0 --image-size 32 --num-classes 9 \
  --train-samples 0 --test-samples 7180 \
  --batch-size 128 --metric-batch-size 128 --num-workers 2 \
  --base-channels 96 --latent-channels 32 --ae-checkpoint "$AE_CKPT" \
  --sigma-values 0.05 0.15 0.30 --guidance-scales 1.0 1.5 2.0 \
  --prior-epochs 80 --prior-lr 2e-4 --prior-width 256 --prior-depth 6 \
  --time-dim 128 --time-sampling logit_normal --label-drop-prob 0.10 \
  --ema-decay 0.999 --sample-steps 100 --metric-samples 7180 \
  --pr-samples 5000 --recon-metric-samples 5000 --pr-chunk-size 256 \
  --pr-nearest-k 5 --checkpoint-every 5 --amp --resume
