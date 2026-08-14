#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 OUT_ROOT AUTOENCODER_CHECKPOINT DATA_CACHE" >&2
  exit 2
fi

OUT_ROOT="$1"
AE_CHECKPOINT="$2"
DATA_CACHE="$3"
[[ -f "$AE_CHECKPOINT" ]] || { echo "Checkpoint not found: $AE_CHECKPOINT" >&2; exit 2; }
mkdir -p "$DATA_CACHE"

python src/srul_cross_dataset_knob_cfg_sweep.py \
  --dataset celeba64 \
  --data-root "$DATA_CACHE" \
  --out-dir "$OUT_ROOT/CelebA64_knob_cfg" \
  --seed 0 \
  --image-size 64 \
  --num-classes 2 \
  --train-samples 30000 \
  --test-samples 5000 \
  --batch-size 64 \
  --metric-batch-size 64 \
  --num-workers 2 \
  --base-channels 64 \
  --latent-channels 32 \
  --ae-checkpoint "$AE_CHECKPOINT" \
  --sigma-values 0.05 0.15 0.30 \
  --guidance-scales 1.0 1.5 2.0 \
  --prior-epochs 60 \
  --prior-lr 2e-4 \
  --prior-width 256 \
  --prior-depth 6 \
  --time-dim 128 \
  --time-sampling logit_normal \
  --label-drop-prob 0.10 \
  --ema-decay 0.999 \
  --sample-steps 100 \
  --metric-samples 5000 \
  --pr-samples 5000 \
  --recon-metric-samples 5000 \
  --pr-chunk-size 256 \
  --pr-nearest-k 5 \
  --checkpoint-every 5 \
  --hf-dataset flwrlabs/celeba \
  --celeba-attribute Smiling \
  --hf-shuffle-buffer 10000 \
  --amp \
  --resume
