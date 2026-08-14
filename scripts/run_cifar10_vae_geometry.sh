#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-runs}"
: "${VAE_CHECKPOINT:?Set VAE_CHECKPOINT to ldm_autoencoder_final.pt}"
: "${VAE_LATENT_STATS:?Set VAE_LATENT_STATS to ldm_latent_stats.pt}"
: "${PREVIOUS_METRICS_CSV:?Set PREVIOUS_METRICS_CSV to generation_metrics.csv}"

for path in "$VAE_CHECKPOINT" "$VAE_LATENT_STATS" "$PREVIOUS_METRICS_CSV"; do
  [[ -f "$path" ]] || { echo "Required file not found: $path" >&2; exit 2; }
done

python src/srul_vae_prior_geometry_comparison.py \
  --dataset cifar10 \
  --data-root "${DATA_ROOT:-./data}" \
  --out-dir "$OUT_ROOT/CIFAR10_vae_prior_geometry" \
  --seed 0 \
  --num-classes 10 \
  --train-samples 0 \
  --test-samples 10000 \
  --vae-checkpoint "$VAE_CHECKPOINT" \
  --vae-latent-stats-checkpoint "$VAE_LATENT_STATS" \
  --previous-metrics-csv "$PREVIOUS_METRICS_CSV" \
  --methods vae_euclidean_fm vae_projected_rfm \
  --base-channels 96 \
  --latent-channels 32 \
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
  --amp \
  --resume
