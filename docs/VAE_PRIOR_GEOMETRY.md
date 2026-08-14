# VAE Prior-Geometry Comparison for SRUL

## Purpose

This experiment keeps one trained KL/VAE autoencoder fixed and compares two
priors on the same VAE latent distribution:

1. **VAE + standard linear Flow Matching** — models the complete Euclidean latent,
   including radius and direction.
2. **VAE + projected spherical RFM** — normalizes each latent token to a unit
   direction, learns only the direction distribution on a product of spheres,
   and restores a class/token mean radius before decoding.

This directly tests whether a direction-only spherical prior is compatible
with a VAE latent that may use variable radii and posterior variance.

The VAE experiment is the main support-matching comparison in the report. A deterministic-autoencoder repetition is kept in the repository as supplementary evidence.

## What is measured

### VAE support diagnostics

- token-radius coefficient of variation;
- average posterior standard deviation;
- reconstruction from posterior mean;
- reconstruction from posterior sample;
- reconstruction after replacing sample-specific radius with a class/token
  mean radius;
- reconstruction from unit directions only.

### Prior comparison

- FID and KID;
- feature precision and recall;
- exact unit-norm trajectory diagnostics for projected RFM;
- qualitative class-balanced sample grids.

## Required files

Place these files in the same Colab folder:

```text
srul_vae_prior_geometry_comparison.py
srul_medmnist_conditional_experiment.py
```

The package ZIP already contains both files.

## Recommended CIFAR-10 checkpoints

Use the VAE autoencoder and latent statistics produced by the compact LDM
baseline:

```text
/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/checkpoints/ldm_autoencoder_final.pt
/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/checkpoints/ldm_latent_stats.pt
```

The existing comparison table is:

```text
/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/generation_metrics.csv
```

## Colab command

```bash
python /content/SRUL_VAE_RFM_Geometry_Package/srul_vae_prior_geometry_comparison.py \
  --dataset cifar10 \
  --data-root /content/data \
  --out-dir "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10_vae_prior_geometry" \
  --seed 0 \
  --num-classes 10 \
  --train-samples 0 \
  --test-samples 10000 \
  --vae-checkpoint \
    "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/checkpoints/ldm_autoencoder_final.pt" \
  --vae-latent-stats-checkpoint \
    "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/checkpoints/ldm_latent_stats.pt" \
  --previous-metrics-csv \
    "/content/drive/MyDrive/SRUL_Final_Comparisons/CIFAR10/seed_0/generation_metrics.csv" \
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
```

With a CUDA memory error, change both batch sizes to `64`.

## Main outputs

```text
.../CIFAR10_vae_prior_geometry/seed_0/
├── vae_radius_diagnostics.json
├── vae_support_reconstruction_metrics.json
├── generation_metrics.csv
├── vae_prior_geometry_comparison.csv
├── summary.json
├── figures/vae_support_reconstruction_diagnostic.png
├── samples/vae_euclidean_fm.png
├── samples/vae_projected_rfm.png
├── logs/
└── checkpoints/
```

## Reported result

| Method | FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| **SRUL** | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | 56.04 | 0.848 | 0.063 |

The VAE token radius has coefficient of variation `0.291`. Replacing sample-specific radii by class/token means worsens reconstruction FID from `16.90` to `22.46`; unit-radius projection gives `113.00`.

## Interpretation

The strongest support-mismatch evidence would be:

1. token radius varies meaningfully;
2. replacing sample-specific radius with a mean radius worsens reconstruction;
3. VAE + projected RFM performs worse than VAE + standard FM.

That would show that the VAE latent uses information outside direction alone,
so post-hoc spherical projection is not equivalent to learning a spherical
autoencoder from the start.

The reported result instead shows a clear gap: standard FM preserves the complete VAE latent, while projected RFM loses radial information used by the decoder.
