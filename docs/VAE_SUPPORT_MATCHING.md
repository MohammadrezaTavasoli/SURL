# VAE support-matching experiment

This experiment compares two priors while keeping the same trained VAE encoder and decoder:

1. **VAE + standard linear Flow Matching** models the complete Euclidean latent.
2. **VAE + projected RFM** normalizes each token to a unit direction, learns the direction distribution, restores a class/token mean radius, and decodes with the unchanged VAE decoder.

The comparison tests whether a VAE trained in Euclidean latent space can be paired with a direction-only spherical prior without losing information.

## Required files

The source files are included in `src/`, but the trained checkpoints are not included. The experiment requires:

```text
ldm_autoencoder_final.pt
ldm_latent_stats.pt
generation_metrics.csv
```

These files can be generated with `scripts/run_cifar10_matched_baselines.sh`. See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the full command sequence.

## Run command

```bash
export VAE_CHECKPOINT=/absolute/path/to/ldm_autoencoder_final.pt
export VAE_LATENT_STATS=/absolute/path/to/ldm_latent_stats.pt
export PREVIOUS_METRICS_CSV=/absolute/path/to/generation_metrics.csv
bash scripts/run_cifar10_vae_geometry.sh runs
```

## Outputs

```text
runs/CIFAR10_vae_prior_geometry/seed_0/
├── vae_radius_diagnostics.json
├── vae_support_reconstruction_metrics.json
├── generation_metrics.csv
├── vae_prior_geometry_comparison.csv
├── summary.json
├── figures/
├── samples/
├── logs/
└── checkpoints/
```

## Reported result

| Method | FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| SRUL | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | 56.04 | 0.848 | 0.063 |

The VAE token-radius coefficient of variation is 0.291. Replacing sample-specific radii with class/token mean radii changes reconstruction FID from 16.90 to 22.46. The result indicates that the VAE decoder uses radial information, so post-training projection to unit directions is not equivalent to training on spherical support from the start.
