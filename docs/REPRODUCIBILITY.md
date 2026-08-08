# Reproducibility

## Environment

- Python 3.10+
- PyTorch with CUDA
- Google Colab T4 or a comparable NVIDIA GPU
- Dependencies listed in `requirements.txt`

## Common settings

- Optimizer: AdamW
- Latent channels: 32
- CIFAR-10 and PathMNIST token grid: 4x4
- CelebA-64 token grid: 8x8
- Sampling steps: 100
- EMA decay for conditional priors: 0.999
- Label-drop probability: 0.10
- Default seed in reported runs: 0

## Output structure

Each script writes a seed-specific directory containing some or all of:

```text
config.json
summary.json
reconstruction_metrics.json
generation_metrics.csv
geometry_metrics.csv
checkpoints/
logs/
samples/
figures/
```

The `--resume` flag restores the latest checkpoint after a Colab interruption.

## Main experiment settings

### CIFAR-10 path comparison

- 50,000 training images
- 10,000 test images
- Autoencoder: 60 epochs
- Prior: 120 epochs
- Base channels: 96
- Prior width/depth: 256/6
- Methods: `chord`, `rfm`

### CIFAR-10 VAE support-matching comparison

This is the main support-matching experiment. It keeps one KL-regularized VAE encoder and decoder fixed and trains both standard linear Flow Matching on the complete latent and projected direction-only RFM. The reported table also includes the existing SRUL result as the matched spherical reference.

First create or supply the SRUL and VAE checkpoints using `scripts/run_cifar10_matched_baselines.sh`. Then run:

```bash
export VAE_CHECKPOINT=runs/final_comparison/CIFAR10/seed_0/checkpoints/ldm_autoencoder_final.pt
export VAE_LATENT_STATS=runs/final_comparison/CIFAR10/seed_0/checkpoints/ldm_latent_stats.pt
export PREVIOUS_METRICS_CSV=runs/final_comparison/CIFAR10/seed_0/generation_metrics.csv
bash scripts/run_cifar10_vae_geometry.sh runs/final_comparison
```

The output includes `vae_radius_diagnostics.json`, `vae_support_reconstruction_metrics.json`, `generation_metrics.csv`, and `vae_prior_geometry_comparison.csv`.

### Supplementary deterministic-autoencoder mismatch test

The same post-hoc projection test is also available for a deterministic unconstrained autoencoder:

```bash
export EUCLIDEAN_AE_CHECKPOINT=runs/final_comparison/CIFAR10/seed_0/checkpoints/euclidean_fm_autoencoder_final.pt
export MATCHED_METRICS_CSV=runs/final_comparison/CIFAR10/seed_0/generation_metrics.csv
bash archive/scripts/run_cifar10_support_mismatch.sh runs/final_comparison
```

This repetition is kept in the repository as supplementary evidence and is not part of the main report table.

### PathMNIST

- Official train and test splits
- Autoencoder: 50 epochs
- Prior: 80 epochs
- Conditional RFM with tissue-class labels

### CelebA-64

- 30,000 training images
- 5,000 evaluation images
- Autoencoder: 40 epochs
- Prior: 60 epochs
- 8x8 token grid
- Conditional RFM with the Smiling attribute

## Metrics

- FID and KID: TorchMetrics / torch-fidelity
- Precision and recall: k-nearest-neighbor feature manifolds using ResNet-18 features
- Reconstruction: MSE, PSNR, SSIM, LPIPS, FID/KID
- Geometry: final token norm, minimum path norm, path-norm error, and radial velocity fraction
