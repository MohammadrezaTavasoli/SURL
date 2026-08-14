# Reproducibility

## Environment

- Python 3.10+
- CUDA-enabled PyTorch
- Google Colab T4 or comparable NVIDIA GPU
- packages in `requirements.txt`

## Common settings

- AdamW optimizer
- 32 latent channels
- 4x4 token grid for CIFAR-10 and PathMNIST
- 8x8 token grid for CelebA-64
- 100 sampling steps
- EMA decay 0.999
- label-drop probability 0.10
- reported seed 0

## Main runs

```bash
bash scripts/run_cifar10_geometry.sh runs
bash scripts/run_cifar10_conditional.sh runs /path/to/cifar_autoencoder.pt
bash scripts/run_cifar10_sigma_sweep.sh runs /path/to/cifar_autoencoder.pt
bash scripts/run_cifar10_vae_geometry.sh runs
bash scripts/run_pathmnist.sh runs
bash scripts/run_celeba64.sh runs /path/to/celeba_cache
```

## Cross-dataset controlled sweeps

The PathMNIST and CelebA scripts evaluate a full 3x3 grid, but results are reported as two controlled slices:

1. information control: vary `sigma_enc`, fix CFG `s=2`;
2. CFG: vary `s`, fix `sigma_enc=0.15`.

```bash
bash scripts/run_pathmnist_knob_cfg.sh runs /path/to/pathmnist_autoencoder.pt
bash scripts/run_celeba64_knob_cfg.sh runs /path/to/celeba_autoencoder.pt /path/to/celeba_cache
```

Outputs include `knob_cfg_sweep_metrics.csv`. The reporting slices are stored in:

```text
results/controlled_noise_sweep_cfg2.csv
results/controlled_cfg_sweep_sigma015.csv
```

## Broad spherical coverage training

```bash
bash scripts/run_cifar10_spherical_coverage.sh runs /path/to/base_autoencoder.pt
```

This branch adds small/large tangent rotations, reconstruction consistency, and latent consistency to encourage approximately uniform token marginals. Current headline FID tables come from the base autoencoder runs; do not mix checkpoints from the two training procedures in one comparison.

## Output structure

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

Long jobs support `--resume`.

## Metrics

- FID/KID: TorchMetrics and torch-fidelity
- precision/recall: k-nearest-neighbor feature manifolds
- reconstruction: MSE, PSNR, SSIM, LPIPS, FID/KID
- geometry: token norms, path-norm error, and radial velocity fraction
