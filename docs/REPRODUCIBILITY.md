# Reproducibility

## Environment

- Python 3.10 or later
- CUDA-enabled PyTorch
- packages listed in `requirements.txt`
- Google Colab T4 or a comparable NVIDIA GPU for the full runs

## Common settings

- AdamW optimizer
- 32 latent channels
- 4x4 token grid for CIFAR-10 and PathMNIST
- 8x8 token grid for CelebA-64
- 100 sampling steps
- EMA decay 0.999
- label-drop probability 0.10
- reported random seed 0

## Checkpoints

Model weights are not included in the repository. See [`CHECKPOINTS.md`](CHECKPOINTS.md) for the order in which checkpoints are generated and consumed.

## Training commands

```bash
bash scripts/run_cifar10_geometry.sh runs
bash scripts/run_cifar10_conditional.sh runs runs/cifar10_geometry/seed_0/checkpoints/autoencoder_final.pt
bash scripts/run_cifar10_sigma_sweep.sh runs runs/cifar10_geometry/seed_0/checkpoints/autoencoder_final.pt
bash scripts/run_pathmnist.sh runs
bash scripts/run_celeba64.sh runs data/celeba64
```

## Cross-dataset controlled sweeps

Each PathMNIST and CelebA run evaluates a 3x3 grid. The report uses two one-variable slices:

1. vary `sigma_enc` while CFG is fixed at `s=2`;
2. vary CFG while `sigma_enc` is fixed at `0.15`.

```bash
bash scripts/run_pathmnist_knob_cfg.sh \
  runs \
  runs/pathmnist/seed_0/checkpoints/autoencoder_final.pt

bash scripts/run_celeba64_knob_cfg.sh \
  runs \
  runs/celeba64/seed_0/checkpoints/autoencoder_final.pt \
  data/celeba64
```

The full grids are saved as `knob_cfg_sweep_metrics.csv`. The controlled tables used in the report are stored in:

```text
results/controlled_noise_sweep_cfg2.csv
results/controlled_cfg_sweep_sigma015.csv
```

## Spherical coverage training

```bash
bash scripts/run_cifar10_spherical_coverage.sh \
  runs \
  runs/cifar10_geometry/seed_0/checkpoints/autoencoder_final.pt
```

This procedure adds small and large tangent rotations, reconstruction consistency, and latent consistency. Keep these checkpoints separate from the base autoencoder checkpoints when comparing results.

## Output layout

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

All long jobs support `--resume`.

## Metrics

- FID and KID: TorchMetrics/torch-fidelity
- precision and recall: k-nearest-neighbor feature manifolds
- reconstruction: MSE, PSNR, SSIM, LPIPS, FID, and KID
- geometry: token norms, path-norm error, and radial velocity fraction
