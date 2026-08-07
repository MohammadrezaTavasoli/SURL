# Reproducibility

## Environment

- Python 3.10+
- PyTorch with CUDA
- Google Colab T4 or a comparable NVIDIA GPU
- Dependencies listed in `requirements.txt`

## Common settings

- Optimizer: AdamW
- Latent channels: 32
- CIFAR-10 and PathMNIST token grid: 4×4
- CelebA-64 token grid: 8×8
- Sampling steps: 100
- EMA decay for conditional priors: 0.999
- Label-drop probability: 0.10
- Default seed in reported runs: 0

## Output structure

Each script writes a seed-specific directory containing:

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

### CIFAR-10 geometry comparison

- 50,000 training images
- 10,000 test images
- Autoencoder: 60 epochs
- Prior: 120 epochs
- Base channels: 96
- Prior width/depth: 256/6
- Methods: `chord`, `rfm`

### PathMNIST

- Official train and test splits
- Autoencoder: 50 epochs
- Prior: 80 epochs
- Conditional RFM with tissue class labels

### CelebA-64

- 30,000 training images
- 5,000 evaluation images
- Autoencoder: 40 epochs
- Prior: 60 epochs
- 8×8 token grid
- Conditional RFM with the Smiling attribute

## Metrics

- FID and KID: TorchMetrics / torch-fidelity
- Precision and recall: k-nearest-neighbor feature manifolds using ResNet-18 features
- Reconstruction: MSE, PSNR, SSIM, LPIPS, FID/KID
- Geometry: final token norm, minimum path norm, path-norm error, and radial velocity fraction
