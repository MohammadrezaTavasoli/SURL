# Learning Spatial Unit-Norm Latents with Riemannian Flow Matching

**SRUL** stands for **Spherical Riemannian Unit-Norm Latents**. The project studies one design rule:

> The prior should match the support and geometry learned by the autoencoder.

The encoder produces a spatial grid of unit-norm tokens. Small tangent perturbations act as an information-control knob. A larger perturbation branch makes the decoder work over wider spherical neighborhoods and encourages approximately uniform coverage of each token sphere. Riemannian Flow Matching (RFM) then learns the remaining structured distribution on the same product-of-spheres support.

![SRUL pipeline](assets/srul_pipeline.png)

## Architecture used in the main CIFAR-10 experiments

- CNN encoder: `3 -> 96 -> 192 -> 384 -> 32`
- Latent grid: `32 x 4 x 4`, normalized independently at every spatial token
- CNN decoder: `32 -> 384 -> 192 -> 96 -> 3`
- Conditional RFM prior: `32 -> 256 -> 32`, six conditioned residual blocks
- Time/class embedding: 128 dimensions
- Sampling: 100 tangent-projected exponential-map steps

CelebA-64 uses an `8 x 8` token grid.

## Main evidence

### 1. Path matching on spherical support

The same spherical autoencoder is used in both rows; only the transport path changes.

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Minimum token norm ↑ |
|---|---:|---:|---:|---:|---:|
| Chord baseline | 67.44 | 0.0740 | 0.840 | 0.085 | 0.705 |
| **SRUL (geodesic RFM)** | **63.84** | **0.0667** | **0.857** | **0.091** | **1.000** |

![Chord and geodesic paths](assets/geometry_schematic.png)

### 2. Support matching with the same VAE

The two VAE rows keep the trained VAE encoder and decoder unchanged. Standard linear FM models the full VAE latent. Projected RFM keeps only token directions and restores a class/token mean radius.

| Method | Latent support and prior | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|
| **SRUL** | Spherical autoencoder + RFM | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | Full VAE latent + linear FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | Directions + class/token mean radius | 56.04 | 0.848 | 0.063 |

## Controlled ablations

The two controls are reported separately.

### Information-control sweep

CFG is fixed at `s = 2`; only `sigma_enc` changes.

| Dataset | Gen. FID at 0.05 | Gen. FID at 0.15 | Gen. FID at 0.30 |
|---|---:|---:|---:|
| CIFAR-10 | **48.71** | 51.61 | 58.16 |
| PathMNIST | 29.88 | **29.35** | 34.33 |
| CelebA-64 | **37.68** | 41.10 | 55.21 |

Large tangent noise (`0.30`) degrades every dataset. The preferred smaller value is dataset dependent.

### Classifier-free-guidance sweep

`sigma_enc` is fixed at `0.15`; only the guidance scale changes.

| Dataset | FID at s=1.0 | FID at s=1.5 | FID at s=2.0 |
|---|---:|---:|---:|
| CIFAR-10 | 62.44 | 54.83 | **49.39** |
| PathMNIST | 29.16 | **28.17** | 29.35 |
| CelebA-64 | 41.13 | 41.15 | **41.10** |

CIFAR-10 benefits strongly from larger guidance, PathMNIST peaks at `1.5`, and CelebA-64 changes little. Recall generally decreases as guidance becomes stronger.

The full PathMNIST and CelebA `3 x 3` grids are retained in `results/`, but the report uses controlled slices so one variable changes at a time.

## Cross-domain reference setting

For a simple transfer comparison, the report uses the same operating point in every row: `sigma_enc = 0.15`, CFG `s = 2`.

| Dataset | Image size | Token grid | Reconstruction FID ↓ | Generation FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 32x32 | 4x4 | 17.31 | 49.39 | 0.884 | 0.059 |
| PathMNIST | 32x32 | 4x4 | 7.13 | 29.36 | 0.829 | 0.309 |
| CelebA-64 | 64x64 | 8x8 | 9.08 | 41.22 | 0.657 | 0.198 |

![Generated samples](assets/generated_samples.png)

## Repository structure

```text
src/         SRUL, VAE-support, coverage-training, and sweep scripts
scripts/     Reproducible command-line launchers
notebooks/   Colab quick-start notebooks
results/     Full grids and controlled reporting slices
assets/      Pipeline, geometry, ablation, and sample figures
docs/        Method, experiments, and reproducibility notes
report/      Final report PDF and LaTeX source
slides/      Final 5-slide deck, PDF, and speaker script
archive/     Earlier and supplementary experiments
```

## Installation

```bash
python -m pip install -r requirements.txt
```

A CUDA GPU is recommended. All long experiments support resumable checkpoints.

## Main launchers

```bash
bash scripts/run_cifar10_geometry.sh /path/to/output/root
bash scripts/run_cifar10_conditional.sh /path/to/output/root /path/to/autoencoder_final.pt
bash scripts/run_cifar10_sigma_sweep.sh /path/to/output/root /path/to/autoencoder_final.pt
bash scripts/run_cifar10_vae_geometry.sh /path/to/output/root
bash scripts/run_pathmnist.sh /path/to/output/root
bash scripts/run_celeba64.sh /path/to/output/root /path/to/dataset/cache
bash scripts/run_pathmnist_knob_cfg.sh /path/to/output/root /path/to/pathmnist_autoencoder.pt
bash scripts/run_celeba64_knob_cfg.sh /path/to/output/root /path/to/celeba_autoencoder.pt /path/to/dataset/cache
bash scripts/run_cifar10_spherical_coverage.sh /path/to/output/root /path/to/base_autoencoder.pt
```

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for complete commands and checkpoint requirements.

## Final artifacts

- [Report](report/SRUL_Final_Report_v33.pdf)
- [Slides](slides/SRUL_Final_5_Slides_v35.pdf)
- [Speaker script](slides/SRUL_5min_Speaker_Script_v33.md)

## Notes

- Checkpoints and datasets are excluded because they are large.
- The current measured FID tables use the base autoencoder runs. The stronger broad-coverage training branch is included as the final method implementation but requires its own rerun before its results can replace the measured tables.
- The main claim is geometry/support matching, not state-of-the-art FID.
