# Results

## Path matching on spherical support

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Min. norm ↑ |
|---|---:|---:|---:|---:|---:|
| Chord baseline | 67.44 | 0.0740 | 0.840 | 0.085 | 0.705 |
| **SRUL** | **63.84** | **0.0667** | **0.857** | **0.091** | **1.000** |

The spherical autoencoder is unchanged. The improvement isolates the benefit of geodesic transport over an off-manifold chord.

## Support matching with the same VAE

| Method | FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| **SRUL** | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | 56.04 | 0.848 | 0.063 |

The projected method loses radial information because the VAE was not trained on spherical support.

## Controlled information-control sweep

CFG is fixed at `s=2`.

| Dataset | FID at sigma=0.05 | FID at sigma=0.15 | FID at sigma=0.30 |
|---|---:|---:|---:|
| CIFAR-10 | **48.71** | 51.61 | 58.16 |
| PathMNIST | 29.88 | **29.35** | 34.33 |
| CelebA-64 | **37.68** | 41.10 | 55.21 |

Noise scale `0.30` is too destructive on all three datasets.

## Controlled CFG sweep

`sigma_enc=0.15` is fixed.

| Dataset | FID at s=1.0 | FID at s=1.5 | FID at s=2.0 | Recall at s=1.0 | Recall at s=1.5 | Recall at s=2.0 |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 62.44 | 54.83 | **49.39** | **0.083** | 0.071 | 0.059 |
| PathMNIST | 29.16 | **28.17** | 29.35 | **0.353** | 0.333 | 0.295 |
| CelebA-64 | 41.13 | 41.15 | **41.10** | 0.214 | **0.215** | 0.208 |

The best guidance scale is dataset dependent. Stronger guidance generally lowers recall, but its FID effect varies.

## Common cross-domain reference setting

All rows use `sigma_enc=0.15` and CFG `s=2`.

| Dataset | Token grid | Reconstruction FID ↓ | Generation FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| CIFAR-10 | 4x4 | 17.31 | 49.39 | 0.884 | 0.059 |
| PathMNIST | 4x4 | 7.13 | 29.36 | 0.829 | 0.309 |
| CelebA-64 | 8x8 | 9.08 | 41.22 | 0.657 | 0.198 |

The complete PathMNIST and CelebA 3x3 grids are stored in `results/pathmnist_knob_cfg_full_grid.csv` and `results/celeba64_knob_cfg_full_grid.csv`.
