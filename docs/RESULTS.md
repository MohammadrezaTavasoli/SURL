# Final results

## 1. Encoder-prior matching

### A. Path match on spherical support

The spherical encoder, decoder, source distribution, latent size, prior capacity, and training budget are the same. Only the path changes.

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Min. norm ↑ |
|---|---:|---:|---:|---:|---:|
| Chord baseline | 67.44 | 0.0740 | 0.840 | 0.085 | 0.705 |
| **SRUL** | **63.84** | **0.0667** | **0.857** | **0.091** | **1.000** |

The chord enters the sphere interior. SRUL follows the geodesic and keeps every token at unit norm.

### B. Support match with the same VAE

The two VAE variants use the same trained encoder and decoder. Standard FM models the complete VAE latent. Projected RFM keeps only token directions and restores a class/token mean radius.

| Method | Latent support and prior | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|
| **SRUL** | Spherical autoencoder + RFM | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | Full VAE latent + linear FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | Directions + class/token mean radius | 56.04 | 0.848 | 0.063 |

The generation table is the main support-matching result. A separate radius diagnostic explains the gap: the VAE uses variable token radii, so post-hoc direction-only projection removes part of the representation. The detailed reconstruction values are kept as supporting evidence in `docs/VAE_PRIOR_GEOMETRY.md`.

## 2. Information-control ablation

| sigma_enc | Reconstruction FID ↓ | Generation FID ↓ | Recall ↑ |
|---:|---:|---:|---:|
| **0.05** | **17.53** | **48.71** | 0.090 |
| 0.15 | 19.56 | 51.61 | **0.090** |
| 0.30 | 27.85 | 58.16 | 0.073 |

The tangent-noise scale is the information-control knob. Larger values perturb the latent directions more strongly and retain less image-specific information. In this sweep, increasing the knob worsens both reconstruction and generation; `0.05` is the best tested value.

## 3. Classifier-free-guidance ablation

| CFG scale | FID ↓ | Precision ↑ | Recall ↑ |
|---:|---:|---:|---:|
| 1.0 | 62.44 | 0.854 | **0.083** |
| 1.5 | 54.83 | 0.865 | 0.071 |
| **2.0** | **49.39** | **0.884** | 0.059 |

## 4. Evaluation settings

CIFAR-10 is used for controlled geometry, support-matching, information-control, and guidance studies. PathMNIST tests histopathology texture. CelebA-64 tests the larger `64x64` setting with an `8x8` token grid.

| Dataset | Image size | Token grid | Reconstruction FID ↓ | Generation FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 32x32 | 4x4 | 17.53 | 48.71 | 0.861 | 0.090 |
| PathMNIST | 32x32 | 4x4 | 7.13 | 29.36 | 0.829 | 0.309 |
| CelebA-64 | 64x64 | 8x8 | 9.08 | 41.22 | 0.657 | 0.198 |
