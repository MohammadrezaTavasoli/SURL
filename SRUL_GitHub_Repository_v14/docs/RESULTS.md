# Final results

## Controlled geometry comparison on CIFAR-10

The encoder, decoder, source distribution, latent size, prior capacity, and training budget are fixed. Only the probability path changes.

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Min. norm ↑ |
|---|---:|---:|---:|---:|---:|
| Chord baseline | 67.44 | 0.0740 | 0.840 | 0.085 | 0.705 |
| **SRUL** | **63.84** | **0.0667** | **0.857** | **0.091** | **1.000** |

## Tangent-noise ablation

| sigma_enc | Reconstruction FID ↓ | Generation FID ↓ | Recall ↑ |
|---:|---:|---:|---:|
| **0.05** | **17.53** | **48.71** | 0.090 |
| 0.15 | 19.56 | 51.61 | **0.090** |
| 0.30 | 27.85 | 58.16 | 0.073 |

## Classifier-free-guidance ablation

| CFG scale | FID ↓ | Precision ↑ | Recall ↑ |
|---:|---:|---:|---:|
| 1.0 | 62.44 | 0.854 | **0.083** |
| 1.5 | 54.83 | 0.865 | 0.071 |
| **2.0** | **49.39** | **0.884** | 0.059 |

## Cross-domain evaluation

CIFAR-10 is used for the controlled geometry and sampling studies. PathMNIST evaluates transfer to histopathology textures, and CelebA-64 evaluates the larger `64x64` setting with an `8x8` token grid.

| Dataset | Image size | Token grid | Reconstruction FID ↓ | Generation FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 32x32 | 4x4 | 17.53 | 48.71 | 0.861 | 0.090 |
| PathMNIST | 32x32 | 4x4 | 7.13 | 29.36 | 0.829 | 0.309 |
| CelebA-64 | 64x64 | 8x8 | 9.08 | 41.22 | 0.657 | 0.198 |

On CIFAR-10, the chord-to-geodesic comparison measures the effect of transport geometry. PathMNIST shows that the same `4x4` token design can represent texture-dominated tissue images. CelebA-64 shows that the model can scale to a larger image resolution and `8x8` spherical-token grid.
