# Learning Spatial Unit-Norm Latents with Riemannian Flow Matching

Latent generative models often use a Gaussian prior, but an autoencoder does not have to produce Gaussian latents. Its architecture and normalization determine the latent geometry. Normalized features can have nearly constant norm, so direction carries most of the information.

We call the proposed framework **SRUL**, short for **Spherical Riemannian Unit-Norm Latents**. The name reflects the unit-norm support learned by the autoencoder and the Riemannian prior used to model it.

SRUL studies a simple design rule:

> **The prior should match the latent support learned by the autoencoder.**

SRUL trains an autoencoder to produce a spatial grid of unit-norm tokens, uses tangent-space noise as an explicit information-control knob, and then learns their distribution with Riemannian Flow Matching (RFM) on the same product-of-spheres support. The knob changes how much image-specific information is retained without changing the spherical support.

![SRUL pipeline](assets/srul_pipeline.png)

## Main evidence

### 1. The transport path should match spherical support

The spherical encoder, decoder, source distribution, latent size, prior network, and training budget are the same. Only the path changes.

| Method | FID ↓ | KID ↓ | Precision ↑ | Recall ↑ | Minimum token norm ↑ |
|---|---:|---:|---:|---:|---:|
| Chord baseline | 67.44 | 0.0740 | 0.840 | 0.085 | 0.705 |
| **SRUL (geodesic RFM)** | **63.84** | **0.0667** | **0.857** | **0.091** | **1.000** |

![Chord and geodesic paths](assets/geometry_schematic.png)

### 2. The prior should match the support learned by the autoencoder

A VAE is trained in Euclidean latent space. We keep the same trained VAE encoder and decoder, then compare two priors. Standard FM models the complete VAE latent. Projected RFM keeps only token directions and restores a class/token mean radius. SRUL is shown in the same conditional evaluation because its spherical support is learned before RFM training.

| Method | Latent support and prior | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---:|---:|---:|
| **SRUL** | Spherical autoencoder + RFM | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | Full VAE latent + linear FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | Directions + class/token mean radius | 56.04 | 0.848 | 0.063 |

The projected-RFM result is worse because the VAE was trained in Euclidean latent space and uses information beyond token direction. Radius diagnostics support this interpretation; the full reconstruction test is reported in [`docs/VAE_PRIOR_GEOMETRY.md`](docs/VAE_PRIOR_GEOMETRY.md). SRUL avoids post-hoc projection by learning spherical support before its prior is trained.

## Information and sampling controls

### Tangent-noise information control

| `sigma_enc` | Reconstruction FID ↓ | Generation FID ↓ |
|---:|---:|---:|
| **0.05** | **17.53** | **48.71** |
| 0.15 | 19.56 | 51.61 |
| 0.30 | 27.85 | 58.16 |

`sigma_enc` is a controlled information knob rather than an exact bitrate in bits. Larger values rotate tokens farther from their clean directions. In the tested range, this removed too much useful information, so `0.05` gave the best reconstruction and generation quality.

### Classifier-free guidance

| CFG scale | FID ↓ | Precision ↑ | Recall ↑ |
|---:|---:|---:|---:|
| 1.0 | 62.44 | 0.854 | **0.083** |
| 1.5 | 54.83 | 0.865 | 0.071 |
| **2.0** | **49.39** | **0.884** | 0.059 |

## Evaluation settings

CIFAR-10 is used for controlled geometry, support-matching, information-control, and guidance studies. PathMNIST tests histopathology textures. CelebA-64 tests a larger `64x64` setting with an `8x8` token grid.

| Dataset | Image size | Token grid | Reconstruction FID ↓ | Generation FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 32x32 | 4x4 | 17.53 | 48.71 | 0.861 | 0.090 |
| PathMNIST | 32x32 | 4x4 | 7.13 | 29.36 | 0.829 | 0.309 |
| CelebA-64 | 64x64 | 8x8 | 9.08 | 41.22 | 0.657 | 0.198 |

![Generated samples](assets/generated_samples.png)

## Method summary

A fuller derivation of the autoencoder losses, standard Flow Matching, SLERP, exponential-map sampling, and CFG is available in [`docs/MATHEMATICAL_DETAILS.md`](docs/MATHEMATICAL_DETAILS.md).


For each spatial token,

```text
h = Encoder(x)
z[i,j] = h[i,j] / ||h[i,j]||
```

Tangent noise changes direction without changing radius and controls retained image information:

```text
u = xi - <xi,z>z
z_sigma = Exp_z(sigma_enc * u)
```

The prior follows token-wise SLERP paths, predicts tangent velocities, and uses exponential-map integration during sampling. Class labels and classifier-free guidance are optional conditional inputs.

## Repository structure

```text
src/         Final SRUL, VAE-support, conditional, and ablation scripts
scripts/     Reproducible command-line launchers
notebooks/   Colab quickstart notebook
results/     Final numerical tables
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

A CUDA GPU is strongly recommended. The scripts were developed for Google Colab and save resumable checkpoints when `--resume` is used.

## Reproduce the reported experiments

```bash
bash scripts/run_cifar10_geometry.sh /path/to/output/root
bash scripts/run_cifar10_conditional.sh /path/to/output/root /path/to/autoencoder_final.pt
bash scripts/run_cifar10_sigma_sweep.sh /path/to/output/root /path/to/autoencoder_final.pt
bash scripts/run_cifar10_matched_baselines.sh /path/to/output/root
bash scripts/run_cifar10_vae_geometry.sh /path/to/output/root
bash scripts/run_pathmnist.sh /path/to/output/root
bash scripts/run_celeba64.sh /path/to/output/root /path/to/dataset/cache
```

The VAE comparison requires checkpoint paths described in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Report and presentation

- [Final report](report/SRUL_Final_Report_v26.pdf)
- [Final slides](slides/SRUL_Final_5_Slides_v26.pdf)
- [Speaker script](slides/SRUL_5min_Speaker_Script_v26.md)

## Notes

- Checkpoints and datasets are excluded from Git because they are large.
- The main claim is support matching between the autoencoder and prior, not state-of-the-art FID.
- The deterministic autoencoder mismatch check remains in `archive/` as supplementary evidence.
- CelebA is subject to its dataset license and is intended for non-commercial research use.
