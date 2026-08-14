# VAE support-matching result

The VAE support experiment keeps the same trained VAE encoder and decoder and changes only the prior treatment.

| Method | FID ↓ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|
| **SRUL** | **49.42** | **0.857** | **0.110** |
| VAE + standard FM | 50.70 | 0.855 | 0.091 |
| VAE + projected RFM | 56.04 | 0.848 | 0.063 |

The main result is the generation gap between VAE + standard FM and VAE + projected RFM. Radius and reconstruction diagnostics are supporting evidence: the VAE uses variable token magnitudes, so a direction-only projection does not preserve the full representation. Full values are documented in `VAE_PRIOR_GEOMETRY.md`.

Full FID, KID, precision, and recall values are available in `results/cifar10_vae_prior_geometry.csv`.
