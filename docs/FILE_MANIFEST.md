# File manifest

## Source code (`src/`)

- `srul_cifar_spatial_tokens_experiment.py`: CIFAR-10 spherical autoencoder, chord baseline, RFM prior, and geometry metrics.
- `srul_cifar_conditional_spatial_experiment.py`: class conditioning, EMA, and classifier-free guidance on CIFAR-10.
- `srul_medmnist_conditional_experiment.py`: PathMNIST and other MedMNIST datasets.
- `srul_celeba64_conditional_experiment.py`: CelebA-64 data cache and 8x8-token experiments.
- `srul_sigma_knob_sweep.py`: CIFAR-10 information-control sweep.
- `srul_cross_dataset_knob_cfg_sweep.py`: PathMNIST/CelebA noise-and-guidance grid.
- `collect_cross_dataset_results.py`: result aggregation.
- `srul_spherical_coverage_training.py`: small/large tangent-rotation coverage training.
- `srul_baseline_comparison.py`: standard latent-space reference models and VAE checkpoint preparation.
- `srul_vae_prior_geometry_comparison.py`: VAE + standard FM versus VAE + projected RFM.

## Launchers (`scripts/`)

- `run_cifar10_geometry.sh`
- `run_cifar10_conditional.sh`
- `run_cifar10_sigma_sweep.sh`
- `run_cifar10_matched_baselines.sh`
- `run_cifar10_vae_geometry.sh`
- `run_pathmnist.sh`
- `run_celeba64.sh`
- `run_pathmnist_knob_cfg.sh`
- `run_celeba64_knob_cfg.sh`
- `run_cifar10_spherical_coverage.sh`

## Results (`results/`)

- `cifar10_geometry.csv`
- `cifar10_support_matching_main.csv`
- `cifar10_vae_prior_geometry.csv`
- `cifar10_sigma_sweep.csv`
- `cifar10_cfg_sweep.csv`
- `controlled_noise_sweep_cfg2.csv`
- `controlled_cfg_sweep_sigma015.csv`
- `cross_domain_reference_sigma015_cfg2.csv`
- `pathmnist_knob_cfg_full_grid.csv`
- `celeba64_knob_cfg_full_grid.csv`

## Documentation (`docs/`)

- `METHOD.md`
- `RESULTS.md`
- `REPRODUCIBILITY.md`
- `CHECKPOINTS.md`
- `MATHEMATICAL_DETAILS.md`
- `VAE_SUPPORT_MATCHING.md`
- `FILE_MANIFEST.md`
- `GITHUB_CHECKLIST.md`

## Report and slides

- `report/SRUL_Final_Report.pdf`
- `report/SRUL_Final_Report.tex`
- `slides/SRUL_Final_5_Slides.pptx`
- `slides/SRUL_Final_5_Slides.pdf`
- `slides/SRUL_5min_Speaker_Script.md`

## Model weights

No trained checkpoint files are included. See `docs/CHECKPOINTS.md`.
