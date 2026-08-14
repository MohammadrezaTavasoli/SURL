# File manifest

## Final code (`src/`)

- `srul_cifar_spatial_tokens_experiment.py`: spatial spherical autoencoder, chord baseline, RFM, and geometry metrics.
- `srul_cifar_conditional_spatial_experiment.py`: conditional RFM, EMA, and CFG on CIFAR-10.
- `srul_medmnist_conditional_experiment.py`: PathMNIST and other MedMNIST datasets.
- `srul_celeba64_conditional_experiment.py`: CelebA-64 cache and 8x8-token experiments.
- `srul_sigma_knob_sweep.py`: CIFAR-10 information-control sweep.
- `srul_cross_dataset_knob_cfg_sweep.py`: PathMNIST/CelebA full 3x3 noise-and-guidance grid.
- `collect_cross_dataset_results.py`: result aggregation helper.
- `srul_sphere_style_uniformization.py`: small/large tangent-rotation training for broader, approximately uniform token coverage.
- `srul_final_baseline_comparison_fixed.py`: checkpoint preparation for SRUL/VAE baselines.
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
- `controlled_noise_sweep_cfg2.csv`: all datasets, CFG fixed at 2.
- `controlled_cfg_sweep_sigma015.csv`: all datasets, noise fixed at 0.15.
- `cross_domain_reference_sigma015_cfg2.csv`: common reference operating point.
- `pathmnist_knob_cfg_full_grid.csv`
- `celeba64_knob_cfg_full_grid.csv`

## Documentation (`docs/`)

- `METHOD.md`
- `RESULTS.md`
- `REPRODUCIBILITY.md`
- `MATHEMATICAL_DETAILS.md`
- `VAE_PRIOR_GEOMETRY.md`
- `FILE_MANIFEST.md`
- `GITHUB_CHECKLIST.md`

## Final artifacts

- `report/SRUL_Final_Report_v33.pdf`
- `report/SRUL_Final_Report_v33.tex`
- `slides/SRUL_Final_5_Slides_v35.pptx`
- `slides/SRUL_Final_5_Slides_v35.pdf`
- `slides/SRUL_5min_Speaker_Script_v33.md`

## Archive

Earlier global-latent and deterministic-autoencoder mismatch experiments are retained under `archive/` and are not part of the main report tables.
