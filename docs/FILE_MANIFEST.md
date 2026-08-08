# File manifest

## Final code (`src/`)

- `srul_cifar_spatial_tokens_experiment.py`: spherical autoencoder, chord baseline, RFM prior, and geometry evaluation.
- `srul_cifar_conditional_spatial_experiment.py`: class conditioning, EMA, and classifier-free guidance on CIFAR-10.
- `srul_medmnist_conditional_experiment.py`: PathMNIST and other MedMNIST datasets.
- `srul_celeba64_conditional_experiment.py`: CelebA-64 data cache, 8x8 spherical tokens, and conditional RFM.
- `srul_sigma_knob_sweep.py`: separate prior training for each tangent-noise value.
- `srul_final_baseline_comparison_fixed.py`: prepares the SRUL and KL-regularized VAE checkpoints used by the final comparison.
- `srul_vae_prior_geometry_comparison.py`: same-VAE comparison of standard linear FM and projected direction-only RFM.

## Launchers (`scripts/`)

- `run_cifar10_geometry.sh`
- `run_cifar10_conditional.sh`
- `run_cifar10_sigma_sweep.sh`
- `run_cifar10_matched_baselines.sh`
- `run_cifar10_vae_geometry.sh`
- `run_pathmnist.sh`
- `run_celeba64.sh`

## Notebooks (`notebooks/`)

- `SRUL_Colab_Quickstart.ipynb`: short smoke test and main commands.

## Results (`results/`)

- `cifar10_geometry.csv`: chord versus geodesic path comparison.
- `cifar10_support_matching_main.csv`: SRUL, VAE + standard FM, and VAE + projected RFM.
- `cifar10_vae_prior_geometry.csv`: complete VAE prior-geometry output table.
- `cifar10_vae_radius_diagnostic.csv`: VAE reconstruction after preserving, averaging, or removing token radius.
- `cifar10_vae_radius_stats.csv`: VAE radius and posterior statistics.
- `cifar10_sigma_sweep.csv`: tangent-noise ablation.
- `cifar10_cfg_sweep.csv`: classifier-free-guidance ablation.
- `cross_domain.csv`: CIFAR-10, PathMNIST, and CelebA-64 results.

## Assets (`assets/`)

- `srul_pipeline.png`: complete representation-learning, RFM-training, and sampling pipeline.
- `geometry_schematic.png`: chord versus geodesic geometry.
- `ablations.png`: tangent-noise and classifier-free-guidance plots.
- `generated_samples.png`: CIFAR-10, PathMNIST, and CelebA-64 samples.

## Extended documentation (`docs/`)

- `METHOD.md`: final method and geometry-matching design.
- `RESULTS.md`: reported numerical results.
- `REPRODUCIBILITY.md`: commands and settings.
- `VAE_PRIOR_GEOMETRY.md`: detailed same-VAE standard-FM versus projected-RFM experiment.

## Archive (`archive/`)

Earlier global-latent experiments and supplementary deterministic-autoencoder mismatch tests. These are not part of the main report or slide deck.

## Final artifacts

- `report/SRUL_Final_Report_v23.pdf`
- `report/SRUL_Final_Report_v23.tex`
- `slides/SRUL_Final_5_Slides_v23.pptx`
- `slides/SRUL_Final_5_Slides_v23.pdf`
- `slides/SRUL_5min_Speaker_Script_v23.md`
