# File manifest

## Final code (`src/`)

- `srul_cifar_spatial_tokens_experiment.py`: spherical autoencoder, chord baseline, RFM prior, and geometry evaluation.
- `srul_cifar_conditional_spatial_experiment.py`: class conditioning, EMA, and classifier-free guidance on CIFAR-10.
- `srul_medmnist_conditional_experiment.py`: PathMNIST and other MedMNIST datasets.
- `srul_celeba64_conditional_experiment.py`: CelebA-64 data streaming/cache, 8x8 spherical tokens, and conditional RFM.
- `srul_sigma_knob_sweep.py`: separate prior training for each tangent-noise value.

## Launchers (`scripts/`)

Shell scripts containing the reported hyperparameters.

## Notebooks (`notebooks/`)

- `SRUL_Colab_Quickstart.ipynb`: short smoke test and main commands.

## Assets (`assets/`)

- `srul_pipeline.png`: complete representation-learning, RFM-training, and sampling pipeline.
- `geometry_schematic.png`: chord versus geodesic geometry.
- `ablations.png`: tangent-noise and classifier-free-guidance plots.
- `generated_samples.png`: CIFAR-10, PathMNIST, and CelebA-64 samples.

## Archive (`archive/`)

Earlier global-latent experiments, external reference baselines, and the full development log. These are not the final method.

## Final artifacts

- `assets/srul_pipeline.png`: complete representation, prior-training, and sampling pipeline.
- `assets/geometry_schematic.png`: controlled chord-versus-geodesic diagram.
- `report/SRUL_Final_Report.pdf`: final report.
- `slides/SRUL_Final_5_Slides.pdf`: final five-slide presentation.
- `slides/SRUL_5min_Speaker_Script.md`: presentation script.
