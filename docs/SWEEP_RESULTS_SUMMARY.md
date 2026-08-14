# Cross-dataset sweep summary

The PathMNIST and CelebA-64 runs evaluated the complete Cartesian grid of three `sigma_enc` values and three CFG scales. For the final paper, the results are interpreted through two controlled one-variable slices.

## Information control: CFG fixed at 2

- CIFAR-10: best generation FID 48.71 at `sigma_enc = 0.05`.
- PathMNIST: best generation FID 29.35 at `sigma_enc = 0.15`; reconstruction is slightly better at `0.05`.
- CelebA-64: best generation FID 37.68 at `sigma_enc = 0.05`.
- `sigma_enc = 0.30` is clearly worse on all three datasets.

## Guidance: `sigma_enc = 0.15` fixed

- CIFAR-10: best FID 49.39 at `s = 2.0`.
- PathMNIST: best FID 28.17 at `s = 1.5`.
- CelebA-64: FID changes little; the lowest value is 41.10 at `s = 2.0`.
- Recall generally decreases as guidance becomes stronger.

## Reporting rule

Do not combine the separate minima into one joint optimum. The paper reports the two controlled slices independently. For the cross-domain transfer table, all datasets use the same reference setting, `sigma_enc = 0.15` and CFG `s = 2`, stored in `results/cross_domain_reference_sigma015_cfg2.csv`.
